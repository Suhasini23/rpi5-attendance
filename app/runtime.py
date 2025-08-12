import os
import json
import time
import sqlite3
import re
import numpy as np
import cv2
from queue import Queue
from threading import Lock
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename
from utils.align import align_face
from utils.pose import head_pose_angles

"""
This runtime module starts a Flask web application that serves video frames,
performs face detection/recognition, and logs attendance.  It uses a
CameraManager instance to abstract away the complexities of configuring
Picamera2 on Raspberry Pi 5 industrial systems (and falls back to OpenCV
automatically if Picamera2 is unavailable).

The CameraManager handles pixel format detection, colour-space correction
and transforms, so we avoid hard-coding a single format (e.g. RGB888) and
fix the common red/blue swap and purple-tint issues.

To customise the camera settings (resolution, framerate, flips), adjust the
CameraManager instantiation below.
"""

# Import our camera manager
from camera_manager import CameraManager

# Paths / models
MODEL_IN    = "app/models/lbph_model.yml"
LABELS_IN   = "app/models/label_map.json"
DB_PATH     = "app/db/attendance.sqlite"
YUNET_MODEL = "app/models/yunet.onnx"

# Training storage
TRAIN_DIR   = "app/data/training_data"
MODEL_PATH  = "app/models/lbph_model.yml"
LABELS_PATH = "app/models/label_map.json"
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")  # tuple for endswith

# Runtime knobs
YAW_TH, PITCH_TH = 30.0, 30.0  # More lenient thresholds for better recognition
PROC_W = 320

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # ~5MB uploads

# ----------------- Camera initialisation -----------------
# Configure the camera.  Adjust width/height/framerate/flip as needed.
camera_manager = CameraManager(
    width=640,
    height=480,
    framerate=20,
    camera_port='CSI0',
    autofocus=False,
    awb_mode='auto',
    exposure_mode='auto',
    horizontal_flip=False,
    vertical_flip=False,
    debug=False
)

# ----------------- Realtime (SSE + snapshot) -----------------
events_q = Queue(maxsize=200)
latest = {"name": None, "conf": None, "yaw": None, "pitch": None, "box": None, "ts": None}
latest_lock = Lock()

def publish(evt: dict):
    evt.setdefault("ts", int(time.time()))
    try:
        events_q.put_nowait(evt)
    except Exception:
        try: events_q.get_nowait()
        except Exception: pass
        try: events_q.put_nowait(evt)
        except Exception: pass
    with latest_lock:
        latest.update({
            "name":  evt.get("name"),
            "conf":  evt.get("conf"),
            "yaw":   evt.get("yaw"),
            "pitch": evt.get("pitch"),
            "box":   evt.get("box"),
            "ts":    evt["ts"]
        })

# ----------------- DB -----------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        ts INTEGER
    )""")
    con.commit(); con.close()

def log_attendance(name):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    ts = int(time.time())
    cur.execute("INSERT INTO logs(name, ts) VALUES (?,?)", (name, ts))
    con.commit(); con.close()
    publish({"type": "attendance_logged", "name": name, "ts": ts})

# ----------------- Models -----------------
def load_models():
    if not os.path.exists(MODEL_IN) or not os.path.exists(LABELS_IN):
        raise RuntimeError("Model or labels missing. Run: python app/enroll.py")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_IN)
    label_map = json.load(open(LABELS_IN))
    inv_labels = {v: k for k, v in label_map.items()}
    # Ensure the YuNet model exists
    from utils.yunet import ensure_model
    ensure_model(YUNET_MODEL)
    # Load YuNet detector
    from utils.yunet import YuNet
    det = YuNet(YUNET_MODEL, input_size=(PROC_W, PROC_W), conf_threshold=0.6)
    return recognizer, inv_labels, det

recognizer, inv_labels, det = load_models()
init_db()
from utils.debounce import Debounce
deb = Debounce(cooldown_sec=120)

# ----------------- Video loop (MJPEG producer) -----------------
def gen_frames():
    """
    Generator that yields JPEG frames for the MJPEG stream.  Frames are
    captured using the CameraManager and passed through the detection
    pipeline.  Colour conversion is handled inside CameraManager, so no
    additional conversion is necessary here.
    """
    # Ensure the camera is initialised (will do nothing if already done)
    if not camera_manager.is_initialized:
        camera_manager.initialize()

    while True:
        frame = camera_manager.read_frame()
        if frame is None:
            # If no frame is captured, wait a bit and try again
            time.sleep(0.01)
            continue

        # ---- detection + overlay ----
        process_and_emit(frame)

        # MJPEG encode & yield
        try:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            chunk = buf.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(chunk)).encode() + b"\r\n\r\n"
                + chunk + b"\r\n"
            )
        except Exception:
            # On encoding error, skip frame
            continue

# ----------------- Vision pipeline -----------------
def process_and_emit(frame):
    """Run detection/recognition on the frame in-place and emit SSE events."""
    h, w = frame.shape[:2]
    scale = PROC_W / float(w)
    small = cv2.resize(frame, (PROC_W, int(h * scale)))

    detections = det.detect(small)
    if not detections:
        return

    detections.sort(key=lambda d: d[2], reverse=True)
    (x, y, ww, hh), lmk, score = detections[0]

    sx, sy = w / PROC_W, h / small.shape[0]
    X, Y, W, H = int(x * sx), int(y * sy), int(ww * sx), int(hh * sy)

    roi_small = small[y:y+hh, x:x+ww].copy()
    pts5_small = [(int(px - x), int(py - y)) for (px, py) in lmk]

    name_txt = "Not frontal"
    color = (0, 0, 255)
    yaw = pitch = None

    yaw, pitch, _ = head_pose_angles(roi_small, pts5_small)
    
    # Debug: Show the detected angles
    if yaw is not None and pitch is not None:
        print(f"Debug: Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}° (thresholds: ±{YAW_TH}°, ±{PITCH_TH}°)")
    
    if yaw is not None and abs(yaw) <= YAW_TH and abs(pitch) <= PITCH_TH:
        aligned = align_face(roi_small, pts5_small, out_size=100)
        pred_id, conf = recognizer.predict(aligned)
        if conf < 80:  # LBPH: lower values are better, so increase threshold
            name = inv_labels.get(pred_id, "Unknown")
            name_txt = f"{name} ({conf:.1f})"
            color = (0, 255, 0)
            if deb.ok(name):
                log_attendance(name)
        else:
            name = "Unknown"
            name_txt = f"{name} ({conf:.1f})"
            color = (0, 0, 255)

        publish({
            "type": "detection",
            "name": name,
            "conf": float(conf),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "box": {"x": X, "y": Y, "w": W, "h": H}
        })
    else:
        publish({
            "type": "detection",
            "name": "NotFrontal",
            "conf": None,
            "yaw": float(yaw) if yaw is not None else None,
            "pitch": float(pitch) if pitch is not None else None,
            "box": {"x": X, "y": Y, "w": W, "h": H}
        })

    cv2.rectangle(frame, (X, Y), (X+W, Y+H), color, 2)
    cv2.putText(frame, name_txt, (X, Y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ----------------- Upload & retrain helpers -----------------
def _safe_person_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip()
    return s or "Unknown"

def _save_uploaded_images(person_name: str, files):
    os.makedirs(os.path.join(TRAIN_DIR, person_name), exist_ok=True)
    saved = []
    ts = int(time.time())
    suffixes = ["front", "left", "right"]
    for i, f in enumerate(files[:3]):
        filename = secure_filename(f.filename or f"img{i+1}.jpg")
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise ValueError(f"Unsupported file type: {ext}")
        out_name = f"{suffixes[i]}_{ts}{ext}"
        out_path = os.path.join(TRAIN_DIR, person_name, out_name)
        f.save(out_path)
        saved.append(out_path)
    return saved

def _detect_align_for_training(det_local: any, img_bgr):
    dets = det_local.detect(img_bgr)
    if not dets:
        return None
    dets.sort(key=lambda d: d[2], reverse=True)
    (x, y, w, h), lmk, _ = dets[0]
    roi = img_bgr[y:y+h, x:x+w]
    pts5 = [(int(px - x), int(py - y)) for (px, py) in lmk]
    return align_face(roi, pts5, out_size=100)

def _retrain_lbph_from_disk():
    # Ensure model exists
    from utils.yunet import ensure_model
    ensure_model(YUNET_MODEL)
    from utils.yunet import YuNet
    det_local = YuNet(YUNET_MODEL, input_size=(PROC_W, PROC_W), conf_threshold=0.6)

    faces, labels = [], []
    label_map, next_id = {}, 0
    os.makedirs(TRAIN_DIR, exist_ok=True)

    for person in sorted(os.listdir(TRAIN_DIR)):
        person_dir = os.path.join(TRAIN_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        if person not in label_map:
            label_map[person] = next_id; next_id += 1
        lbl = label_map[person]
        for fn in os.listdir(person_dir):
            if not fn.lower().endswith(ALLOWED_EXT):
                continue
            path = os.path.join(person_dir, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = PROC_W / float(w)
            small = cv2.resize(img, (PROC_W, int(h * scale)))
            aligned = _detect_align_for_training(det_local, small)
            if aligned is None:
                continue
            faces.append(aligned)
            labels.append(lbl)

    if not faces:
        raise RuntimeError("No usable faces found in training data.")

    rec = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    rec.train(faces, np.array(labels, dtype=np.int32))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    rec.write(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    global recognizer, inv_labels
    recognizer = rec
    inv_labels = {v: k for k, v in label_map.items()}

    return {"persons": len(label_map), "samples": len(faces)}

# ----------------- Routes -----------------
@app.route("/video")
def video():
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
        direct_passthrough=True,
    )

@app.route("/events")
def events():
    @stream_with_context
    def gen():
        yield "retry: 1000\n\n"
        while True:
            evt = events_q.get()
            yield f"data: {json.dumps(evt)}\n\n"
    return Response(gen(), mimetype="text/event-stream")

@app.route("/api/users/<name>/photos", methods=["POST"])
def api_upload_photos(name):
    person = _safe_person_name(name)
    files = request.files.getlist("images[]") or request.files.getlist("images")
    if not files or len(files) < 3:
        return jsonify({"ok": False, "error": "Provide 3 images as images[]"}), 400
    try:
        saved_paths = _save_uploaded_images(person, files)
        stats = _retrain_lbph_from_disk()
        return jsonify({"ok": True, "saved": saved_paths, "retrained": stats})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/logs")
def logs():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT name, ts FROM logs ORDER BY ts DESC LIMIT 100")
    rows = [{"name": r[0], "ts": r[1]} for r in cur.fetchall()]
    con.close()
    return rows

@app.route("/latest")
def latest_detection():
    with latest_lock:
        return jsonify(latest)

if __name__ == "__main__":
    # Optionally pre-initialize the camera here
    # camera_manager.initialize()
    app.run(host="0.0.0.0", port=5000, threaded=True)
