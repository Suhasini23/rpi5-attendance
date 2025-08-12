import os, json, time, sqlite3
import numpy as np
import cv2
from flask import Flask, Response, jsonify

from utils.yunet import YuNet, ensure_model
from utils.align import align_face
from utils.pose import head_pose_angles
from utils.debounce import Debounce

MODEL_IN = "app/models/lbph_model.yml"
LABELS_IN = "app/models/label_map.json"
DB_PATH = "app/db/attendance.sqlite"
YUNET_MODEL = "app/models/yunet.onnx"

YAW_TH, PITCH_TH = 15.0, 15.0
PROC_W = 320

app = Flask(__name__)

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
    cur.execute("INSERT INTO logs(name, ts) VALUES (?,?)", (name, int(time.time())))
    con.commit(); con.close()

def load_models():
    # LBPH + labels
    if not os.path.exists(MODEL_IN) or not os.path.exists(LABELS_IN):
        raise RuntimeError("Model or labels missing. Run: python app/enroll.py")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_IN)
    label_map = json.load(open(LABELS_IN))
    inv_labels = {v:k for k,v in label_map.items()}

    ensure_model(YUNET_MODEL)
    det = YuNet(YUNET_MODEL, input_size=(PROC_W, PROC_W), conf_threshold=0.6)
    return recognizer, inv_labels, det

recognizer, inv_labels, det = load_models()
init_db()
deb = Debounce(cooldown_sec=120)

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    while True:
        ok, frame = cap.read()
        if not ok: break

        # downscale for speed
        h, w = frame.shape[:2]
        scale = PROC_W / float(w)
        small = cv2.resize(frame, (PROC_W, int(h*scale)))

        detections = det.detect(small)
        if detections:
            # pick top detection only (attendance scenario – close-up kiosk)
            detections.sort(key=lambda d: d[2], reverse=True)
            (x,y,ww,hh), lmk, score = detections[0]

            # Map back to full-res coords (for drawing)
            sx, sy = w/PROC_W, h/small.shape[0]
            X,Y,W,H = int(x*sx), int(y*sy), int(ww*sx), int(hh*sy)
            color = (0,0,255); name_txt = "Ignoring"

            # Landmarks in small image space -> crop ROI from small for alignment & pose
            roi_small = small[y:y+hh, x:x+ww].copy()
            # shift landmarks to ROI space
            pts5_small = [(int(px - x), int(py - y)) for (px,py) in lmk]

            # Head pose on ROI (frontal-only gate)
            yaw, pitch, roll = head_pose_angles(roi_small, pts5_small)
            if yaw is not None and abs(yaw) <= YAW_TH and abs(pitch) <= PITCH_TH:
                # Align and recognize
                aligned = align_face(roi_small, pts5_small, out_size=100)
                pred_id, conf = recognizer.predict(aligned)
                if conf < 60:
                    name = inv_labels.get(pred_id, "Unknown")
                    name_txt = f"{name} ({conf:.1f})"
                    color = (0,255,0)
                    if deb.ok(name):
                        log_attendance(name)
                else:
                    name_txt = f"Unknown ({conf:.1f})"
                    color = (0,0,255)
            else:
                name_txt = "Not frontal"

            # draw on full frame
            cv2.rectangle(frame, (X,Y), (X+W,Y+H), color, 2)
            cv2.putText(frame, name_txt, (X, Y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buf = cv2.imencode(".jpg", frame)
        if not ret: 
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logs")
def logs():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT name, ts FROM logs ORDER BY ts DESC LIMIT 100")
    rows = [{"name": r[0], "ts": r[1]} for r in cur.fetchall()]
    con.close()
    return rows

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
