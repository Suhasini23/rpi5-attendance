# app/runtime.py
import os, json, time, sqlite3, re
import numpy as np
import cv2
from queue import Queue
from threading import Lock
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

# Import picamera2 for better Pi camera support
try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available, falling back to OpenCV")

from utils.yunet import YuNet, ensure_model
from utils.align import align_face
from utils.pose import head_pose_angles
from utils.debounce import Debounce

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
YAW_TH, PITCH_TH = 15.0, 15.0
PROC_W = 320

app = Flask(__name__)
# Cap uploads to ~5MB (tune as needed)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# ----------------- Realtime (SSE + snapshot) -----------------
events_q = Queue(maxsize=200)
latest = {"name": None, "conf": None, "yaw": None, "pitch": None, "box": None, "ts": None}
latest_lock = Lock()

def publish(evt: dict):
    """Push event to SSE and update snapshot."""
    evt.setdefault("ts", int(time.time()))
    try:
        events_q.put_nowait(evt)
    except:
        # queue full: drop oldest and retry once
        try: events_q.get_nowait()
        except: pass
        try: events_q.put_nowait(evt)
        except: pass
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

    ensure_model(YUNET_MODEL)
    det = YuNet(YUNET_MODEL, input_size=(PROC_W, PROC_W), conf_threshold=0.6)
    return recognizer, inv_labels, det

recognizer, inv_labels, det = load_models()
init_db()
deb = Debounce(cooldown_sec=120)

# ----------------- Video loop (MJPEG producer) -----------------
def gen_frames():
    if PICAMERA2_AVAILABLE:
        return gen_frames_picamera2()
    else:
        return gen_frames_opencv()

def gen_frames_picamera2():
    """Generate frames using picamera2 (recommended for Pi)"""
    try:
        # Initialize picamera2
        picam2 = Picamera2()
        
        # Configure camera with better color settings
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"},
            controls={
                "FrameDurationLimits": (33333, 33333),  # 30 FPS
                "AeEnable": 1,  # Auto exposure (1 = True, 0 = False)
                "AwbEnable": 1,  # Auto white balance (1 = True, 0 = False)
                "AeExposureMode": 0,  # Normal exposure mode
                "AwbMode": 0  # Auto white balance mode
            }
        )
        picam2.configure(config)
        
        # Start camera
        picam2.start()
        print("[Camera] picamera2 initialized successfully")
        
        # Warm up and set initial controls
        time.sleep(2)
        
        # Try to set better color balance
        try:
            picam2.set_controls({"AwbMode": 0})
            picam2.set_controls({"AeEnable": 1})
        except:
            pass  # Some controls might not be available
        
        while True:
            try:
                # Capture frame
                frame = picam2.capture_array()
                
                if frame is None or frame.size == 0:
                    print("[Camera] Empty frame from picamera2")
                    time.sleep(0.1)
                    continue
                
                # Ensure proper color format (BGR to RGB if needed)
                if len(frame.shape) == 3:
                    # Convert BGR to RGB for better color representation
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert back to BGR for OpenCV compatibility
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Process frame for face detection
                h, w = frame.shape[:2]
                scale = PROC_W / float(w)
                small = cv2.resize(frame, (PROC_W, int(h * scale)))
                
                detections = det.detect(small)
                if detections:
                    detections.sort(key=lambda d: d[2], reverse=True)  # highest score
                    (x, y, ww, hh), lmk, score = detections[0]

                    # Map to original frame coords (for drawing)
                    sx, sy = w / PROC_W, h / small.shape[0]
                    X, Y, W, H = int(x * sx), int(y * sy), int(ww * sx), int(hh * sy)

                    # Work on ROI (small) for alignment/pose
                    roi_small = small[y:y+hh, x:x+ww].copy()
                    pts5_small = [(int(px - x), int(py - y)) for (px, py) in lmk]

                    name_txt = "Not frontal"
                    color = (0, 0, 255)
                    conf_val = None
                    yaw = pitch = None

                    # frontal-only gate via head pose
                    yaw, pitch, roll = head_pose_angles(roi_small, pts5_small)
                    if yaw is not None and abs(yaw) <= YAW_TH and abs(pitch) <= PITCH_TH:
                        aligned = align_face(roi_small, pts5_small, out_size=100)
                        pred_id, conf = recognizer.predict(aligned)
                        conf_val = float(conf)
                        if conf < 60:
                            name = inv_labels.get(pred_id, "Unknown")
                            name_txt = f"{name} ({conf:.1f})"
                            color = (0, 255, 0)
                            if deb.ok(name):
                                log_attendance(name)
                        else:
                            name = "Unknown"
                            name_txt = f"{name} ({conf:.1f})"
                            color = (0, 0, 255)

                        # emit detection event
                        publish({
                            "type": "detection",
                            "name": name,
                            "conf": conf_val,
                            "yaw": float(yaw),
                            "pitch": float(pitch),
                            "box": {"x": X, "y": Y, "w": W, "h": H}
                        })
                    else:
                        # non-frontal
                        publish({
                            "type": "detection",
                            "name": "NotFrontal",
                            "conf": None,
                            "yaw": float(yaw) if yaw is not None else None,
                            "pitch": float(pitch) if pitch is not None else None,
                            "box": {"x": X, "y": Y, "w": W, "h": H}
                        })

                    # draw overlay
                    cv2.rectangle(frame, (X, Y), (X+W, Y+H), color, 2)
                    cv2.putText(frame, name_txt, (X, Y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Encode JPEG
                ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    print("[Camera] Failed to encode frame")
                    continue
                chunk = buf.tobytes()
                
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(chunk)).encode() + b"\r\n"
                    b"\r\n" + chunk + b"\r\n"
                )
                
            except Exception as e:
                print(f"[Camera] Error in picamera2 loop: {e}")
                time.sleep(0.1)
                continue
                
    except Exception as e:
        print(f"[Camera] Failed to initialize picamera2: {e}")
        print("[Camera] Falling back to OpenCV method")
        return gen_frames_opencv()

def gen_frames_opencv():
    """Fallback method using OpenCV (for compatibility)"""
    # Try different camera devices - modern Pi models often use different device numbers
    camera_devices = [0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    cap = None
    for device in camera_devices:
        try:
            print(f"[Camera] Trying OpenCV device {device}...")
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                # Set camera properties for better compatibility
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                time.sleep(0.5)  # warm up
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    print(f"[Camera] Successfully opened OpenCV device {device}")
                    print(f"[Camera] Frame size: {frame.shape}")
                    break
                else:
                    print(f"[Camera] Device {device} opened but couldn't read frame")
                    cap.release()
                    cap = None
            else:
                print(f"[Camera] Device {device} not accessible")
        except Exception as e:
            print(f"[Camera] Error with device {device}: {e}")
            if cap:
                cap.release()
                cap = None

    if not cap or not cap.isOpened():
        error_msg = "No camera available on any device. Please check:"
        error_msg += "\n1. Camera module is enabled in raspi-config"
        error_msg += "\n2. User has video group permissions"
        error_msg += "\n3. Camera is properly connected"
        error_msg += "\n4. No other application is using the camera"
        print(f"[Camera] {error_msg}")
        raise RuntimeError(error_msg)

    print(f"[Camera] Starting OpenCV video stream with device {device}")
    
    while True:
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[Camera] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            # downscale for detection speed
            h, w = frame.shape[:2]
            scale = PROC_W / float(w)
            small = cv2.resize(frame, (PROC_W, int(h * scale)))

            detections = det.detect(small)
            if detections:
                detections.sort(key=lambda d: d[2], reverse=True)  # highest score
                (x, y, ww, hh), lmk, score = detections[0]

                # Map to original frame coords (for drawing)
                sx, sy = w / PROC_W, h / small.shape[0]
                X, Y, W, H = int(x * sx), int(y * sy), int(ww * sx), int(hh * sy)

                # Work on ROI (small) for alignment/pose
                roi_small = small[y:y+hh, x:x+ww].copy()
                pts5_small = [(int(px - x), int(py - y)) for (px, py) in lmk]

                name_txt = "Not frontal"
                color = (0, 0, 255)
                conf_val = None
                yaw = pitch = None

                # frontal-only gate via head pose
                yaw, pitch, roll = head_pose_angles(roi_small, pts5_small)
                if yaw is not None and abs(yaw) <= YAW_TH and abs(pitch) <= PITCH_TH:
                    aligned = align_face(roi_small, pts5_small, out_size=100)
                    pred_id, conf = recognizer.predict(aligned)
                    conf_val = float(conf)
                    if conf < 60:
                        name = inv_labels.get(pred_id, "Unknown")
                        name_txt = f"{name} ({conf:.1f})"
                        color = (0, 255, 0)
                        if deb.ok(name):
                            log_attendance(name)
                    else:
                        name = "Unknown"
                        name_txt = f"{name} ({conf:.1f})"
                        color = (0, 0, 255)

                    # emit detection event
                    publish({
                        "type": "detection",
                        "name": name,
                        "conf": conf_val,
                        "yaw": float(yaw),
                        "pitch": float(pitch),
                        "box": {"x": X, "y": Y, "w": W, "h": H}
                    })
                else:
                    # non-frontal
                    publish({
                        "type": "detection",
                        "name": "NotFrontal",
                        "conf": None,
                        "yaw": float(yaw) if yaw is not None else None,
                        "pitch": float(pitch) if pitch is not None else None,
                        "box": {"x": X, "y": Y, "w": W, "h": H}
                    })

                # draw overlay
                cv2.rectangle(frame, (X, Y), (X+W, Y+H), color, 2)
                cv2.putText(frame, name_txt, (X, Y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Encode JPEG (quality ~80 to save CPU/bandwidth)
            ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                print("[Camera] Failed to encode frame")
                continue
            chunk = buf.tobytes()
            # Include Content-Length; boundary must match route below
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(chunk)).encode() + b"\r\n"
                b"\r\n" + chunk + b"\r\n"
            )
        except Exception as e:
            print(f"[Camera] Error in OpenCV loop: {e}")
            time.sleep(0.1)
            continue

# ----------------- Upload & retrain helpers -----------------
def _safe_person_name(name: str) -> str:
    # allow letters, numbers, spaces, hyphens/underscores
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

def _detect_align_for_training(det: YuNet, img_bgr):
    dets = det.detect(img_bgr)
    if not dets:
        return None
    dets.sort(key=lambda d: d[2], reverse=True)
    (x, y, w, h), lmk, _ = dets[0]
    roi = img_bgr[y:y+h, x:x+w]
    pts5 = [(int(px - x), int(py - y)) for (px, py) in lmk]
    return align_face(roi, pts5, out_size=100)

def _retrain_lbph_from_disk():
    """Rebuild LBPH model from all images in app/data/training_data and hot-swap runtime model."""
    ensure_model(YUNET_MODEL)
    det_local = YuNet(YUNET_MODEL, input_size=(PROC_W, PROC_W), conf_threshold=0.6)

    faces, labels = [], []
    label_map, next_id = {}, 0

    if not os.path.isdir(TRAIN_DIR):
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
        "X-Accel-Buffering": "no",  # disable proxy buffering if behind nginx
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
        yield "retry: 1000\n\n"  # reconnect after 1s
        while True:
            evt = events_q.get()  # blocking
            yield f"data: {json.dumps(evt)}\n\n"
    return Response(gen(), mimetype="text/event-stream")

@app.route("/api/users/<name>/photos", methods=["POST"])
def api_upload_photos(name):
    """
    FE: multipart/form-data with images[] (3 files)
    curl -F "images[]=@front.jpg" -F "images[]=@left.jpg" -F "images[]=@right.jpg" \
         http://<pi-ip>:5000/api/users/Alice/photos
    """
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

# optional snapshot route if you want to poll instead of SSE
@app.route("/latest")
def latest_detection():
    with latest_lock:
        return jsonify(latest)

@app.route("/test-camera")
def test_camera():
    """Comprehensive camera testing endpoint"""
    results = {
        "status": "unknown",
        "message": "",
        "picamera2_available": PICAMERA2_AVAILABLE,
        "available_devices": [],
        "working_devices": [],
        "picamera2_status": "not_tested",
        "errors": [],
        "system_info": {}
    }
    
    try:
        # Check system info
        import platform
        results["system_info"]["platform"] = platform.platform()
        results["system_info"]["python_version"] = platform.python_version()
        results["system_info"]["opencv_version"] = cv2.__version__
        
        # Test picamera2 first (preferred method)
        if PICAMERA2_AVAILABLE:
            try:
                print("[Camera] Testing picamera2...")
                picam2 = Picamera2()
                config = picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "BGR888"}
                )
                picam2.configure(config)
                picam2.start()
                time.sleep(1)
                
                frame = picam2.capture_array()
                if frame is not None and frame.size > 0:
                    results["picamera2_status"] = "working"
                    results["picamera2_info"] = {
                        "frame_shape": frame.shape,
                        "frame_size": frame.size,
                        "dtype": str(frame.dtype)
                    }
                    print("[Camera] picamera2 test successful")
                else:
                    results["picamera2_status"] = "no_frame"
                    results["errors"].append("picamera2: opened but no frame")
                
                picam2.stop()
                picam2.close()
                
            except Exception as e:
                results["picamera2_status"] = "error"
                results["errors"].append(f"picamera2: {str(e)}")
                print(f"[Camera] picamera2 test failed: {e}")
        
        # List all video devices
        import glob
        video_devices = glob.glob("/dev/video*")
        results["available_devices"] = sorted(video_devices)
        
        if not video_devices:
            results["status"] = "error"
            results["message"] = "No video devices found in /dev/"
            results["errors"].append("No /dev/video* devices found")
            return jsonify(results), 500
        
        # Test each OpenCV device
        for device_path in video_devices:
            device_num = int(device_path.split('/dev/video')[-1])
            device_info = {"device": device_path, "number": device_num}
            
            try:
                cap = cv2.VideoCapture(device_num)
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        device_info["status"] = "working"
                        device_info["frame_shape"] = frame.shape
                        device_info["frame_size"] = frame.size
                        results["working_devices"].append(device_info)
                    else:
                        device_info["status"] = "opened_but_no_frame"
                        device_info["error"] = "Device opened but couldn't read frame"
                        results["errors"].append(f"Device {device_num}: opened but no frame")
                else:
                    device_info["status"] = "not_accessible"
                    device_info["error"] = "Device not accessible"
                    results["errors"].append(f"Device {device_num}: not accessible")
                
                cap.release()
                
            except Exception as e:
                device_info["status"] = "error"
                device_info["error"] = str(e)
                results["errors"].append(f"Device {device_num}: {str(e)}")
        
        # Determine overall status
        if results["picamera2_status"] == "working":
            results["status"] = "success"
            results["message"] = "picamera2 is working (recommended method)"
        elif results["working_devices"]:
            results["status"] = "success"
            results["message"] = f"Found {len(results['working_devices'])} working OpenCV camera(s)"
        else:
            results["status"] = "error"
            results["message"] = "No working cameras found"
            
        # Add troubleshooting tips
        if results["status"] == "error":
            results["troubleshooting"] = [
                "1. Enable camera in raspi-config: sudo raspi-config",
                "2. Add user to video group: sudo usermod -a -G video $USER",
                "3. Install picamera2: sudo apt install python3-picamera2",
                "4. Reboot: sudo reboot",
                "5. Check camera connection and cables",
                "6. Ensure no other app is using the camera"
            ]
        
        return jsonify(results)
        
    except Exception as e:
        results["status"] = "error"
        results["message"] = f"Camera test failed: {str(e)}"
        results["errors"].append(str(e))
        return jsonify(results), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
