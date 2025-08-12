# rpi5-attendance (OpenCV YuNet + LBPH, Flask, SQLite)

Fast, free, and low-CPU **attendance** system for Raspberry Pi 5.
- **Detection + 5 landmarks:** OpenCV **YuNet** (`cv2.FaceDetectorYN`) – lightweight and fast on CPU.
- **Frontal-only gate:** head pose (yaw/pitch) from 5 landmarks + `solvePnP`.
- **Recognition:** OpenCV **LBPH** (no dlib build needed).
- **Server:** Flask MJPEG stream (`/video`) and JSON logs (`/logs`) backed by SQLite.

## Install (on RPi 5)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** This uses OpenCV **contrib** for YuNet + LBPH (`opencv-contrib-python`).

## Prepare data
Put 3 images per user (front/left/right) in:
```
app/data/training_data/<Name>/{front.jpg,left.jpg,right.jpg}
```

## Train
```bash
python app/enroll.py
```
- Downloads `yunet.onnx` model on first run (if missing).
- Detects + aligns faces to 100×100 grayscale.
- Trains `app/models/lbph_model.yml` and `app/models/label_map.json`.

## Run
```bash
python app/runtime.py
```
- Video stream: `http://<pi-ip>:5000/video`
- Last 100 logs: `http://<pi-ip>:5000/logs`
- SQLite DB: `app/db/attendance.sqlite`

## Tuning
- **Head pose thresholds**: `YAW_TH`, `PITCH_TH` (default ±15°) in `runtime.py`.
- **LBPH confidence**: lower is better; start at `< 60` and tune by lighting/camera.
- **Processing width**: `PROC_W=320` (speed/accuracy tradeoff).

## Why YuNet?
OpenCV’s YuNet detector is tiny, CPU‑friendly, and returns **5 landmarks** (eyes, nose, mouth corners) needed for both alignment and head pose – no extra dependencies like dlib/mediapipe.

