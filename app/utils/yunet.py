import os
import cv2
import urllib.request

DEFAULT_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def ensure_model(model_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        try:
            print(f"[YuNet] Downloading model to {model_path} ...")
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
            print("[YuNet] Download complete.")
        except Exception as e:
            print(f"[YuNet] Could not download model automatically: {e}")
            print(f"Please download manually from:\n{DEFAULT_MODEL_URL}\n and place it at: {model_path}")

class YuNet:
    def __init__(self, model_path: str, input_size=(320, 320), conf_threshold=0.6, nms_threshold=0.3, top_k=5000, backend_id=0, target_id=0):
        ensure_model(model_path)
        self.input_size = input_size
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=backend_id,
            target_id=target_id
        )

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        if (w, h) != self.input_size:
            self.detector.setInputSize((w, h))
        # result: N x 15 (x, y, w, h, 5 landmarks(x,y)*5, score)
        dets = self.detector.detect(bgr)[1]
        if dets is None or len(dets) == 0:
            return []
        faces = []
        for d in dets:
            x, y, ww, hh = d[:4].astype(int)
            lmk = d[4:14].reshape(-1, 2).astype(int)  # 5 landmarks
            score = float(d[14])
            faces.append(((x, y, ww, hh), lmk, score))
        return faces
