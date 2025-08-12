import os, json
import numpy as np
import cv2

from utils.yunet import YuNet, ensure_model
from utils.align import align_face

DATA_DIR = "app/data/training_data"
MODEL_OUT = "app/models/lbph_model.yml"
LABELS_OUT = "app/models/label_map.json"
YUNET_MODEL = "app/models/yunet.onnx"

def iter_person_images():
    for person in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                yield person, os.path.join(pdir, fn)

if __name__ == "__main__":
    ensure_model(YUNET_MODEL)
    det = YuNet(YUNET_MODEL, input_size=(320,320), conf_threshold=0.6)

    faces, labels = [], []
    label_map, next_id = {}, 0

    for person, path in iter_person_images():
        img = cv2.imread(path)
        if img is None: 
            print(f"Skip unreadable: {path}")
            continue
        dets = det.detect(img)
        if not dets:
            print(f"No face in: {path}")
            continue
        # Take the highest-score detection
        dets.sort(key=lambda d: d[2], reverse=True)
        (x,y,w,h), lmk, score = dets[0]

        # YuNet landmark order: [x1,y1,x2,y2,...] typically left_eye, right_eye, nose, left_mouth, right_mouth
        pts5 = [(int(px), int(py)) for (px,py) in lmk]
        aligned = align_face(img, pts5, out_size=100)

        if person not in label_map:
            label_map[person] = next_id; next_id += 1
        faces.append(aligned)
        labels.append(label_map[person])

    if not faces:
        raise SystemExit("No faces found to train.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    recognizer.write(MODEL_OUT)
    with open(LABELS_OUT, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Trained LBPH on {len(faces)} samples across {len(label_map)} people.")
