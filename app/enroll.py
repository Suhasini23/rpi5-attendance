# app/enroll_fixed.py
import os, json, cv2, numpy as np
from utils.yunet import YuNet, ensure_model
from utils.align import align_face

DATA_DIR   = "app/data/training_data"
MODEL_OUT  = "app/models/lbph_model.yml"
LABELS_OUT = "app/models/label_map.json"
YUNET_MODEL= "app/models/yunet.onnx"

PROC_W = 320          # must match runtime
DET_CONF = 0.5        # a bit looser than 0.6 for training

def iter_person_images():
    for person in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(pdir): continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                yield person, os.path.join(pdir, fn)

def best_detection(det, img_bgr):
    """Try multiple scales/rotations; return best (box, lmk, score, used_image)."""
    candidates = []
    rotations = [0, 90, 270]  # try upright, left, right
    sizes = [PROC_W, 480, 640]
    for rot in rotations:
        if rot == 0:
            base = img_bgr
        elif rot == 90:
            base = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        else:
            base = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h0, w0 = base.shape[:2]
        for W in sizes:
            scale = W / float(w0)
            small = cv2.resize(base, (W, int(h0*scale)))
            dets = det.detect(small)
            if not dets: continue
            dets.sort(key=lambda d: d[2], reverse=True)
            (x,y,w,h), lmk, score = dets[0]
            candidates.append(((x,y,w,h), lmk, score, small))
    if not candidates:
        return None
    # choose highest score
    return max(candidates, key=lambda c: c[2])

if __name__ == "__main__":
    ensure_model(YUNET_MODEL)
    det = YuNet(YUNET_MODEL, input_size=(PROC_W, PROC_W), conf_threshold=DET_CONF)

    faces, labels = [], []
    label_map, next_id = {}, 0

    for person, path in iter_person_images():
        img = cv2.imread(path)
        if img is None:
            print(f"Skip unreadable: {path}")
            continue

        found = best_detection(det, img)
        if not found:
            print(f"No face in: {path}")
            continue

        (x,y,w,h), lmk, score, used = found
        roi = used[y:y+h, x:x+w].copy()
        pts5 = [(int(px - x), int(py - y)) for (px, py) in lmk]  # RELATIVE to ROI

        aligned = align_face(roi, pts5, out_size=100)
        if aligned is None or aligned.size == 0:
            print(f"Align failed: {path}")
            continue

        # LBPH likes 8-bit grayscale; also normalize a bit
        if aligned.ndim == 3:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        aligned = cv2.equalizeHist(aligned)

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
