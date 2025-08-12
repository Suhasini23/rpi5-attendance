#!/usr/bin/env python3
"""
Simple test script to verify the trained face recognition model
"""
import cv2
import json
import numpy as np
from utils.yunet import YuNet, ensure_model
from utils.align import align_face

# Paths
MODEL_IN = "app/models/lbph_model.yml"
LABELS_IN = "app/models/label_map.json"
YUNET_MODEL = "app/models/yunet.onnx"
TEST_IMAGE = "app/data/training_data/Suhasini/front.png"  # Use one of the training images

def test_model():
    """Test the trained model on a known image"""
    print("Testing face recognition model...")
    
    # Load the trained model
    if not os.path.exists(MODEL_IN) or not os.path.exists(LABELS_IN):
        print("❌ Model files not found. Run enrollment first!")
        return False
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_IN)
    
    label_map = json.load(open(LABELS_IN))
    inv_labels = {v: k for k, v in label_map.items()}
    
    print(f"✅ Model loaded successfully")
    print(f"✅ Labels: {label_map}")
    
    # Load face detector
    ensure_model(YUNET_MODEL)
    det = YuNet(YUNET_MODEL, input_size=(320, 320), conf_threshold=0.6)
    
    # Load test image
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print(f"❌ Could not load test image: {TEST_IMAGE}")
        return False
    
    print(f"✅ Test image loaded: {TEST_IMAGE}")
    
    # Detect face
    detections = det.detect(img)
    if not detections:
        print("❌ No face detected in test image")
        return False
    
    (x, y, w, h), lmk, score = detections[0]
    print(f"✅ Face detected with confidence: {score:.3f}")
    
    # Extract ROI and align
    roi = img[y:y+h, x:x+w].copy()
    pts5 = [(int(px - x), int(py - y)) for (px, py) in lmk]
    
    aligned = align_face(roi, pts5, out_size=100)
    if aligned is None:
        print("❌ Face alignment failed")
        return False
    
    print(f"✅ Face aligned successfully")
    
    # Predict
    pred_id, conf = recognizer.predict(aligned)
    predicted_name = inv_labels.get(pred_id, "Unknown")
    
    print(f"✅ Prediction: {predicted_name} (ID: {pred_id}, confidence: {conf:.1f})")
    
    # Check if prediction is correct
    expected_name = "Suhasini"
    if predicted_name == expected_name:
        print(f"🎉 SUCCESS: Model correctly identified {expected_name}!")
        return True
    else:
        print(f"❌ FAILED: Expected {expected_name}, got {predicted_name}")
        return False

if __name__ == "__main__":
    import os
    success = test_model()
    if success:
        print("\n✅ Face recognition model is working correctly!")
    else:
        print("\n❌ Face recognition model has issues.")
