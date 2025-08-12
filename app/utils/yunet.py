import os
import cv2
import numpy as np

class YuNet:
    def __init__(self, model_path: str, input_size=(320, 320), conf_threshold=0.6, nms_threshold=0.3, top_k=5000, backend_id=0, target_id=0):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Use OpenCV's built-in face detection instead of problematic ONNX models
        # This is more reliable on Raspberry Pi
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Also try to load a more accurate face cascade
        try:
            self.face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        except:
            self.face_cascade_alt = None
            
        print("[YuNet] Using OpenCV built-in face detection (more compatible with Pi)")

    def detect(self, bgr):
        """Detect faces using OpenCV's built-in cascade classifiers"""
        # Convert to grayscale for detection
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Try the primary cascade first
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces found, try alternative cascade
        if len(faces) == 0 and self.face_cascade_alt:
            faces = self.face_cascade_alt.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        results = []
        for (x, y, w, h) in faces:
            # Calculate confidence based on face size and position
            # Larger faces in center are more confident
            center_x, center_y = x + w//2, y + h//2
            img_center_x, img_center_y = bgr.shape[1]//2, bgr.shape[0]//2
            
            # Distance from center (closer = higher confidence)
            dist_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_dist = np.sqrt((img_center_x)**2 + (img_center_y)**2)
            center_confidence = 1.0 - (dist_from_center / max_dist)
            
            # Size confidence (larger faces = higher confidence)
            size_confidence = min(w * h / (bgr.shape[0] * bgr.shape[1] * 0.1), 1.0)
            
            # Combined confidence
            confidence = (center_confidence + size_confidence) / 2.0
            
            if confidence >= self.conf_threshold:
                # Generate synthetic landmarks (approximate positions)
                # This is a simplified approach but works for basic face alignment
                landmarks = [
                    (x + int(w * 0.3), y + int(h * 0.4)),  # Left eye
                    (x + int(w * 0.7), y + int(h * 0.4)),  # Right eye
                    (x + int(w * 0.5), y + int(h * 0.6)),  # Nose
                    (x + int(w * 0.3), y + int(h * 0.8)),  # Left mouth
                    (x + int(w * 0.7), y + int(h * 0.8))   # Right mouth
                ]
                
                results.append(((x, y, w, h), landmarks, confidence))
        
        # Sort by confidence
        results.sort(key=lambda x: x[2], reverse=True)
        return results

def ensure_model(model_path: str):
    """Dummy function for compatibility - we don't need external models"""
    pass