import cv2
import numpy as np

# 5-point alignment to 100x100 output
REF_5PT = np.array([
    (30.2946, 51.6963),
    (65.5318, 51.5014),
    (48.0252, 71.7366),
    (33.5493, 92.3655),
    (62.7299, 92.2041)
], dtype=np.float32)  # reference for 112x112; we'll scale to out_size

def align_face(bgr, pts5, out_size=100):
    try:
        ref = REF_5PT * (out_size / 112.0)
        src = np.array(pts5, dtype=np.float32)
        
        # Check if we have enough points
        if len(src) != 5:
            print(f"    Warning: Expected 5 points, got {len(src)}")
            return None
            
        # Estimate affine transformation
        result = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
        if result is None or result[0] is None:
            print(f"    Warning: Failed to estimate affine transformation")
            return None
            
        M = result[0]
        aligned = cv2.warpAffine(bgr, M, (out_size, out_size))
        
        # Convert to grayscale if needed
        if len(aligned.shape) == 3:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        
        return aligned
    except Exception as e:
        print(f"    Warning: Alignment failed with error: {e}")
        return None
