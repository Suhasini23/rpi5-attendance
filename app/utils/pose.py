import cv2
import numpy as np

# Simple 3D template for 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
MODEL_3D = np.array([
    (-30.0, -30.0,  30.0),  # left eye
    ( 30.0, -30.0,  30.0),  # right eye
    (  0.0,   0.0,   0.0),  # nose tip
    (-20.0,  30.0,  10.0),  # left mouth
    ( 20.0,  30.0,  10.0)   # right mouth
], dtype=np.float64)

def head_pose_angles(image, pts5):
    """
    Estimate head pose from 5 facial landmarks.
    Returns (yaw, pitch, roll) in degrees.
    """
    try:
        if len(pts5) != 5:
            return None, None, None
            
        h, w = image.shape[:2]
        
        # Create camera matrix (simplified)
        cam = np.array([[w, 0, w/2],
                        [0, w, h/2],
                        [0, 0,   1 ]], dtype=np.float64)
        
        # No distortion
        dist = np.zeros((5,1))
        
        # Convert points to numpy array
        image_pts = np.array(pts5, dtype=np.float64)
        
        # Use SOLVEPNP_EPNP for better 5-point handling
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_3D, 
            image_pts, 
            cam, 
            dist, 
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not ok:
            return None, None, None
            
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create projection matrix
        P = np.hstack([R, tvec])
        
        # Decompose to get Euler angles
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(P)
        
        # Extract angles and convert to degrees
        yaw, pitch, roll = [float(a) for a in euler]
        
        # Normalize angles to [-180, 180] range
        yaw = ((yaw + 180) % 360) - 180
        pitch = ((pitch + 180) % 360) - 180
        roll = ((roll + 180) % 360) - 180
        
        return yaw, pitch, roll
        
    except Exception as e:
        # Return None values if pose estimation fails
        return None, None, None
