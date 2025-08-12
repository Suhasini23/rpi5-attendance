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
    h, w = image.shape[:2]
    cam = np.array([[w, 0, w/2],
                    [0, w, h/2],
                    [0, 0,   1 ]], dtype=np.float64)
    dist = np.zeros((4,1))
    image_pts = np.array(pts5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, image_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    P = np.hstack([R, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(P)
    yaw, pitch, roll = [float(a) for a in euler]
    return yaw, pitch, roll
