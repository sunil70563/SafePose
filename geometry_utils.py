import math

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) between three points.
    Point 'b' is the vertex (pivot point).
    """
    # a, b, c are [x, y] coordinates
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    
    # Calculate angle using arctan2 to get the angle relative to the X-axis
    # Then subtract them to get the internal angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Normalize the angle to be between 0 and 360, then keep it positive
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
        
    return angle

def check_posture(keypoints):
    """
    Analyzes body keypoints to detect unsafe postures.
    Input: YOLOv8 Keypoints Tensor (17, 3) -> [x, y, confidence]
    """
    # YOLOv8-Pose Keypoint Map:
    # 5: Left Shoulder,  7: Left Elbow,  9: Left Wrist
    # 6: Right Shoulder, 8: Right Elbow, 10: Right Wrist
    # 11: Left Hip, 13: Left Knee, 15: Left Ankle
    
    # --- LOGIC 1: OVER-REACHING (Shoulder-Elbow-Wrist) ---
    # We check the Right Arm (Indices 6, 8, 10)
    
    # Ensure confidence is high (> 0.5) before math
    if keypoints[6][2] < 0.5 or keypoints[8][2] < 0.5 or keypoints[10][2] < 0.5:
        return "Analyzing...", (255, 255, 255)

    # Extract [x, y] only
    p_shoulder = keypoints[6][:2]
    p_elbow = keypoints[8][:2]
    p_wrist = keypoints[10][:2]
    
    angle = calculate_angle(p_shoulder, p_elbow, p_wrist)
    
    # Industrial Safety Heuristics:
    # - Straight Arm (> 160): Risk of over-extension during lifting
    # - Acute Flexion (< 50): Risk of strain or awkward grip
    
    if angle > 160:
        return f"RISK: Over-Reach ({int(angle)}deg)", (0, 0, 255) # Red
    elif angle < 50:
        return f"RISK: Tight Grip ({int(angle)}deg)", (0, 165, 255) # Orange
    else:
        return f"Safe Posture ({int(angle)}deg)", (0, 255, 0) # Green