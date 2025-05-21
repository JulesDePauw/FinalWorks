
import json
import time
import cv2
import numpy as np
import mediapipe as mp

MAX_ANGLE_DIFF   = 45
JOINT_RADIUS     = 10
LINE_THICKNESS   = 5
LINE_SEGMENTS    = 10

ANGLE_DEFINITIONS = {
    "left_elbow":      [11, 13, 15],
    "right_elbow":     [12, 14, 16],
    "left_knee":       [23, 25, 27],
    "right_knee":      [24, 26, 28],
    "neck":            [11, "neck", 12],
    "hip":             [23, "hip_center", 24],
    "head":            ["neck", "head", 0],
    "left_shoulder":   [23, 11, 13],
    "right_shoulder":  [24, 12, 14],
    "left_wrist":      [13, 15, 19],
    "right_wrist":     [14, 16, 20],
    "left_ankle":      [25, 27, 31],
    "right_ankle":     [26, 28, 32],
}

mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

frame_counter = 0
last_render   = None
last_score    = None
last_heavy    = time.time()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc   = a - b, c - b
    cosine   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def gradient_color(norm):
    if not isinstance(norm, (int, float)) or np.isnan(norm):
        return (200, 200, 200)
    if norm <= 0.5:
        r = int(255 * (norm * 2)); g = 255
    else:
        r = 255; g = int(255 * (1 - (norm - 0.5) * 2))
    return (0, g, r)

def midpoint(p1, p2):
    return tuple(int((p1[i] + p2[i]) / 2) for i in range(2))

def draw_gradient_line(img, pt1, pt2, color1, color2, segments=LINE_SEGMENTS, thickness=LINE_THICKNESS):
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        x1 = int(pt1[0] * (1 - t1) + pt2[0] * t1)
        y1 = int(pt1[1] * (1 - t1) + pt2[1] * t1)
        x2 = int(pt1[0] * (1 - t2) + pt2[0] * t2)
        y2 = int(pt1[1] * (1 - t2) + pt2[1] * t2)
        c  = [int(color1[j] * (1 - t1) + color2[j] * t1) for j in range(3)]
        cv2.line(img, (x1, y1), (x2, y2), c, thickness)

def render_skeleton_frame(frame, model_json, mode="full"):
    global frame_counter, last_render, last_score, last_heavy
    now = time.time()

    if last_render is not None and (now - last_heavy) < 0.05:
        return last_render, last_score

    last_heavy  = now
    frame_counter += 1

    h, w = frame.shape[:2]

    small     = cv2.resize(frame, (320, 240))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results   = pose.process(rgb_small)

    h_ratio = h / 240
    w_ratio = w / 320
    pose_px = {}
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            x_full = int(lm.x * 320 * w_ratio)
            y_full = int(lm.y * 240 * h_ratio)
            pose_px[i] = (x_full, y_full)

    virtual_joints = {}
    if 11 in pose_px and 12 in pose_px:
        virtual_joints["neck"] = midpoint(pose_px[11], pose_px[12])
    if 7 in pose_px and 8 in pose_px:
        virtual_joints["head"] = midpoint(pose_px[7], pose_px[8])
    if 23 in pose_px and 24 in pose_px:
        virtual_joints["hip_center"] = midpoint(pose_px[23], pose_px[24])

    def get_coord(x):
        return pose_px.get(x) if isinstance(x, int) else virtual_joints.get(x)

    raw_colors = {}
    total_diff = 0
    counted    = 0
    if mode == "full" and results.pose_landmarks:
        for name, (a, b, c) in ANGLE_DEFINITIONS.items():
            pa, pb, pc = get_coord(a), get_coord(b), get_coord(c)
            if pa and pb and pc:
                angle = calculate_angle(pa, pb, pc)
                model = model_json["angles"].get(name + "_angle", angle)
                diff = abs(angle - model)
                norm = min(diff / MAX_ANGLE_DIFF, 1.0)
                col  = gradient_color(norm)
                for joint in [a, b, c]:
                    raw_colors.setdefault(joint, []).append(col)
                total_diff += norm
                counted    += 1

    joint_colors = {
        j: tuple(int(np.mean([c[i] for c in clist])) for i in range(3))
        for j, clist in raw_colors.items()
    } if raw_colors else {}

    links = [
        ("head","neck"), ("neck",11), ("neck",12),
        (11,13),(13,15),(12,14),(14,16),
        ("neck","hip_center"),("hip_center",23),("hip_center",24),
        (23,25),(25,27),(24,26),(26,28)
    ]
    for p1, p2 in links:
        c1 = joint_colors.get(p1, (200,200,200))
        c2 = joint_colors.get(p2, (200,200,200))
        pt1 = get_coord(p1); pt2 = get_coord(p2)
        if pt1 and pt2:
            draw_gradient_line(frame, pt1, pt2, c1, c2)

    visible = {"head","neck","hip_center",11,12,13,14,15,16,23,24,25,26,27,28}
    for idx in visible:
        coord = get_coord(idx)
        if coord:
            col = joint_colors.get(idx, (255,255,255))
            cv2.circle(frame, coord, JOINT_RADIUS, col, -1)

    score = None
    if mode == "full" and counted > 0:
        score = max(0, (1 - total_diff / counted) * 100)

    last_render = frame.copy()
    last_score  = score
    return last_render, last_score
