import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

# === Configuratie ===
MAX_ANGLE_DIFF = 45
JOINT_RADIUS = 10
LINE_THICKNESS = 5
LINE_SEGMENTS = 10

# === Helperfuncties ===
def generate_friendly_feedback(*args, **kwargs):
    return "🤖 Hier komt later AI-feedback voor deze gewrichtsafwijking!"

def midpoint(p1, p2):
    return tuple(int((p1[i] + p2[i]) / 2) for i in range(2))

def gradient_color(norm):
    if not isinstance(norm, (int, float)) or np.isnan(norm):
        return (200, 200, 200)
    if norm <= 0.5:
        r = int(255 * (norm * 2))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (norm - 0.5) * 2))
    return (0, g, r)

def draw_gradient_line(img, pt1, pt2, color1, color2, segments=10, thickness=6):
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        x1 = int(pt1[0] * (1 - t1) + pt2[0] * t1)
        y1 = int(pt1[1] * (1 - t1) + pt2[1] * t1)
        x2 = int(pt1[0] * (1 - t2) + pt2[0] * t2)
        y2 = int(pt1[1] * (1 - t2) + pt2[1] * t2)
        c = [int(color1[j] * (1 - t1) + color2[j] * t1) for j in range(3)]
        cv2.line(img, (x1, y1), (x2, y2), c, thickness)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
# === Verbeterde skeletroutine met gradient en virtuele gewrichten ===
def run_yoga_routine(routine_steps, camera_placeholder, feedback_text_area, timer_area, sidebar_placeholder, title_placeholder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
    cap = cv2.VideoCapture(0)

    ANGLE_DEFINITIONS = {
        "left_elbow": [11, 13, 15],
        "right_elbow": [12, 14, 16],
        "left_knee": [23, 25, 27],
        "right_knee": [24, 26, 28],
        "neck": [11, "neck", 12],
        "hip": [23, "hip_center", 24],
        "head": ["neck", "head", 0],
        "left_shoulder": [23, 11, 13],
        "right_shoulder": [24, 12, 14],
        "left_wrist": [13, 15, 19],
        "right_wrist": [14, 16, 20],
        "left_ankle": [25, 27, 31],
        "right_ankle": [26, 28, 32]
    }

    try:
        for i, step in enumerate(routine_steps):
            with open(step['pose_json']) as f:
                model_json = json.load(f)

            model_angles = model_json["angles"]

            if sidebar_placeholder:
                sidebar_placeholder.image(step["image_path"], caption=step["label"], use_container_width=True)
            if title_placeholder:
                title_placeholder.markdown(f"## Pose {i+1}: {step['label']}")

            # === Voorbereidingstijd ===
            prep_time = step.get("prep_time", 5)
            for sec in range(prep_time, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB")
                timer_area.markdown(f"🔀 **Voorbereiding**: start in {sec}s")
                feedback_text_area.markdown("Neem de startpositie in. Kijk naar het voorbeeld in de zijbalk.")
                time.sleep(1)

            # === Pose vasthouden ===
            hold_time = step.get("hold_time", 30)
            pose_start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                elapsed = time.time() - pose_start_time
                remaining = max(0, int(hold_time - elapsed))
                timer_area.markdown(f"⏳ **Pose actief: {remaining} seconden**")

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    pose_px = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))}
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
                    counted = 0

                    for name, (a, b, c) in ANGLE_DEFINITIONS.items():
                        pa, pb, pc = get_coord(a), get_coord(b), get_coord(c)
                        if pa and pb and pc:
                            angle = calculate_angle(pa, pb, pc)
                            model = model_angles.get(name + "_angle", angle)
                            diff = abs(angle - model)
                            norm = min(diff / MAX_ANGLE_DIFF, 1.0)
                            color = gradient_color(norm)
                            for joint in [a, b, c]:
                                raw_colors.setdefault(joint, []).append(color)
                            total_diff += norm
                            counted += 1

                    joint_colors = {
                        j: tuple(int(np.mean([c[i] for c in clist])) for i in range(3))
                        for j, clist in raw_colors.items()
                    }

                    def pt(x): return pose_px.get(x) if isinstance(x, int) else virtual_joints.get(x)

                    links = [
                        ("head", "neck"),
                        ("neck", 11), ("neck", 12),
                        (11, 13), (13, 15),
                        (12, 14), (14, 16),
                        ("neck", "hip_center"),
                        ("hip_center", 23), ("hip_center", 24),
                        (23, 25), (25, 27),
                        (24, 26), (26, 28),
                    ]

                    for p1_id, p2_id in links:
                        p1, p2 = pt(p1_id), pt(p2_id)
                        if p1 and p2:
                            c1 = joint_colors.get(p1_id, (200, 200, 200))
                            c2 = joint_colors.get(p2_id, (200, 200, 200))
                            draw_gradient_line(frame, p1, p2, c1, c2, segments=LINE_SEGMENTS, thickness=LINE_THICKNESS)

                    visible_joints = set([
                        "head", "neck", "hip_center",
                        11, 12, 13, 14, 15, 16,
                        23, 24, 25, 26, 27, 28
                    ])
                    for idx in visible_joints:
                        coord = pt(idx)
                        if coord:
                            color = joint_colors.get(idx, (255, 255, 255))
                            cv2.circle(frame, coord, JOINT_RADIUS, color, -1)

                    if counted > 0:
                        score = max(0, int((1 - total_diff / counted) * 100))
                        cv2.putText(frame, f"Pose Score: {score}%", (30, h - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB")

                if elapsed >= hold_time:
                    break
                time.sleep(1/30)

            feedback_text_area.empty()
            timer_area.markdown("🧑‍🔭 **Overgang**: bereid je voor op de volgende pose")
            time.sleep(5)
    finally:
        cap.release()
