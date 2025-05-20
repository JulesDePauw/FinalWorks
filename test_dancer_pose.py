JOINT_RADIUS = 10
import cv2
import mediapipe as mp
import numpy as np
import json
import streamlit as st
import os

MODEL_PATH = "modelposes_json/DancerWrongHR.json"
IMAGE_PATH = "modelposes/DancerWrongHR.jpg"

st.title("Dancer Pose Vergelijking")

# Laad afbeelding
image = cv2.imread(IMAGE_PATH)
if image is None:
    st.error(f"❌ Afbeelding niet gevonden: {IMAGE_PATH}")
    st.stop()

# Laad modelpose JSON
with open(MODEL_PATH) as f:
    model = json.load(f)

model_angles = model.get("angles", {})
model_priority = model.get("priority", {})
model_head_turn = model.get("head_turn", None)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(rgb)

image_placeholder = st.empty()

if not results.pose_landmarks:
    st.error("⚠️ Geen pose landmarks gevonden in afbeelding.")
    st.stop()

landmarks = results.pose_landmarks.landmark
h, w = image.shape[:2]

POSE_LANDMARKS = mp_pose.PoseLandmark

KEYPOINTS = {i.name.lower(): {"x": lm.x, "y": lm.y} for i, lm in zip(POSE_LANDMARKS, landmarks)}

ANGLE_DEFINITIONS = {
    "left_elbow_angle": ["left_shoulder", "left_elbow", "left_wrist"],
    "right_elbow_angle": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_knee_angle": ["left_hip", "left_knee", "left_ankle"],
    "right_knee_angle": ["right_hip", "right_knee", "right_ankle"],
    "left_shoulder_angle": ["left_elbow", "left_shoulder", "left_hip"],
    "right_shoulder_angle": ["right_elbow", "right_shoulder", "right_hip"],
    "left_hip_angle": ["left_shoulder", "left_hip", "left_knee"],
    "right_hip_angle": ["right_shoulder", "right_hip", "right_knee"],
    "left_ankle_angle": ["left_knee", "left_ankle", "left_foot_index"],
    "right_ankle_angle": ["right_knee", "right_ankle", "right_foot_index"]
}

def calculate_angle(a, b, c):
    ba = [a['x'] - b['x'], a['y'] - b['y']]
    bc = [c['x'] - b['x'], c['y'] - b['y']]
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    magnitude = (np.hypot(*ba) * np.hypot(*bc))
    if magnitude == 0:
        return 0.0
    angle_rad = np.arccos(dot_product / magnitude)
    return np.degrees(angle_rad)

def calculate_head_turn():
    try:
        left = KEYPOINTS["left_ear"]["x"]
        right = KEYPOINTS["right_ear"]["x"]
        nose = KEYPOINTS["nose"]["x"]
        center = (left + right) / 2
        return round(nose - center, 3)
    except:
        return None

angles = {}
feedback_lines = []

for label, (a, b, c) in ANGLE_DEFINITIONS.items():
    try:
        angle = calculate_angle(KEYPOINTS[a], KEYPOINTS[b], KEYPOINTS[c])
        angles[label] = angle
        model_val = model_angles.get(label)
        if model_val is not None:
            diff = abs(angle - model_val)
            prio = model_priority.get(label, 5)
            if prio >= 7 and diff >= 10:
                feedback_lines.append(f"{label}: {round(angle)}° vs {round(model_val)}° (prio {prio})")
    except KeyError:
        angles[label] = None

# Head turn
head_turn = calculate_head_turn()
if head_turn is not None and model_head_turn is not None:
    diff = abs(head_turn - model_head_turn)
    prio = model_priority.get("head_turn", 5)
    if prio >= 7 and diff >= 0.05:
        feedback_lines.append(f"head_turn offset: {round(diff, 3)} (prio {prio})")

# Teken skelet
for label, (a, b, c) in ANGLE_DEFINITIONS.items():
    try:
        pa = (int(KEYPOINTS[a]['x'] * w), int(KEYPOINTS[a]['y'] * h))
        pb = (int(KEYPOINTS[b]['x'] * w), int(KEYPOINTS[b]['y'] * h))
        pc = (int(KEYPOINTS[c]['x'] * w), int(KEYPOINTS[c]['y'] * h))
        cv2.line(image, pa, pb, (0, 255, 0), 2)
        cv2.line(image, pb, pc, (0, 255, 0), 2)
        for p in [pa, pb, pc]:
            cv2.circle(image, p, JOINT_RADIUS, (0, 255, 0), -1)
    except KeyError:
        continue

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_placeholder.image(image_rgb, channels="RGB", caption="Input Pose")

if feedback_lines:
    st.markdown("### Live Feedback")
    for line in feedback_lines:
        st.markdown(f"- {line}")
else:
    st.info("✅ Geen significante afwijkingen gevonden bij belangrijke gewrichten.")
