import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

# === Configuratie ===
FEEDBACK_INTERVAL = 0.5
MAX_ANGLE_DIFF = 45
JOINT_RADIUS = 10
LINE_THICKNESS = 5
LINE_SEGMENTS = 10

# === Helpers ===
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

def load_routine():
    ROUTINE_DIR = "routines_json"
    routine_files = [f for f in os.listdir(ROUTINE_DIR) if f.endswith(".json")]
    routine_name = st.selectbox("Kies een yoga-routine:", routine_files)
    if not routine_name:
        st.stop()
    with open(os.path.join(ROUTINE_DIR, routine_name), "r") as f:
        return json.load(f)

def run_yoga_routine(routine_steps, camera_placeholder, feedback_text_area, timer_area, sidebar_placeholder, title_placeholder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
    cap = cv2.VideoCapture(0)

    try:
        for i, step in enumerate(routine_steps):
            if "pose_json" not in step:
                step["pose_json"] = f"modelposes_json/{step['pose']}.json"
            if "image_path" not in step:
                step["image_path"] = f"modelposes/{step['pose']}.jpg"
            if "label" not in step:
                step["label"] = step["pose"]
            if "transition_text" not in step:
                step["transition_text"] = step.get("transition", "Bereid je voor op de volgende pose.")

            try:
                with open(step['pose_json']) as f:
                    model_json = json.load(f)
            except FileNotFoundError:
                st.error(f"⚠️ Bestand niet gevonden: {step['pose_json']}. Sla deze pose over.")
                continue

            model_angles = model_json["angles"]
            model_priority = model_json.get("priority", {})
            model_head_turn = model_json.get("head_turn")
            model_label = model_json.get("label", "deze pose")
            model_description = model_json.get("description", "")

            if sidebar_placeholder:
                sidebar_placeholder.image(step["image_path"], caption=step["label"], use_container_width=True)
            if title_placeholder:
                title_placeholder.markdown(f"## Pose {i+1}: {step['label']}")

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
                    for lm in results.pose_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

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

# === Streamlit UI ===
st.set_page_config(page_title="Yoga Routine", layout="wide")
st.title("🧘‍♀️ Live Yoga Routine met Feedback")
routine = load_routine()

if st.button("Start routine"):
    camera_placeholder = st.empty()
    feedback_text_area = st.empty()
    timer_area = st.empty()
    sidebar_placeholder = st.sidebar.empty()
    title_placeholder = st.empty()
    run_yoga_routine(
        routine,
        camera_placeholder,
        feedback_text_area,
        timer_area,
        sidebar_placeholder,
        title_placeholder
    )
