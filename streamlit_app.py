
import os
import time
import json
import cv2
import streamlit as st
from compare_pose_streamlit import render_skeleton_frame

st.set_page_config(page_title="🧘‍♀️ Live Yoga Routine", layout="wide")
st.title("🧘‍♀️ Live Yoga Routine met Permanente Webcam")

# Init session state
if "pose_models" not in st.session_state:
    st.session_state.pose_models = {}
    st.session_state.fps_text = st.sidebar.empty()
    st.session_state.score_placeholder = st.sidebar.empty()
    st.session_state.last_pose_img = None
    st.session_state.phase_start = None
    st.session_state.running = False

# Load poses
MODELPOSE_DIR = "modelposes_json"
for fn in os.listdir(MODELPOSE_DIR):
    if fn.endswith(".json") and fn not in st.session_state.pose_models:
        st.session_state.pose_models[fn] = json.load(open(os.path.join(MODELPOSE_DIR, fn)))

ROUTINE_DIR = "routines_json"
routine_files = [f for f in os.listdir(ROUTINE_DIR) if f.endswith(".json")]
selected_routine = st.sidebar.selectbox("Kies een yoga-routine:", routine_files)

# Containers
image_caption = st.sidebar.empty()
image_display = st.sidebar.empty()
frame_placeholder = st.empty()
timer_placeholder = st.sidebar.empty()
text_placeholder = st.sidebar.empty()

# Start button
if st.sidebar.button("Start routine"):
    st.session_state.routine = json.load(open(os.path.join(ROUTINE_DIR, selected_routine)))
    st.session_state.current_step = 0
    st.session_state.phase = "prepare"
    st.session_state.phase_start = time.time()
    st.session_state.running = True
    st.session_state.last_pose_img = None

# Begin while loop
if st.session_state.get("running", False):
    cap = cv2.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Kan geen frame van de camera lezen.")
            break
        frame = cv2.flip(frame, 1)

        step = st.session_state.routine[st.session_state.current_step]
        pose_name = step["pose"]
        model_fn = os.path.basename(step.get("pose_json", f"{pose_name}.json"))
        model_data = st.session_state.pose_models.get(model_fn, {})
        img_path = step.get("image_path", f"modelposes/{pose_name}.jpg")

        # Thumbnail bij posewissel
        if st.session_state.last_pose_img != img_path:
            image_caption.markdown(f"📌 **Huidige pose:** {pose_name}")
            image_display.image(img_path, use_container_width=True)
            st.session_state.last_pose_img = img_path

        mode = "full" if st.session_state.phase == "hold" else "light"
        t0 = time.time()
        annotated_frame, score = render_skeleton_frame(frame.copy(), model_data, mode=mode)
        t1 = time.time()
        fps = 1 / (t1 - t0) if (t1 - t0) > 0 else 0
        st.session_state.fps_text.text(f"Proc: {(t1 - t0)*1000:.0f} ms — {fps:.1f} fps")

        frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        if score is not None:
            st.session_state.score_placeholder.metric("Pose Score", f"{score:.1f}%")

        now = time.time()
        phase = st.session_state.phase

        if phase == "prepare":
            elapsed = now - st.session_state.phase_start
            remaining = step.get("prep_time", 5) - elapsed
            timer_placeholder.markdown(f"🔄 Voorbereiding: {int(remaining)} s")
            text_placeholder.markdown("Neem de houding aan.")
            if remaining <= 0:
                st.session_state.phase = "hold"
                st.session_state.phase_start = now

        elif phase == "hold":
            elapsed = now - st.session_state.phase_start
            remaining = step.get("hold_time", 30) - elapsed
            timer_placeholder.markdown(f"⏳ Houd vast: {int(remaining)} s")
            text_placeholder.markdown("Goed bezig!")
            if remaining <= 0:
                st.session_state.phase = "transition"
                st.session_state.phase_start = now

        elif phase == "transition":
            st.session_state.current_step += 1
            if st.session_state.current_step >= len(st.session_state.routine):
                st.session_state.running = False
                st.success("Routine voltooid! 🧘‍♀️")
                break
            else:
                st.session_state.phase = "prepare"
                st.session_state.phase_start = now
                st.session_state.last_pose_img = None

        time.sleep(0.03)

    cap.release()
