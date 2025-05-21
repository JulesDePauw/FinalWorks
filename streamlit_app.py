import streamlit as st
import cv2
import json
import os
import time
import numpy as np
from compare_pose_streamlit import render_skeleton_frame, load_model_data, ANGLE_DEFINITIONS, calculate_angle
from compare_pose_streamlit import reset_pose_state

st.set_page_config(layout="wide", page_title="🧘 Yoga App Live")

if "routine" not in st.session_state:
    st.session_state.routine = []
    st.session_state.current_step = 0
    st.session_state.phase = "idle"
    st.session_state.step_start_time = None
    st.session_state.score = None
    st.session_state.running = False
    st.session_state.last_annotated_frame = None
    st.session_state.model_cache = {}
    st.session_state.last_frame_time = 0
    st.session_state.last_fps_update = 0
    st.session_state.pose_models = {}
    st.session_state.fps_text = st.sidebar.empty()

MODELPOSE_DIR = "modelposes_json"
for file in os.listdir(MODELPOSE_DIR):
    if file.endswith(".json") and file not in st.session_state.pose_models:
        with open(os.path.join(MODELPOSE_DIR, file)) as f:
            st.session_state.pose_models[file] = json.load(f)

st.title("🧘‍♀️ Live Yoga Routine met Permanente Webcam")

ROUTINE_DIR = "routines_json"
IMAGE_DIR = "modelposes"

routine_files = [f for f in os.listdir(ROUTINE_DIR) if f.endswith(".json")]
selected_routine = st.sidebar.selectbox("Kies een yoga-routine:", routine_files)

frame_placeholder = st.empty()
timer_placeholder = st.empty()
text_placeholder = st.empty()
title_placeholder = st.empty()
image_sidebar = st.sidebar.empty()

if st.button("Start routine"):
    with open(os.path.join(ROUTINE_DIR, selected_routine)) as f:
        st.session_state.routine = json.load(f)
    st.session_state.current_step = 0
    st.session_state.phase = "prepare"
    st.session_state.step_start_time = time.time()
    st.session_state.running = True

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Kan camera niet openen.")
    st.stop()

while st.session_state.running:
    loop_start = time.time()
    print(f"[LOOP] {loop_start:.3f}")

    ret, frame = cap.read()
    if not ret:
        st.warning("Geen frame ontvangen van camera.")
        break

    frame = cv2.flip(frame, 1)
    phase = st.session_state.phase

    if st.session_state.routine:
        current_step = st.session_state.routine[st.session_state.current_step]
        model_path = current_step.get("pose_json") or os.path.join(MODELPOSE_DIR, f"{current_step['pose']}.json")
        model_filename = os.path.basename(model_path)
        model_data = st.session_state.pose_models.get(model_filename, {})

        mode = "full" if phase == "hold" else "light"
        annotated_frame, temp_score = render_skeleton_frame(frame.copy(), model_data, mode=mode)
        st.session_state.last_annotated_frame = annotated_frame

        if phase == "hold":
            score = temp_score
        else:
            score = None

        elapsed = time.time() - st.session_state.step_start_time
        hold_time = current_step.get("hold_time", 30)
        prep_time = current_step.get("prep_time", 5)

        if phase == "transition" and st.session_state.current_step + 1 < len(st.session_state.routine):
            next_step = st.session_state.routine[st.session_state.current_step + 1]
            next_image_path = next_step.get("image_path") or os.path.join(IMAGE_DIR, f"{next_step['pose']}.jpg")
            image_sidebar.image(next_image_path, caption=f"Volgende: {next_step['pose']}", use_container_width=True)
        else:
            image_path = current_step.get("image_path") or os.path.join(IMAGE_DIR, f"{current_step['pose']}.jpg")
            image_sidebar.image(image_path, caption=current_step['pose'], use_container_width=True)

        if phase == "prepare":
            remaining = prep_time - elapsed
            timer_placeholder.markdown(f"🔄 Voorbereiding: {int(remaining)}s")
            text_placeholder.markdown("Neem de houding aan. Kijk naar het voorbeeld hiernaast.")
            if remaining <= 0:
                st.session_state.phase = "hold"
                st.session_state.step_start_time = time.time()

        elif phase == "hold":
            remaining = hold_time - elapsed
            timer_placeholder.markdown(f"⏳ Houd vast: {int(remaining)}s")
            if remaining <= 0:
                st.session_state.score = score
                st.session_state.phase = "transition"
                st.session_state.step_start_time = time.time()

        elif phase == "transition":
            text_placeholder.markdown("🧘‍♂️ Overgang naar volgende pose...")
            timer_placeholder.markdown("")
            if elapsed >= 5:
                st.session_state.current_step += 1
                reset_pose_state()
                if st.session_state.current_step >= len(st.session_state.routine):
                    st.session_state.running = False
                    st.balloons()
                    title_placeholder.markdown("✅ Routine voltooid")
                    break
                else:
                    st.session_state.phase = "prepare"
                    st.session_state.step_start_time = time.time()


        title_placeholder.markdown(f"## Pose {st.session_state.current_step + 1}: {current_step['pose']}")
        frame_rgb = cv2.cvtColor(st.session_state.last_annotated_frame, cv2.COLOR_BGR2RGB)

        if time.time() - st.session_state.last_frame_time > 1/30:
            st.session_state.last_frame_time = time.time()
            frame_placeholder.image(frame_rgb, channels="RGB")

    time.sleep(1 / 30)

cap.release()