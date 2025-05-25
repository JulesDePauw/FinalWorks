import os
import time
import json
import cv2
import numpy as np
import streamlit as st
from compare_pose_streamlit import render_skeleton_frame
import matplotlib.pyplot as plt

# --- Configuratie ---
st.set_page_config(page_title="🧘‍♀️ Yoga Routine", layout="wide")

# Directories voor modelposes en routines
MODELPOSE_DIR = "modelposes_json"
ROUTINE_DIR = "routines_json"

# Main-window placeholders
title_placeholder = st.empty()
frame_placeholder = st.empty()
metrics_placeholder = st.empty()

# Laad modelposes in session state
if "pose_models" not in st.session_state:
    st.session_state.pose_models = {}
    for fn in os.listdir(MODELPOSE_DIR):
        if fn.endswith(".json"):
            path = os.path.join(MODELPOSE_DIR, fn)
            with open(path) as f:
                st.session_state.pose_models[fn] = json.load(f)
    st.session_state.phase_start = None
    st.session_state.running = False
    st.session_state.score_log = {}
    st.session_state.final_results_shown = False
    st.session_state.last_displayed_pose = None

# Sidebar: routine selectie en startknop
routine_files = [f for f in os.listdir(ROUTINE_DIR) if f.endswith(".json")]
selected_routine = st.sidebar.selectbox("Kies een yoga-routine:", routine_files)
if st.sidebar.button("Start routine"):
    st.session_state.routine = json.load(open(os.path.join(ROUTINE_DIR, selected_routine)))
    st.session_state.current_step = 0
    st.session_state.phase = "prepare"
    st.session_state.phase_start = time.time()
    st.session_state.running = True
    st.session_state.score_log = {}
    st.session_state.final_results_shown = False
    st.session_state.last_displayed_pose = None

# Model pose onderaan de sidebar
IMAGE_CAPTION = st.sidebar.empty()
IMAGE_DISPLAY = st.sidebar.empty()

# Hoofdloop: video feed en UI updates
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
        phase = st.session_state.phase

        # Titel updaten
        title_placeholder.title(f"🧘‍♀️ {pose_name}")

        # Model-pose tonen in sidebar tijdens prepare en hold
        if phase in ["prepare", "hold"]:
            if pose_name != st.session_state.last_displayed_pose:
                IMAGE_CAPTION.markdown(f"📌 **Model pose:** {pose_name}")
                IMAGE_DISPLAY.image(
                    step.get("image_path", f"modelposes/{pose_name}.jpg"),
                    use_container_width=True
                )
                st.session_state.last_displayed_pose = pose_name
        else:
            IMAGE_CAPTION.empty()
            IMAGE_DISPLAY.empty()

        # Pose rendering + score overlay
        model_fn = os.path.basename(step.get("pose_json", f"{pose_name}.json"))
        model_data = st.session_state.pose_models.get(model_fn, {})
        mode = "full" if phase == "hold" else "light"
        t0 = time.time()
        annotated_frame, score = render_skeleton_frame(frame.copy(), model_data, mode=mode)
        t1 = time.time()
        if score is not None:
            cv2.putText(
                annotated_frame,
                f"{score:.1f}%",
                (20, 50),  # Y-position iets hoger voor grotere tekst
                cv2.FONT_HERSHEY_SIMPLEX,
                2,       # Vergroot de font scale naar 2
                (0, 255, 0),
                5,       # Verhoog de dikte naar 3
                cv2.LINE_AA
            )
            if phase == "hold":
                st.session_state.score_log.setdefault(pose_name, []).append(score)

        frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

        # Metrics onder camera feed
        fps = 1 / (t1 - t0) if (t1 - t0) > 0 else 0
        elapsed = time.time() - st.session_state.phase_start
        if phase == "prepare":
            remaining = step.get("prep_time", 5) - elapsed
            phase_text = f"🔄 **Voorbereiding:** {int(remaining)} s — Neem de houding aan."
        elif phase == "hold":
            remaining = step.get("hold_time", 30) - elapsed
            phase_text = f"⏳ **Houd vast:** {int(remaining)} s — Goed bezig!"
        else:
            phase_text = ""
        metrics_placeholder.markdown(
            f"**Proc:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps**  \n{phase_text}"
        )

        # Fase-overgang logica
        now = time.time()
        if phase == "prepare" and remaining <= 0:
            st.session_state.phase = "hold"
            st.session_state.phase_start = now
        elif phase == "hold" and remaining <= 0:
            st.session_state.phase = "transition"
            st.session_state.phase_start = now
        elif phase == "transition":
            st.session_state.current_step += 1
            if st.session_state.current_step >= len(st.session_state.routine):
                st.session_state.running = False
            else:
                st.session_state.phase = "prepare"
                st.session_state.phase_start = now
                st.session_state.last_displayed_pose = None

        time.sleep(0.03)
    cap.release()

# Eindreeks: scoregrafiek tonen en UI clearen
if (
    not st.session_state.get("running", True)
    and not st.session_state.final_results_shown
    and any(st.session_state.score_log.values())
):
    frame_placeholder.empty()
    title_placeholder.empty()
    metrics_placeholder.empty()
    IMAGE_CAPTION.empty()
    IMAGE_DISPLAY.empty()

    st.header("Gemiddelde score per pose")
    fig, ax = plt.subplots()
    for pose, scores in st.session_state.score_log.items():
        if scores:
            seconds = np.linspace(0, len(scores)/30, num=len(scores))
            ax.plot(seconds, scores, label=pose)
    ax.set_xlabel("Tijd (seconden)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
 #   ax.set_xlim(0, 30)
    ax.legend()
    st.pyplot(fig)

    st.session_state.final_results_shown = True