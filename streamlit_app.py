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
# Pastel achtergrond + tekst en knoppen styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFBF2;
        color: #333333;
    }
    /* Knop-styling */
    div.stButton > button {
        background-color: #4C9A2A;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    div.stButton > button:hover {
        background-color: #3E7D22;
    }
    /* Dropdown styling */
    div.stSelectbox > div {
        background-color: #FFFFFF;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True
)

# Directories
MODELPOSE_DIR = "modelposes_json"
ROUTINE_DIR = "routines_json"

# Laad modelposes
if "pose_models" not in st.session_state:
    st.session_state.pose_models = {}
    for fn in os.listdir(MODELPOSE_DIR):
        if fn.endswith(".json"):
            with open(os.path.join(MODELPOSE_DIR, fn)) as f:
                st.session_state.pose_models[fn] = json.load(f)
    st.session_state.running = False
    st.session_state.score_log = {}
    st.session_state.final_results_shown = False
    st.session_state.current_step = 0
    st.session_state.phase = None
    st.session_state.phase_start = None

# Layout: twee gelijke kolommen
col1, col2 = st.columns(2)

# Create placeholders for controls
with col1:
    routine_ph = st.empty()
    start_ph = st.empty()
    pose_ph = st.empty()

# Create placeholders for video and text on right
with col2:
    frame_ph = st.empty()
    title_ph = st.empty()
    metrics_ph = st.empty()

# Controls: toon selecties enkel als niet running
with col1:
    if not st.session_state.running:
        selected = routine_ph.selectbox(
            "Kies een yoga-routine:",
            [f for f in os.listdir(ROUTINE_DIR) if f.endswith(".json")]
        )
        if start_ph.button("Start routine"):
            st.session_state.routine = json.load(
                open(os.path.join(ROUTINE_DIR, selected))
            )
            st.session_state.running = True
            st.session_state.current_step = 0
            st.session_state.phase = "prepare"
            st.session_state.phase_start = time.time()
            st.session_state.score_log = {}
            st.session_state.final_results_shown = False
            # (Re)initialize camera
            if "cap" in st.session_state:
                st.session_state.cap.release()
            st.session_state.cap = cv2.VideoCapture(0)
    else:
        # Clear controls once running
        routine_ph.empty()
        start_ph.empty()

# Always display model pose in left when running in left when running in left when running
if st.session_state.running:
    step = st.session_state.routine[st.session_state.current_step]
    pose_name = step["pose"]
    if st.session_state.phase in ["prepare", "hold"]:
        with col1:
            pose_ph.markdown(f"📌 **Model pose:** {pose_name}")
            pose_ph.image(
                step.get("image_path", f"modelposes/{pose_name}.jpg"),
                use_container_width=True
            )
else:
    pose_ph.empty()

# Main loop for video
if st.session_state.running:
    cap = st.session_state.cap
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Kan geen frame van de camera lezen.")
            break
        frame = cv2.flip(frame, 1)

        step = st.session_state.routine[st.session_state.current_step]
        pose = step["pose"]
        phase = st.session_state.phase

        # Render + overlay score
        model_fn = os.path.basename(step.get("pose_json", f"{pose}.json"))
        mode = "full" if phase == "hold" else "light"
        t0 = time.time()
        annotated, score = render_skeleton_frame(
            frame.copy(), st.session_state.pose_models.get(model_fn, {}), mode=mode
        )
        t1 = time.time()
        if score is not None:
            txt = f"{score:.1f}%"
            (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 15), (5 + w + 10, 15 + h + 10), (50, 50, 50), -1)
            annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
            cv2.putText(annotated, txt, (10, 15 + h), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 255, 255), 3, cv2.LINE_AA)
            if phase == "hold":
                st.session_state.score_log.setdefault(pose, []).append(score)

        # Show frame
        frame_ph.image(annotated, channels="BGR", use_container_width=True)
        # Title under frame
        title_ph.header(f"🧘‍♀️ {pose}")

                # Metrics under title
        fps = 1 / (t1 - t0) if t1 > t0 else 0
        elapsed = time.time() - st.session_state.phase_start
        if phase == "prepare":
            rem = step.get("prep_time", 5) - elapsed
            if st.session_state.current_step > 0:
                prev = st.session_state.routine[st.session_state.current_step - 1]
                tr = prev.get("transition", "Neem houding aan.")
            else:
                tr = "Neem houding aan."
            phase_text = f"🔄 Voorbereiding: {int(rem)} s — {tr}"
        elif phase == "hold":
            rem = step.get("hold_time", 30) - elapsed
            phase_text = f"⏳ Houd vast: {int(rem)} s — Goed bezig!"
        else:
            phase_text = ""
        # Markdown voor metrics met newline
        metrics_text = (
            f"**Proc:** {(t1 - t0)*1000:.0f} ms — **{fps:.1f} fps**"
            f"{phase_text}"
        )
        metrics_ph.markdown(metrics_text)

        # Phase transitions
        now = time.time()
        if phase == "prepare" and rem <= 0:
            st.session_state.phase = "hold"
            st.session_state.phase_start = now
        elif phase == "hold" and rem <= 0:
            st.session_state.phase = "transition"
            st.session_state.phase_start = now
        elif phase == "transition":
            st.session_state.current_step += 1
            if st.session_state.current_step >= len(st.session_state.routine):
                st.session_state.running = False
            else:
                st.session_state.phase = "prepare"
                st.session_state.phase_start = now

        time.sleep(0.03)
    # Release camera when done
    st.session_state.cap.release()

# End graph
if (not st.session_state.running and not st.session_state.final_results_shown and
    any(st.session_state.score_log.values())):
    frame_ph.empty()
    title_ph.empty()
    metrics_ph.empty()
    with col2:
        st.header("Score per pose")
        fig, ax = plt.subplots()
        for p, scores in st.session_state.score_log.items():
            ax.plot(np.linspace(0, len(scores)/30, len(scores)), scores, label=p)
        ax.set(xlabel="Tijd (seconden)", ylabel="Score (%)", ylim=(0, 100))
        ax.legend()
        st.pyplot(fig)
    st.session_state.final_results_shown = True
