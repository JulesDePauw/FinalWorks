import streamlit as st
import os
import json
import time
from compare_pose_streamlit import run_streamlit_feedback

MODELPOSE_DIR = "modelposes_json"
IMAGE_DIR = "modelposes"
ROUTINE_DIR = "routines_json"

st.set_page_config(layout="wide", page_title="🧘 Yoga Routine Feedback")

# === Sidebar ===
st.sidebar.title("📌 Kies een yoga-routine")
routine_files = sorted([f for f in os.listdir(ROUTINE_DIR) if f.endswith(".json")])
selected_routine = st.sidebar.selectbox("Selecteer een routine:", routine_files)

# === Placeholders voor dynamische content ===
title_placeholder = st.empty()
camera_placeholder = st.empty()
timer_placeholder = st.empty()
text_placeholder = st.empty()

# === Startknop ===
if st.button("Start routine") and selected_routine:
    routine_path = os.path.join(ROUTINE_DIR, selected_routine)
    with open(routine_path, "r") as f:
        routine = json.load(f)

    session_log = []

    for i, step in enumerate(routine):
        pose_label = step["pose"]
        modelpose_path = os.path.join(MODELPOSE_DIR, f"{pose_label}.json")

        # === Sidebar modelpose image ===
        with open(modelpose_path, "r") as f:
            model_data = json.load(f)
        image_path = os.path.join(IMAGE_DIR, model_data["filename"])
        st.sidebar.image(image_path, caption=pose_label, use_container_width=True)

        # === Titel aanpassen ===
        title_placeholder.markdown(f"## Pose {i+1}: {pose_label}")

        # === Voorbereidingstijd ===
        prep_time = step.get("prep_time", 5)
        for sec in range(prep_time, 0, -1):
            timer_placeholder.markdown(f"🌀 **Voorbereiding**: start in {sec}s")
            time.sleep(1)

        # === Pose vasthouden ===
        hold_time = step.get("hold_time", 30)
        timer_placeholder.markdown(f"⏳ **Pose actief: {hold_time} seconden**")
        text_placeholder.markdown("Live feedback wordt gegenereerd...")

        score = run_streamlit_feedback(
            modelpose_path=modelpose_path,
            streamlit_placeholder=camera_placeholder,
            hold_time=hold_time,
            feedback_text_area=text_placeholder,
            timer_area=timer_placeholder
        )

        session_log.append({
            "pose": pose_label,
            "score": score,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })

        # === Overgang naar volgende ===
        transition_text = step.get("transition", "Bereid je voor op de volgende pose.")
        for sec in range(5, 0, -1):
            timer_placeholder.markdown(f"🧭 **Overgang**: {transition_text} ({sec}s)")
            text_placeholder.empty()
            time.sleep(1)

    st.balloons()
    title_placeholder.markdown("## ✅ Routine voltooid")
    timer_placeholder.markdown("")
    text_placeholder.markdown("Je hebt alle poses afgewerkt. 👏")
