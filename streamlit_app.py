import streamlit as st
import os
import json
from compare_pose_streamlit import run_camera_loop

MODELPOSE_DIR = "modelposes_json"
IMAGE_DIR = "modelposes"
ROUTINE_DIR = "routines_json"

st.set_page_config(layout="wide", page_title="🧘 Yoga Routine Feedback")

st.sidebar.title("📌 Kies een yoga-routine")
routine_files = sorted([f for f in os.listdir(ROUTINE_DIR) if f.endswith(".json")])
selected_routine = st.sidebar.selectbox("Selecteer een routine:", routine_files)

title_placeholder = st.empty()
camera_placeholder = st.empty()
timer_placeholder = st.empty()
text_placeholder = st.empty()
modelpose_sidebar_placeholder = st.sidebar.empty()

if st.button("Start routine") and selected_routine:
    routine_path = os.path.join(ROUTINE_DIR, selected_routine)
    with open(routine_path, "r") as f:
        routine = json.load(f)

    routine_steps = []
    for step in routine:
        pose_label = step["pose"]
        modelpose_path = os.path.join(MODELPOSE_DIR, f"{pose_label}.json")
        with open(modelpose_path, "r") as f:
            model_data = json.load(f)
        image_path = os.path.join(IMAGE_DIR, model_data["filename"])
        prep_time = step.get("prep_time", 5)
        hold_time = step.get("hold_time", 30)
        transition_text = step.get("transition", "Bereid je voor op de volgende pose.")
        routine_steps.append({
            'pose_json': modelpose_path,
            'hold_time': hold_time,
            'label': pose_label,
            'transition_text': transition_text,
            'image_path': image_path,
            'prep_time': prep_time
        })

    run_camera_loop(
        routine_steps,
        camera_placeholder,
        feedback_text_area=text_placeholder,
        timer_area=timer_placeholder,
        sidebar_modelpose_placeholder=modelpose_sidebar_placeholder,
        title_placeholder=title_placeholder
    )

    st.balloons()
    title_placeholder.markdown("## ✅ Routine voltooid")
    timer_placeholder.markdown("")
    text_placeholder.markdown("Je hebt alle poses afgewerkt. 👏")
