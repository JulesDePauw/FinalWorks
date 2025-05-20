import streamlit as st
import time
import json
import os
import cv2
import numpy as np

ROUTINE_DIR = "routines_json"

# Dropdown met beschikbare routines
routine_files = [f for f in os.listdir(ROUTINE_DIR) if f.endswith(".json")]
routine_name = st.selectbox("Kies een yoga-routine:", routine_files)

# Inladen geselecteerde routine
with open(os.path.join(ROUTINE_DIR, routine_name), "r") as f:
    routine = json.load(f)

# Startknop
if st.button("Start routine"):
    for i, step in enumerate(routine):
        st.markdown(f"### Pose {i+1}: {step['pose']}")

        phase = st.empty()
        timer = st.empty()
        camera_placeholder = st.empty()

        # === VOORBEREIDING ===
        phase.markdown("**Voorbereiding**")
        prep_time = step.get("prep_time", 5)
        for sec in range(prep_time, 0, -1):
            timer.markdown(f"Neem de houding aan... **{sec}s**")
            time.sleep(1)

        # === HOUDING VASTHOUDEN (met camerafeed) ===
        phase.markdown("**Houding vasthouden**")
        hold_time = step.get("hold_time", 30)

        cap = cv2.VideoCapture(0)
        start_time = time.time()

        while time.time() - start_time < hold_time:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame, channels="RGB")

            remaining = int(hold_time - (time.time() - start_time))
            timer.markdown(f"Houd vast... **{remaining}s**")
            time.sleep(1 / 30)

        cap.release()

        # === OVERGANG ===
        phase.markdown("**Overgang**")
        timer.markdown(step['transition'])
        time.sleep(5)

        st.markdown("---")

    st.balloons()
    st.success("Routine voltooid!")
