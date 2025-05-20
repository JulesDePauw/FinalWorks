import streamlit as st
import os
import json
from compare_pose import compare_poses  # dit moet een functie worden in je script

# Paden
MODELPOSE_DIR = "modelposes_json"

# Modelpose kiezen
model_files = sorted(f for f in os.listdir(MODELPOSE_DIR) if f.endswith(".json"))
model_choice = st.selectbox("Kies een modelpose", model_files)

# Gebruikerspose uploaden
user_file = st.file_uploader("Upload je gebruikerspose (JSON)", type="json")

if model_choice and user_file:
    with open(os.path.join(MODELPOSE_DIR, model_choice)) as f:
        model_data = json.load(f)
    user_data = json.load(user_file)

    # Vergelijk
    feedback = compare_poses(model_data, user_data)

    # Toon resultaat
    st.subheader("Feedback op pose vergelijking")
    for angle, diff in feedback.items():
        st.write(f"{angle}: verschil = {diff:.2f}°")
