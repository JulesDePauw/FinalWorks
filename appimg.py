import os
import time
import json
import cv2
import numpy as np
import requests
import streamlit as st
import tempfile
import uuid
import base64
import queue
from compare_pose_streamlit import render_skeleton_frame

from ui import apply_styles, render_selection_screen, create_main_layout
apply_styles()

MODELPOSE_DIR  = "modelposes_json"
ROUTINE_DIR    = "routines_json"
IMAGE_DIR      = "modelposes"
STATIC_IMAGE_NAME = "Dancer.jpg"
STATIC_IMAGE_PATH = os.path.join(IMAGE_DIR, STATIC_IMAGE_NAME)

DEFAULT_SAY_VOICE = "Ava"
FEEDBACK_COUNT = 3

def generate_tts_wav_with_say(text: str, voice: str = DEFAULT_SAY_VOICE) -> bytes:
    tmp_dir = tempfile.gettempdir()
    aiff_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.aiff")
    wav_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.wav")
    say_cmd = f'say -v "{voice}" -o "{aiff_path}" "{text}"'
    if os.system(say_cmd) != 0:
        print(f"[LOG] ⚠️ 'say' command failed for text: {text}")
        return b""
    time.sleep(0.1)
    afconvert_cmd = f'afconvert -f WAVE -d LEI16@22050 "{aiff_path}" "{wav_path}"'
    if os.system(afconvert_cmd) != 0:
        print(f"[LOG] ⚠️ 'afconvert' failed for file: {aiff_path}")
        return b""
    time.sleep(0.1)
    try:
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
    except Exception as e:
        print(f"[LOG] ⚠️ Cannot read WAV file: {e}")
        wav_bytes = b""
    finally:
        for p in [aiff_path, wav_path]:
            try:
                os.remove(p)
            except:
                pass
    return wav_bytes

def play_tts(text: str):
    wav_bytes = generate_tts_wav_with_say(text)
    if not wav_bytes:
        return
    b64 = base64.b64encode(wav_bytes).decode()
    html = (
        "<audio autoplay='true' style='display:none'>"
        f"<source src='data:audio/wav;base64,{b64}' type='audio/wav'></audio>"
    )
    st.markdown(html, unsafe_allow_html=True)

feedback_queue = queue.Queue()

def get_summary_feedback(pose_name: str, joint_norms: dict) -> str:
    sorted_joints = sorted(joint_norms.items(), key=lambda kv: kv[1], reverse=True)
    top_three = sorted_joints[:3]
    joint_list_str = ", ".join(f"{name}={norm:.2f}" for name, norm in top_three)
    prompt = (
        f"You are a yoga coach. For the pose '{pose_name}', "
        f"the top joint deviations are: {joint_list_str}. "
        f"Give one very concise tip (max 10 words) to correct the worst deviation."
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[LOG] Error in get_summary_feedback: {e}")
        return "⚠️ Feedback not available"

def init_state():
    defaults = {
        'pose_models': {}, 'running': False, 'selected': None,
        'poses': [], 'phase': None, 'phase_start': None,
        'img_shape': None, 'score_log': {}, 'feedback_history': [],
        'feedback_triggered': [], 'current_scores': [], 'tts_feedback_played': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if not st.session_state.pose_models:
        for fn in os.listdir(MODELPOSE_DIR):
            if fn.endswith('.json'):
                st.session_state.pose_models[fn] = json.load(
                    open(os.path.join(MODELPOSE_DIR, fn))
                )
init_state()

def on_select(fn):
    data = json.load(open(os.path.join(ROUTINE_DIR, fn)))
    st.session_state.selected = fn
    st.session_state.routine_meta = data
    st.session_state.poses = data.get('poses', [])
    st.session_state.running = True
    st.session_state.current_step = 0
    st.session_state.score_log = {}
    st.session_state.feedback_history = []
    st.session_state.feedback_triggered = []
    st.session_state.current_scores = []
    st.session_state.tts_feedback_played = False
    static_img = cv2.imread(STATIC_IMAGE_PATH)
    if static_img is None:
        st.error(f"Kan de afbeelding niet laden: {STATIC_IMAGE_PATH}")
        st.stop()
    st.session_state.static_img = cv2.flip(static_img, 1)
    if st.session_state.poses:
        first_img = st.session_state.poses[0].get('image_path', '')
        img0 = cv2.imread(os.path.join(IMAGE_DIR, first_img))
        st.session_state.img_shape = img0.shape[:2] if img0 is not None else None
    play_tts(f"Welcome to {data.get('name', fn.replace('.json',''))}, using static image mode.")

if not st.session_state.running:
    render_selection_screen(on_select)
pose_ph, frame_ph, title_ph, metrics_ph, timer_ph, feedback_ph = create_main_layout()

if st.session_state.running:
    idx = st.session_state.current_step
    step = st.session_state.poses[idx]
    pose_name = step['pose']
    pose_ph.markdown(f"📌 **Model pose:** {pose_name}")
    raw = step.get('image_path', '')
    for p in [raw, os.path.join(IMAGE_DIR, raw), os.path.join(IMAGE_DIR, f"{pose_name}.jpg")]:
        if p and os.path.isfile(p):
            pose_ph.image(p, use_container_width=True)
            break
    frame = cv2.flip(st.session_state.static_img.copy(), 1)
    if st.session_state.img_shape:
        th, tw = st.session_state.img_shape
        frame = cv2.resize(frame, (tw, th))
    annotated, score = render_skeleton_frame(
        frame.copy(),
        st.session_state.pose_models.get(f"{pose_name}.json", {}),
        mode='full'
    )
    disp_w = annotated.shape[1]
    frame_ph.image(annotated, channels='BGR', use_container_width=False, width=disp_w)
    if score is not None:
        title_ph.header(f"🧘‍♀️ {pose_name} – Score: {score:.1f}%")
    else:
        title_ph.header(f"🧘‍♀️ {pose_name}")
    if st.button("⇦ Kies een andere routine"):
        st.session_state.running = False
        st.experimental_rerun()
