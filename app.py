import os
import time
import json
import cv2
import numpy as np
import requests
import streamlit as st
from compare_pose_streamlit import render_skeleton_frame
import matplotlib.pyplot as plt
import threading
import uuid
import tempfile
import subprocess
import base64
import queue

# UI styling
from ui import apply_styles, render_selection_screen, create_main_layout
apply_styles()

# Directories
MODELPOSE_DIR = "modelposes_json"
ROUTINE_DIR   = "routines_json"
IMAGE_DIR     = "modelposes"
FEEDBACK_COUNT = 3

# --- Helper functions for TTS via macOS 'say' with autoplay ---
DEFAULT_SAY_VOICE = "Ava"

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

# Play TTS via HTML <audio autoplay> tag
def play_tts(text: str):
    wav_bytes = generate_tts_wav_with_say(text)
    if not wav_bytes:
        print(f"[LOG] TTS failed: {text}")
        return
    b64 = base64.b64encode(wav_bytes).decode()
    html = f"<audio autoplay='true' style='display:none'><source src='data:audio/wav;base64,{b64}' type='audio/wav'></audio>"
    st.markdown(html, unsafe_allow_html=True)
    print(f"[LOG] TTS played (autoplay): {text}")

# Queues for feedback
feedback_queue = queue.Queue()

def get_summary_feedback(pose_name: str, avg_score: float) -> str:
    prompt = (
        f"You are a friendly yoga coach. "
        f"Provide one very concise tip (max 5 words) to improve the user's posture."
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[LOG] Error in get_summary_feedback: {e}")
        return "⚠️ Feedback not available"().get("response", "").strip()
    except Exception as e:
        print(f"[LOG] Error in get_summary_feedback: {e}")
        return "⚠️ Feedback not available".get("response", "").strip()
    except Exception as e:
        print(f"[LOG] Error in get_summary_feedback: {e}")
        return "⚠️ Feedback not available"

# Initialize session_state defaults
def init_state():
    defaults = {
        'pose_models': {},
        'running': False,
        'selected': None,
        'poses': [],
        'phase': None,
        'phase_start': None,
        'img_shape': None,
        'score_log': {},
        'feedback_history': [],
        'feedback_triggered': [],
        'current_scores': [],
        'tts_feedback_played': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if not st.session_state.pose_models:
        for fn in os.listdir(MODELPOSE_DIR):
            if fn.endswith('.json'):
                st.session_state.pose_models[fn] = json.load(open(os.path.join(MODELPOSE_DIR, fn)))

init_state()

# Callback to start routine
def on_select(fn):
    print(f"[LOG] on_select called with file: {fn}")
    st.session_state.selected = fn
    data = json.load(open(os.path.join(ROUTINE_DIR, fn)))
    st.session_state.routine_meta = data
    st.session_state.poses = data.get('poses', [])
    st.session_state.running = True
    st.session_state.current_step = 0
    st.session_state.prev_step = None
    st.session_state.phase = 'prepare'
    st.session_state.phase_start = time.time()
    st.session_state.score_log = {}
    st.session_state.feedback_history = []
    st.session_state.feedback_triggered = []
    st.session_state.current_scores = []
    st.session_state.tts_feedback_played = False

    # Start camera
    cap = cv2.VideoCapture(0)
    st.session_state.cap = cap
    print("[LOG] Camera started")

    if st.session_state.poses:
        first_img = st.session_state.poses[0].get('image_path', '')
        img0 = cv2.imread(os.path.join(IMAGE_DIR, first_img))
        st.session_state.img_shape = img0.shape[:2] if img0 is not None else None
    else:
        st.session_state.img_shape = None

    # Welcome TTS
    routine_name = data.get('name', fn.replace('.json', ''))
    welcome_text = f"Welcome to {routine_name}, please ensure you are fully visible to the camera."
    play_tts(welcome_text)
    print("[LOG] on_select completed")

# Main UI
if not st.session_state.running:
    render_selection_screen(on_select)

pose_ph, frame_ph, title_ph, metrics_ph, timer_ph, feedback_ph = create_main_layout()
VISUAL_FEEDBACK_ENABLED = True

while st.session_state.running:
    idx = st.session_state.current_step
    step = st.session_state.poses[idx]
    pose_name = step['pose']
    now = time.time()
    elapsed = now - st.session_state.phase_start

    if st.session_state.prev_step != idx:
        pose_ph.markdown(f"📌 **Model pose:** {pose_name}")
        raw = step.get('image_path', '')
        candidates = [ raw, os.path.join(IMAGE_DIR, raw), os.path.join(IMAGE_DIR, f"{pose_name}.jpg") ]
        for p in candidates:
            if p and os.path.isfile(p):
                pose_ph.image(p, use_container_width=True)
                break
        if idx == 0 and step.get('prep_time', 5) > 0:
            feedback_ph.markdown(f"ℹ️ {st.session_state.routine_meta.get('description','')} ")
        st.session_state.prev_step = idx

    if st.session_state.phase == 'prepare':
        prep_time = step.get('prep_time', 5)
        if elapsed >= prep_time:
            st.session_state.phase = 'hold'
            st.session_state.phase_start = now
            st.session_state.feedback_history = []
            st.session_state.feedback_triggered = []
            st.session_state.current_scores = []
            st.session_state.tts_feedback_played = False
            continue
        else:
            timer_ph.markdown(f"⏳ Preparation: {int(prep_time - elapsed)} s")

    elif st.session_state.phase == 'hold':
        hold_time = step.get('hold_time', 30)
        ret_sb, frame_sb = st.session_state.cap.read()
        if ret_sb:
            frame_sb = cv2.flip(frame_sb, 1)
            # Crop and resize camera frame to match model pose dimensions before annotation
            if st.session_state.img_shape:
                th, tw = st.session_state.img_shape
            else:
                ref_path = os.path.join(IMAGE_DIR, f"{pose_name}.jpg")
                ref_img = cv2.imread(ref_path)
                if ref_img is not None:
                    th, tw = ref_img.shape[:2]
                    st.session_state.img_shape = (th, tw)
                else:
                    th, tw = frame_sb.shape[:2]
            h_sb, w_sb = frame_sb.shape[:2]
            target_ratio = tw / th
            current_ratio = w_sb / h_sb
            if current_ratio > target_ratio:
                new_w = int(h_sb * target_ratio)
                x0 = (w_sb - new_w) // 2
                frame_cropped = frame_sb[:, x0:x0 + new_w]
            else:
                new_h = int(w_sb / target_ratio)
                y0 = (h_sb - new_h) // 2
                frame_cropped = frame_sb[y0:y0 + new_h, :]
            frame_resized = cv2.resize(frame_cropped, (tw, th))
            annotated_sb, score_sb = render_skeleton_frame(
                frame_resized.copy(),
                st.session_state.pose_models.get(f"{pose_name}.json", {}),
                mode='full'
            )
            # Display hold-phase camera feed with fixed width
            frame_ph.image(annotated_sb, channels='BGR', use_container_width=False, width=tw)
            if score_sb is not None:
                st.session_state.current_scores.append(score_sb)

        if VISUAL_FEEDBACK_ENABLED:
            thresholds = [(i+1) * hold_time / FEEDBACK_COUNT for i in range(FEEDBACK_COUNT)]
            for it, thr in enumerate(thresholds):
                if elapsed >= thr and it not in st.session_state.feedback_triggered:
                    st.session_state.feedback_triggered.append(it)
                    def fetch_feedback(pose_name_inner, avg_s_inner):
                        tip_full = get_summary_feedback(pose_name_inner, avg_s_inner)
                        feedback_queue.put(tip_full)
                    avg_s = (sum(st.session_state.current_scores) / len(st.session_state.current_scores)) if st.session_state.current_scores else 0
                    threading.Thread(target=fetch_feedback, args=(pose_name, avg_s), daemon=True).start()
                    break

        if not st.session_state.tts_feedback_played and not feedback_queue.empty():
            new_tip = feedback_queue.get()
            st.session_state.feedback_history = [f"💡 {new_tip}"]
            play_tts(new_tip)
            st.session_state.tts_feedback_played = True

        if VISUAL_FEEDBACK_ENABLED and st.session_state.feedback_history:
            feedback_ph.markdown(st.session_state.feedback_history[-1])

        timer_ph.markdown(f"⏳ Hold: {int(hold_time - elapsed)} s")
        if elapsed >= hold_time:
            final_avg = (sum(st.session_state.current_scores) / len(st.session_state.current_scores)) if st.session_state.current_scores else 0
            st.session_state.score_log.setdefault(pose_name, []).append(final_avg)
            st.session_state.phase = 'transition'
            st.session_state.phase_start = now
        # Skip main frame display when in hold
        continue

    elif st.session_state.phase == 'transition':
        transition_time = step.get('transition_time', 3)
        if elapsed >= transition_time:
            feedback_ph.markdown(f"🔄 {step.get('transition', '')}")
            st.session_state.feedback_history = []
            st.session_state.feedback_triggered = []
            st.session_state.current_scores = []
            st.session_state.current_step += 1
            if st.session_state.current_step >= len(st.session_state.poses):
                st.session_state.running = False
                break
            st.session_state.phase = 'prepare'
            st.session_state.phase_start = time.time()
            st.session_state.prev_step = None
            continue
        else:
            timer_ph.markdown(f"⏳ Next pose in: {int(transition_time - elapsed)} s")

    ret, frame = st.session_state.cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Crop and resize camera to match model pose aspect ratio and dimensions
    # Load model reference image dimensions if not already loaded
    if st.session_state.img_shape:
        th, tw = st.session_state.img_shape  # target height, width from model image
    else:
        # Fallback: read the current pose reference image directly
        ref_path = os.path.join(IMAGE_DIR, f"{pose_name}.jpg")
        ref_img = cv2.imread(ref_path)
        if ref_img is not None:
            th, tw = ref_img.shape[:2]
            st.session_state.img_shape = (th, tw)
        else:
            th, tw = frame.shape[:2]
    h, w = frame.shape[:2]
    target_ratio = tw / th
    current_ratio = w / h
    # Crop to match aspect ratio
    if current_ratio > target_ratio:
        # frame is wider; crop sides
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        # frame is taller; crop top/bottom
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    # Finally, resize to exactly model image dimensions
    frame = cv2.resize(frame, (tw, th))
    t0 = time.time()
    annot, score = render_skeleton_frame(
        frame.copy(),
        st.session_state.pose_models.get(f"{pose_name}.json", {}),
        mode='full'
    )
    t1 = time.time()
    if score is not None:
        txt = f"{score:.1f}%"
        (wt, ht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        ov = annot.copy()
        cv2.rectangle(ov, (5,5), (5+wt+10, 5+ht+10), (50,50,50), -1)
        annot = cv2.addWeighted(ov, 0.6, annot, 0.4, 0)
        cv2.putText(annot, txt, (10, 10+ht), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
    # Display camera with fixed width to prevent resizing each frame
        frame_ph.image(annot, channels='BGR', use_container_width=False, width=tw)
    title_ph.header(f"🧘‍♀️ {pose_name}")
    fps = 1/(t1-t0) if t1>t0 else 0
    metrics_ph.markdown(f"**Processing time:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps**")
    time.sleep(0.03)

# End review
_ = st.session_state.cap.release()
if st.session_state.score_log:
    for pn, sc in st.session_state.score_log.items():
        img_col, plot_col = st.columns([1,1])
        with img_col:
            tp = os.path.join(IMAGE_DIR, f"{pn}.jpg")
            if os.path.isfile(tp):
                st.image(tp, use_container_width=True)
        with plot_col:
            avg = (sum(sc)/len(sc))
            top = max(sc)
            st.markdown(f"## {pn}")
            st.write(f"**Top score:** {top:.1f}% — **Avg. score:** {avg:.1f}%")
            feedback = get_summary_feedback(pn, avg)
            st.info(feedback)

if st.button("⇦ Choose another routine"):
    st.session_state.running = False
    st.experimental_rerun()
