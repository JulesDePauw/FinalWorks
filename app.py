import os
import time
import json
import cv2
import numpy as np
import requests
import streamlit as st
from compare_pose_streamlit import render_skeleton_frame
  # we will patch this to also return joint_norms
import threading
import uuid
import tempfile
import base64
import queue

# UI styling
from ui import apply_styles, render_selection_screen, create_main_layout
apply_styles()

# Directories
MODELPOSE_DIR  = "modelposes_json"
ROUTINE_DIR    = "routines_json"
IMAGE_DIR      = "modelposes"
FEEDBACK_COUNT = 3

# --- Helper functions for TTS via macOS 'say' with autoplay ---
DEFAULT_SAY_VOICE = "Ava"

def generate_tts_wav_with_say(text: str, voice: str = DEFAULT_SAY_VOICE) -> bytes:
    tmp_dir   = tempfile.gettempdir()
    aiff_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.aiff")
    wav_path  = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.wav")
    say_cmd   = f'say -v "{voice}" -o "{aiff_path}" "{text}"'
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
        print(f"[LOG] TTS failed: {text}")
        return
    b64  = base64.b64encode(wav_bytes).decode()
    html = (
        "<audio autoplay='true' style='display:none'>"
        f"<source src='data:audio/wav;base64,{b64}' type='audio/wav'></audio>"
    )
    st.markdown(html, unsafe_allow_html=True)
    print(f"[LOG] TTS played (autoplay): {text}")

# Queues for feedback
feedback_queue = queue.Queue()

def get_summary_feedback(pose_name: str, joint_norms: dict) -> str:
    """
    Build a prompt listing each joint’s normalized deviation (0–1),
    then ask the LLM to give a single, concise tip on the worst one.
    """
    # Sort joints by descending norm:
    sorted_joints = sorted(joint_norms.items(), key=lambda kv: kv[1], reverse=True)

    # Build a short list: e.g. "left_elbow=0.72, right_knee=0.56, left_ankle=0.33"
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

# Initialize session_state defaults
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

# Callback to start routine
def on_select(fn):
    print(f"[LOG] on_select called with file: {fn}")
    st.session_state.selected = fn
    data = json.load(open(os.path.join(ROUTINE_DIR, fn)))
    st.session_state.routine_meta      = data
    st.session_state.poses              = data.get('poses', [])
    st.session_state.running            = True
    st.session_state.current_step       = 0
    st.session_state.prev_step          = None
    st.session_state.phase              = 'prepare'
    st.session_state.phase_start        = time.time()
    st.session_state.score_log          = {}
    st.session_state.feedback_history   = []
    st.session_state.feedback_triggered = []
    st.session_state.current_scores     = []
    st.session_state.tts_feedback_played= False

    # Start camera
    cap = cv2.VideoCapture(0)
    st.session_state.cap = cap
    print("[LOG] Camera started")

    if st.session_state.poses:
        first_img = st.session_state.poses[0].get('image_path', '')
        img0      = cv2.imread(os.path.join(IMAGE_DIR, first_img))
        st.session_state.img_shape = img0.shape[:2] if img0 is not None else None
    else:
        st.session_state.img_shape = None

    # Welcome TTS
    routine_name = data.get('name', fn.replace('.json', ''))
    welcome_text = (
        f"Welcome to {routine_name}, please ensure you are fully visible to the camera."
    )
    play_tts(welcome_text)
    print("[LOG] on_select completed")

# Main UI
if not st.session_state.running:
    render_selection_screen(on_select)

pose_ph, frame_ph, title_ph, metrics_ph, timer_ph, feedback_ph = create_main_layout()
VISUAL_FEEDBACK_ENABLED = True

while st.session_state.running:
    idx       = st.session_state.current_step
    step      = st.session_state.poses[idx]
    pose_name = step['pose']
    now       = time.time()
    elapsed   = now - st.session_state.phase_start

    if st.session_state.prev_step != idx:
        pose_ph.markdown(f"📌 **Model pose:** {pose_name}")
        raw = step.get('image_path', '')
        candidates = [
            raw,
            os.path.join(IMAGE_DIR, raw),
            os.path.join(IMAGE_DIR, f"{pose_name}.jpg")
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                pose_ph.image(p, use_container_width=True)
                break
        if idx == 0 and step.get('prep_time', 5) > 0:
            feedback_ph.markdown(
                f"ℹ️ {st.session_state.routine_meta.get('description','')}"
            )
        st.session_state.prev_step = idx

    if st.session_state.phase == 'prepare':
        prep_time = step.get('prep_time', 5)
        if elapsed >= prep_time:
            st.session_state.phase = 'hold'
            st.session_state.phase_start = now
            st.session_state.feedback_history   = []
            st.session_state.feedback_triggered = []
            st.session_state.current_scores     = []
            st.session_state.tts_feedback_played= False
            continue
        else:
            timer_ph.markdown(f"⏳ Preparation: {int(prep_time - elapsed)} s")

    elif st.session_state.phase == 'hold':
        hold_time  = step.get('hold_time', 30)
        ret_sb, frame_sb = st.session_state.cap.read()
        if ret_sb:
            frame_sb = cv2.flip(frame_sb, 1)

            # Crop & resize camera frame to match model dimensions
            if st.session_state.img_shape:
                th, tw = st.session_state.img_shape
            else:
                ref_path = os.path.join(IMAGE_DIR, f"{pose_name}.jpg")
                ref_img  = cv2.imread(ref_path)
                if ref_img is not None:
                    th, tw = ref_img.shape[:2]
                    st.session_state.img_shape = (th, tw)
                else:
                    th, tw = frame_sb.shape[:2]

            h_sb, w_sb       = frame_sb.shape[:2]
            target_ratio     = tw / th
            current_ratio    = w_sb / h_sb

            if current_ratio > target_ratio:
                new_w = int(h_sb * target_ratio)
                x0    = (w_sb - new_w) // 2
                frame_cropped = frame_sb[:, x0 : x0 + new_w]
            else:
                new_h = int(w_sb / target_ratio)
                y0    = (h_sb - new_h) // 2
                frame_cropped = frame_sb[y0 : y0 + new_h, :]

            frame_resized  = cv2.resize(frame_cropped, (tw, th))

            # *** Here we expect render_skeleton_frame to return joint_norms ***
            annotated_sb, score_sb, joint_norms = render_skeleton_frame(
                frame_resized.copy(),
                st.session_state.pose_models.get(f"{pose_name}.json", {}),
                mode='full'
            )

            # Display hold-phase camera feed at fixed width
            frame_ph.image(
                annotated_sb,
                channels='BGR',
                use_container_width=False,
                width=tw
            )

                        # Append overall score (if present)
            if score_sb is not None:
                st.session_state.current_scores.append(score_sb)
            # Schedule feedback at intervals
            # Custom intervals: at 1/6, 3/6, 5/6 of hold_time
            multipliers = [i*2+1 for i in range(FEEDBACK_COUNT)]
            thresholds = [m * hold_time / (2 * FEEDBACK_COUNT) for m in multipliers]
            for it, thr in enumerate(thresholds):
                if elapsed >= thr and it not in st.session_state.feedback_triggered:
                    st.session_state.feedback_triggered.append(it)
                    def fetch_feedback(pose_name_inner, joint_norms_inner):
                        tip_full = get_summary_feedback(pose_name_inner, joint_norms_inner)
                        feedback_queue.put(tip_full)
                    threading.Thread(
                        target=fetch_feedback,
                        args=(pose_name, joint_norms),
                        daemon=True
                    ).start()
                    break
            # Read feedback from queue and display
            if not feedback_queue.empty():
                new_tip = feedback_queue.get()
                st.session_state.feedback_history.append(new_tip)
                feedback_ph.markdown(f"💡 {new_tip}")
                play_tts(new_tip)

        # Show timer
        timer_ph.markdown(f"⏳ Hold: {int(hold_time - elapsed)} s")
        if elapsed >= hold_time:
            st.session_state.phase       = 'transition'
            st.session_state.phase_start = now
        continue  # skip main camera loop during hold

    elif st.session_state.phase == 'transition':
        transition_time = step.get('transition_time', 3)
        if elapsed >= transition_time:
            feedback_ph.markdown(f"🔄 {step.get('transition', '')}")
            st.session_state.feedback_history   = []
            st.session_state.feedback_triggered = []
            st.session_state.current_scores     = []
            st.session_state.current_step       += 1
            if st.session_state.current_step >= len(st.session_state.poses):
                st.session_state.running = False
                break
            st.session_state.phase       = 'prepare'
            st.session_state.phase_start = time.time()
            st.session_state.prev_step   = None
            continue
        else:
            timer_ph.markdown(f"⏳ Next pose in: {int(transition_time - elapsed)} s")

    # Main camera loop (for view before 'prepare', or fallback)
    ret, frame = st.session_state.cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Crop & resize camera to match model aspect ratio
    if st.session_state.img_shape:
        th, tw = st.session_state.img_shape
    else:
        ref_path = os.path.join(IMAGE_DIR, f"{pose_name}.jpg")
        ref_img  = cv2.imread(ref_path)
        if ref_img is not None:
            th, tw = ref_img.shape[:2]
            st.session_state.img_shape = (th, tw)
        else:
            th, tw = frame.shape[:2]

    h, w         = frame.shape[:2]
    target_ratio = tw / th
    current_ratio= w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0    = (w - new_w) // 2
        frame = frame[:, x0 : x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0    = (h - new_h) // 2
        frame = frame[y0 : y0 + new_h, :]

    frame = cv2.resize(frame, (tw, th))
    t0 = time.time()
    annot, score, _ = render_skeleton_frame(
        frame.copy(),
        st.session_state.pose_models.get(f"{pose_name}.json", {}),
        mode='full'
    )
    t1 = time.time()

    if score is not None:
        txt         = f"{score:.1f}%"
        (wt, ht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        ov          = annot.copy()
        cv2.rectangle(ov, (5,5), (5+wt+10, 5+ht+10), (50,50,50), -1)
        annot = cv2.addWeighted(ov, 0.6, annot, 0.4, 0)
        cv2.putText(
            annot, txt, (10, 10+ht),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2
        )
        # Display camera with fixed width
        frame_ph.image(
            annot,
            channels='BGR',
            use_container_width=False,
            width=tw
        )

    title_ph.header(f"🧘‍♀️ {pose_name}")
    fps = 1/(t1-t0) if t1 > t0 else 0
    metrics_ph.markdown(
        f"**Processing time:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps**"
    )
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
            # For review, show final feedback for each pose at end:
            feedback = get_summary_feedback(pn, {"overall": 1.0})  # filled just to show syntax
            st.info(feedback)

if st.button("⇦ Choose another routine"):
    st.session_state.running = False
    st.experimental_rerun()
