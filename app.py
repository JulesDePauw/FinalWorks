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

# --- Helperfuncties voor TTS via macOS 'say' met autoplay ---
DEFAULT_SAY_VOICE = "Ava"

def generate_tts_wav_with_say(text: str, voice: str = DEFAULT_SAY_VOICE) -> bytes:
    tmp_dir = tempfile.gettempdir()
    aiff_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.aiff")
    wav_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.wav")
    say_cmd = f'say -v "{voice}" -o "{aiff_path}" "{text}"'
    if os.system(say_cmd) != 0:
        print(f"[LOG] ⚠️ 'say' commandeerfout voor tekst: {text}")
        return b""
    time.sleep(0.1)
    afconvert_cmd = f'afconvert -f WAVE -d LEI16@22050 "{aiff_path}" "{wav_path}"'
    if os.system(afconvert_cmd) != 0:
        print(f"[LOG] ⚠️ 'afconvert' fout voor bestand: {aiff_path}")
        return b""
    time.sleep(0.1)
    try:
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
    except Exception as e:
        print(f"[LOG] ⚠️ Kan WAV niet lezen: {e}")
        wav_bytes = b""
    finally:
        for p in [aiff_path, wav_path]:
            try:
                os.remove(p)
            except:
                pass
    return wav_bytes

# Speel TTS via HTML <audio autoplay> tag
def play_tts(text: str):
    wav_bytes = generate_tts_wav_with_say(text)
    if not wav_bytes:
        print(f"[LOG] TTS falen: {text}")
        return
    b64 = base64.b64encode(wav_bytes).decode()
    html = f"<audio autoplay='true' style='display:none'><source src='data:audio/wav;base64,{b64}' type='audio/wav'></audio>"
    st.markdown(html, unsafe_allow_html=True)
    print(f"[LOG] TTS afgespeeld (autoplay): {text}")

# Queues voor feedback
import queue
feedback_queue = queue.Queue()

def get_summary_feedback(pose_name: str, avg_score: float) -> str:
    prompt = (
        f"Je bent een vriendelijke yoga-coach die specifiek ingaat op lichaamshouding en gewrichtsuitlijning. "
        f"De gebruiker voerde de pose '{pose_name}' uit met een gemiddelde nauwkeurigheid van {avg_score:.1f}% volgens onze metingen. "
        f"Noem in één beknopte, positieve zin een concrete tip gebaseerd op de gemeten nauwkeurigheid."
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[LOG] fout bij get_summary_feedback: {e}")
        return "⚠️ Feedback niet beschikbaar"

# Initialiseer session_state defaults
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
        'tts_test_done': False,
        'tts_hold_test_done': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if not st.session_state.pose_models:
        for fn in os.listdir("modelposes_json"):
            if fn.endswith('.json'):
                st.session_state.pose_models[fn] = json.load(open(os.path.join("modelposes_json", fn)))

init_state()

# UI-styling
from ui import apply_styles, render_selection_screen, create_main_layout
apply_styles()

# Directories
MODELPOSE_DIR = "modelposes_json"
ROUTINE_DIR   = "routines_json"
IMAGE_DIR     = "modelposes"
FEEDBACK_COUNT = 3

# Callback om routine te starten
def on_select(fn):
    print(f"[LOG] on_select wordt aangeroepen met bestand: {fn}")
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

    # Start camera
    cap = cv2.VideoCapture(0)
    st.session_state.cap = cap
    print("[LOG] camera gestart")

    if st.session_state.poses:
        first_img = st.session_state.poses[0].get('image_path', '')
        img0 = cv2.imread(os.path.join(IMAGE_DIR, first_img))
        st.session_state.img_shape = img0.shape[:2] if img0 is not None else None
    else:
        st.session_state.img_shape = None

    # Welkomst-TTS
    practiceoutine = data.get('name', fn.replace('.json', ''))
    welcome_text = f"Welcome to {practiceoutine}, please make sure you are fully visible for the camera."
    play_tts(welcome_text)
    print("[LOG] on_select voltooid")

# Hoofd UI
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
            st.session_state.tts_hold_test_done = False
            continue
        else:
            timer_ph.markdown(f"⏳ Voorbereiding: {int(prep_time - elapsed)} s")

    elif st.session_state.phase == 'hold':
        if not st.session_state.tts_hold_test_done:
            play_tts("Audio Test 2.")
            st.session_state.tts_hold_test_done = True
        hold_time = step.get('hold_time', 30)
        ret_sb, frame_sb = st.session_state.cap.read()
        if ret_sb:
            frame_sb = cv2.flip(frame_sb, 1)
            annotated_sb, score_sb = render_skeleton_frame(
                frame_sb.copy(),
                st.session_state.pose_models.get(f"{pose_name}.json", {}),
                mode='full'
            )
            frame_ph.image(annotated_sb, channels='BGR', use_container_width=True)
            if score_sb is not None:
                st.session_state.current_scores.append(score_sb)

        if VISUAL_FEEDBACK_ENABLED:
            thresholds = [(i+1) * hold_time / FEEDBACK_COUNT for i in range(FEEDBACK_COUNT)]
            for it, thr in enumerate(thresholds):
                if elapsed >= thr and it not in st.session_state.feedback_triggered:
                    st.session_state.feedback_triggered.append(it)
                    def fetch_and_play(pose_name_inner, avg_s_inner):
                        tip_full = get_summary_feedback(pose_name_inner, avg_s_inner)
                        feedback_queue.put(tip_full)
                        play_tts(tip_full)
                    avg_s = sum(st.session_state.current_scores)/len(st.session_state.current_scores) if st.session_state.current_scores else 0
                    threading.Thread(target=fetch_and_play, args=(pose_name, avg_s), daemon=True).start()
                    break

        if not feedback_queue.empty():
            new_tip = feedback_queue.get()
            st.session_state.feedback_history = [f"💡 {new_tip}"]
        if VISUAL_FEEDBACK_ENABLED and st.session_state.feedback_history:
            feedback_ph.markdown(st.session_state.feedback_history[-1])

        timer_ph.markdown(f"⏳ Houd vast: {int(hold_time - elapsed)} s")
        if elapsed >= hold_time:
            final_avg = sum(st.session_state.current_scores)/len(st.session_state.current_scores) if st.session_state.current_scores else 0
            st.session_state.score_log.setdefault(pose_name, []).append(final_avg)
            st.session_state.phase = 'transition'
            st.session_state.phase_start = now
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
            timer_ph.markdown(f"⏳ Volgende pose over: {int(transition_time - elapsed)} s")

    ret, frame = st.session_state.cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if st.session_state.img_shape:
        th, tw = st.session_state.img_shape
        h, w = frame.shape[:2]
        if w/h > tw/th:
            nw = int(h * tw / th)
            x0 = (w - nw)//2
            frame = frame[:, x0:x0+nw]
        else:
            nh = int(w * th / tw)
            y0 = (h - nh)//2
            frame = frame[y0:y0+nh, :]
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
        cv2.putText(
            annot,
            txt,
            (10, 10+ht),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255,255,255),
            2
        )
    frame_ph.image(annot, channels='BGR', use_container_width=True)
    title_ph.header(f"🧘‍♀️ {pose_name}")
    fps = 1/(t1-t0) if t1>t0 else 0
    metrics_ph.markdown(f"**Proc:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps**")
    time.sleep(0.03)

# Eindreview
_ = st.session_state.cap.release()
if st.session_state.score_log:
    for pn, sc in st.session_state.score_log.items():
        img_col, plot_col = st.columns([1,1])
        with img_col:
            tp = os.path.join(IMAGE_DIR, f"{pn}.jpg")
            if os.path.isfile(tp):
                st.image(tp, use_container_width=True)
        with plot_col:
            avg = sum(sc)/len(sc)
            top = max(sc)
            st.markdown(f"## {pn}")
            st.write(f"**Topscore:** {top:.1f}% — **Gem.score:** {avg:.1f}%")
            feedback = get_summary_feedback(pn, avg)
            st.info(feedback)

if st.button("⇦ Kies een andere routine"):
    st.session_state.running = False
    st.experimental_rerun()
