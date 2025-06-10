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
    time.sleep(0.1) # Give system a moment
    afconvert_cmd = f'afconvert -f WAVE -d LEI16@22050 "{aiff_path}" "{wav_path}"'
    if os.system(afconvert_cmd) != 0:
        print(f"[LOG] ⚠️ 'afconvert' fout voor bestand: {aiff_path}")
        return b""
    time.sleep(0.1) # Give system a moment
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

def play_tts(text: str):
    wav_bytes = generate_tts_wav_with_say(text)
    if not wav_bytes:
        print(f"[LOG] TTS falen: {text}")
        return
    b64 = base64.b64encode(wav_bytes).decode()
    html = f"<audio autoplay='true' style='display:none'><source src='data:audio/wav;base64,{b64}' type='audio/wav'></audio>"
    st.markdown(html, unsafe_allow_html=True)
    print(f"[LOG] TTS afgespeeld (autoplay): {text}")

feedback_queue = queue.Queue()

def get_summary_feedback(pose_name: str, avg_score: float) -> str:
    prompt = (
        f"You are a friendly yoga coach. The user performed '{pose_name}'. "
        f"Based on their performance, provide one concise, positive tip for improvement. "
        f"Ensure the tip is in English, less than 10 words, and contains no numbers."
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[LOG] fout bij get_summary_feedback: {e}")
        return "⚠️ Feedback not available"

# --- Globale configuratievariabelen ---
# Pas de onderstaande waarde aan naar de exacte duur van 'Camera_test.mp4' in seconden.
# Bijvoorbeeld, als de video 7.5 seconden duurt, zet CAMERA_CHECK_DURATION = 7.5
CAMERA_CHECK_DURATION = 11 # VOER HIER DE WERKELIJKE DUUR VAN JE VIDEO IN!
MODELPOSE_DIR = "modelposes_json"
ROUTINE_DIR   = "routines_json"
IMAGE_DIR     = "modelposes"
FEEDBACK_COUNT = 3
CAMERA_TEST_VIDEO_PATH = "Camera_test.mp4" # Path to your video file

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
        'tts_hold_test_done': False,
        'prev_step': None,
        'cap': None,
        'paused': False,
        'pause_start_time': None,
        're_render_pose_image': False # Add this new state variable
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if not st.session_state.pose_models:
        for fn in os.listdir("modelposes_json"):
            if fn.endswith('.json'):
                st.session_state.pose_models[fn] = json.load(open(os.path.join("modelposes_json", fn)))

init_state()

from ui import apply_styles, render_selection_screen, create_main_layout
apply_styles()

def on_select(fn):
    print(f"[LOG] on_select wordt aangeroepen met bestand: {fn}")
    st.session_state.selected = fn
    data = json.load(open(os.path.join(ROUTINE_DIR, fn)))
    st.session_state.routine_meta = data
    st.session_state.poses = data.get('poses', [])
    st.session_state.running = True
    st.session_state.current_step = 0
    st.session_state.prev_step = None
    st.session_state.phase = 'camera_check'
    st.session_state.phase_start = time.time()
    st.session_state.score_log = {}
    st.session_state.feedback_history = []
    st.session_state.feedback_triggered = []
    st.session_state.current_scores = []
    st.session_state.paused = False
    st.session_state.pause_start_time = None
    st.session_state.re_render_pose_image = False # Reset this on routine start

    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        st.session_state.cap = cap
        print("[LOG] camera gestart")

    st.session_state.img_shape = (427, 640)

    print(f"[DEBUG] Vaste bijsnijding camera naar (hoogte, breedte): {st.session_state.img_shape}")

    print("[LOG] on_select voltooid")

if not st.session_state.running:
    render_selection_screen(on_select)

# Bovenste rij voor knoppen en titel
top_cols = st.columns([1, 4, 1])
with top_cols[0]:
    if st.session_state.running:
        if st.button("⏯️ Pauze / Hervat"):
            st.session_state.paused = not st.session_state.paused
            if st.session_state.paused:
                st.session_state.pause_start_time = time.time()
            else:
                if st.session_state.pause_start_time is not None:
                    paused_duration = time.time() - st.session_state.pause_start_time
                    st.session_state.phase_start += paused_duration
                    st.session_state.pause_start_time = None
                    # Set flag to re-render pose image when resuming
                    st.session_state.re_render_pose_image = True
with top_cols[2]:
    if st.session_state.running:
        if st.button("⏹️ Stop Routine"):
            st.session_state.running = False
            st.session_state.cap.release()
            st.session_state.cap = None
            st.experimental_rerun()

pose_ph, frame_ph, title_ph, metrics_ph, timer_ph, feedback_ph = create_main_layout(top_cols[1])
VISUAL_FEEDBACK_ENABLED = True

if st.session_state.running:
    while st.session_state.running:
        if st.session_state.paused:
            title_ph.header("⏸️ Routine Gepauzeerd")
            time.sleep(0.1)
            continue

        idx       = st.session_state.current_step
        step      = st.session_state.poses[idx]
        pose_name = step['pose']
        now       = time.time()
        elapsed   = now - st.session_state.phase_start
        t0 = time.time()

        ret, frame = st.session_state.cap.read()
        if not ret:
            print("[LOG] Kan geen frame van camera lezen. Afsluiten.")
            st.session_state.running = False
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
            
        else:
            print("[DEBUG] st.session_state.img_shape is None, bijsnijden wordt overgeslagen. Dit zou niet moeten gebeuren met expliciete instelling.")


        annotated_frame = frame.copy()
        score = None

        if st.session_state.phase == 'camera_check':
            if st.session_state.current_step != 0:
                st.session_state.phase = 'prepare'
                st.session_state.phase_start = time.time()
                continue

            if st.session_state.prev_step != 'camera_check_init':
                title_ph.header(f"🔍 Camera Check")
                if os.path.exists(CAMERA_TEST_VIDEO_PATH):
                    pose_ph.video(CAMERA_TEST_VIDEO_PATH, format="video/mp4", start_time=0, loop=False, autoplay=True)
                else:
                    pose_ph.empty()
                    st.warning(f"Video file not found: {CAMERA_TEST_VIDEO_PATH}")
                metrics_ph.empty()
                feedback_ph.empty()
                st.session_state.prev_step = 'camera_check_init'


            remaining = max(0, int(CAMERA_CHECK_DURATION - elapsed))
            timer_ph.markdown(f"⏳ Zorg dat je goed zichtbaar bent ({remaining}s)...")

            frame_ph.image(frame, channels="BGR", use_container_width=True)

            if elapsed >= CAMERA_CHECK_DURATION:
                st.session_state.phase = 'prepare'
                st.session_state.phase_start = time.time()
                st.session_state.prev_step = None
                pose_ph.empty()
                timer_ph.empty()
                feedback_ph.empty() 
                metrics_ph.empty()
                continue

        else: # Phases: 'prepare', 'hold', 'transition'
            # --- START: Modified rendering logic for pose image ---
            if st.session_state.prev_step != idx or st.session_state.re_render_pose_image:
                title_ph.header(f"🧘‍♀️ {pose_name}")
                pose_ph.markdown(f"📌 **Model pose:** {pose_name}")
                raw = step.get('image_path', '')
                candidates = [ raw, os.path.join(IMAGE_DIR, raw), os.path.join(IMAGE_DIR, f"{pose_name}.jpg") ]
                found_image = False
                for p in candidates:
                    if p and os.path.isfile(p):
                        pose_ph.image(p, use_container_width=True)
                        found_image = True
                        break
                if not found_image:
                    pose_ph.empty()
                
                if idx == 0 and step.get('prep_time', 5) > 0 and st.session_state.phase == 'prepare':
                    feedback_ph.markdown(f"ℹ️ {st.session_state.routine_meta.get('description','')} ")
                
                st.session_state.prev_step = idx
                st.session_state.re_render_pose_image = False # Reset the flag after rendering
            # --- END: Modified rendering logic for pose image ---

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
                
                annotated_frame, _ = render_skeleton_frame(
                    frame.copy(),
                    {},
                    mode="landmarks"
                )

            elif st.session_state.phase == 'hold':
                hold_time = step.get('hold_time', 30)
                
                annotated_frame, score = render_skeleton_frame(
                    frame.copy(),
                    st.session_state.pose_models.get(f"{pose_name}.json", {}),
                    mode='full'
                )
                if score is not None:
                    st.session_state.current_scores.append(score)

                if VISUAL_FEEDBACK_ENABLED:
                    thresholds = [(i+1) * hold_time / FEEDBACK_COUNT for i in range(FEEDBACK_COUNT)]
                    for it, thr in enumerate(thresholds):
                        if elapsed >= thr and it not in st.session_state.feedback_triggered:
                            st.session_state.feedback_triggered.append(it)
                            def fetch_feedback_async(pose_name_inner, avg_s_inner):
                                tip_full = get_summary_feedback(
                                    pose_name_inner, avg_s_inner
                                )
                                feedback_queue.put(tip_full)

                            avg_s = (
                                sum(st.session_state.current_scores)
                                / len(st.session_state.current_scores)
                                if st.session_state.current_scores
                                else 0
                            )
                            threading.Thread(
                                target=fetch_feedback_async,
                                args=(pose_name, avg_s),
                                daemon=True,
                            ).start()
                            break

                if not feedback_queue.empty():
                    new_tip = feedback_queue.get()
                    st.session_state.feedback_history = [f"💡 {new_tip}"]
                    play_tts(new_tip)

                if VISUAL_FEEDBACK_ENABLED and st.session_state.feedback_history:
                    feedback_ph.markdown(st.session_state.feedback_history[-1])

                timer_ph.markdown(f"⏳ Houd vast: {int(hold_time - elapsed)} s")
                if elapsed >= hold_time:
                    if st.session_state.current_scores:
                        st.session_state.score_log.setdefault(pose_name, []).append(list(st.session_state.current_scores))
                    else:
                        st.session_state.score_log.setdefault(pose_name, []).append([])

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
                
                annotated_frame, _ = render_skeleton_frame(
                    frame.copy(),
                    {},
                    mode="landmarks"
                )

            if score is not None:
                txt = f"{score:.1f}%"
                (wt, ht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                ov = annotated_frame.copy()
                cv2.rectangle(ov, (5,5), (5+wt+10, 5+ht+10), (50,50,50), -1)
                annotated_frame = cv2.addWeighted(ov, 0.6, annotated_frame, 0.4, 0)
                cv2.putText(
                    annotated_frame,
                    txt,
                    (10, 10+ht),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255,255,255),
                    2
                )
            frame_ph.image(annotated_frame, channels='BGR', use_container_width=True)
            
        t1 = time.time()
        fps = 1/(t1-t0) if t1>t0 else 0
        metrics_ph.markdown(f"**Proc:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps**")
        
# --- START: Schoonmaken van de live-weergave na routine ---
pose_ph.empty()
frame_ph.empty()
title_ph.empty()
metrics_ph.empty()
timer_ph.empty()
feedback_ph.empty()

if st.session_state.cap is not None:
    _ = st.session_state.cap.release()
    st.session_state.cap = None
# --- EINDE: Schoonmaken van de live-weergave na routine ---

# Eindreview
if st.session_state.score_log:
    st.header("✨ Jouw Routine Overzicht ✨")
    for pn, scores_for_pose_attempts in st.session_state.score_log.items():
        for i, attempt_scores in enumerate(scores_for_pose_attempts):
            img_col, plot_col = st.columns([1,1])
            with img_col:
                tp = os.path.join(IMAGE_DIR, f"{pn}.jpg")
                if os.path.isfile(tp):
                    st.image(tp, use_container_width=True)
                else:
                    st.markdown(f"Model afbeelding voor **{pn}** niet gevonden.")
            with plot_col:
                avg = sum(attempt_scores)/len(attempt_scores) if attempt_scores else 0
                top = max(attempt_scores) if attempt_scores else 0
                
                st.markdown(f"## {pn} (Poging {i+1})")
                st.markdown(f"<p style='font-size:1.2em; color:gray;'>Today</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:1.5em; color:#00D09B; margin-bottom: 0.1em;'>Average Score <span style='float:right; font-weight:bold;'>{avg:.0f}%</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:1.5em; color:#00D09B;'>High Score <span style='float:right; font-weight:bold;'>{top:.0f}%</span></p>", unsafe_allow_html=True)
                
                if attempt_scores:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    ax.plot(attempt_scores, color='#00D09B', linewidth=2)
                    
                    ax.set_facecolor('#F8F8F8')
                    ax.grid(False)
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("Score (%)")
                    ax.set_xlabel("Meting (frames)")
                    
                    ax.axhline(y=70, color='#E0E0E0', linestyle='-', linewidth=1)
                    ax.axhline(y=80, color='#E0E0E0', linestyle='-', linewidth=1)
                    ax.set_yticks([0, 20, 40, 60, 70, 80, 100])
                    
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_color('gray')
                    ax.spines['bottom'].set_color('gray')
                    
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("Geen scoregegevens beschikbaar voor deze poging.")

                feedback = get_summary_feedback(pn, avg)
                st.info(feedback)
            st.markdown("---")

if st.button("⇦ Kies een andere routine"):
    st.session_state.running = False
    st.experimental_rerun()