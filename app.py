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
    
    try:
        subprocess.run(say_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[LOG] ⚠️ 'say' command error for text: {text}. Stderr: {e.stderr.decode()}")
        return b""
    except FileNotFoundError:
        print(f"[LOG] ⚠️ 'say' command not found. Is this macOS?")
        return b""
    
    time.sleep(0.5) # Increased sleep for stability

    afconvert_cmd = f'afconvert -f WAVE -d LEI16@22050 "{aiff_path}" "{wav_path}"'
    try:
        subprocess.run(afconvert_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[LOG] ⚠️ 'afconvert' error for file: {aiff_path}. Stderr: {e.stderr.decode()}")
        return b""
    except FileNotFoundError:
        print(f"[LOG] ⚠️ 'afconvert' command not found.")
        return b""

    time.sleep(0.5) # Increased sleep for stability

    try:
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
    except Exception as e:
        print(f"[LOG] ⚠️ Cannot read WAV: {e}")
        wav_bytes = b""
    finally:
        for p in [aiff_path, wav_path]:
            try:
                os.remove(p)
            except Exception as e:
                print(f"[LOG] ⚠️ Failed to remove temp file {p}: {e}")
                pass
    return wav_bytes

def play_tts(text: str):
    ENABLE_TTS_PLAYBACK = True 
    if not ENABLE_TTS_PLAYBACK:
        print(f"[LOG] TTS playback disabled for debugging: {text}")
        return

    if st.session_state.get('tts_audio_placeholder') is None:
        st.session_state.tts_audio_placeholder = st.empty() 

    wav_bytes = generate_tts_wav_with_say(text)
    if not wav_bytes:
        print(f"[LOG] TTS failed to generate WAV: {text}")
        return
    try:
        b64 = base64.b64encode(wav_bytes).decode()
        html = f"<audio autoplay='true' style='display:none'><source src='data:audio/wav;base64,{b64}' type='audio/wav'></audio>"
        
        # Clear the placeholder before updating it to ensure new audio plays
        st.session_state.tts_audio_placeholder.empty() 
        st.session_state.tts_audio_placeholder.markdown(html, unsafe_allow_html=True)
        print(f"[LOG] TTS played (autoplay): {text}")
    except Exception as e:
        print(f"[LOG] Error rendering TTS HTML: {e}")

feedback_queue = queue.Queue() # Re-enabled feedback queue

def get_summary_feedback(pose_name: str, avg_score: float) -> str:
    ENABLE_LLM_FEEDBACK = True # Re-enabled LLM feedback
    if not ENABLE_LLM_FEEDBACK:
        return "LLM feedback disabled for debugging."

    prompt = (
        f"You are a friendly yoga coach. The user performed '{pose_name}'. "
        f"Based on their performance, provide one concise, positive tip for improvement. "
        f"Ensure the tip is in English, less than 10 words, and contains no numbers."
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=10 
        )
        resp.raise_for_status() 
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        print(f"[LOG] LLM request timed out for pose: {pose_name}")
        return "⚠️ Feedback: Request timed out."
    except requests.exceptions.ConnectionError:
        print(f"[LOG] LLM connection error. Is http://localhost:11434 running?")
        return "⚠️ Feedback: Connection error."
    except Exception as e:
        print(f"[LOG] Error in get_summary_feedback: {e}")
        return "⚠️ Feedback not available"

# --- Globale configuratievariabelen ---
CAMERA_CHECK_DURATION = 11
MODELPOSE_DIR = "modelposes_json"
ROUTINE_DIR   = "routines_json"
IMAGE_DIR      = "modelposes"
FEEDBACK_COUNT = 3 # Re-enabled feedback count
CAMERA_TEST_VIDEO_PATH = "Camera_test.mp4"

# Drempel voor handgebaar pauze
PAUSE_GESTURE_THRESHOLD_X_RIGHT = 100 
PAUSE_GESTURE_COOLDOWN = 2 

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
        'feedback_history': [], # Re-enabled
        'feedback_triggered': [], # Re-enabled
        'current_scores': [], # Will now store (time, score) tuples
        'tts_test_done': False,
        # 'tts_hold_test_done': False, # This was removed in previous versions and is not needed for current logic
        'prev_step': None,
        'cap': None,
        'paused': False,
        'pause_start_time': None,
        're_render_pose_image': False,
        'last_pause_gesture_time': 0, 
        'tts_audio_placeholder': None 
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

def toggle_pause():
    current_time = time.time()
    if current_time - st.session_state.last_pause_gesture_time > PAUSE_GESTURE_COOLDOWN:
        st.session_state.paused = not st.session_state.paused
        if st.session_state.paused:
            st.session_state.pause_start_time = time.time()
            play_tts("Routine gepauzeerd. Beweeg je hand naar rechts om te hervatten.") 
        else:
            if st.session_state.pause_start_time is not None:
                paused_duration = time.time() - st.session_state.pause_start_time
                st.session_state.phase_start += paused_duration
                st.session_state.pause_start_time = None
                st.session_state.re_render_pose_image = True 
                # Re-add feedback reset on resume for 'hold' phase only
                if st.session_state.phase == 'hold': 
                    st.session_state.feedback_triggered = [] 
                play_tts("Routine hervat.") 
        st.session_state.last_pause_gesture_time = current_time 
        st.rerun() 

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
    st.session_state.feedback_history = [] # Re-enabled
    st.session_state.feedback_triggered = [] # Re-enabled
    st.session_state.current_scores = []
    st.session_state.paused = False
    st.session_state.pause_start_time = None
    st.session_state.re_render_pose_image = False
    st.session_state.last_pause_gesture_time = 0 

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

top_cols = st.columns([1, 4, 1])
with top_cols[0]: 
    if st.session_state.running:
        if st.button("⏹️ Stop Routine"):
            st.session_state.running = False
            if st.session_state.cap is not None and st.session_state.cap.isOpened():
                st.session_state.cap.release()
            st.session_state.cap = None
            st.rerun()
with top_cols[2]: 
    if st.session_state.running:
        if st.button("⏯️ Pauze / Hervat"):
            toggle_pause() 

pose_ph, frame_ph, title_ph, metrics_ph, timer_ph, feedback_ph = create_main_layout(top_cols[1])

if 'tts_audio_placeholder' not in st.session_state or st.session_state.tts_audio_placeholder is None:
    st.session_state.tts_audio_placeholder = st.empty()


# Re-enable VISUAL_FEEDBACK_ENABLED (if not already defined as a global, it will default to True here)
# For clarity, let's explicitly define it if it was removed
VISUAL_FEEDBACK_ENABLED = True 


if st.session_state.running:
    while st.session_state.running:
        t0 = time.time() 

        ret, frame = st.session_state.cap.read()
        if not ret:
            print("[LOG] Kan geen frame van camera lezen. Afsluiten.")
            st.session_state.running = False
            break
        frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]

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
        
        current_pose_model = {}
        if st.session_state.current_step < len(st.session_state.poses) and st.session_state.phase == 'hold':
            current_pose_model = st.session_state.pose_models.get(f"{st.session_state.poses[st.session_state.current_step]['pose']}.json", {})

        annotated_frame, score, hand_keypoints = render_skeleton_frame(
            frame.copy(),
            current_pose_model,
            mode='full' if st.session_state.phase == 'hold' else 'landmarks'
        )

        if st.session_state.phase != 'camera_check': 
            for h_kp_x, h_kp_y in hand_keypoints:
                if h_kp_x > (frame_w - PAUSE_GESTURE_THRESHOLD_X_RIGHT):
                    toggle_pause()
                    break 
        
        if not st.session_state.paused:
            idx       = st.session_state.current_step
            if idx >= len(st.session_state.poses):
                st.session_state.running = False
                continue 

            step      = st.session_state.poses[idx]
            pose_name = step['pose']
            now       = time.time()
            elapsed   = now - st.session_state.phase_start
            
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

                if elapsed >= CAMERA_CHECK_DURATION:
                    st.session_state.phase = 'prepare'
                    st.session_state.phase_start = time.time()
                    st.session_state.prev_step = None
                    pose_ph.empty()
                    timer_ph.empty()
                    feedback_ph.empty() 
                    metrics_ph.empty()
                    continue

            else: 
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
                        feedback_ph.markdown(f"ℹ️ {st.session_state.routine_meta.get('description','')}")
                    # Keep feedback_ph populated with last LLM feedback during prep if available
                    elif VISUAL_FEEDBACK_ENABLED and st.session_state.feedback_history:
                        feedback_ph.markdown(st.session_state.feedback_history[-1])
                    else:
                        feedback_ph.empty() 
                    
                    st.session_state.prev_step = idx
                    st.session_state.re_render_pose_image = False
                
                if st.session_state.phase == 'prepare':
                    prep_time = step.get('prep_time', 5)
                    if elapsed >= prep_time:
                        st.session_state.phase = 'hold'
                        st.session_state.phase_start = now
                        st.session_state.feedback_history = [] # Re-enabled reset
                        st.session_state.feedback_triggered = [] # Re-enabled reset
                        st.session_state.current_scores = [] 
                        play_tts(f"Start met {pose_name}.") 
                        continue
                    else:
                        timer_ph.markdown(f"⏳ Voorbereiding: {int(prep_time - elapsed)} s")
                    
                elif st.session_state.phase == 'hold':
                    hold_time = step.get('hold_time', 30)
                    
                    if score is not None:
                        st.session_state.current_scores.append((elapsed, score)) # Store (time, score) pair

                    if VISUAL_FEEDBACK_ENABLED: # Re-enabled LLM feedback logic during hold
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
                                    sum(s for t, s in st.session_state.current_scores) # Adjusted for (time, score) tuples
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

                    if not feedback_queue.empty(): # Re-enabled
                        new_tip = feedback_queue.get()
                        st.session_state.feedback_history = [f"💡 {new_tip}"]
                        play_tts(new_tip)

                    if VISUAL_FEEDBACK_ENABLED and st.session_state.feedback_history: # Re-enabled
                        feedback_ph.markdown(st.session_state.feedback_history[-1])
                    else:
                        feedback_ph.empty() # Clear if no feedback or feedback not enabled

                    timer_ph.markdown(f"⏳ Houd vast: {int(hold_time - elapsed)} s")
                    if elapsed >= hold_time:
                        if st.session_state.current_scores:
                            st.session_state.score_log.setdefault(pose_name, []).append(list(st.session_state.current_scores))
                        else:
                            st.session_state.score_log.setdefault(pose_name, []).append([])

                        st.session_state.phase = 'transition'
                        st.session_state.phase_start = now
                        play_tts(f"Goed gedaan. {step.get('transition', 'Volgende pose.')}") 
                        continue

                elif st.session_state.phase == 'transition':
                    transition_time = step.get('transition_time', 3)
                    if elapsed >= transition_time:
                        feedback_ph.markdown(f"🔄 {step.get('transition', '')}")
                        st.session_state.feedback_history = [] # Re-enabled reset
                        st.session_state.feedback_triggered = [] # Re-enabled reset
                        st.session_state.current_scores = [] 
                        st.session_state.current_step += 1
                        if st.session_state.current_step >= len(st.session_state.poses):
                            st.session_state.running = False
                            play_tts("Routine voltooid! Goed gedaan!") 
                            break
                        st.session_state.phase = 'prepare'
                        st.session_state.phase_start = time.time()
                        st.session_state.prev_step = None
                        continue
                    else:
                        timer_ph.markdown(f"⏳ Volgende pose over: {int(transition_time - elapsed)} s")
            
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
            
            t1 = time.time()
            fps = 1/(t1-t0) if t1>t0 else 0
            metrics_ph.markdown(f"**Proc:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps**")

        else: 
            title_ph.header("⏸️ Routine Gepauzeerd")
            timer_ph.markdown("---") 
            metrics_ph.markdown("---") 
            feedback_ph.markdown("👋 Beweeg je hand naar de rechterkant van het scherm om te hervatten") 
            time.sleep(0.1) 
        
        frame_ph.image(annotated_frame, channels='BGR', use_container_width=True)
        
pose_ph.empty()
frame_ph.empty()
title_ph.empty()
metrics_ph.empty()
timer_ph.empty()
feedback_ph.empty()

if st.session_state.get('tts_audio_placeholder') is not None:
    st.session_state.tts_audio_placeholder.empty()

if st.session_state.cap is not None:
    if st.session_state.cap.isOpened(): 
        _ = st.session_state.cap.release()
    st.session_state.cap = None

if st.session_state.score_log:
    st.header("✨ Jouw Routine Overzicht ✨")
    for pn, scores_for_pose_attempts in st.session_state.score_log.items():
        for i, attempt_scores_with_time in enumerate(scores_for_pose_attempts): 
            img_col, plot_col = st.columns([1,1]) 
            with img_col:
                tp = os.path.join(IMAGE_DIR, f"{pn}.jpg")
                if os.path.isfile(tp):
                    st.image(tp, use_container_width=True)
                else:
                    st.markdown(f"Model afbeelding voor **{pn}** niet gevonden.")
            with plot_col:
                actual_scores = [s for t, s in attempt_scores_with_time]
                avg = sum(actual_scores)/len(actual_scores) if actual_scores else 0
                top = max(actual_scores) if actual_scores else 0
                
                st.markdown(f"## {pn} (Poging {i+1})")
                st.markdown(f"<p style='font-size:1.2em; color:gray;'>Today</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:1.5em; color:#00D09B; margin-bottom: 0.1em;'>Average Score <span style='float:right; font-weight:bold;'>{avg:.0f}%</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:1.5em; color:#00D09B;'>High Score <span style='float:right; font-weight:bold;'>{top:.0f}%</span></p>", unsafe_allow_html=True)
                
                if actual_scores:
                    fig, ax = plt.subplots(figsize=(8, 5)) 
                    
                    times_for_plot = [t for t, s in attempt_scores_with_time]
                    scores_for_plot = [s for t, s in attempt_scores_with_time]

                    ax.plot(times_for_plot, scores_for_plot, color='#00D09B', linewidth=2)
                    
                    ax.set_facecolor('#F8F8F8')
                    ax.grid(False)
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("Score (%)")
                    ax.set_xlabel("Tijd (seconden)") 
                    
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

                # LLM Feedback re-enabled
                feedback = get_summary_feedback(pn, avg)
                st.info(feedback) 
            st.markdown("---")

if st.button("⇦ Kies een andere routine"):
    st.session_state.running = False
    st.rerun()