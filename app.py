import os
import time
import json
import cv2
import numpy as np
import streamlit as st
from compare_pose_streamlit import render_skeleton_frame
import matplotlib.pyplot as plt

# --- Configuratie ---
st.set_page_config(page_title="🧘‍♀️ Yoga Routine", layout="wide")
# Styling
st.markdown(
    """
    <style>
    .stApp { background-color: #FFFBF2; color: #333333; }
    div.stButton > button { background-color: #4C9A2A; color: white; border: none; border-radius: 5px; padding: 0.5em 1em; }
    div.stButton > button:hover { background-color: #3E7D22; }
    div.stSelectbox > div { background-color: #FFFFFF; color: #000000; }
    </style>
    """, unsafe_allow_html=True
)

# Directories
MODELPOSE_DIR = "modelposes_json"
ROUTINE_DIR = "routines_json"

# Laad modelposes
def load_models():
    st.session_state.pose_models = {}
    for fn in os.listdir(MODELPOSE_DIR):
        if fn.endswith('.json'):
            with open(os.path.join(MODELPOSE_DIR, fn)) as f:
                st.session_state.pose_models[fn] = json.load(f)

# Callback om routine te starten
def start_routine():
    # Laad geselecteerde routine JSON
    routine_file = os.path.join(ROUTINE_DIR, st.session_state.selected)
    data = json.load(open(routine_file))
    # Sla poses en metadata op
    st.session_state.poses = data.get('poses', [])
    st.session_state.routine_meta = {
        'title': data.get('title'),
        'description': data.get('description'),
        'thumbnail': data.get('thumbnail'),
        'difficulty': data.get('difficulty')
    }
    # Init status
    st.session_state.running = True
    st.session_state.current_step = 0
    st.session_state.prev_step = None
    st.session_state.phase = 'prepare'
    st.session_state.phase_start = time.time()
    st.session_state.score_log = {}
    st.session_state.final_results_shown = False
    # Camera opstarten
    if 'cap' in st.session_state:
        st.session_state.cap.release()
    st.session_state.cap = cv2.VideoCapture(0)
    # Bepaal crop-target van eerste pose
    if st.session_state.poses:
        first_pose = st.session_state.poses[0]
        first_img = first_pose.get('image_path', f"modelposes/{first_pose['pose']}.jpg")
        img0 = cv2.imread(first_img)
        st.session_state.img_shape = img0.shape[:2] if img0 is not None else None

# Callback voor selectieknop
def on_select(fn):
    st.session_state.selected = fn
    start_routine()
def on_select(fn):
    st.session_state.selected = fn
    start_routine()

# Init session_state defaults
def init_state():
    if 'pose_models' not in st.session_state:
        load_models()
        st.session_state.running = False
        st.session_state.current_step = 0
        st.session_state.prev_step = None
        st.session_state.phase = None
        st.session_state.phase_start = None
        st.session_state.img_shape = None
        st.session_state.score_log = {}
        st.session_state.final_results_shown = False
        st.session_state.selected = None
        st.session_state.poses = []

# Roep init_state() aan
init_state()

# SELECTIESCHERM: toon alleen zolang de routine niet draait: toon alleen zolang de routine niet draait
if not st.session_state.running:
    sel_container = st.container()
    sel_container.title("Kies een yoga-routine:")
    routines = [f for f in os.listdir(ROUTINE_DIR) if f.endswith('.json')]
    meta = {fn: json.load(open(os.path.join(ROUTINE_DIR, fn))) for fn in routines}
    for fn in routines:
        cols = sel_container.columns(2)
        data = meta[fn]
        with cols[0]:
            thumb = data.get('thumbnail')
            if thumb and os.path.exists(os.path.join('modelposes', thumb)):
                st.image(os.path.join('modelposes', thumb), use_container_width=True)
        with cols[1]:
            st.subheader(data.get('title', fn))
            total = sum(p['prep_time'] + p['hold_time'] for p in data.get('poses', []))
            m, s = divmod(total, 60)
            st.write(f"**Duur:** {m}m {s}s")
            st.write(f"**Beschrijving:** {data.get('description','')}")
            if st.button(f"Start {data.get('title', fn)}", key=fn, on_click=on_select, args=(fn,)):
                sel_container.empty()
    st.stop()

# Zodra gestart, toon twee kolommen voor camera en metrics
col1, col2 = st.columns(2)
# Placeholder in kolom 1 voor modelpose
pose_ph = col1.empty()

# Kolom 2: camera en metrics
# Placeholder in kolom 1 voor modelpose
pose_ph = col1.empty()

# Kolom 2: camera en metrics
with col2:
    frame_ph = st.empty()
    title_ph = st.empty()
    metrics_ph = st.empty()

# Hoofdloop
enumerate_loop = True
if st.session_state.running and st.session_state.poses:

    cap = st.session_state.cap
    target = st.session_state.img_shape
    while st.session_state.running:
        idx = st.session_state.current_step
        # Update modelpose alleen bij step change
        if st.session_state.prev_step != idx:
            step = st.session_state.poses[idx]
            pose_name = step['pose']
            pose_ph.markdown(f"📌 **Model pose:** {pose_name}")
            pose_ph.image(step.get('image_path', f"modelposes/{pose_name}.jpg"), use_container_width=True)
            st.session_state.prev_step = idx
        # Camera frame
        ret, frame = cap.read()
        if not ret:
            st.warning('Kan geen frame van de camera lezen.')
            break
        frame = cv2.flip(frame, 1)
        # Crop voor gelijke AR
        if target:
            th, tw = target
            h, w = frame.shape[:2]
            if w/h > tw/th:
                new_w = int(h * (tw/th)); x0 = (w-new_w)//2
                frame = frame[:, x0:x0+new_w]
            else:
                new_h = int(w * (th/tw)); y0 = (h-new_h)//2
                frame = frame[y0:y0+new_h, :]
        # Render & score
        mode = 'full'  # Always use full mediapipe model
        t0 = time.time()
        annotated, score = render_skeleton_frame(
            frame.copy(), st.session_state.pose_models.get(f"{pose_name}.json", {}), mode='full'  # Always full model
        )
        t1 = time.time()
        if score is not None:
            txt = f"{score:.1f}%"
            (wt, ht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            ov = annotated.copy()
            cv2.rectangle(ov, (5,15), (5+wt+10,15+ht+10), (50,50,50), -1)
            annotated = cv2.addWeighted(ov, 0.6, annotated, 0.4, 0)
            cv2.putText(annotated, txt, (10,15+ht), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            if st.session_state.phase=='hold':
                st.session_state.score_log.setdefault(pose_name, []).append(score)
        # Show frame + metrics
        frame_ph.image(annotated, channels='BGR', use_container_width=True)
        title_ph.header(f"🧘‍♀️ {pose_name}")
        fps = 1/(t1-t0) if t1>t0 else 0; elapsed = time.time()-st.session_state.phase_start
        if st.session_state.phase=='prepare':
            rem = step.get('prep_time',5)-elapsed
            tr = (st.session_state.poses[idx-1].get('transition','Neem houding aan.') if idx>0 else 'Neem houding aan.')
            phase_text = f"🔄 Voorbereiding: {int(rem)} s — {tr}"
        elif st.session_state.phase=='hold':
            rem = step.get('hold_time',30)-elapsed
            phase_text = f"⏳ Houd vast: {int(rem)} s — Goed bezig!"
        else: phase_text = ''
        metrics_ph.markdown(f"**Proc:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps** {phase_text}")
        # Transities
        now = time.time()
        if st.session_state.phase=='prepare' and rem<=0:
            st.session_state.phase='hold'; st.session_state.phase_start=now
        elif st.session_state.phase=='hold' and rem<=0:
            st.session_state.phase='transition'; st.session_state.phase_start=now
        elif st.session_state.phase=='transition':
            st.session_state.current_step +=1
            if st.session_state.current_step>=len(st.session_state.poses):
                st.session_state.running=False
            else:
                nxt = st.session_state.poses[st.session_state.current_step]
                img_n = cv2.imread(nxt.get('image_path', f"modelposes/{nxt['pose']}.jpg"))
                if img_n is not None: target = img_n.shape[:2]
                st.session_state.phase='prepare'; st.session_state.phase_start=now
        time.sleep(0.03)
    cap.release()

# Toon eindgrafieken
if not st.session_state.running and not st.session_state.final_results_shown and any(st.session_state.score_log.values()):
    frame_ph.empty(); title_ph.empty(); metrics_ph.empty(); pose_ph.empty()
    for pose_name, scores in st.session_state.score_log.items():
        # kolommen voor eindresultaten
        img_col, plot_col = st.columns([1,1])
        # model afbeelding links
        with img_col:
            model_img = next(
                (
                    s.get('image_path', f"modelposes/{pose_name}.jpg")
                    for s in st.session_state.poses if s.get('pose')==pose_name
                ),
                f"modelposes/{pose_name}.jpg"
            )
            if model_img and os.path.exists(model_img):
                st.image(model_img, use_container_width=True)
        # grafiek en stats rechts
        with plot_col:
            st.markdown(f"### {pose_name}")
            fig, ax = plt.subplots(figsize=(4,2), facecolor="#FFFBF2")
            ax.set_facecolor("#FFFBF2")
            times = np.linspace(0, len(scores)/30, len(scores))
            ax.plot(times, scores)
            ax.set_xticks([])
            yticks = ax.get_yticks()
            ax.set_yticklabels([f"{int(v)}%" for v in yticks])
            ax.set_ylim(0,100)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            top = max(scores)
            avg = sum(scores)/len(scores)
            st.markdown(
                f"**Topscore:** {top:.1f}%  **Gem.score:** {avg:.1f}%  "
                f"**Feedback:** Placeholder feedback."
            )
    st.session_state.final_results_shown = True
    # Knop om terug te keren naar routine-selectie
    if st.button("⇦ Kies een andere routine"):
        # Reset state
        st.session_state.running = False
        st.session_state.selected = None
        st.session_state.poses = []
        st.session_state.final_results_shown = False
        # Stop hier zodat selectie opnieuw getoond wordt
        st.stop()
