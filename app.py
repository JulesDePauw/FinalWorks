import os
import time
import json
import cv2
import numpy as np
import requests
import streamlit as st
from compare_pose_streamlit import run_streamlit_feedback, render_skeleton_frame
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
IMAGE_DIR = "modelposes"

# Aantal AI feedbackmomenten tijdens hold-fase
FEEDBACK_COUNT = 3

# Callback om routine te starten
def on_select(fn):
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
    cap = cv2.VideoCapture(0)
    st.session_state.cap = cap
    first_img = st.session_state.poses[0].get('image_path', '')
    img0 = cv2.imread(os.path.join(IMAGE_DIR, first_img))
    st.session_state.img_shape = img0.shape[:2] if img0 is not None else None

# Helper voor AI feedback tip
def get_summary_feedback(pose_name: str, avg_score: float) -> str:
    prompt = (
        f"Je bent een vriendelijke yoga-coach."
        f" De gebruiker voerde de pose '{pose_name}' uit met een gemiddelde nauwkeurigheid van {avg_score:.1f}%."
        f" Geef een korte tip om deze houding te verbeteren."
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        return resp.json().get("response", "").strip()
    except:
        return "⚠️ Feedback niet beschikbaar"

# Initialiseer session_state defaults
def init_state():
    if 'pose_models' not in st.session_state:
        st.session_state.pose_models = {}
        for fn in os.listdir(MODELPOSE_DIR):
            if fn.endswith('.json'):
                st.session_state.pose_models[fn] = json.load(open(os.path.join(MODELPOSE_DIR, fn)))
        st.session_state.running = False
        st.session_state.selected = None
        st.session_state.poses = []
        st.session_state.phase = None
        st.session_state.phase_start = None
        st.session_state.img_shape = None
        st.session_state.score_log = {}
        st.session_state.feedback_history = []
init_state()

# Selectiescherm
if not st.session_state.running:
    sel = st.container()
    sel.title("Kies een yoga-routine:")
    routines = sorted([f for f in os.listdir(ROUTINE_DIR) if f.endswith('.json')])
    for fn in routines:
        meta = json.load(open(os.path.join(ROUTINE_DIR, fn)))
        cols = sel.columns([1, 2])
        thumb = meta.get('thumbnail', '')
        path = os.path.join(IMAGE_DIR, thumb)
        with cols[0]:
            if thumb and os.path.isfile(path):
                st.image(path, use_container_width=True)
            else:
                st.write("(Geen thumbnail)")
        with cols[1]:
            st.subheader(meta.get('title', fn))
            dur = sum(p['prep_time'] + p['hold_time'] for p in meta.get('poses', []))
            m, s = divmod(dur, 60)
            st.write(f"**Duur:** {m}m {s}s")
            st.write(meta.get('description', ''))
            if st.button(f"Start {meta.get('title', fn)}", key=fn, on_click=on_select, args=(fn,)):
                sel.empty()
    st.stop()

# Layout
display_col, controls_col = st.columns(2)
pose_ph = display_col.empty()
with controls_col:
    frame_ph = st.empty()
    title_ph = st.empty()
    metrics_ph = st.empty()
    feedback_ph = st.empty()
    timer_ph = st.empty()

# Hoofdloop
while st.session_state.running:
    idx = st.session_state.current_step
    step = st.session_state.poses[idx]
    pose_name = step['pose']
    now = time.time()
    elapsed = now - st.session_state.phase_start

    # Toon model pose en init feedback block bij nieuwe stap
    if st.session_state.prev_step != idx:
        pose_ph.markdown(f"📌 **Model pose:** {pose_name}")
        raw = step.get('image_path', '')
        candidates = [raw, os.path.join(IMAGE_DIR, raw), os.path.join(IMAGE_DIR, f"{pose_name}.jpg")]
        for p in candidates:
            if p and os.path.isfile(p):
                pose_ph.image(p, use_container_width=True)
                break
        # Feedback tijdens voorbereiding van de eerste pose
        if idx == 0 and step['prep_time'] > 0:
            feedback_ph.markdown(f"ℹ️ {st.session_state.routine_meta.get('description','')}")
        st.session_state.prev_step = idx

    # Fase wissel
    if st.session_state.phase == 'prepare' and elapsed >= step.get('prep_time', 5):
        st.session_state.phase = 'hold'
        st.session_state.phase_start = now
        elapsed = 0
        st.session_state.feedback_history = []
    elif st.session_state.phase == 'hold':
        # Gesegmenteerde feedback
        segment = step.get('hold_time', 30) / FEEDBACK_COUNT
        for i in range(FEEDBACK_COUNT):
            avg_seg = run_streamlit_feedback(
                os.path.join(MODELPOSE_DIR, f"{pose_name}.json"),
                frame_ph,
                hold_time=segment,
                feedback_text_area=None,
                timer_area=timer_ph
            )
            st.session_state.score_log.setdefault(pose_name, []).append(avg_seg)
            tip = get_summary_feedback(pose_name, avg_seg)
            st.session_state.feedback_history.append(f"💡 {tip}")
            feedback_ph.markdown("\n".join(st.session_state.feedback_history))
        st.session_state.phase = 'transition'
        st.session_state.phase_start = time.time()
        elapsed = 0
    elif st.session_state.phase == 'transition' and elapsed >= step.get('transition_time', 3):
        # Toon transition tekst
        feedback_ph.markdown(f"🔄 {step.get('transition','')} ")
        st.session_state.current_step += 1
        if st.session_state.current_step >= len(st.session_state.poses):
            st.session_state.running = False
            break
        st.session_state.phase = 'prepare'
        st.session_state.phase_start = time.time()
        st.session_state.prev_step = None
        elapsed = 0

    # Live camera en metrics op elk moment
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.warning('Kan geen frame lezen')
        break
    frame = cv2.flip(frame, 1)
    if st.session_state.img_shape:
        th, tw = st.session_state.img_shape
        h, w = frame.shape[:2]
        if w/h > tw/th:
            nw = int(h * tw/th); x = (w-nw)//2; frame = frame[:, x:x+nw]
        else:
            nh = int(w * th/tw); y = (h-nh)//2; frame = frame[y:y+nh]
    t0 = time.time()
    annot, score = render_skeleton_frame(frame.copy(), st.session_state.pose_models.get(f"{pose_name}.json", {}), mode='full')
    t1 = time.time()
    # Overlay score
    if score is not None:
        txt = f"{score:.1f}%"
        (wt, ht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        ov = annot.copy(); cv2.rectangle(ov, (5,5), (5+wt+10,5+ht+10), (50,50,50), -1)
        annot = cv2.addWeighted(ov, 0.6, annot, 0.4, 0)
        cv2.putText(annot, txt, (10,10+ht), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
    frame_ph.image(annot, channels='BGR', use_container_width=True)
    title_ph.header(f"🧘‍♀️ {pose_name}")
    fps = 1/(t1-t0) if t1>t0 else 0
    phase_label = {
        'prepare': f"🔄 Voorbereiding: {int(step.get('prep_time',5)-elapsed)} s",
        'hold':    f"⏳ Houd vast segments",
        'transition': f"🔁 Volgende pose wordt geladen"
    }.get(st.session_state.phase, '')
    metrics_ph.markdown(f"**Proc:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps** {phase_label}")
    time.sleep(0.03)

# Eindreview na afloop
st.session_state.cap.release()
if st.session_state.score_log:
    for pose_name, scores in st.session_state.score_log.items():
        img_col, plot_col = st.columns([1,1])
        with img_col:
            thumb = os.path.join(IMAGE_DIR, f"{pose_name}.jpg")
            if os.path.isfile(thumb): st.image(thumb, use_container_width=True)
        with plot_col:
            st.markdown(f"## {pose_name}")
            avg = sum(scores)/len(scores)
            top = max(scores)
            st.write(f"**Topscore:** {top:.1f}% — **Gem.score:** {avg:.1f}%")
            st.info(get_summary_feedback(pose_name, avg))
    if st.button("⇦ Kies een andere routine"):
        st.session_state.running = False
        st.experimental_rerun()
