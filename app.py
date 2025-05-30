import os
import time
import json
import cv2
import numpy as np
import requests
import streamlit as st
from compare_pose_streamlit import run_streamlit_feedback, render_skeleton_frame
import matplotlib.pyplot as plt
import threading

# Function to fetch feedback asynchronously
def _async_fetch_feedback(pose_name, avg_score, idx_thr):
    print(f"[LOG] Async feedback thread started for pose '{pose_name}', segment {idx_thr}, avg_score={avg_score:.1f}%")
    tip = get_summary_feedback(pose_name, avg_score)
    print(f"[LOG] Received AI feedback for pose '{pose_name}': {tip}")
    # Append to history on completion
    # Use session_state to store feedback and trigger UI rerun
    st.session_state.feedback_history.append(f"💡 {tip}")
    st.session_state.feedback_triggered.append(idx_thr)
    print(f"[LOG] feedback_history updated: {st.session_state.feedback_history}")
    # Trigger Streamlit rerun to refresh UI
    try:
        st.experimental_rerun()
    except AttributeError:
        # Can't rerun, ignore
        pass

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
    # Feedback state
    st.session_state.feedback_history = []
    st.session_state.feedback_triggered = []
    st.session_state.current_scores = []
    st.session_state.feedback_pending = None
    cap = cv2.VideoCapture(0)
    st.session_state.cap = cap
    first_img = st.session_state.poses[0].get('image_path', '')
    img0 = cv2.imread(os.path.join(IMAGE_DIR, first_img))
    st.session_state.img_shape = img0.shape[:2] if img0 is not None else None
    first_img = st.session_state.poses[0].get('image_path', '')
    img0 = cv2.imread(os.path.join(IMAGE_DIR, first_img))
    st.session_state.img_shape = img0.shape[:2] if img0 is not None else None

# Helper voor AI feedback tip
def get_summary_feedback(pose_name: str, avg_score: float) -> str:
    # Prompt includes accuracy and instructs to use pose specifics
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
    except:
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
        'feedback_pending': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    # Load pose models
    if not st.session_state.pose_models:
        for fn in os.listdir(MODELPOSE_DIR):
            if fn.endswith('.json'):
                st.session_state.pose_models[fn] = json.load(open(os.path.join(MODELPOSE_DIR, fn)))
init_state()

# Selectiescherm
if not st.session_state.running:
    sel = st.container()
    sel.title("Kies een yoga-routine:")
    routines = sorted([f for f in os.listdir(ROUTINE_DIR) if f.endswith('.json')])
    for fn in routines:
        try:
            meta = json.load(open(os.path.join(ROUTINE_DIR, fn)))
        except json.JSONDecodeError:
            continue
        cols = sel.columns([1, 2])
        thumb = meta.get('thumbnail', '')
        path = os.path.join(IMAGE_DIR, thumb)
        with cols[0]:
            if thumb and os.path.isfile(path): st.image(path, use_container_width=True)
            else: st.write("(Geen thumbnail)")
        with cols[1]:
            st.subheader(meta.get('title', fn))
            dur = sum(p['prep_time'] + p['hold_time'] for p in meta.get('poses', []))
            m, s = divmod(dur, 60)
            st.write(f"**Duur:** {m}m {s}s")
            st.write(meta.get('description', ''))
            if st.button(f"Start {meta.get('title', fn)}", key=fn, on_click=on_select, args=(fn,)): sel.empty()
    st.stop()

# Layout
display_col, controls_col = st.columns(2)
pose_ph = display_col.empty()
with controls_col:
    frame_ph = st.empty()
    title_ph = st.empty()
    metrics_ph = st.empty()
    timer_ph = st.empty()
    feedback_ph = st.empty()

# Hoofdloop
while st.session_state.running:
    idx = st.session_state.current_step; step = st.session_state.poses[idx]; pose_name = step['pose']; now = time.time(); elapsed = now - st.session_state.phase_start
    if st.session_state.prev_step != idx:
        pose_ph.markdown(f"📌 **Model pose:** {pose_name}")
        raw = step.get('image_path',''); candidates = [raw, os.path.join(IMAGE_DIR, raw), os.path.join(IMAGE_DIR,f"{pose_name}.jpg")]
        for p in candidates:
            if p and os.path.isfile(p): pose_ph.image(p,use_container_width=True); break
        if idx==0 and step['prep_time']>0: feedback_ph.markdown(f"ℹ️ {st.session_state.routine_meta.get('description','')}")
        st.session_state.prev_step = idx
    if st.session_state.phase=='prepare' and elapsed>=step.get('prep_time',5):
        # Transition to hold and prefetch AI feedback in parallel
        st.session_state.phase='hold'
        st.session_state.phase_start=now
        elapsed=0
        # Reset feedback state
        st.session_state.feedback_history=[]
        st.session_state.current_scores=[]
                # Prefetch FEEDBACK_COUNT feedbacks asynchronously
        for i in range(FEEDBACK_COUNT):
            def worker(idx=i, pose=pose_name):
                tip = get_summary_feedback(pose, 0.0)
                # Overwrite feedback: only show latest tip
                st.session_state.feedback_history = [f"💡 {tip}"]
                print(f"[LOG] Prefetched tip: {tip}")
            threading.Thread(target=worker, daemon=True).start()
        continue
    elif st.session_state.phase=='hold':
        # Lees frame en score
        ret_sb, frame_sb = st.session_state.cap.read()
        if ret_sb:
            frame_sb = cv2.flip(frame_sb,1)
            _, score_sb = render_skeleton_frame(
                frame_sb.copy(),
                st.session_state.pose_models.get(f"{pose_name}.json", {}),
                mode='full'
            )
            if score_sb is not None:
                st.session_state.current_scores.append(score_sb)
        # AI-feedback op intervals (synchronous)
        hold_time = step.get('hold_time', 30)
        thresholds = [(i+1) * hold_time / FEEDBACK_COUNT for i in range(FEEDBACK_COUNT)]
        for it, thr in enumerate(thresholds):
            if elapsed >= thr and it not in st.session_state.feedback_triggered:
                avg_s = sum(st.session_state.current_scores) / len(st.session_state.current_scores) if st.session_state.current_scores else 0
                tip = get_summary_feedback(pose_name, avg_s)
                # Overwrite history to only latest tip
                st.session_state.feedback_history = [f"💡 {tip}"]
                st.session_state.feedback_triggered.append(it)
        # Toon enkel de meest recente feedback
        if st.session_state.feedback_history:
            feedback_ph.markdown(st.session_state.feedback_history[-1])
        # Einde hold-fase
        if st.session_state.feedback_history:
            feedback_ph.markdown("".join(st.session_state.feedback_history))
        # Einde hold-fase
        if elapsed >= hold_time:
            final_avg = sum(st.session_state.current_scores) / len(st.session_state.current_scores) if st.session_state.current_scores else 0
            st.session_state.score_log.setdefault(pose_name, []).append(final_avg)
            st.session_state.phase = 'transition'
            st.session_state.phase_start = now
            elapsed = 0
        if elapsed >= hold_time:
            final_avg = sum(st.session_state.current_scores) / len(st.session_state.current_scores) if st.session_state.current_scores else 0
            st.session_state.score_log.setdefault(pose_name, []).append(final_avg)
            st.session_state.phase = 'transition'
            st.session_state.phase_start = now
            elapsed = 0
    elif st.session_state.phase=='transition' and elapsed>=step.get('transition_time',3):
        feedback_ph.markdown(f"🔄 {step.get('transition','')}")
        # Reset feedback for next pose
        st.session_state.feedback_history = []
        st.session_state.feedback_triggered = []
        st.session_state.current_scores = []
        # Advance step
        st.session_state.current_step += 1
        if st.session_state.current_step >= len(st.session_state.poses):
            st.session_state.running = False
            break
        # Prepare for next pose
        st.session_state.phase = 'prepare'
        st.session_state.phase_start = time.time()
        st.session_state.prev_step = None
        elapsed = 0
    # Live feed + metrics
    ret, frame = st.session_state.cap.read()
    if not ret: break
    frame=cv2.flip(frame,1)
    if st.session_state.img_shape:
        th,tw=st.session_state.img_shape;h,w=frame.shape[:2]
        if w/h>tw/th: nw=int(h*tw/th); x=(w-nw)//2; frame=frame[:,x:x+nw]
        else: nh=int(w*th/tw); y=(h-nh)//2; frame=frame[y:y+nh]
    # Render skeleton and compute score
    t0=time.time(); annot, score = render_skeleton_frame(frame.copy(), st.session_state.pose_models.get(f"{pose_name}.json", {}), mode='full'); t1=time.time()
    # Overlay score
    if score is not None:
        txt=f"{score:.1f}%"; (wt,ht),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,1.5,2)
        ov=annot.copy();cv2.rectangle(ov,(5,5),(5+wt+10,5+ht+10),(50,50,50),-1);annot=cv2.addWeighted(ov,0.6,annot,0.4,0)
        cv2.putText(annot,txt,(10,10+ht),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    frame_ph.image(annot,channels='BGR',use_container_width=True)
    # Title
    title_ph.header(f"🧘‍♀️ {pose_name}")
    # Update timer display
    if st.session_state.phase=='prepare':
        rem_prep = step.get('prep_time',5) - elapsed
        timer_ph.markdown(f"⏳ Voorbereiding: {int(rem_prep)} s")
    elif st.session_state.phase=='hold':
        rem_hold = step.get('hold_time',30) - elapsed
        timer_ph.markdown(f"⏳ Houd vast: {int(rem_hold)} s")
    else:
        timer_ph.empty()
    # Metrics
    fps=1/(t1-t0) if t1>t0 else 0
    phase_txt={'prepare':'','hold':'','transition':'🔁 Volgende pose'}
    metrics_ph.markdown(f"**Proc:** {(t1-t0)*1000:.0f} ms — **{fps:.1f} fps** {phase_txt.get(st.session_state.phase,'')}" )
    time.sleep(0.03)
# Eindreview
deprecated = st.session_state.cap.release()
if st.session_state.score_log:
    for pn, sc in st.session_state.score_log.items():
        img_col, plot_col = st.columns([1,1])
        with img_col:
            tp = os.path.join(IMAGE_DIR, f"{pn}.jpg")
            if os.path.isfile(tp):
                st.image(tp, use_container_width=True)
        with plot_col:
            avg = sum(sc) / len(sc)
            top = max(sc)
            st.markdown(f"## {pn}")
            st.write(f"**Topscore:** {top:.1f}% — **Gem.score:** {avg:.1f}%")
            st.info(get_summary_feedback(pn, avg))
# Restart button
if st.button("⇦ Kies een andere routine"):
    st.session_state.running = False
    st.experimental_rerun()
