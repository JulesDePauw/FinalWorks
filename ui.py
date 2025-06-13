import os
import json
import streamlit as st

IMAGE_DIR = "modelposes"
ROUTINE_DIR = "routines_json"

# --- Paginalayout & Styling ---

def apply_styles():
    """
    Stel pagina-configuratie en basis styling in.
    """
    st.set_page_config(page_title="🧘‍♀️ Yoga Routine", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background-color: #DCF0EB; color: #707070; }
        div.stButton > button { background-color: #3CF4C5; color: white; border: none; border-radius: 5px; padding: 0.5em 1em; }
        div.stButton > button:hover { background-color: #FFFFFF; color: #3CF4C5;}
        div.stSelectbox > div { background-color: #FFFFFF; color: #000000; }
        </style>
        """, unsafe_allow_html=True
    )

# --- Selectiescherm ---

def render_selection_screen(on_select_callback):
    """
    Render de selectiepagina met alle yoga-routines.
    on_select_callback wordt aangeroepen zodra de gebruiker op een 'Start'-knop klikt.
    """
    sel = st.container()
    sel.title("Kies een yoga-routine:")
    routines = sorted([f for f in os.listdir(ROUTINE_DIR) if f.endswith('.json')])
    for fn in routines:
        try:
            meta = json.load(open(os.path.join(ROUTINE_DIR, fn)))
        except json.JSONDecodeError:
            continue
        # Kolomverdeling: afbeelding 1/3, details 2/3
        cols = sel.columns([1, 2], gap="large")
        thumb = meta.get('thumbnail', '')
        img_path = os.path.join(IMAGE_DIR, thumb)
        with cols[0]:
            if thumb and os.path.isfile(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.write("(Geen afbeelding)")
        with cols[1]:
            title = meta.get('title', fn)
            total_dur = sum(p.get('prep_time', 0) + p.get('hold_time', 0) for p in meta.get('poses', []))
            minutes, seconds = divmod(total_dur, 60)
            st.subheader(title)
            st.write(f"**Duur:** {minutes}m {seconds}s")
            st.write(f"**Beschrijving:** {meta.get('description', '')}")
            if st.button(f"Start {title}", key=fn, on_click=on_select_callback, args=(fn,)):
                sel.empty()
        sel.markdown("---")
    st.stop()

# --- Hoofd layout placeholders ---

def create_main_layout():
    """
    Zet de hoofdindeling (twee kolommen) op en retourneer placeholder-objecten voor:
    pose_ph, frame_ph, title_ph, metrics_ph, timer_ph, feedback_ph
    Deze placeholders gebruiken we in app.py om dynamisch content te vullen.
    """
    display_col, controls_col = st.columns(2)
    pose_ph = display_col.empty()
    with controls_col:
        frame_ph   = st.empty()
        title_ph   = st.empty()
        metrics_ph = st.empty()
        timer_ph   = st.empty()
        feedback_ph= st.empty()
    return pose_ph, frame_ph, title_ph, metrics_ph, timer_ph, feedback_ph