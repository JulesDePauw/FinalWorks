import os
import time
import uuid
import tempfile
import sys
import streamlit as st
import subprocess
import base64

# --- Helperfuncties voor TTS ---
# Haalt beschikbare stemmen op via 'say -v ?' (macOS)
def get_available_say_voices():
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        voices = []
        for line in lines:
            parts = line.split()
            if parts:
                voices.append(parts[0])
        return sorted(voices)
    except Exception as e:
        st.error(f"⚠️ Fout bij ophalen beschikbare voices: {e}")
        return []

DEFAULT_SAY_VOICE = "Ava"

# Genereert wav bytes via 'say' voor gegeven text en voice
def generate_tts_wav_with_say(text: str, voice: str = DEFAULT_SAY_VOICE) -> bytes:
    tmp_dir = tempfile.gettempdir()
    aiff_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.aiff")
    wav_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4().hex}.wav")
    say_cmd = f'say -v "{voice}" -o "{aiff_path}" "{text}"'
    if os.system(say_cmd) != 0:
        st.error(f"⚠️ 'say' commandeerfout voor tekst: {text}")
        return b""
    time.sleep(0.1)
    afconvert_cmd = f'afconvert -f WAVE -d LEI16@22050 "{aiff_path}" "{wav_path}"'
    if os.system(afconvert_cmd) != 0:
        st.error(f"⚠️ 'afconvert' fout voor bestand: {aiff_path}")
        return b""
    time.sleep(0.1)
    try:
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
    except Exception as e:
        st.error(f"⚠️ Kan WAV niet lezen: {e}")
        wav_bytes = b""
    finally:
        for p in [aiff_path, wav_path]:
            try: os.remove(p)
            except: pass
    return wav_bytes

# --- Streamlit UI ---
st.set_page_config(page_title="TTS Sequencer met Ava", layout="centered")
st.title("TTS Sequencer met Ava")
st.write("Klik op de knop om drie berichten achter elkaar te laten uitspreken met 2s pauze.")

if st.button("Play Sequence"):
    # Drie berichten hardcoded
    messages = [
        "Dit is het eerste bericht.",
        "Dit is het tweede bericht.",
        "Dit is het derde bericht."
    ]
    base64_list = []
    for msg in messages:
        wav = generate_tts_wav_with_say(msg, DEFAULT_SAY_VOICE)
        if not wav:
            st.error("⚠️ Fout bij genereren van een van de berichten.")
            st.stop()
        b64 = base64.b64encode(wav).decode()
        base64_list.append(f"data:audio/wav;base64,{b64}")

    # Bouw HTML met <audio> en JavaScript voor sequentiële afspelen
    html = """
<audio id='audioPlayer' style='display:none'></audio>
<script>
  const messages = %s;
  let player = document.getElementById('audioPlayer');
  let index = 0;
  function playNext() {
    if (index < messages.length) {
      player.src = messages[index];
      player.play();
      index++;
      player.onended = function() {
        setTimeout(playNext, 2000);
      };
    }
  }
  playNext();
</script>
""" % base64_list
    st.components.v1.html(html, height=0)
    st.success("🔊 Sequencing gestart!")
