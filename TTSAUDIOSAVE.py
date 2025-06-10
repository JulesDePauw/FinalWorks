import subprocess
import os

DEFAULT_SAY_VOICE = "Ava"

def text_to_audio(text, output_filename, voice=DEFAULT_SAY_VOICE):
    """
    Converteert tekst naar spraak met behulp van de macOS 'say' command
    en slaat de audio op als een bestand.

    Args:
        text (str): De tekst die moet worden voorgelezen.
        output_filename (str): De naam van het bestand waarin de audio moet worden opgeslagen
                                (bijv. "mijn_audio.aiff", "mijn_audio.wav", "mijn_audio.mp3").
                                De extensie bepaalt het outputformaat.
        voice (str): De stem die moet worden gebruikt (standaard is "Ava").
    """

    # Bepaal het outputformaat op basis van de bestandsextensie
    _, ext = os.path.splitext(output_filename)
    ext = ext.lower()

    if ext == ".aiff":
        # AIFF is het native formaat van 'say'
        command = ["say", "-v", voice, "-o", output_filename, "--data-format=LEF32@22050", text]
    elif ext == ".wav":
        # Say kan direct naar WAV exporteren
        command = ["say", "-v", voice, "-o", output_filename, "--file-format=WAVE", "--data-format=LEF32@22050", text]
    elif ext == ".mp3":
        # Direct opslaan als MP3 is niet standaard mogelijk met 'say'.
        # We slaan eerst op als AIFF en converteren dan naar MP3 met afconvert.
        temp_aiff_file = output_filename.replace(".mp3", ".aiff")
        print(f"Let op: MP3 vereist een tijdelijk AIFF-bestand: {temp_aiff_file}")

        # Eerst opslaan als AIFF
        aiff_command = ["say", "-v", voice, "-o", temp_aiff_file, "--data-format=LEF32@22050", text]
        try:
            subprocess.run(aiff_command, check=True)
            print(f"Tijdelijk AIFF-bestand '{temp_aiff_file}' succesvol aangemaakt.")
        except subprocess.CalledProcessError as e:
            print(f"Fout bij het aanmaken van AIFF-bestand: {e}")
            return

        # Vervolgens converteren naar MP3 met afconvert
        # afconvert is een macOS-tool voor audioformaatconversie
        mp3_command = ["afconvert", temp_aiff_file, output_filename, "-f", "mp3", "-d", "LEF32@22050"]
        try:
            subprocess.run(mp3_command, check=True)
            print(f"Bestand '{output_filename}' succesvol geconverteerd naar MP3.")
            os.remove(temp_aiff_file) # Verwijder het tijdelijke AIFF-bestand
            print(f"Tijdelijk AIFF-bestand '{temp_aiff_file}' verwijderd.")
        except subprocess.CalledProcessError as e:
            print(f"Fout bij het converteren naar MP3: {e}")
            return
        return # Afgehandeld voor MP3
    else:
        print(f"Niet-ondersteund audioformaat: {ext}. Ondersteunde formaten zijn .aiff, .wav, .mp3.")
        return

    try:
        subprocess.run(command, check=True)
        print(f"Bestand '{output_filename}' succesvol aangemaakt.")
    except FileNotFoundError:
        print("De 'say' command is niet gevonden. Dit script werkt alleen op macOS.")
    except subprocess.CalledProcessError as e:
        print(f"Fout bij het uitvoeren van de 'say' command: {e}")

if __name__ == "__main__":
    tekst_voorlezen = "Hi, Welcome to pose to pose. Make sure you are fully visible to the camera so we can analyse your pose. Make sure your hands are still visible to the camera when reaching up, or sideways."

    # Voorbeeld 1: Opslaan als AIFF
    output_aiff = "test_audio_ava.aiff"
    text_to_audio(tekst_voorlezen, output_aiff)

    # Voorbeeld 2: Opslaan als WAV
    output_wav = "test_audio_ava.wav"
    text_to_audio(tekst_voorlezen, output_wav)

    # Voorbeeld 3: Opslaan als MP3 (vereist afconvert, standaard op macOS)
    output_mp3 = "test_audio_ava.mp3"
    text_to_audio(tekst_voorlezen, output_mp3)

    print("\nProbeer ook met een andere stem (indien beschikbaar op je systeem, bijv. 'Daniel'):")
    text_to_audio("Dit is een test met een andere stem, Daniel.", "test_audio_daniel.wav", voice="Daniel")

    print("\nAlle bestanden zijn aangemaakt in de huidige directory.")