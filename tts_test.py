#!/usr/bin/env python3
"""
Interactive text-to-speech test script with pyttsx3.
- Lists available voices (with index, name, and id)
- Lets user select a voice by index or id (stable identifier)
- Optionally plays example audio cues in English
"""
import pyttsx3


def list_voices(engine):
    voices = engine.getProperty('voices')
    print("Available voices:")
    for i, voice in enumerate(voices):
        langs = ''.join(voice.languages) if voice.languages else ''
        print(f"{i}: {voice.name} | id={voice.id} | ({langs})")


def select_voice(engine):
    voices = engine.getProperty('voices')
    while True:
        choice = input("Enter voice index or full id to select (or 'q' to quit): ")
        if choice.lower() == 'q':
            return None
        # try index selection
        try:
            idx = int(choice)
            if 0 <= idx < len(voices):
                return voices[idx].id
            else:
                print("Index out of range.")
                continue
        except ValueError:
            # treat choice as id
            for voice in voices:
                if choice.strip() == voice.id:
                    return voice.id
            print("No voice found with that id. Please try again.")


def main():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)    # Speaking rate
    engine.setProperty('volume', 1.0)  # Volume (0.0–1.0)

    list_voices(engine)
    selected_voice_id = select_voice(engine)
    if not selected_voice_id:
        print("No voice selected. Exiting.")
        return

    engine.setProperty('voice', selected_voice_id)
    print(f"Selected voice id: {selected_voice_id}")

    # Optionally play sample cues
    play_cues = input("Play example cues with selected voice? (y/n): ")
    if play_cues.lower() == 'y':
        cues = [
            "Voice test successful.",
            "Welcome to the text to speech test.",
            "Prepare for your yoga session.",
            "Starting in three, two, one.",
            "Hold the pose.",
            "And relax, release your muscles."
        ]
        for cue in cues:
            print(f"Speaking: {cue}")
            engine.say(cue)
            engine.runAndWait()

    print("Test completed.")

if __name__ == "__main__":
    main()
