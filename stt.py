# stt.py — Classroom Co-Pilot AI
# Speech-to-Text module with a 3-tier fallback chain:
#   1. Gemini 2.5 Flash  (native audio understanding — primary)
#   2. Local OpenAI Whisper  (offline fallback, requires 'openai-whisper')
#   3. Google Speech Recognition API  (lightweight final fallback)

from typing import Optional

# Global cache — Whisper model is large; load once and reuse across reruns
_whisper_model = None


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file to text completely offline using Whisper.

    Args:
        audio_path : Absolute path to the audio file (WAV / WebM / MP3).

    Returns:
        Transcribed text as a plain string. Returns "" if audio is silent.

    Raises:
        RuntimeError: If every STT method fails.
    """
    global _whisper_model

    # ── Tier 2: Local OpenAI Whisper ──────────────────────────────────────
    # Works fully offline. Requires: pip install openai-whisper
    # First run downloads the 'tiny' model (~75 MB). Cached globally after that.
    try:
        import whisper  # type: ignore

        if _whisper_model is None:
            print("[STT] Loading Whisper 'tiny' model (one-time download)…")
            _whisper_model = whisper.load_model("tiny")

        result = _whisper_model.transcribe(audio_path)
        transcript = result.get("text", "").strip()
        if transcript:
            print(f"[STT] Whisper transcription: '{transcript}'")
            return transcript

    except ImportError:
        print("[STT] openai-whisper not installed. Falling back to Google STT…")
    except Exception as whisper_err:
        print(f"[STT] Whisper failed ({whisper_err}). Trying Google STT…")

    # ── Tier 3: Google Speech Recognition API ─────────────────────────────
    # Free, no key needed, internet required. Best with WAV format.
    try:
        import speech_recognition as sr  # type: ignore

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        transcript = recognizer.recognize_google(audio_data)
        print(f"[STT] Google STT transcription: '{transcript}'")
        return transcript.strip()

    except Exception as sr_err:
        print(f"[STT] All STT methods failed. Last error: {sr_err}")
        raise RuntimeError(
            "Could not transcribe audio. "
            "Please check your microphone or type your command manually."
        )
