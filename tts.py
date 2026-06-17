# tts.py — Classroom Co-Pilot AI
# Text-to-Speech module using gTTS.
# Returns an in-memory BytesIO MP3 buffer for st.audio() playback.

import io
from gtts import gTTS

from utils import clean_tts_text, truncate_for_tts


def generate_audio(text: str, lang: str = "en", max_chars: int = 420) -> io.BytesIO:
    """
    Converts text to speech and returns an MP3 audio buffer.

    Args:
        text      : Text to synthesise. Markdown is stripped automatically.
        lang      : BCP-47 language code (default 'en').
        max_chars : Hard clip — keeps classroom TTS to ~30-45 seconds.

    Returns:
        io.BytesIO MP3 buffer ready for st.audio().

    Raises:
        ValueError   : If text is empty after cleaning.
        RuntimeError : If gTTS synthesis fails.
    """
    clean = clean_tts_text(text)
    if not clean.strip():
        raise ValueError("No speakable text after Markdown cleaning.")

    clip = truncate_for_tts(clean, max_chars=max_chars)

    try:
        tts_obj = gTTS(text=clip, lang=lang, slow=False)
        buf = io.BytesIO()
        tts_obj.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as exc:
        raise RuntimeError(f"gTTS synthesis failed: {exc}") from exc


def build_tts_script(content) -> str:
    """
    Builds the most appropriate TTS narration text for a ClassroomContent object.

    Selects the right field based on intent (quiz / story / concept)
    and prepends a friendly classroom introduction.
    """
    if content.intent == "QUIZ":
        n = len(content.quiz or [])
        return (
            f"Quiz time! Aaj ka quiz hai {content.topic} par. "
            f"Total {n} sawaal hain. "
            "Dhyan se suno aur apne jawab likhte jao. "
            "Teacher sahi jawab baad mein reveal karenge."
        )

    if content.intent == "STORY" and content.story:
        return (
            f"Aaj hum {content.topic} ke baare mein ek kahaani ke zariye seekhenge. "
            f"{content.story}"
        )

    if content.explanation:
        return f"Aaj ka topic hai {content.topic}. {content.explanation}"

    return f"{content.topic} ka content ab board pe ready hai."
