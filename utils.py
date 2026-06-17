# utils.py — Classroom Co-Pilot AI
# Utility helpers: intent detection, text sanitisation, topic extraction.

import re


# ── Intent keyword dictionaries ─────────────────────────────────────────────
_QUIZ_KEYWORDS = {
    "quiz", "test", "questions", "question", "mcq", "mcqs",
    "assessment", "assess", "examine", "exam",
}
_STORY_KEYWORDS = {
    "story", "stories", "narrative", "tale", "analogy",
    "as a story", "tell me a story", "explain as story",
    "explain with a story", "story mode",
}


def detect_intent(command: str) -> str:
    """
    Analyses a teacher's command and returns the detected content intent.

    Returns one of:
        "QUIZ"    — teacher wants a multiple-choice quiz
        "STORY"   — teacher wants a narrative / analogy explanation
        "CONCEPT" — teacher wants a standard concept explanation (default)

    Examples:
        >>> detect_intent("Create a quiz on Photosynthesis")
        'QUIZ'
        >>> detect_intent("Explain Gravity as a story")
        'STORY'
        >>> detect_intent("Explain Newton's Laws")
        'CONCEPT'
    """
    cmd = command.lower()

    # Quiz has highest priority
    for kw in _QUIZ_KEYWORDS:
        if kw in cmd:
            return "QUIZ"

    # Story / narrative mode
    for kw in _STORY_KEYWORDS:
        if kw in cmd:
            return "STORY"

    return "CONCEPT"


def clean_tts_text(text: str) -> str:
    """
    Strips Markdown formatting from text so gTTS reads it naturally.
    Removes: **bold**, *italic*, # headings, `code`, bullet markers.
    """
    # Remove markdown headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold / italic markers
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Remove inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Remove bullet/dash markers
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_for_tts(text: str, max_chars: int = 420) -> str:
    """
    Truncates text to max_chars at a word boundary for TTS playback.
    Adds a gentle prompt to look at the board for the rest.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "… Board pe poori explanation dekho."


def extract_topic(command: str) -> str:
    """
    Heuristically extracts the topic from a teacher command.
    Used for display labels when the LLM response is not yet available.

    Examples:
        "Explain Photosynthesis"            → "Photosynthesis"
        "Create a quiz on Newton's Laws"    → "Newton's Laws"
        "Tell me a story about Gravity"     → "Gravity"
        "Ask a socratic question about Photosynthesis to make students think" → "Photosynthesis"
        "Give a real world example of Photosynthesis" → "Photosynthesis"
    """
    cmd = command.strip()
    patterns = [
        r"^(?:explain|describe|tell me about|teach me about|batao|samjhao)\s+",
        r"^(?:create a quiz on|quiz me on|generate a quiz on|make a quiz on)\s+",
        r"^(?:what is|what are|what's)\s+",
        r"^(?:tell me a story about|explain as a story)\s+",
        r"^(?:ask a socratic question about|socratic question about|socratic question on)\s+",
        r"^(?:give a real world example of|give a real-world example of|real world example of|real-world example of|real world example on|real-world example on)\s+",
        r"^(?:create a debate topic around|debate topic around|debate on|debate about)\s+",
        r"\s+as\s+a\s+story$",
        r"\s+as\s+a\s+narrative$",
        r"\s+ki\s+story\s+batao$",
        r"\s+explain\s+karo$",
        r"\s+to\s+make\s+students\s+think$",
    ]
    for pat in patterns:
        cmd = re.sub(pat, "", cmd, flags=re.IGNORECASE).strip()
    return cmd.title() if cmd else command.title()

