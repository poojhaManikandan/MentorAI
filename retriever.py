# retriever.py — Classroom Co-Pilot AI
# ─────────────────────────────────────────────────────────────────────────────
# Keyword-Based Document Retrieval (RAG without a vector database).
#
# Strategy:
#   1. Tokenise the query into meaningful keywords (stop-words removed).
#   2. Score each document chunk by keyword frequency (TF-lite).
#   3. Apply a topic bonus when a chapter/subject is specified.
#   4. Return the top-k highest-scoring chunks as grounded context.
#
# This works fully offline, requires no extra API calls, and runs
# fast enough for real-time classroom use.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from typing import List, Optional

from doc_processor import DocumentStore, TextChunk


# ── Common English + Hindi stop words to ignore during scoring ─────────────
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "then", "once", "and", "but", "or", "nor", "so", "yet", "both",
    "either", "neither", "not", "only", "own", "same", "than", "too",
    "very", "just", "because", "as", "until", "while", "that", "this",
    "these", "those", "what", "which", "who", "whom", "when", "where",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "its", "it", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "they", "them", "their",
    # Common Hinglish filler words
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "ek", "aur",
    "kya", "karo", "karein", "explain", "batao", "tell",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stop words and short tokens."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


def _score_chunk(chunk: TextChunk, query_tokens: List[str], topic_tokens: List[str]) -> float:
    """
    Scores a single chunk against query and topic tokens based on frequency.
    """
    chunk_lower = chunk.text.lower()
    chunk_tokens_list = _tokenize(chunk.text)
    score = 0.0

    for token in query_tokens:
        count = chunk_tokens_list.count(token)
        if count > 0:
            score += (count * 1.0)
        else:
            # Substring match (e.g. photosynthesis matching photosynthesizing)
            occurrences = chunk_lower.count(token)
            score += (occurrences * 0.5)

    for token in topic_tokens:
        count = chunk_tokens_list.count(token)
        if count > 0:
            score += (count * 2.0)
        else:
            occurrences = chunk_lower.count(token)
            score += (occurrences * 1.0)

    # Normalise by chunk length to avoid bias toward very long chunks
    if len(chunk.text) > 0:
        score = score / (len(chunk.text) ** 0.3)  # Slightly stronger length penalty

    return score


def search(
    query: str,
    doc_store: DocumentStore,
    topic: Optional[str] = None,
    chapter: Optional[str] = None,
    subject: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.05,
) -> List[TextChunk]:
    """
    Returns the top-k most relevant document chunks for a given query.

    Args:
        query     : Teacher's command / question (e.g. "Explain photosynthesis").
        doc_store : Populated DocumentStore from doc_processor.
        topic     : Optional topic name (boosts relevance of matching chunks).
        chapter   : Optional chapter name (boosts relevance).
        subject   : Optional subject name (boosts relevance).
        top_k     : Maximum number of chunks to return.
        min_score : Minimum relevance score — chunks below this are excluded.

    Returns:
        List of TextChunk objects, sorted by relevance (highest first).
    """
    if doc_store.is_empty():
        return []

    # Build query token set
    query_tokens = _tokenize(query)

    # Build topic token set from topic + chapter + subject
    topic_text = " ".join(filter(None, [topic, chapter, subject]))
    topic_tokens = _tokenize(topic_text)

    # Score all chunks
    scored = [
        (chunk, _score_chunk(chunk, query_tokens, topic_tokens))
        for chunk in doc_store.chunks
    ]

    # Filter and sort
    scored = [(c, s) for c, s in scored if s >= min_score]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored[:top_k]]


def build_context_string(
    chunks: List[TextChunk],
    max_chars: int = 6_000,
) -> str:
    """
    Formats retrieved chunks into a clean context string for the LLM prompt.

    Args:
        chunks    : Retrieved TextChunk objects from search().
        max_chars : Hard cap to stay within LLM context window.

    Returns:
        Formatted string with source labels, ready to inject into the prompt.
    """
    if not chunks:
        return ""

    parts = ["=== TEXTBOOK CONTENT (use this as primary source) ==="]
    total = len(parts[0])

    for chunk in chunks:
        header = f"\n[Source: {chunk.source}" + (f", Page {chunk.page}" if chunk.page else "") + "]"
        entry = f"{header}\n{chunk.text}\n"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    parts.append("=== END OF TEXTBOOK CONTENT ===")
    return "\n".join(parts)
