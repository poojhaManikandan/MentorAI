# doc_processor.py — Classroom Co-Pilot AI
# ─────────────────────────────────────────────────────────────────────────────
# Document Processing Module.
# Extracts text from teacher-uploaded educational materials:
#   • PDF textbooks   (via PyPDF2)
#   • DOCX lesson notes  (via python-docx)
#   • TXT worksheets  (native)
# Chunks text into overlapping segments for RAG retrieval.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextChunk:
    """A single chunk of text from an uploaded document."""
    text: str
    source: str         # Original filename
    page: Optional[int] = None   # PDF page number (1-indexed), None for other formats


@dataclass
class DocumentStore:
    """
    Collection of processed text chunks from one or more uploaded files.
    Stored in Streamlit session_state and passed to the retriever.
    """
    chunks: List[TextChunk] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)   # Unique filenames
    total_chars: int = 0

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def summary(self) -> str:
        n = len(self.chunks)
        s = len(self.sources)
        return f"{s} file{'s' if s != 1 else ''} · {n} chunks · {self.total_chars:,} chars"


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[TextChunk]:
    """
    Extracts text from a PDF file page by page.
    Returns one TextChunk per non-empty page.
    """
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF support. Run: pip install PyPDF2")

    chunks: List[TextChunk] = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = _clean_extracted_text(text)
            if len(text.strip()) > 50:   # Skip near-empty pages
                page_chunks = _split_into_chunks(text, filename)
                for pc in page_chunks:
                    pc.page = page_num
                chunks.extend(page_chunks)
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed for '{filename}': {exc}") from exc

    return chunks


def extract_text_from_docx(file_bytes: bytes, filename: str) -> List[TextChunk]:
    """
    Extracts text from a DOCX file, grouping paragraphs into chunks.
    """
    try:
        from docx import Document  # type: ignore
    except ImportError:
        raise ImportError("python-docx is required for DOCX support. Run: pip install python-docx")

    try:
        doc = Document(io.BytesIO(file_bytes))
        full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        cleaned = _clean_extracted_text(full_text)
        return _split_into_chunks(cleaned, filename)
    except Exception as exc:
        raise RuntimeError(f"DOCX extraction failed for '{filename}': {exc}") from exc


def extract_text_from_txt(file_bytes: bytes, filename: str) -> List[TextChunk]:
    """Reads plain text, trying common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(encoding)
            cleaned = _clean_extracted_text(text)
            return _split_into_chunks(cleaned, filename)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode '{filename}' — try saving as UTF-8.")


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    existing_store: Optional[DocumentStore] = None,
) -> DocumentStore:
    """
    Processes one uploaded file and adds its chunks to a DocumentStore.

    Args:
        file_bytes     : Raw bytes of the uploaded file.
        filename       : Original filename (used to determine format).
        existing_store : Existing store to append to; creates new if None.

    Returns:
        Updated DocumentStore with new chunks appended.
    """
    store = existing_store or DocumentStore()
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        new_chunks = extract_text_from_pdf(file_bytes, filename)
    elif ext in ("docx", "doc"):
        new_chunks = extract_text_from_docx(file_bytes, filename)
    elif ext == "txt":
        new_chunks = extract_text_from_txt(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file format: .{ext}. Use PDF, DOCX, or TXT.")

    if not new_chunks:
        raise ValueError(f"No readable text found in '{filename}'. Check the file is not image-only.")

    store.chunks.extend(new_chunks)
    if filename not in store.sources:
        store.sources.append(filename)
    store.total_chars = sum(len(c.text) for c in store.chunks)

    return store


def get_full_context(store: DocumentStore, max_chars: int = 12_000) -> str:
    """
    Concatenates all chunks into a single context string for the LLM,
    capped at max_chars to stay within token limits.
    """
    parts = []
    total = 0
    for chunk in store.chunks:
        header = f"[{chunk.source}" + (f" p.{chunk.page}" if chunk.page else "") + "]"
        entry = f"{header}\n{chunk.text}\n"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_extracted_text(text: str) -> str:
    """Normalises whitespace and removes common PDF extraction artefacts."""
    # Replace form feeds and carriage returns
    text = text.replace("\f", "\n").replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive whitespace within lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse more than 2 consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_chunks(
    text: str,
    filename: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[TextChunk]:
    """
    Splits text into overlapping chunks of ~chunk_size characters,
    breaking at paragraph boundaries where possible.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[TextChunk] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunk_text = "\n\n".join(current)
            chunks.append(TextChunk(text=chunk_text, source=filename))
            # Overlap: keep last paragraph(s) for continuity
            overlap_text = para if len(para) <= overlap else para[-overlap:]
            current = [overlap_text]
            current_len = len(overlap_text)
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append(TextChunk(text="\n\n".join(current), source=filename))

    return chunks
