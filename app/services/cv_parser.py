"""
CV Parser — extract plain text from PDF / DOCX files.

Works with in-memory bytes (from FastAPI UploadFile) so we don't need temp files
on disk for most cases.
"""

from __future__ import annotations

import io
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _is_scanned_pdf(file_bytes: bytes) -> bool:
    """Quick heuristic: check if a PDF is mostly scanned images (very little extractable text)."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return True
            total_chars = 0
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                total_chars += len(page_text.strip())
            avg_chars = total_chars / len(pdf.pages)
            # Less than 50 chars per page on average = likely scanned
            return avg_chars < 50
    except Exception:
        return False


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes. Tries multiple strategies and picks the best result.
    For scanned PDFs, uses Gemini Vision OCR.
    """
    # Early detection: if PDF is scanned, skip directly to Vision OCR
    if _is_scanned_pdf(file_bytes):
        logger.info("Detected scanned PDF — skipping text extraction, going to Vision OCR.")
        vision_text = _extract_text_from_pdf_vision(file_bytes)
        if vision_text.strip():
            return _clean_text(vision_text)
        # If Vision OCR also fails, continue with normal strategies anyway

    fitz_text = ""
    plumber_text = ""

    # Strategy 1: pymupdf (fitz) — better word spacing for most CVs
    try:
        import fitz  # pymupdf

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        fitz_text = "\n\n".join(pages)
    except ImportError:
        logger.debug("pymupdf not installed")
    except Exception as e:
        logger.warning("pymupdf failed: %s", e)

    # Strategy 2: pdfplumber (excellent table/layout handling)
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            plumber_text = "\n\n".join(pages)
    except ImportError:
        logger.debug("pdfplumber not installed")
    except Exception as e:
        logger.warning("pdfplumber failed: %s", e)

    # Pick the best result: prefer fitz (better word spacing) when available
    # Use pdfplumber only if fitz failed or produced less content
    text = ""
    if fitz_text.strip() and len(fitz_text.strip()) > 100:
        text = fitz_text
    elif plumber_text.strip():
        text = plumber_text

    if text.strip() and len(text.strip()) > 100:
        return _clean_text(text)

    # Strategy 3: LlamaIndex SimpleDirectoryReader (writes temp file)
    try:
        text = _extract_via_llamaindex(file_bytes, suffix=".pdf")
    except Exception as e:
        logger.warning("LlamaIndex PDF reader failed: %s", e)

    if text.strip() and len(text.strip()) > 100:
        return _clean_text(text)

    # Strategy 4: Vision API OCR (for scanned PDFs)
    logger.info("Normal extraction yielded too little text. Attempting Vision OCR fallback...")
    vision_text = _extract_text_from_pdf_vision(file_bytes)
    if vision_text.strip():
        return _clean_text(vision_text)

    logger.error("All PDF extraction strategies failed — returning empty string")
    # Return whatever we got initially (even if it's less than 100 chars, it's better than nothing)
    return _clean_text(text) if text.strip() else ""


def _extract_text_from_pdf_vision(file_bytes: bytes) -> str:
    """
    Convert PDF pages to images and use Google Gemini Vision API to OCR the text.
    Handles up to 10 pages. Supports Vietnamese and English documents.
    """
    import base64
    import os
    import requests

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY is missing, skipping Vision OCR.")
        return ""
    
    # We will use the model specified in .env or fallback to a known model
    model_name = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash")

    OCR_PROMPT = (
        "You are an OCR system. Extract ALL text from this document image. "
        "The document may be in Vietnamese or English. "
        "Output only the extracted text, preserving its structure: "
        "headings, bullet points, tables, and all diacritical marks (dấu tiếng Việt). "
        "Do not add conversational text, commentary, or interpretation. "
        "Do not skip any sections."
    )

    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []

        # Process up to 10 pages
        max_pages = min(10, len(doc))
        for page_idx in range(max_pages):
            page = doc[page_idx]
            # Zoom x3 for better resolution (important for small text & Vietnamese diacritics)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": OCR_PROMPT},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": b64_img
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 8192
                }
            }

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            try:
                resp = requests.post(url, json=payload, timeout=90)
                if resp.status_code != 200:
                    logger.error(f"Gemini API Error on page {page_idx+1}: {resp.text[:300]}")
                    continue
                
                data = resp.json()
                extracted = data["candidates"][0]["content"]["parts"][0]["text"]
                pages_text.append(extracted)
            except (KeyError, IndexError):
                logger.error(f"Unexpected Gemini response format on page {page_idx+1}")
            except requests.exceptions.Timeout:
                logger.warning(f"Vision OCR timeout on page {page_idx+1}, skipping")
            except Exception as e:
                logger.error(f"Vision OCR error on page {page_idx+1}: {e}")

        doc.close()
        return "\n\n".join(pages_text)
    except Exception as e:
        logger.error(f"Vision OCR extraction failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from DOCX bytes using python-docx.
    """
    text = ""

    try:
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
    except ImportError:
        logger.warning("python-docx not installed, trying LlamaIndex fallback")
    except Exception as e:
        logger.warning("python-docx failed: %s, trying fallback", e)

    if text.strip():
        return _clean_text(text)

    # Fallback: LlamaIndex reader
    try:
        text = _extract_via_llamaindex(file_bytes, suffix=".docx")
    except Exception as e:
        logger.warning("LlamaIndex DOCX reader failed: %s", e)

    if text.strip():
        return _clean_text(text)

    logger.error("All DOCX extraction strategies failed — returning empty string")
    return ""


# ---------------------------------------------------------------------------
# Generic DOC extraction (legacy .doc format)
# ---------------------------------------------------------------------------

def extract_text_from_doc(file_bytes: bytes) -> str:
    """Best-effort extraction for legacy .doc files."""
    try:
        text = _extract_via_llamaindex(file_bytes, suffix=".doc")
        if text.strip():
            return _clean_text(text)
    except Exception as e:
        logger.warning("DOC extraction failed: %s", e)
    return ""


# ---------------------------------------------------------------------------
# Router — detect format and dispatch
# ---------------------------------------------------------------------------

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Auto-detect file format from filename extension and extract text.
    Returns cleaned plain text suitable for LLM processing.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_bytes)
    elif ext == ".doc":
        return extract_text_from_doc(file_bytes)
    elif ext in (".txt", ".md"):
        # Plain text — just decode
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return file_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return file_bytes.decode("utf-8", errors="replace")
    else:
        logger.warning("Unsupported file format: %s — attempting PDF extraction", ext)
        return extract_text_from_pdf(file_bytes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_via_llamaindex(file_bytes: bytes, *, suffix: str) -> str:
    """Write bytes to temp file and use LlamaIndex SimpleDirectoryReader."""
    from llama_index.core import SimpleDirectoryReader

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        reader = SimpleDirectoryReader(input_files=[tmp_path])
        docs = reader.load_data()
        return "\n\n".join(d.text for d in docs if d.text)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _clean_text(text: str) -> str:
    """Normalize whitespace, remove CID references, fix merged words, strip control chars."""
    # Remove null bytes and control chars (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Remove PDF CID references like (cid:131), (cid:239) — common in fonts with custom encoding
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Fix words merged by PDF kerning: insert space between lowercase+Uppercase transitions
    # e.g. "reducingunnecessary" → "reducing unnecessary"
    # Only apply to sequences of 15+ chars without spaces (likely merged words)
    def _fix_merged(match: re.Match) -> str:
        word = match.group(0)
        if len(word) >= 15:
            # Insert space before uppercase letters that follow lowercase
            return re.sub(r"([a-z])([A-Z])", r"\1 \2", word)
        return word
    text = re.sub(r"[A-Za-z]{15,}", _fix_merged, text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def guess_candidate_name(cv_text: str, filename: str) -> str:
    """
    Best-effort: extract candidate name from CV text or filename.
    Heuristic: first non-empty line that looks like a name (2-5 words, title-case).
    Falls back to cleaned filename stem.
    """
    lines = [ln.strip() for ln in cv_text.splitlines() if ln.strip()]
    for line in lines[:5]:
        # Skip lines that are clearly headers/labels
        lower = line.lower()
        if any(kw in lower for kw in ("curriculum", "resume", "cv", "tóm tắt", "hồ sơ", "ứng viên")):
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w.isalpha()):
            return line
    # Fallback: filename
    stem = Path(filename).stem
    return stem.replace("-", " ").replace("_", " ").strip().title()
