from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionOptions:
    """
    Day-2 ingestion options:
    - Prefer LlamaParse for complex PDFs (tables/forms), output markdown.
    - Parse markdown into elements (heading/table/list) before chunking.
    """

    pdf_engine: str = "auto"  # auto | llamaparse | simple
    use_markdown_element_parser: bool = True
    # If True: split markdown into large sections (e.g., by "##") before element parsing / chunking.
    # This helps PDFs that already have a clean structure (brochure/course catalog style).
    section_chunking: bool = True
    section_heading_level: int = 2  # split by headings like "## ..."
    llamaparse_result_type: str = "markdown"
    language: str = "vi"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception as e:
        logger.debug("ingestion_modern: failed to load .env: %s", e)


def _split_files(data_path: str, input_files: Optional[List[str]]) -> List[Path]:
    if input_files:
        return [Path(p) for p in input_files]
    base = Path(data_path)
    if not base.exists():
        raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {base}")
    exts = {".pdf", ".md", ".txt", ".docx", ".rtf"}
    return [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _simple_reader_load(files: List[Path]):
    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(input_files=[str(p) for p in files])
    return reader.load_data()


def _llamaparse_load_pdf(path: Path, *, result_type: str, language: str):
    """
    Parse a single PDF via LlamaParse -> list[Document].
    Requires LLAMA_CLOUD_API_KEY (or LLAMAPARSE_API_KEY).
    """
    import os

    _load_env()
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu LLAMA_CLOUD_API_KEY (hoặc LLAMAPARSE_API_KEY) để dùng LlamaParse.")

    # Prefer llama_parse SDK; fall back to llama-index reader if needed.
    try:
        from llama_parse import LlamaParse  # type: ignore

        parser = LlamaParse(api_key=api_key, result_type=result_type, language=language)
        docs = parser.load_data(str(path))
        return docs
    except Exception as e:
        logger.debug("ingestion_modern: llama_parse SDK failed for %s: %s (fallback to LlamaParseReader)", path, e)
        from llama_index.readers.llama_parse import LlamaParseReader  # type: ignore

        reader = LlamaParseReader(api_key=api_key, result_type=result_type, language=language)
        return reader.load_data(str(path))


def load_documents_for_ingestion(
    data_path: str,
    *,
    input_files: Optional[List[str]] = None,
    opts: IngestionOptions,
):
    """
    Load Documents for ingestion with an optional LlamaParse path for PDFs.

    Strategy:
    - Non-PDF always uses SimpleDirectoryReader.
    - PDF uses:
      - `llamaparse` if forced, or if `auto` and key exists.
      - otherwise SimpleDirectoryReader.
    """
    import os

    files = _split_files(data_path, input_files)
    if not files:
        return []

    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    others = [p for p in files if p.suffix.lower() != ".pdf"]

    # Always load non-PDF via simple reader
    documents = []
    if others:
        documents.extend(_simple_reader_load(others))

    engine = (opts.pdf_engine or "auto").strip().lower()
    _load_env()
    has_key = bool(os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY"))
    use_llamaparse = engine == "llamaparse" or (engine == "auto" and has_key)

    if pdfs and use_llamaparse:
        for pdf in pdfs:
            parsed = _llamaparse_load_pdf(pdf, result_type=opts.llamaparse_result_type, language=opts.language)
            # Ensure source metadata exists for traceability
            for d in parsed:
                try:
                    md = d.metadata or {}
                except Exception:
                    md = {}
                md.setdefault("file_name", pdf.name)
                md.setdefault("source", pdf.name)
                try:
                    d.metadata = md
                except Exception as e:
                    logger.debug("ingestion_modern: failed to set metadata for %s: %s", pdf.name, e)
            documents.extend(parsed)
    elif pdfs:
        documents.extend(_simple_reader_load(pdfs))

    return documents


def _looks_like_markdown(text: str) -> bool:
    if not text:
        return False
    # A few cheap indicators; keep this permissive.
    if "## " in text or "\n## " in text or "\n# " in text:
        return True
    if "\n- " in text or "\n* " in text:
        return True
    if "\n| " in text and "|\n" in text:
        return True
    return False


def _extract_doc_meta(d) -> Dict[str, object]:
    try:
        md = d.metadata or {}
    except Exception:
        md = {}
    return md if isinstance(md, dict) else {}


def _group_documents_by_source(documents) -> List[Tuple[str, str, Dict[str, object]]]:
    """
    Combine potentially multiple Documents (e.g., per page) into one per `source/file_name`.
    Returns list of (source, combined_text, merged_meta).
    """
    groups: Dict[str, Dict[str, object]] = {}
    for d in documents or []:
        try:
            text = getattr(d, "text", "") or ""
        except Exception:
            try:
                text = d.get("text", "")
            except Exception:
                text = ""
        meta = _extract_doc_meta(d)
        src = meta.get("source") or meta.get("file_name") or meta.get("file_path") or "unknown"
        if not isinstance(src, str) or not src:
            src = "unknown"
        g = groups.get(src)
        if g is None:
            groups[src] = {"texts": [text], "meta": dict(meta)}
        else:
            g["texts"].append(text)
            # best-effort keep first values; don't overwrite
            gm = g.get("meta")
            if isinstance(gm, dict):
                for k, v in meta.items():
                    if k not in gm and v is not None:
                        gm[k] = v

    out: List[Tuple[str, str, Dict[str, object]]] = []
    for src, g in groups.items():
        texts = g.get("texts", [])
        if not isinstance(texts, list):
            texts = []
        combined = "\n\n".join([t for t in texts if isinstance(t, str) and t.strip()])
        meta = g.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        out.append((src, combined, meta))
    return out


def _split_markdown_sections(markdown: str, *, heading_level: int) -> List[Tuple[str, str]]:
    """
    Split markdown into sections by a chosen heading level, keeping headings inside their section.
    Returns list of (heading, section_markdown).
    """
    import re

    md = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    if not md.strip():
        return []

    level = max(1, min(6, int(heading_level)))
    pat = re.compile(rf"^(#{{{level}}})\s+(.+?)\s*$")
    lines = md.splitlines()

    sections: List[Tuple[str, List[str]]] = []
    cur_heading = ""
    cur_lines: List[str] = []

    def _flush():
        nonlocal cur_heading, cur_lines
        txt = "\n".join(cur_lines).strip()
        if txt:
            heading = cur_heading.strip() or "Thông tin chung"
            sections.append((heading, cur_lines.copy()))
        cur_lines = []

    for ln in lines:
        m = pat.match(ln)
        if m:
            _flush()
            cur_heading = m.group(2).strip()
            cur_lines = [ln]  # keep heading line in section
        else:
            cur_lines.append(ln)
    _flush()

    # Post: convert to markdown strings
    return [(h, "\n".join(ls).strip()) for h, ls in sections if ls]


def _split_plaintext_sections(text: str) -> List[Tuple[str, str]]:
    """
    Heuristic section splitter for plain text (e.g., PDF extract without markdown headings).

    A "heading line" is detected when it looks like:
    - Numbered outline: "1.", "1.1", "2)", "I.", "II."
    - Uppercase / short title-like line.
    - Vietnamese labels like "PHẦN", "CHƯƠNG", "MỤC".
    """
    import re

    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return []
    lines = [ln.rstrip() for ln in s.splitlines()]

    re_numbered = re.compile(r"^\s*(\d+(\.\d+){0,3}|\d+)\s*[\)\.\-]\s+\S")
    re_roman = re.compile(r"^\s*[IVXLCDM]{1,8}\s*[\)\.\-]\s+\S", re.IGNORECASE)
    re_label = re.compile(r"^\s*(PHẦN|CHƯƠNG|MỤC|CHUYÊN ĐỀ|GIỚI THIỆU)\b", re.IGNORECASE)

    def is_heading(line: str) -> bool:
        ln = (line or "").strip()
        if not ln:
            return False
        if len(ln) > 90:
            return False
        if re_numbered.match(ln) or re_roman.match(ln) or re_label.match(ln):
            return True
        # Uppercase-ish title lines (ignore digits/punct)
        letters = [ch for ch in ln if ch.isalpha()]
        if letters and len(letters) >= 6:
            upper_ratio = sum(1 for ch in letters if ch == ch.upper()) / float(len(letters))
            if upper_ratio >= 0.85 and len(ln.split()) <= 12:
                return True
        # Short title ending with ":" (common label lines)
        if ln.endswith(":") and len(ln.split()) <= 10:
            return True
        return False

    sections: List[Tuple[str, List[str]]] = []
    cur_heading = ""
    cur_lines: List[str] = []

    def flush():
        nonlocal cur_heading, cur_lines
        txt = "\n".join(cur_lines).strip()
        if txt:
            heading = cur_heading.strip() or "Thông tin chung"
            sections.append((heading, cur_lines.copy()))
        cur_lines = []

    for ln in lines:
        if is_heading(ln):
            flush()
            cur_heading = ln.strip().strip("#").strip()
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
    flush()

    # Filter extremely tiny sections (often noise)
    out: List[Tuple[str, str]] = []
    for h, ls in sections:
        body = "\n".join(ls).strip()
        if len(body) < 80 and len(out) > 0:
            # merge into previous
            ph, pb = out[-1]
            out[-1] = (ph, (pb + "\n\n" + body).strip())
        else:
            out.append((h, body))
    return out


def build_nodes_for_ingestion(
    documents,
    *,
    chunk_size: int,
    chunk_overlap: int,
    use_markdown_elements: bool,
    section_chunking: bool = True,
    section_heading_level: int = 2,
):
    """
    Build nodes with structure preservation when possible.

    - If markdown element parser is available and enabled, parse into elements first.
    - Then chunk large nodes with SentenceSplitter to keep chunk size stable.
    """
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # If input is markdown-ish, split into larger "sections" (e.g., brochure headings) first.
    # This ensures we don't mix content across major headings when we later chunk.
    if section_chunking and documents:
        try:
            from llama_index.core import Document
        except Exception:
            Document = None  # type: ignore

        if Document is not None:
            grouped = _group_documents_by_source(documents)
            section_docs = []
            for src, combined, base_meta in grouped:
                if _looks_like_markdown(combined):
                    secs = _split_markdown_sections(combined, heading_level=section_heading_level)
                else:
                    secs = _split_plaintext_sections(combined)
                if not secs:
                    section_docs.append(Document(text=combined, metadata=dict(base_meta)))
                    continue
                for i, (heading, sec_md) in enumerate(secs, 1):
                    md = dict(base_meta)
                    md["source"] = str(md.get("source") or md.get("file_name") or src)
                    md["section_heading"] = heading
                    md["section_index"] = i
                    section_docs.append(Document(text=sec_md, metadata=md))
            documents = section_docs

    nodes = None
    if use_markdown_elements:
        try:
            from llama_index.core.node_parser import MarkdownElementNodeParser  # type: ignore

            md_parser = MarkdownElementNodeParser()
            nodes = md_parser.get_nodes_from_documents(documents)
        except Exception:
            nodes = None

    if nodes is None:
        base_nodes = splitter.get_nodes_from_documents(documents)
        # Context Injection Step: Prepend metadata context to text for better retrieval
        for n in base_nodes:
            ctx = []
            if n.metadata.get("source"):
                src_name = Path(str(n.metadata["source"])).stem  # "Bang_gia_IELTS"
                ctx.append(src_name)
            if n.metadata.get("section_heading"):
                ctx.append(str(n.metadata["section_heading"]))
            
            if ctx:
                breadcrumb = " > ".join(ctx)
                # Format: [Breadcrumb] Original text...
                n.text = f"[{breadcrumb}]\n{n.text}"
                n.excluded_embed_metadata_keys.extend(["source", "section_heading"])
                n.excluded_llm_metadata_keys.extend(["source", "section_heading"])
        return base_nodes

    # Chunk large element-nodes while keeping metadata (element_type, heading, page, ...)
    try:
        from llama_index.core import Document
    except Exception:
        Document = None  # type: ignore

    out = []
    for n in nodes:
        try:
            text = n.get_text()
        except Exception:
            text = getattr(n, "text", "") or ""
        if len(text) <= chunk_size * 2 or Document is None:
            out.append(n)
            continue
        try:
            md = getattr(n, "metadata", {}) or {}
            d = Document(text=text, metadata=dict(md))
            out.extend(splitter.get_nodes_from_documents([d]))
        except Exception:
            out.extend(splitter.get_nodes_from_documents([d]))
        except Exception:
            out.append(n)

    # Post-process for element nodes as well: inject context if missing
    for n in out:
        if not n.text.startswith("["):
            ctx = []
            if n.metadata.get("source"):
                ctx.append(Path(str(n.metadata["source"])).stem)
            if n.metadata.get("section_heading"):
                ctx.append(str(n.metadata["section_heading"]))
            
            if ctx:
                breadcrumb = " > ".join(ctx)
                n.text = f"[{breadcrumb}]\n{n.text}"
                # Avoid double-counting tokens in metadata
                if "source" not in n.excluded_embed_metadata_keys:
                    n.excluded_embed_metadata_keys.append("source")
                if "section_heading" not in n.excluded_embed_metadata_keys:
                    n.excluded_embed_metadata_keys.append("section_heading")

    return out
