import sys
from pathlib import Path
import os
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =========================
# CONFIG (edit if needed)
# =========================
INPUT_FILE = "data/knowledge_base/tenant_flexenglish.pdf"
OUTPUT_MD = "data/.cache/chunk_preview_tenant_flexenglish.md"

# Use "simple" to avoid network (LlamaParse). Use "llamaparse" if you have keys + network access.
PDF_ENGINE = "simple"  # auto | llamaparse | simple

# Chunking settings
USE_MD_ELEMENTS = True
SECTION_CHUNKING = True
SECTION_HEADING_LEVEL = 2
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _safe_text_preview(text: str, n: int = 600) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + " ..."


def main() -> None:
    # Prefer the real ingestion pipeline (requires llama-index).
    have_llama_index = True
    try:
        import llama_index  # type: ignore  # noqa: F401
    except Exception:
        have_llama_index = False

    in_path = Path(INPUT_FILE)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    nodes = []
    docs = []
    fallback_note = None

    if have_llama_index:
        from app.services.ingestion_modern import IngestionOptions, build_nodes_for_ingestion, load_documents_for_ingestion

        opts = IngestionOptions(
            pdf_engine=PDF_ENGINE,
            use_markdown_element_parser=USE_MD_ELEMENTS,
            section_chunking=SECTION_CHUNKING,
            section_heading_level=SECTION_HEADING_LEVEL,
        )

        docs = load_documents_for_ingestion(str(in_path.parent), input_files=[str(in_path)], opts=opts)
        nodes = build_nodes_for_ingestion(
            docs,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            use_markdown_elements=USE_MD_ELEMENTS,
            section_chunking=SECTION_CHUNKING,
            section_heading_level=SECTION_HEADING_LEVEL,
        )
    else:
        # Fallback mode: no llama-index available. Try to extract text via pypdf and apply
        # the same "section-aware" splitting + a simple character-based chunker.
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as e:
            # Convenience: if we're running in a bare env, re-exec via conda env `agent`
            # (which is expected to have llama-index dependencies installed).
            if not os.getenv("PREVIEW_CHUNKING_NO_REEXEC"):
                try:
                    cmd = [
                        "conda",
                        "run",
                        "-n",
                        "agent",
                        "python",
                        str(Path(__file__).resolve()),
                    ]
                    print("Re-running with: " + " ".join(cmd))
                    env = dict(os.environ)
                    env["PREVIEW_CHUNKING_NO_REEXEC"] = "1"
                    r = subprocess.run(cmd, env=env)
                    raise SystemExit(r.returncode)
                except FileNotFoundError:
                    pass

            msg = (
                "This preview script needs either `llama-index` (preferred) or `pypdf` (fallback).\n"
                "You're running in an environment without `llama-index`.\n\n"
                "Recommended:\n"
                "  conda run -n agent python scripts/preview_chunking.py\n\n"
                "Or install deps into current env:\n"
                "  pip install llama-index pypdf\n"
            )
            raise RuntimeError(msg) from e

        reader = PdfReader(str(in_path))
        text_parts = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                text_parts.append(t)
        full_text = "\n\n".join(text_parts).strip()

        from app.services.ingestion_modern import _split_plaintext_sections  # type: ignore

        if SECTION_CHUNKING:
            sections = _split_plaintext_sections(full_text)
        else:
            sections = [("Document", full_text)]

        def char_chunks(s: str, *, max_chars: int, overlap: int) -> list[str]:
            s = (s or "").strip()
            if not s:
                return []
            if len(s) <= max_chars:
                return [s]
            out = []
            start = 0
            while start < len(s):
                end = min(len(s), start + max_chars)
                out.append(s[start:end])
                if end >= len(s):
                    break
                start = max(0, end - overlap)
            return out

        fake_nodes = []
        for section_index, (heading, section_text) in enumerate(sections, 1):
            for ci, chunk in enumerate(
                char_chunks(section_text, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP), 1
            ):
                fake_nodes.append(
                    {
                        "node_id": f"fallback:{section_index}:{ci}",
                        "text": chunk,
                        "metadata": {
                            "source": in_path.name,
                            "section_heading": heading,
                            "section_index": section_index,
                        },
                    }
                )

        nodes = fake_nodes
        fallback_note = "fallback_mode=pypdf+plaintext_sections+char_chunker (no llama-index)"

    out_path = Path(OUTPUT_MD)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Chunk preview: `{in_path.as_posix()}`")
    lines.append("")
    if fallback_note:
        lines.append(f"> Note: {fallback_note}")
        lines.append("")
    lines.append("## Settings")
    lines.append(f"- pdf_engine: `{PDF_ENGINE}`")
    lines.append(f"- section_chunking: `{SECTION_CHUNKING}` (heading_level={SECTION_HEADING_LEVEL})")
    lines.append(f"- use_markdown_element_parser: `{USE_MD_ELEMENTS}`")
    lines.append(f"- chunk_size: `{CHUNK_SIZE}`, chunk_overlap: `{CHUNK_OVERLAP}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- documents_loaded: `{len(docs) if isinstance(docs, list) else 0}`")
    lines.append(f"- nodes_built: `{len(nodes)}`")
    lines.append("")

    for i, n in enumerate(nodes, 1):
        if isinstance(n, dict):
            node_id = n.get("node_id") or n.get("id") or n.get("id_") or ""
            text = n.get("text") or ""
            meta = n.get("metadata") or {}
        else:
            node_id = None
            for attr in ("node_id", "id_", "id"):
                try:
                    v = getattr(n, attr, None)
                except Exception:
                    v = None
                if isinstance(v, str) and v:
                    node_id = v
                    break

            try:
                text = n.get_text()
            except Exception:
                text = getattr(n, "text", "") or ""

            try:
                meta = getattr(n, "metadata", {}) or {}
            except Exception:
                meta = {}
        if not isinstance(text, str):
            text = str(text)
        if not isinstance(meta, dict):
            meta = {}

        source = meta.get("source") or meta.get("file_name") or meta.get("file_path") or "unknown"
        section_heading = meta.get("section_heading")
        section_index = meta.get("section_index")
        element_type = meta.get("element_type")

        lines.append(f"## Node {i}")
        lines.append(f"- id: `{node_id or ''}`")
        lines.append(f"- source: `{source}`")
        if section_heading:
            lines.append(f"- section_heading: `{section_heading}`")
        if section_index:
            lines.append(f"- section_index: `{section_index}`")
        if element_type:
            lines.append(f"- element_type: `{element_type}`")
        lines.append(f"- text_len: `{len(text)}`")
        lines.append("")
        lines.append("```text")
        lines.append(_safe_text_preview(text, 800))
        lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    # Keep stdout ASCII-only to avoid Windows console encoding issues.
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()
