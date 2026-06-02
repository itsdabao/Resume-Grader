import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =========================
# CONFIG (edit if needed)
# =========================
INPUT_DIR = "data/knowledge_base"  # only *.pdf directly inside this folder
OUTPUT_DIR = "data/knowledge_base/preprocessed_markdown"  # markdown outputs per PDF


def _ensure_env_has_pypdf() -> None:
    """
    If run in an env without pypdf, re-exec via conda env `agent` (expected to have deps).
    """
    if os.getenv("PDF2MD_NO_REEXEC"):
        return
    try:
        import pypdf  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    cmd = ["conda", "run", "-n", "agent", "python", str(Path(__file__).resolve())]
    env = dict(os.environ)
    env["PDF2MD_NO_REEXEC"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("Re-running with: " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, env=env)
        raise SystemExit(r.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Missing dependency `pypdf` and cannot find `conda` to re-exec into env `agent`.\n"
            "Run:\n"
            "  conda run -n agent python scripts/pdf_to_markdown_heuristic.py\n"
            "Or install:\n"
            "  pip install pypdf"
        ) from e


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _looks_like_markdown(text: str) -> bool:
    t = text or ""
    return ("\n# " in t) or ("\n## " in t) or t.lstrip().startswith("# ")


def _split_markdown_sections(markdown: str, *, heading_level: int = 2) -> List[Tuple[str, str]]:
    md = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    if not md.strip():
        return []
    level = max(1, min(6, int(heading_level)))
    pat = re.compile(rf"^(#{{{level}}})\s+(.+?)\s*$")
    lines = md.splitlines()

    out: List[Tuple[str, List[str]]] = []
    cur_h = ""
    cur: List[str] = []

    def flush():
        nonlocal cur, cur_h
        body = "\n".join(cur).strip()
        if body:
            out.append((cur_h.strip() or "Thông tin chung", cur.copy()))
        cur = []

    for ln in lines:
        m = pat.match(ln)
        if m:
            flush()
            cur_h = m.group(2).strip()
            cur = [ln]
        else:
            cur.append(ln)
    flush()
    return [(h, "\n".join(ls).strip()) for h, ls in out]


def _split_plaintext_sections(text: str) -> List[Tuple[str, str]]:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return []
    lines = [ln.rstrip() for ln in s.splitlines()]

    re_numbered = re.compile(r"^\s*(\d+(\.\d+){0,3}|\d+)\s*[\)\.\-:]\s+\S")
    re_roman = re.compile(r"^\s*[IVXLCDM]{1,8}\s*[\)\.\-:]\s+\S", re.IGNORECASE)
    re_label = re.compile(r"^\s*(PHẦN|CHƯƠNG|MỤC|CHUYÊN ĐỀ|GIỚI THIỆU)\b", re.IGNORECASE)
    re_semantic_heading = re.compile(
        r"^\s*(hoc\s*phi|phi\s*tai\s*lieu|lich(\s*khai\s*giang|\s*hoc)?|khai\s*giang|chinh\s*sach|quy\s*dinh|cam\s*ket|khoa\s*hoc|chuong\s*trinh|lo\s*trinh)\b",
        re.IGNORECASE,
    )

    def is_heading(line: str) -> bool:
        ln = (line or "").strip()
        if not ln or len(ln) > 90:
            return False
        if re_semantic_heading.match(_strip_accents(ln).lower()):
            return True
        if re_numbered.match(ln) or re_roman.match(ln) or re_label.match(ln):
            return True
        letters = [ch for ch in ln if ch.isalpha()]
        if letters and len(letters) >= 6:
            upper_ratio = sum(1 for ch in letters if ch == ch.upper()) / float(len(letters))
            if upper_ratio >= 0.85 and len(ln.split()) <= 12:
                return True
        if ln.endswith(":") and len(ln.split()) <= 10:
            return True
        return False

    sections: List[Tuple[str, List[str]]] = []
    cur_h = ""
    cur: List[str] = []

    def flush():
        nonlocal cur, cur_h
        body = "\n".join(cur).strip()
        if body:
            sections.append((cur_h.strip() or "Thông tin chung", cur.copy()))
        cur = []

    for ln in lines:
        if is_heading(ln):
            flush()
            cur_h = ln.strip()
            cur = [ln]
        else:
            cur.append(ln)
    flush()

    return [(h, "\n".join(ls).strip()) for h, ls in sections if ls]


def _split_sections(text: str) -> List[Tuple[str, str]]:
    if _looks_like_markdown(text):
        secs = _split_markdown_sections(text, heading_level=2)
        return secs if secs else [("Document", text)]
    secs = _split_plaintext_sections(text)
    return secs if secs else [("Document", text)]


def _extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


def _render_markdown(pdf_name: str, sections: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    lines.append(f"# Extracted Markdown: {pdf_name}")
    lines.append("")
    for heading, body in sections:
        h = (heading or "Thông tin chung").strip()
        b = (body or "").strip()
        if not b:
            continue

        # Avoid duplicating heading if it's already the first line of body.
        first_line = b.splitlines()[0].strip() if b.splitlines() else ""
        if _strip_accents(first_line).lower() == _strip_accents(h).lower():
            b = "\n".join(b.splitlines()[1:]).strip()

        lines.append(f"## {h}")
        lines.append("")
        lines.append(b)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    _ensure_env_has_pypdf()

    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in in_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {in_dir}")

    for pdf in pdfs:
        text = _extract_pdf_text(pdf)
        sections = _split_sections(text)
        md = _render_markdown(pdf.name, sections)
        out_path = out_dir / f"{pdf.stem}.md"
        out_path.write_text(md, encoding="utf-8")

    print(f"OK: wrote {len(pdfs)} markdown files to {out_dir}")


if __name__ == "__main__":
    main()

