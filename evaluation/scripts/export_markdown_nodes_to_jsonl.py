import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Section:
    heading: str
    text: str
    index: int


def _derive_tenant_id(md_path: Path) -> str:
    name = md_path.stem
    if name.lower().startswith("tenant_"):
        name = name[len("tenant_") :]
    name = re.sub(r"[^a-z0-9_]+", "", name.lower())
    return name or md_path.stem.lower()


def _split_markdown_sections(markdown: str, heading_level: int = 2) -> List[Section]:
    md = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    if not md.strip():
        return []

    level = max(1, min(6, int(heading_level)))
    pat = re.compile(rf"^(#{{{level}}})\s+(.+?)\s*$")
    lines = md.splitlines()

    sections: List[Tuple[str, List[str]]] = []
    cur_heading = "Thông tin chung"
    cur_lines: List[str] = []

    def flush():
        nonlocal cur_heading, cur_lines
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append((cur_heading, cur_lines.copy()))
        cur_lines = []

    for ln in lines:
        m = pat.match(ln)
        if m:
            flush()
            cur_heading = m.group(2).strip()
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
    flush()

    out: List[Section] = []
    for idx, (h, ls) in enumerate(sections, 1):
        body = "\n".join(ls).strip()
        if not body:
            continue
        out.append(Section(heading=h.strip() or "Thông tin chung", text=body, index=idx))
    return out


def _find_split_point(s: str, start: int, end: int) -> int:
    """
    Pick a nicer split point near `end` to avoid cutting in the middle of lines/words.
    """
    if end >= len(s):
        return len(s)
    lo = start + int((end - start) * 0.6)
    window = s[lo:end]

    # Prefer newline boundary
    nl = window.rfind("\n")
    if nl != -1:
        return lo + nl + 1

    # Fallback to whitespace boundary
    ws = max(window.rfind(" "), window.rfind("\t"))
    if ws != -1:
        return lo + ws + 1

    return end


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> List[str]:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return []
    if chunk_size <= 0:
        return [s]
    overlap = max(0, int(chunk_overlap))
    if len(s) <= chunk_size:
        return [s]

    chunks: List[str] = []
    start = 0
    while start < len(s):
        end = min(len(s), start + chunk_size)
        split_end = _find_split_point(s, start, end)
        if split_end <= start:
            split_end = end
        chunk = s[start:split_end].strip()
        if chunk:
            chunks.append(chunk)
        if split_end >= len(s):
            break
        start = max(0, split_end - overlap)
    return chunks


def export_nodes_from_markdown(
    md_path: Path,
    *,
    output_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    heading_level: int,
) -> int:
    text = md_path.read_text(encoding="utf-8")
    sections = _split_markdown_sections(text, heading_level=heading_level) or [
        Section(heading="Document", text=text, index=1)
    ]
    tenant_id = _derive_tenant_id(md_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for sec in sections:
            for ci, chunk in enumerate(
                _chunk_text(sec.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap), 1
            ):
                node_id = f"{tenant_id}:{sec.index}:{ci}"
                rec = {
                    "id": node_id,
                    "text": chunk,
                    "metadata": {
                        "tenant_id": tenant_id,
                        "file_name": md_path.name,
                        "file_path": str(md_path.as_posix()),
                        "source": md_path.name,
                        "section_heading": sec.heading,
                        "section_index": sec.index,
                        "chunk_index": ci,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count


def iter_markdown_files(input_dir: Path, pattern: str) -> Iterable[Path]:
    if "*" in pattern or "?" in pattern or "[" in pattern:
        yield from sorted(input_dir.glob(pattern))
    else:
        # Treat as extension shorthand
        if pattern.startswith("."):
            yield from sorted(input_dir.glob(f"*{pattern}"))
        else:
            yield from sorted(input_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chunked node JSONL from preprocessed markdown files.")
    parser.add_argument(
        "--input_dir",
        default="data/knowledge_base/preprocessed_markdown",
        help="Folder containing preprocessed markdown files",
    )
    parser.add_argument(
        "--output_dir",
        default="data/knowledge_base/new_jsonl",
        help="Output folder for JSONL node files",
    )
    parser.add_argument("--pattern", default="tenant_*.md", help="Glob pattern for markdown files")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--heading_level", type=int, default=2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input_dir: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = [p for p in iter_markdown_files(input_dir, args.pattern) if p.is_file()]
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {input_dir} with pattern={args.pattern}")

    total_nodes = 0
    for md in md_files:
        tenant_id = _derive_tenant_id(md)
        out_path = output_dir / f"nodes_{tenant_id}.jsonl"
        n = export_nodes_from_markdown(
            md,
            output_path=out_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            heading_level=args.heading_level,
        )
        print(f"Wrote {n} nodes -> {out_path}")
        total_nodes += n

    print(f"Done. Total nodes: {total_nodes}. Output dir: {output_dir}")


if __name__ == "__main__":
    main()

