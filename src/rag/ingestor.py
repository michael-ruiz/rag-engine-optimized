"""Document loading and section-aware recursive chunking for .md, .txt, .pdf and .py files."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".py"}
_SEPARATORS = ["\n\n", "\n", ". ", " "]
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_PY_DEF = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)")


@dataclass(frozen=True)
class Chunk:
    text: str
    source_file: str
    chunk_index: int
    section_header: str

    @property
    def citation(self) -> str:
        return f"{self.source_file}:{self.chunk_index}"

    def to_payload(self) -> dict:
        return asdict(self)


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    """Split text into pieces <= size, preferring the coarsest separator that occurs."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep in text:
            parts = text.split(sep)
            out: list[str] = []
            for j, part in enumerate(parts):
                piece = part + (sep if j < len(parts) - 1 else "")
                out.extend(_recursive_split(piece, size, separators[i + 1:]))
            return out
    return [text[i:i + size] for i in range(0, len(text), size)]  # no separator left: hard cut


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merge recursive pieces into chunks of <= chunk_size chars, carrying an overlap tail."""
    chunks: list[str] = []
    current = ""
    for piece in _recursive_split(text, chunk_size, _SEPARATORS):
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            if " " in tail:
                tail = tail[tail.index(" ") + 1:]  # start the overlap on a word boundary
            current = tail[max(0, len(tail) + len(piece) - chunk_size):]
        current += piece
    chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def _split_sections(text: str, heading: re.Pattern, md_fences: bool) -> list[tuple[str, str]]:
    """Split text at heading lines into (header, body) pairs; body keeps its heading line."""
    sections: list[tuple[str, str]] = []
    header, buf, in_fence = "", [], False
    for line in text.splitlines(keepends=True):
        if md_fences and line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else heading.match(line)
        if match:
            if "".join(buf).strip():
                sections.append((header, "".join(buf)))
            header, buf = match.group(1).strip(), []
            if heading is _PY_DEF:
                header = line.split("(")[0].split(":")[0].strip()  # "def foo" / "class Bar"
        buf.append(line)
    if "".join(buf).strip():
        sections.append((header, "".join(buf)))
    return sections


class DocumentIngestor:
    """Load supported files and turn them into metadata-rich chunks.

    chunk_size and chunk_overlap are measured in characters.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Need chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_sections(self, path: Path) -> list[tuple[str, str]]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return [(f"page {i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".md":
            return _split_sections(text, _MD_HEADING, md_fences=True)
        if suffix == ".py":
            return _split_sections(text, _PY_DEF, md_fences=False)
        if suffix == ".txt":
            return [("", text)]
        raise ValueError(f"Unsupported file type '{suffix}' for {path}")

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        path = Path(path).resolve()
        try:  # cwd-relative names are unique across ingest roots and readable in citations
            source = path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            source = path.as_posix()
        chunks: list[Chunk] = []
        for header, body in self.load_sections(path):
            for text in split_text(body, self.chunk_size, self.chunk_overlap):
                chunks.append(Chunk(text, source, len(chunks), header))
        logger.info("chunked file=%s sections->chunks=%d", source, len(chunks))
        return chunks

    def ingest_paths(self, paths: list[str | Path]) -> list[Chunk]:
        """Chunk files and directories (recursively). Hidden paths and unsupported types are skipped."""
        chunks: list[Chunk] = []
        for p in map(Path, paths):
            if not p.exists():
                raise FileNotFoundError(p)
            if p.is_file():
                chunks.extend(self.chunk_file(p))
                continue
            for f in sorted(p.rglob("*")):
                rel = f.relative_to(p)
                if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES and not any(
                    part.startswith(".") for part in rel.parts
                ):
                    chunks.extend(self.chunk_file(f))
        return chunks
