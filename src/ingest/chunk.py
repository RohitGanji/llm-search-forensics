from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    header_path: str
    source_type: str  # "html"

def chunk_text(text: str, max_chars: int = 2400, overlap_chars: int = 300) -> List[Tuple[int, int, str]]:
    t = text.strip()
    if not t:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")
    out: List[Tuple[int, int, str]] = []
    n = len(t)
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        out.append((start, end, t[start:end]))
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return out
