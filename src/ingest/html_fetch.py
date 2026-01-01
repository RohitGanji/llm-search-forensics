from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

import httpx
from bs4 import BeautifulSoup, Tag

WHITESPACE_RE = re.compile(r"\s+")

@dataclass(frozen=True)
class Section:
    header_path: str
    text: str

def _clean_text(s: str) -> str:
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s

def fetch_html(url: str, timeout_s: float = 20.0) -> str:
    headers = {"User-Agent": "llm-search-forensics/0.1"}
    with httpx.Client(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.text

def html_to_sections(html: str) -> List[Section]:
    """
    Convert HTML into ordered sections keyed by heading path (h1/h2/h3).
    This is intentionally simple but effective for docs/help-center pages.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove obvious boilerplate
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    body = soup.body or soup
    headings = body.find_all(["h1", "h2", "h3"])
    if not headings:
        # fallback: all visible text as one section
        txt = _clean_text(body.get_text(" ", strip=True))
        return [Section(header_path="", text=txt)] if txt else []

    stack: List[Tuple[int, str]] = []
    sections: List[Section] = []
    current_lines: List[str] = []

    def path() -> str:
        return " > ".join([t for _, t in stack]) if stack else ""

    def flush():
        txt = _clean_text(" ".join(current_lines))
        if txt:
            sections.append(Section(header_path=path(), text=txt))

    # Walk the document in order, collecting text under headings
    for node in body.descendants:
        if not isinstance(node, Tag):
            continue

        if node.name in ["h1", "h2", "h3"]:
            flush()
            current_lines.clear()
            level = int(node.name[1])
            title = _clean_text(node.get_text(" ", strip=True))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue

        # capture useful text blocks
        if node.name in ["p", "li", "pre", "code"]:
            txt = _clean_text(node.get_text(" ", strip=True))
            if txt:
                current_lines.append(txt)

    flush()

    # drop tiny garbage sections
    sections = [s for s in sections if len(s.text) >= 40]
    return sections
