"""Chunking strategy: split markdown by heading structure.

Each top-level section (#) becomes one document; subsections (##, ###) are
kept together unless they exceed a character budget, in which case they are
split on paragraph boundaries.
"""
from __future__ import annotations

import re
from typing import List, Tuple


def chunk_markdown(text: str, max_chars: int = 2000) -> List[Tuple[str, str]]:
    """Split markdown into (heading_path, chunk_text) pairs.

    Args:
        text: markdown source.
        max_chars: soft budget per chunk; long sections are split by paragraph.
    """
    sections = _split_sections(text)
    chunks: List[Tuple[str, str]] = []
    for heading, body in sections:
        if len(body) <= max_chars:
            chunks.append((heading, body))
            continue
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        acc, acc_len = [], 0
        for para in paragraphs:
            if acc_len + len(para) > max_chars and acc:
                chunks.append((heading, "\n\n".join(acc)))
                acc, acc_len = [para], len(para)
            else:
                acc.append(para)
                acc_len += len(para) + 2
        if acc:
            chunks.append((heading, "\n\n".join(acc)))
    return chunks


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Group markdown into top-level sections; subtitles attach to their parent.

    `#` headings start a new section; `##`/`###` content (with its heading
    kept) is appended to the current top-level section so that a subtitle is
    not lost when the parent chunk is later split by budget.
    """
    matches = list(_HEADING_RE.finditer(text))
    sections: List[Tuple[str, str]] = []
    title: str | None = None
    body_parts: List[str] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg_body = text[start:end].strip()
        heading = f"{m.group(1)} {m.group(2)}"
        if level == 1:
            if title is not None and body_parts:
                sections.append((title, "\n\n".join(body_parts)))
            title = heading
            body_parts = [seg_body] if seg_body else []
        else:
            if title is None:
                title = heading
            if seg_body:
                body_parts.append(f"{heading}\n{seg_body}")
    if title is not None and body_parts:
        sections.append((title, "\n\n".join(body_parts)))
    return sections
