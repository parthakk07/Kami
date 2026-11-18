"""Cheap, LLM-free per-section signals."""
from __future__ import annotations

import re

from .parse import Document, Section

GREEK = re.compile(r"[\u0370-\u03FF]")
MATH = re.compile(r"[\u2200-\u22FF\u2190-\u21FF\u27C0-\u27EF\u2A00-\u2AFF\U0001D400-\U0001D7FF=+^_\u221A\u2211\u220F\u222B\u00B1\u00D7\u00B7]")
EQNUM = re.compile(r"\(\d{1,3}\)\s*$")
TERM_PATTERNS = [
    re.compile(r"\bwe (?:call|term|refer to|define|denote|name|dub)\b", re.I),
    re.compile(r"\b(?:called|termed|referred to as|known as|denoted (?:by|as)|is defined as|defined as)\b", re.I),
]
ACRONYM_DEF = re.compile(r"\(([A-Z][A-Za-z\-]*[A-Z][a-z]?s?)\)")
CITE_NUM = re.compile(r"\[(\d+(?:\s*[,\u2013\-]\s*\d+)*)\]")
CITE_AUTHOR = re.compile(r"\b[A-Z][A-Za-z\-]+ et al\.?,?\s*(?:\(?\d{4}\)?)?")


def section_lines(doc: Document, s: Section) -> list[str]:
    seen: dict[int, list[str]] = {}
    for tok in doc.tokens[s.start:s.end]:
        for wi in tok.words:
            w = doc.words[wi]
            seen.setdefault(w.line, []).append(w.text)
    return [" ".join(v) for v in seen.values()]


def count_citations(text: str) -> int:
    n = 0
    for m in CITE_NUM.finditer(text):
        n += len(re.split(r"[,]", m.group(1)))
    n += len(CITE_AUTHOR.findall(text))
    return n


def compute(doc: Document, s: Section) -> dict:
    text = doc.section_text(s)
    lines = section_lines(doc, s)
    eq_lines = 0
    for ln in lines:
        m = len(MATH.findall(ln)) + len(GREEK.findall(ln))
        letters = sum(c.isalpha() for c in ln) or 1
        if EQNUM.search(ln) and m >= 1:
            eq_lines += 1
        elif m >= 3 and m / letters > 0.08:
            eq_lines += 1
    new_terms = sum(len(p.findall(text)) for p in TERM_PATTERNS) + len(set(ACRONYM_DEF.findall(text)))
    n_words = max(1, s.end - s.start)
    cites = count_citations(text)
    return {
        "n_words": n_words,
        "n_lines": len(lines),
        "n_equations": eq_lines,
        "equation_density": round(eq_lines / max(1, len(lines)), 3),
        "n_new_terms": new_terms,
        "n_citations": cites,
        "citations_per_100_words": round(100 * cites / n_words, 2),
    }
