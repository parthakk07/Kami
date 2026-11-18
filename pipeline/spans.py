"""Map quoted text from the LLM back to exact token ranges and page rectangles."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from .parse import Document

_NON = re.compile(r"[^0-9a-z]+")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    t = t.replace("\ufb01", "fi").replace("\ufb02", "fl")
    return _NON.sub("", t)


class SpanIndex:
    def __init__(self, doc: Document):
        self.doc = doc
        self.keys: list[str] = []
        self.key_tok: list[int] = []
        for i, tok in enumerate(doc.tokens):
            k = norm(tok.text)
            if k:
                self.keys.append(k)
                self.key_tok.append(i)
        # char offsets for exact substring search
        self.joined = " ".join(self.keys)
        self.offsets = []
        pos = 0
        for k in self.keys:
            self.offsets.append(pos)
            pos += len(k) + 1

    def _key_range(self, start_tok: int, end_tok: int) -> tuple[int, int]:
        import bisect
        a = bisect.bisect_left(self.key_tok, start_tok)
        b = bisect.bisect_left(self.key_tok, end_tok)
        return a, b

    def _offset_to_key(self, off: int) -> int:
        import bisect
        return bisect.bisect_right(self.offsets, off) - 1

    def find(self, quote: str, start_tok: int = 0, end_tok: int | None = None) -> tuple[int, int, float] | None:
        """Return (tok_start, tok_end, confidence) or None."""
        if end_tok is None:
            end_tok = len(self.doc.tokens)
        q = [k for k in (norm(w) for w in quote.split()) if k]
        if len(q) < 2:
            return None
        a, b = self._key_range(start_tok, end_tok)
        hit = self._exact(q, a, b)
        if hit is None and (a, b) != (0, len(self.keys)):
            hit = self._exact(q, 0, len(self.keys))  # quote may come from a neighbouring section
        if hit is not None:
            return self.key_tok[hit[0]], self.key_tok[hit[1] - 1] + 1, 1.0
        return self._fuzzy(q, a, b)

    def _exact(self, q: list[str], a: int, b: int) -> tuple[int, int] | None:
        needle = " ".join(q)
        lo = self.offsets[a] if a < len(self.offsets) else len(self.joined)
        hi = (self.offsets[b] if b < len(self.offsets) else len(self.joined) + 1)
        idx = self.joined.find(needle, lo, hi)
        if idx < 0:
            return None
        k0 = self._offset_to_key(idx)
        k1 = self._offset_to_key(idx + len(needle) - 1) + 1
        return k0, k1

    def _fuzzy(self, q: list[str], a: int, b: int) -> tuple[int, int, float] | None:
        hay = self.keys[a:b]
        if not hay:
            return None
        # anchor on the first and last few words
        n = min(5, len(q))
        head = self._exact(q[:n], a, b)
        tail = self._exact(q[-n:], a, b)
        if head and tail and 0 < tail[1] - head[0] <= 2 * len(q) + 5:
            return self.key_tok[head[0]], self.key_tok[tail[1] - 1] + 1, 0.8
        sm = SequenceMatcher(None, hay, q, autojunk=False)
        blocks = [m for m in sm.get_matching_blocks() if m.size > 0]
        if not blocks:
            return None
        # densest window of length ~len(q)
        best = None
        for m in blocks:
            lo = m.a
            hi = lo + int(len(q) * 1.3) + 2
            score = sum(min(x.a + x.size, hi) - max(x.a, lo) for x in blocks if x.a < hi and x.a + x.size > lo)
            if best is None or score > best[0]:
                best = (score, lo, min(hi, len(hay)))
        score, lo, hi = best
        conf = score / len(q)
        if conf < 0.6:
            return None
        covered = [x for x in blocks if x.a < hi and x.a + x.size > lo]
        lo = min(x.a for x in covered)
        hi = max(x.a + x.size for x in covered)
        return self.key_tok[a + lo], self.key_tok[a + hi - 1] + 1, round(conf, 2)

    def rects(self, tok_start: int, tok_end: int, pad: float = 0.8) -> list[dict]:
        doc = self.doc
        lines: dict[int, list[float]] = {}
        page_of: dict[int, int] = {}
        for tok in doc.tokens[tok_start:tok_end]:
            for wi in tok.words:
                w = doc.words[wi]
                b = lines.get(w.line)
                if b is None:
                    lines[w.line] = list(w.bbox)
                    page_of[w.line] = w.page
                else:
                    b[0] = min(b[0], w.bbox[0]); b[1] = min(b[1], w.bbox[1])
                    b[2] = max(b[2], w.bbox[2]); b[3] = max(b[3], w.bbox[3])
        out = []
        for ln, b in lines.items():
            out.append({
                "page": page_of[ln],
                "x0": round(b[0] - pad, 2), "y0": round(b[1] - pad * 0.5, 2),
                "x1": round(b[2] + pad, 2), "y1": round(b[3] + pad * 0.5, 2),
            })
        return out

    def line_rect(self, line_id: int) -> dict | None:
        if line_id is None or line_id >= len(self.doc.lines):
            return None
        l = self.doc.lines[line_id]
        b = l["bbox"]
        return {"page": l["page"], "x0": b[0] - 2, "y0": b[1] - 1, "x1": b[2] + 2, "y1": b[3] + 1}
