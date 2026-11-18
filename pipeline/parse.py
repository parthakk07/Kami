"""Turn a PDF or an image of a page into a positioned word list and sections.

Everything downstream works on one shared representation:
  - pages:   size of each page in its own coordinate space (PDF points or image pixels)
  - words:   every word with its page, bounding box and visual line id
  - tokens:  words after joining line-end hyphenation ("atten-" + "tion"), each token
             remembers which words it covers. Tokens are what the LLM sees and what
             quotes are matched against.
  - sections: heading-delimited token ranges
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import pymupdf


@dataclass
class Word:
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    line: int  # global visual line id


@dataclass
class Token:
    text: str
    words: list[int]


@dataclass
class Section:
    id: str
    title: str
    level: int
    start: int  # token index, inclusive
    end: int  # token index, exclusive
    heading_line: int | None = None  # visual line id of the heading
    kind: str = "body"  # body | front | references | appendix
    synthetic: bool = False  # a part cut from a long article without headings

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "kind": self.kind,
        }
        if self.synthetic:
            d["synthetic"] = True
        return d


@dataclass
class Document:
    kind: str  # pdf | image
    pages: list[dict] = field(default_factory=list)  # {"width", "height"}
    words: list[Word] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)  # {"page", "bbox", "text"}
    tokens: list[Token] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    title: str = ""
    breaks: set[int] = field(default_factory=set)  # token indices that start a new block (web articles)

    def text(self, start: int, end: int) -> str:
        if not self.breaks:
            return " ".join(t.text for t in self.tokens[start:end])
        out = []
        for i in range(start, min(end, len(self.tokens))):
            if out:
                out.append("\n" if i in self.breaks else " ")
            out.append(self.tokens[i].text)
        return "".join(out)

    def section_text(self, s: Section) -> str:
        return self.text(s.start, s.end)


NUMBERED = re.compile(r"^(?:[A-Z]|\d+)(?:\.\d+){0,3}\.?\s+[A-Z][\w\-]")
NUMBERED_OCR = re.compile(r"^\d+(?:\.\d+){0,3}\.?\s*[A-Z][a-z][\w\-]")
KNOWN = {
    "abstract", "introduction", "background", "related work", "conclusion",
    "conclusions", "discussion", "references", "bibliography", "acknowledgements",
    "acknowledgments", "appendix", "methods", "method", "results", "experiments",
    "evaluation", "limitations", "future work", "summary",
}
REFS = {"references", "bibliography"}


def _is_bold(span: dict) -> bool:
    f = span.get("font", "")
    return bool(span.get("flags", 0) & 16) or any(k in f for k in ("Bold", "Medi", "CMBX", "Semibold", "Black", "Heavy"))


def _build_tokens(doc: Document) -> None:
    words = doc.words
    tokens: list[Token] = []
    i = 0
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else None
        if (
            nxt is not None
            and w.text.endswith("-")
            and len(w.text) > 2
            and nxt.line != w.line
            and nxt.text[:1].islower()
        ):
            keep = w.text[:1].isupper() or "-" in nxt.text or "-" in w.text[:-1]
            tokens.append(Token((w.text if keep else w.text[:-1]) + nxt.text, [i, i + 1]))
            i += 2
            continue
        tokens.append(Token(w.text, [i]))
        i += 1
    doc.tokens = tokens


def _clean_title(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _build_sections(doc: Document, heading_lines: list[tuple[int, str, int]]) -> None:
    """heading_lines: (line_id, title, level), in reading order."""
    line_first_token: dict[int, int] = {}
    for ti, tok in enumerate(doc.tokens):
        ln = doc.words[tok.words[0]].line
        line_first_token.setdefault(ln, ti)

    starts = []
    for line_id, title, level in heading_lines:
        if line_id in line_first_token:
            starts.append((line_first_token[line_id], line_id, title, level))
    starts.sort()

    sections: list[Section] = []
    n = len(doc.tokens)
    first = starts[0][0] if starts else n
    if first > 0:
        sections.append(Section("S0", "Title and authors", 0, 0, first, None, "front"))
    in_refs = False
    for k, (tstart, line_id, title, level) in enumerate(starts):
        tend = starts[k + 1][0] if k + 1 < len(starts) else n
        if tend <= tstart:
            continue
        low = re.sub(r"^[\dA-Z]+(\.\d+)*\.?\s+", "", title).lower() if NUMBERED.match(title) else title.lower()
        kind = "body"
        if low in REFS:
            in_refs = True
            kind = "references"
        elif in_refs or low.startswith("appendix"):
            kind = "appendix" if not low in REFS else "references"
        sections.append(Section(f"S{len(sections)}", title, level, tstart, tend, line_id, kind))
    if not sections:
        sections.append(Section("S0", "Document", 1, 0, n, None, "body"))
    doc.sections = sections


def parse_pdf(data: bytes) -> Document:
    pdf = pymupdf.open(stream=data, filetype="pdf")
    doc = Document(kind="pdf")
    sizes: list[float] = []
    raw_lines: list[dict] = []  # visual lines before heading detection

    for pno, page in enumerate(pdf):
        doc.pages.append({"width": page.rect.width, "height": page.rect.height})
        rd = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT)
        for block in rd["blocks"]:
            if block.get("type") != 0:
                continue
            # merge PyMuPDF lines of a block that sit on the same baseline
            vlines: list[list[dict]] = []
            for line in block["lines"]:
                if line["dir"][0] < 0.9:  # skip rotated text (arXiv side stamp etc.)
                    continue
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if vlines:
                    prev = vlines[-1]
                    py0 = min(l["bbox"][1] for l in prev)
                    py1 = max(l["bbox"][3] for l in prev)
                    overlap = min(y1, py1) - max(y0, py0)
                    if overlap > 0.5 * min(y1 - y0, py1 - py0):
                        prev.append(line)
                        continue
                vlines.append([line])
            for vl in vlines:
                vl.sort(key=lambda l: l["bbox"][0])
                spans = [s for l in vl for s in l["spans"]]
                chars = [(c, s) for s in spans for c in s["chars"]]
                if not chars:
                    continue
                line_id = len(raw_lines)
                # words from chars
                cur, cur_box, line_words = "", None, []
                for c, s in chars:
                    ch = c["c"]
                    if ch.isspace():
                        if cur:
                            line_words.append((cur, cur_box))
                        cur, cur_box = "", None
                        continue
                    b = c["bbox"]
                    if cur and cur_box and b[0] - cur_box[2] > 0.25 * s["size"]:
                        line_words.append((cur, cur_box))
                        cur, cur_box = "", None
                    cur += ch
                    cur_box = b if cur_box is None else (
                        min(cur_box[0], b[0]), min(cur_box[1], b[1]), max(cur_box[2], b[2]), max(cur_box[3], b[3]))
                if cur:
                    line_words.append((cur, cur_box))
                if not line_words:
                    continue
                text = " ".join(t for t, _ in line_words)
                for s in spans:
                    s["text"] = "".join(c["c"] for c in s["chars"])
                tsizes = [s["size"] for s in spans if s["text"].strip()]
                bold_chars = sum(len(s["text"].strip()) for s in spans if _is_bold(s))
                total_chars = sum(len(s["text"].strip()) for s in spans) or 1
                lb = (min(w[1][0] for w in line_words), min(w[1][1] for w in line_words),
                      max(w[1][2] for w in line_words), max(w[1][3] for w in line_words))
                raw_lines.append({
                    "page": pno, "bbox": lb, "text": text,
                    "size": statistics.median(tsizes) if tsizes else 0,
                    "bold": bold_chars / total_chars > 0.8,
                    "words": line_words, "page_h": page.rect.height,
                })
                sizes.extend(tsizes)

    doc.lines = [{"page": l["page"], "bbox": l["bbox"], "text": l["text"]} for l in raw_lines]
    body = statistics.mode([round(s, 0) for s in sizes]) if sizes else 10

    # Words feed the analysis. Footnotes, page footers and bare page numbers are left out
    # (they stay visible on the page image, they just are not analyzed or highlighted).
    for lid, l in enumerate(raw_lines):
        t = l["text"].strip()
        footer = l["bbox"][1] > 0.72 * l["page_h"] and l["size"] <= body - 0.8
        pageno = bool(re.fullmatch(r"\d{1,3}", t)) and (l["bbox"][1] > 0.88 * l["page_h"] or l["bbox"][3] < 0.08 * l["page_h"])
        if footer or pageno:
            continue
        for wt, b in l["words"]:
            doc.words.append(Word(wt, l["page"], tuple(b), lid))

    # title = largest text on page 1
    p0 = [l for l in raw_lines if l["page"] == 0 and len(l["text"]) > 3]
    if p0:
        big = max(l["size"] for l in p0)
        doc.title = _clean_title(" ".join(l["text"] for l in p0 if l["size"] >= big - 0.5))[:200]

    headings: list[tuple[int, str, int]] = []
    for lid, l in enumerate(raw_lines):
        t = _clean_title(l["text"])
        if not (2 < len(t) < 90):
            continue
        if t.endswith(".") and not NUMBERED.match(t):
            continue
        strong = l["bold"] or l["size"] > body + 0.8
        if not strong or l["size"] < body - 1.5:
            continue
        if NUMBERED.match(t) and not re.search(r"\d{3,}|[=<>]", t.split(" ", 1)[1] if " " in t else ""):
            num = t.split()[0].rstrip(".")
            if num.isdigit() and int(num) > 30:
                continue
            level = num.count(".") + 1
            headings.append((lid, t, level))
        elif t.lower() in KNOWN or re.match(r"^appendix\b", t, re.I):
            headings.append((lid, t, 1))

    _build_tokens(doc)
    _build_sections(doc, headings)
    if not doc.title and doc.sections:
        doc.title = doc.section_text(doc.sections[0])[:120]
    return doc


def parse_image(data: bytes) -> Document:
    """OCR an image of a paper page. Uses RapidOCR (ONNX, pip-installable, no system deps)."""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    doc = Document(kind="image", pages=[{"width": w, "height": h}])
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Image input needs rapidocr_onnxruntime (pip install rapidocr_onnxruntime)") from e
    import numpy as np

    engine = _ocr_engine()
    result, _ = engine(np.array(img))
    result = result or []
    # each item: [box(4 points), text, score]; sort by reading order (column aware-ish)
    items = []
    for box, text, score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append(((min(xs), min(ys), max(xs), max(ys)), text))
    items = _ocr_reading_order(items, w)
    heights = [b[3] - b[1] for b, _ in items] or [10]
    body_h = statistics.median(heights)
    headings = []
    for lid, (b, text) in enumerate(items):
        parts = text.split()
        if not parts:
            continue
        total = sum(len(p) for p in parts) + len(parts) - 1
        x = b[0]
        span = b[2] - b[0]
        for p in parts:
            pw = span * len(p) / max(total, 1)
            doc.words.append(Word(p, 0, (x, b[1], x + pw, b[3]), lid))
            x += pw + span / max(total, 1)
        doc.lines.append({"page": 0, "bbox": b, "text": text})
        t = _clean_title(text)
        if 2 < len(t) < 80 and (NUMBERED_OCR.match(t) or t.lower() in KNOWN) and not t.endswith("."):
            t = re.sub(r"^(\d+(?:\.\d+)*\.?)\s*", r"\1 ", t) if NUMBERED_OCR.match(t) else t
            headings.append((lid, t, t.split()[0].count(".") + 1 if NUMBERED_OCR.match(t) else 1))
    _build_tokens(doc)
    _build_sections(doc, headings)
    if doc.sections and doc.sections[0].kind == "front":
        # an image is usually a page from the middle of a paper: analyze the leading text too
        doc.sections[0].kind = "body"
        doc.sections[0].title = "Page text" if headings else "Document"
    if not doc.title:
        doc.title = doc.text(0, min(12, len(doc.tokens)))
    return doc


def _ocr_reading_order(items, page_w):
    """Order OCR boxes: detect one vs two columns, then top to bottom, left to right."""
    if not items:
        return items
    mid = page_w / 2
    spanning = sum(1 for b, _ in items if b[0] < mid - 0.05 * page_w and b[2] > mid + 0.05 * page_w)
    right = sum(1 for b, _ in items if b[0] > mid - 0.02 * page_w)
    two_col = spanning < 0.15 * len(items) and right > 0.25 * len(items)

    def rows(boxes):
        boxes = sorted(boxes, key=lambda it: it[0][1])
        out, cur = [], []
        for it in boxes:
            if cur:
                cy0 = min(x[0][1] for x in cur); cy1 = max(x[0][3] for x in cur)
                ov = min(cy1, it[0][3]) - max(cy0, it[0][1])
                if ov > 0.5 * min(cy1 - cy0, it[0][3] - it[0][1]):
                    cur.append(it); continue
                out.append(sorted(cur, key=lambda x: x[0][0]))
            cur = [it]
        if cur:
            out.append(sorted(cur, key=lambda x: x[0][0]))
        return [x for r in out for x in r]

    if not two_col:
        return rows(items)
    left = [it for it in items if it[0][2] <= mid + 0.05 * page_w or it[0][0] < mid - 0.05 * page_w and it[0][2] - it[0][0] < 0.55 * page_w]
    rest = [it for it in items if it not in left]
    return rows(left) + rows(rest)


_OCR = None


def _ocr_engine():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR
