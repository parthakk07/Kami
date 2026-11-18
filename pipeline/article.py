"""Web articles (blog posts, essays, docs pages) as input.

extract_article() turns a fetched HTML page into a list of clean text blocks
(headings, paragraphs, list items, quotes, code) plus metadata. build_document()
turns those blocks into the same Document the PDF path produces, so the analysis
pipeline, span matching and reading plan run unchanged. Each block becomes one
"line"; words get placeholder boxes because the reader renders HTML text, not images.
"""
from __future__ import annotations

import re

from .parse import REFS, Document, Section, Token, Word

MIN_WORDS = 120
CHUNK_TOKENS = 900  # target size of a part when an article has no headings
EXTRA_REFS = {"citation", "citations", "cite this", "further reading", "footnotes", "notes", "sources", "links"}
WS = re.compile(r"\s+")


class ArticleError(ValueError):
    pass


def _clean(t: str) -> str:
    return WS.sub(" ", t or "").strip()


def _decode(html: bytes | str) -> str:
    if isinstance(html, str):
        return html
    m = re.search(rb'charset=["\']?([\w\-]+)', html[:4000])
    for enc in ([m.group(1).decode()] if m else []) + ["utf-8", "cp1252"]:
        try:
            return html.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return html.decode("utf-8", "replace")


# ---------- trafilatura XML -> blocks ----------

def _own_text(el, skip=("list",)) -> str:
    """Text of an element without nested block children such as sub-lists."""
    parts = [el.text or ""]
    for ch in el:
        if ch.tag not in skip:
            parts.append("".join(ch.itertext()))
        parts.append(ch.tail or "")
    return _clean("".join(parts))


def _paragraphs(el) -> list[str]:
    """Split a <p> on line breaks: many sites (Paul Graham's essays, old blogs) use <br><br> for paragraphs."""
    parts = [el.text or ""]
    for ch in el:
        parts.append("\n" if ch.tag == "lb" else "".join(ch.itertext()))
        parts.append(ch.tail or "")
    raw = "".join(parts)
    chunks = re.split(r"\n\s*\n", raw)
    if len(chunks) == 1 and len(raw.split()) > 400:
        chunks = raw.split("\n")
    return [t for t in (_clean(c) for c in chunks) if t]


def _walk_xml(el, out: list[dict], depth: int = 0, quote: bool = False):
    for ch in el:
        tag = ch.tag
        if tag == "head":
            rend = ch.get("rend") or "h2"
            lvl = int(rend[1]) if re.fullmatch(r"h[1-6]", rend) else 2
            t = _clean("".join(ch.itertext()))
            if t:
                out.append({"tag": "h", "level": lvl, "text": t})
        elif tag == "p":
            for t in _paragraphs(ch):
                out.append({"tag": "quote" if quote else "p", "text": t})
        elif tag == "list":
            for item in ch:
                if item.tag != "item":
                    continue
                t = _own_text(item)
                if t:
                    out.append({"tag": "li", "depth": depth, "text": t, "ordered": (ch.get("rend") == "ol")})
                for sub in item:
                    if sub.tag == "list":
                        _walk_xml(_wrap(sub), out, depth + 1, quote)
        elif tag == "quote":
            if any(c.tag in ("p", "list") for c in ch):
                _walk_xml(ch, out, depth, True)
            else:
                t = _clean("".join(ch.itertext()))
                if t:
                    out.append({"tag": "quote", "text": t})
        elif tag == "code":
            t = "".join(ch.itertext()).strip("\n")
            if t.strip():
                out.append({"tag": "pre", "text": t})
        elif tag == "table":
            if ch.find(".//cell/p") is not None or ch.find(".//cell/head") is not None or ch.find(".//cell/list") is not None:
                for cell in ch.iter("cell"):  # a layout table (old-style sites): read the cells as page content
                    _walk_xml(cell, out, depth, quote)
                continue
            for row in ch.iter("row"):
                cells = [_clean("".join(c.itertext())) for c in row if c.tag == "cell"]
                cells = [c for c in cells if c]
                if cells:
                    out.append({"tag": "row", "text": "  |  ".join(cells)})
        elif tag in ("div", "main", "body", "section"):
            _walk_xml(ch, out, depth, quote)
        # graphic, lb, ref at block level: ignored


def _wrap(lst):
    """Present a nested <list> as a container so _walk_xml handles it."""
    from lxml import etree
    box = etree.Element("div")
    box.append(_copy(lst))
    return box


def _copy(el):
    from copy import deepcopy
    return deepcopy(el)


def _from_trafilatura(html: str, url: str) -> dict | None:
    import trafilatura
    from lxml import etree

    xml = trafilatura.extract(html, url=url or None, output_format="xml", include_tables=True,
                              include_formatting=False, include_links=False, include_images=False,
                              include_comments=False, favor_recall=True, with_metadata=True)
    if not xml:
        return None
    root = etree.fromstring(xml.encode())
    main = root.find("main")
    blocks: list[dict] = []
    _walk_xml(main if main is not None else root, blocks)
    return {"title": _clean(root.get("title", "")), "author": _clean(root.get("author", "")),
            "date": root.get("date", "") or "", "site": _clean(root.get("sitename", "") or root.get("hostname", "")),
            "description": "", "blocks": blocks}


# ---------- readability fallback ----------

def _from_readability(html: str) -> dict | None:
    from lxml import html as lh
    from readability import Document as RDoc

    rd = RDoc(html)
    summary = rd.summary(html_partial=True)
    root = lh.fromstring(summary)
    blocks: list[dict] = []
    stop = {"p", "pre", "h1", "h2", "h3", "h4", "h5", "h6"}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if any(a.tag in stop for a in el.iterancestors()):
            continue
        tag = el.tag
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            t = _clean(el.text_content())
            if t:
                blocks.append({"tag": "h", "level": int(tag[1]), "text": t})
        elif tag == "p":
            t = _clean(el.text_content())
            if t:
                quote = any(a.tag == "blockquote" for a in el.iterancestors())
                blocks.append({"tag": "quote" if quote else "p", "text": t})
        elif tag == "pre":
            t = el.text_content().strip("\n")
            if t.strip():
                blocks.append({"tag": "pre", "text": t})
        elif tag == "li":
            depth = sum(1 for a in el.iterancestors() if a.tag in ("ul", "ol")) - 1
            t = _own_text(el, skip=("ul", "ol", "p", "pre"))
            if t:
                blocks.append({"tag": "li", "depth": max(0, depth), "text": t, "ordered": el.getparent().tag == "ol"})
    return {"title": _clean(rd.short_title() or rd.title()), "author": "", "date": "", "site": "",
            "description": "", "blocks": blocks}


def _n_words(blocks: list[dict]) -> int:
    return sum(len(b["text"].split()) for b in blocks if b["tag"] != "pre")


def extract_article(html: bytes | str, url: str = "") -> dict:
    """Main article text of an HTML page as blocks. Raises ArticleError if there is no readable text."""
    text = _decode(html)
    art = None
    try:
        art = _from_trafilatura(text, url)
    except Exception:  # pragma: no cover - parser edge cases fall through to readability
        art = None
    if art is None or _n_words(art["blocks"]) < MIN_WORDS:
        try:
            alt = _from_readability(text)
        except Exception:
            alt = None
        if alt and (art is None or _n_words(alt["blocks"]) > _n_words(art["blocks"])):
            if art:  # keep the better metadata
                alt.update({k: art[k] for k in ("title", "author", "date", "site") if art.get(k)})
            art = alt
    if art is None or _n_words(art["blocks"]) < MIN_WORDS:
        raise ArticleError(
            "Could not find readable article text on that page. It may be paywalled, need a sign-in, "
            "or be built with JavaScript. Try the PDF version, or save the page as PDF and upload it.")
    blocks = art["blocks"]
    if not art["title"]:
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        art["title"] = _clean(m.group(1)) if m else ""
    # drop a leading heading that only repeats the title
    if blocks and blocks[0]["tag"] == "h" and _clean(blocks[0]["text"]).lower() == art["title"].lower():
        blocks.pop(0)
    # normalise heading levels so the top level used becomes 1
    levels = [b["level"] for b in blocks if b["tag"] == "h"]
    if levels:
        top = min(levels)
        for b in blocks:
            if b["tag"] == "h":
                b["level"] = min(4, b["level"] - top + 1)
    art["url"] = url
    art["n_words"] = _n_words(blocks)
    return art


def build_document(art: dict) -> Document:
    doc = Document(kind="html", title=art.get("title", "")[:200])
    heads: list[tuple[int, str, int, int]] = []  # (token_start, title, level, block)
    for bi, b in enumerate(art["blocks"]):
        doc.lines.append({"page": 0, "bbox": (0.0, float(bi), 1.0, bi + 1.0), "text": b["text"]})
        start = len(doc.tokens)
        if b["tag"] != "pre":  # code is shown but not analysed
            for w in b["text"].split():
                doc.words.append(Word(w, 0, (0.0, float(bi), 1.0, bi + 1.0), bi))
                doc.tokens.append(Token(w, [len(doc.words) - 1]))
        b["start"], b["end"] = start, len(doc.tokens)
        if b["end"] > start:
            doc.breaks.add(start)
        if b["tag"] == "h" and b["end"] > b["start"]:
            heads.append((start, b["text"], b["level"], bi))

    n = len(doc.tokens)
    sections: list[Section] = []
    first = heads[0][0] if heads else n
    if first > 0:
        sections.append(Section("S0", "Opening", 1, 0, first, None, "body"))
    in_refs = False
    for k, (tstart, title, level, bi) in enumerate(heads):
        tend = heads[k + 1][0] if k + 1 < len(heads) else n
        low = re.sub(r"^[\dA-Z]+(\.\d+)*\.?\s+", "", title).lower().strip(" :")
        kind = "body"
        if low in REFS or low in EXTRA_REFS:
            in_refs, kind = True, "references"
        elif in_refs or low.startswith("appendix"):
            kind = "appendix"
        sections.append(Section(f"S{len(sections)}", title, level, tstart, tend, bi, kind))
    if not heads and n > CHUNK_TOKENS * 1.5:
        sections = _chunk(art["blocks"], n)
    if not sections:
        sections.append(Section("S0", "Article", 1, 0, n, None, "body"))
    doc.sections = sections
    return doc


def _chunk(blocks: list[dict], n: int) -> list[Section]:
    """Long essays without headings: cut into parts at paragraph boundaries so every part gets analysed."""
    parts = max(2, round(n / CHUNK_TOKENS))
    target = n / parts
    cuts = [0]
    for b in blocks:
        if b["end"] > b["start"] and b["start"] - cuts[-1] >= target * 0.8 and n - b["start"] >= target * 0.5:
            cuts.append(b["start"])
    cuts.append(n)
    total = len(cuts) - 1
    return [Section(f"S{i}", f"Part {i + 1} of {total}", 1, a, b, None, "body", synthetic=True)
            for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]

