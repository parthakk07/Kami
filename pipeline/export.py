"""Download notes (md/txt) or a highlighted PDF of the actual paper."""
from __future__ import annotations

import html as htmlmod
import io
import re
from pathlib import Path

import pymupdf

HL_COLORS = {
    "claim": (0.89, 0.62, 0.12),
    "key": (0.96, 0.85, 0.15),
    "complex": (0.52, 0.41, 0.77),
    "definition": (0.13, 0.57, 0.53),
}
HL_CSS = {
    "claim": "#f0c45a",
    "key": "#f6de46",
    "complex": "#c4b3e8",
    "definition": "#8fd4cc",
}
PRIORITY = {"claim": 0, "complex": 1, "key": 2, "definition": 3}


def slug(title: str) -> str:
    s = re.sub(r"[^\w]+", "-", (title or "kami").strip(), flags=re.U).strip("-")
    return (s[:60] or "kami")


def notes_markdown(doc: dict) -> str:
    lines = [f"# {doc.get('title') or 'Untitled'}", ""]
    src = doc.get("source") or (doc.get("article") or {}).get("url")
    if src:
        lines += [f"Source: {src}", ""]
    if doc.get("kind") == "html" and doc.get("tokens"):
        lines += ["## Article", "", _article_plain(doc), ""]
    claims = doc.get("claims") or []
    if claims:
        lines.append("## Claims")
        for c in claims:
            lines.append(f"- **{c.get('id', '')}** {c.get('claim', '').strip()}")
            if c.get("quote"):
                lines.append(f"  > {c['quote'].strip()}")
        lines.append("")
    plan = doc.get("plan") or {}
    order = plan.get("order") or []
    sec = {s["id"]: s for s in doc.get("sections") or []}
    if order:
        lines.append("## Reading plan")
        if plan.get("summary"):
            lines.append(plan["summary"].strip())
            lines.append("")
        for o in order:
            s = sec.get(o["section_id"]) or {}
            title = s.get("title") or o["section_id"]
            why = o.get("why") or ""
            lines.append(f"- **{o.get('bucket', '').upper()}** {title}" + (f" — {why}" if why else ""))
        lines.append("")
    groups = {"claim": "Key claims", "key": "Important", "complex": "Complex", "definition": "Definitions"}
    by = {k: [] for k in groups}
    for h in doc.get("highlights") or []:
        by.setdefault(h.get("type"), []).append(h)
    if any(by[k] for k in groups):
        lines.append("## Highlights")
        for typ, label in groups.items():
            if not by[typ]:
                continue
            lines.append(f"### {label}")
            for h in by[typ]:
                note = (h.get("note") or "").strip()
                quote = (h.get("quote") or "").strip()
                if note:
                    lines.append(f"- {note}")
                if quote:
                    lines.append(f"  > {quote}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def notes_text(doc: dict) -> str:
    md = notes_markdown(doc)
    md = re.sub(r"^#+\s*", "", md, flags=re.M)
    md = md.replace("**", "").replace("> ", "")
    return md


def _article_plain(doc: dict) -> str:
    return " ".join(doc.get("tokens") or [])


def _stamp_highlights(pdf: pymupdf.Document, highlights: list) -> None:
    for h in highlights or []:
        color = HL_COLORS.get(h.get("type"), (0.96, 0.85, 0.15))
        for r in h.get("rects") or []:
            pno = int(r.get("page") or 0)
            if not 0 <= pno < pdf.page_count:
                continue
            page = pdf[pno]
            rect = pymupdf.Rect(float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
            # Bake a tint into the page so highlights show even if a viewer hides annotations.
            page.draw_rect(rect, color=None, fill=color, fill_opacity=0.32, overlay=True, width=0)
            try:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()
            except Exception:
                pass


def highlighted_pdf(source: Path | None, kind: str, highlights: list, notes_doc: dict) -> bytes:
    if kind == "pdf" and source and source.exists():
        pdf = pymupdf.open(source)
        _stamp_highlights(pdf, highlights)
        out = pdf.tobytes(deflate=True)
        pdf.close()
        return out
    if kind == "image" and source and source.exists():
        img = pymupdf.open(source)
        pix = img[0].get_pixmap()
        pdf = pymupdf.open()
        page = pdf.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, pixmap=pix)
        img.close()
        _stamp_highlights(pdf, highlights)
        out = pdf.tobytes(deflate=True)
        pdf.close()
        return out
    return _article_pdf(notes_doc)


def _article_pdf(doc: dict) -> bytes:
    """Full article text as a PDF, with Kami highlights marked in the body."""
    tokens = doc.get("tokens") or []
    blocks = doc.get("blocks") or []
    if not tokens or not blocks:
        return _notes_pdf(doc)
    owner = [None] * len(tokens)
    for h in sorted(doc.get("highlights") or [], key=lambda x: PRIORITY.get(x.get("type"), 9)):
        rng = h.get("token_range") or [0, 0]
        a, b = int(rng[0]), int(rng[1])
        for i in range(max(0, a), min(b, len(tokens))):
            if owner[i] is None:
                owner[i] = h.get("type")

    def wrap(a: int, b: int) -> str:
        out, i = [], a
        while i < b:
            t = owner[i]
            j = i + 1
            while j < b and owner[j] == t:
                j += 1
            piece = htmlmod.escape(" ".join(tokens[i:j]))
            if t and t in HL_CSS:
                piece = f'<mark style="background:{HL_CSS[t]}">{piece}</mark>'
            out.append(piece)
            i = j
        return " ".join(out)

    body = [f"<h1>{htmlmod.escape(doc.get('title') or 'Untitled')}</h1>"]
    src = doc.get("source") or (doc.get("article") or {}).get("url")
    if src:
        body.append(f'<p style="color:#666;font-size:10pt">{htmlmod.escape(str(src))}</p>')
    for b in blocks:
        tag = b.get("tag")
        a, c = int(b.get("start") or 0), int(b.get("end") or 0)
        if tag == "pre":
            body.append(f"<pre>{htmlmod.escape(b.get('text') or '')}</pre>")
            continue
        if c <= a:
            continue
        inner = wrap(a, c)
        if tag == "h":
            lvl = min(3, max(1, int(b.get("level") or 1)))
            body.append(f"<h{lvl+1}>{inner}</h{lvl+1}>")
        elif tag == "li":
            body.append(f"<p>• {inner}</p>")
        elif tag == "quote":
            body.append(f"<blockquote>{inner}</blockquote>")
        else:
            body.append(f"<p>{inner}</p>")

    css = (
        "body{font-family:serif;font-size:11pt;line-height:1.45;color:#1f1d1a}"
        "h1{font-size:18pt;margin:0 0 12pt}"
        "h2{font-size:14pt;margin:16pt 0 8pt}"
        "h3,h4{font-size:12pt;margin:12pt 0 6pt}"
        "p{margin:0 0 8pt}"
        "blockquote{margin:0 0 8pt 12pt;color:#555}"
        "pre{font-family:monospace;font-size:9pt;white-space:pre-wrap}"
        "mark{color:inherit}"
    )
    story = pymupdf.Story("".join(body), user_css=css)
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (50, 50, -50, -50)
    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return buf.getvalue()


def _notes_pdf(doc: dict) -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    margin = 54
    y = 64
    width = page.rect.width - 2 * margin

    def line(text: str, fontsize: float = 11):
        nonlocal page, y
        text = text.replace("\t", " ")
        max_chars = max(20, int(width / max(fontsize * 0.5, 1)))
        while True:
            if y > page.rect.height - 56:
                page = pdf.new_page()
                y = 64
            chunk, rest = text[:max_chars], text[max_chars:]
            if rest and " " in chunk:
                cut = chunk.rfind(" ")
                chunk, rest = chunk[:cut], chunk[cut + 1:] + rest
            page.insert_text((margin, y), chunk, fontsize=fontsize, fontname="helv")
            y += fontsize + 5
            if not rest:
                break
            text = rest

    line(doc.get("title") or "Kami", 16)
    y += 10
    for raw in notes_text(doc).splitlines():
        line(raw, 10)
        if not raw.strip():
            y += 4
    buf = io.BytesIO()
    pdf.save(buf)
    pdf.close()
    return buf.getvalue()
