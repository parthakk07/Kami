"""Paper Highlighter: FastAPI backend.

Run:  .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from urllib.parse import quote
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pipeline import analyze
from pipeline.llm import LLMError, get_llm, provider_status
from pipeline.article import ArticleError, build_document, extract_article
from pipeline.parse import parse_image, parse_pdf
from pipeline import export as ex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("paper")

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PAPER_DATA_DIR", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 60 * 1024 * 1024
PAGE_ZOOM = 2.0
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif", "image/bmp": "bmp", "image/tiff": "tiff"}

app = FastAPI(title="Kami")
JOBS: dict[str, dict] = {}


def _sniff(data: bytes, hint: str = "") -> str:
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    head = data[:1024].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html", b"<?xml")) and b"<html" in data[:4096].lower() or "html" in hint:
        return "html"
    return IMAGE_TYPES.get(hint, "")


def _normalize_url(url: str) -> str:
    """Accept bare arXiv ids and arXiv abs/html pages; always fetch the arXiv PDF."""
    url = url.strip()
    m = re.match(r"^(?:https?://)?(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/([\w.\-/]+?)(?:\.pdf)?/?(?:[?#].*)?$", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    if re.match(r"^(?:arxiv:)?\d{4}\.\d{4,5}(v\d+)?$", url, re.I):
        return f"https://arxiv.org/pdf/{url.split(':')[-1]}"
    m = re.match(r"^(?:https?://)?openreview\.net/forum\?id=([\w\-]+)", url)
    if m:
        return f"https://openreview.net/pdf?id={m.group(1)}"
    if not re.match(r"^https?://", url):
        url = "https://" + url
    return url


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}
FIREFOX_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0"
CITATION_PDF = [
    re.compile(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']', re.I),
]


def _pdf_guesses(url: str) -> list[str]:
    """Other URLs to try when the HTML page is blocked."""
    out = []
    u = url.rstrip("/")
    if not u.lower().endswith(".pdf"):
        out.append(u + ".pdf")
        out.append(re.sub(r"/html/", "/pdf/", u, flags=re.I))
        out.append(re.sub(r"/abs/", "/pdf/", u, flags=re.I))
        out.append(re.sub(r"/abstract", "/pdf", u, flags=re.I))
        out.append(re.sub(r"/full(?:text)?", "/pdf", u, flags=re.I))
    # unique, still different from the original
    seen = {url, u}
    return [g for g in out if g not in seen and not (g in seen or seen.add(g))]


def _http_error(code: int) -> HTTPException:
    if code in (401, 402, 403):
        msg = (f"The site refused to share this page (HTTP {code}). It is probably paywalled, needs a sign-in, "
               "or blocks automated readers. Try the PDF, or save the page as PDF and upload it.")
    elif code == 404:
        msg = "That page was not found (HTTP 404). Check the link."
    elif code == 429:
        msg = "The site is rate limiting requests (HTTP 429). Wait a minute and try again."
    else:
        msg = f"Could not download that link (HTTP {code})."
    return HTTPException(400, msg)


async def _fetch(url: str) -> tuple[bytes, str, str]:
    """Download a link. Returns (data, content_type, final_url)."""
    url = _normalize_url(url)
    parsed = httpx.URL(url)
    headers = dict(BROWSER_HEADERS)
    headers["Referer"] = f"{parsed.scheme}://{parsed.host}/"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=45, headers=headers) as c:
            r = await c.get(url)
            if r.status_code in (401, 403):
                r = await c.get(url, headers={**headers, "User-Agent": FIREFOX_UA})
            if r.status_code >= 400:
                for alt in _pdf_guesses(str(parsed)):
                    try:
                        r2 = await c.get(alt)
                    except httpx.HTTPError:
                        continue
                    if r2.status_code < 400 and (_sniff(r2.content) == "pdf" or "pdf" in (r2.headers.get("content-type") or "").lower()):
                        return r2.content, "application/pdf", str(r2.url)
                raise _http_error(r.status_code)
            data = r.content
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            kind = _sniff(data, ctype)
            if kind and kind != "html":
                return data, ctype, str(r.url)
            if kind == "html" or "html" in ctype or "xml" in ctype:
                html = r.text
                for rx in CITATION_PDF:
                    m = rx.search(html)
                    if m:
                        try:
                            r2 = await c.get(httpx.URL(str(r.url)).join(m.group(1)))
                            if r2.status_code < 400 and _sniff(r2.content) == "pdf":
                                return r2.content, "application/pdf", str(r2.url)
                        except httpx.HTTPError:
                            pass
                        break
                return data, "text/html", str(r.url)
            raise HTTPException(400, "That link is not a PDF, an image, or a web page.")
    except httpx.TimeoutException:
        raise HTTPException(400, "The site took too long to respond. Try again, or download the file and upload it.")
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Could not reach that link ({e.__class__.__name__}). Check the address.")


def _doc_dir(doc_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16}", doc_id):
        raise HTTPException(404, "unknown document")
    return DATA / doc_id


def _result_path(doc_id: str, llm) -> Path:
    tag = re.sub(r"[^\w.\-]", "_", f"{llm.name}-{llm.model}")
    return _doc_dir(doc_id) / f"result-{tag}.json"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _find_result(doc_id: str) -> Path | None:
    """Latest analysis on disk. Survives process restart and provider changes."""
    d = DATA / doc_id
    if not d.is_dir():
        return None
    try:
        tagged = _result_path(doc_id, get_llm())
        data = _read_json(tagged) if tagged.exists() else None
        if data and not data.get("partial"):
            return tagged
    except LLMError:
        tagged = None
    complete, partial = [], []
    for p in d.glob("result-*.json"):
        data = _read_json(p)
        if not data:
            continue
        (partial if data.get("partial") else complete).append(p)
    pool = complete or partial
    if not pool:
        return None
    pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pool[0]


def _parse(kind: str, data: bytes):
    if kind == "pdf":
        return parse_pdf(data), None
    if kind == "html":
        art = json.loads(data)
        return build_document(art), art  # build_document adds token ranges to art["blocks"]
    return parse_image(data), None


def _decorate_result(result: dict, doc_id: str, source: str, kind: str, art, tokens, elapsed_s=None) -> dict:
    result = dict(result)
    result.update(doc_id=doc_id, source=source)
    if elapsed_s is not None:
        result["elapsed_s"] = elapsed_s
    if kind == "image" and source:
        result["title"] = Path(source).name if not source.startswith("http") else source.rsplit("/", 1)[-1]
    if kind == "html":
        result["article"] = {k: art.get(k, "") for k in ("author", "date", "site", "url")}
        result["tokens"] = [t.text for t in tokens]
        result["blocks"] = [{k: b[k] for k in ("tag", "level", "depth", "ordered", "start", "end") if k in b}
                            | ({"text": b["text"]} if b["tag"] == "pre" else {}) for b in art["blocks"]]
        for h in result["highlights"]:
            h.pop("rects", None)
    else:
        result["page_image"] = f"/api/doc/{doc_id}/page/{{n}}.png"
    return result


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


async def _process(doc_id: str, data: bytes, kind: str, source: str, force: bool):
    job = JOBS[doc_id]
    try:
        llm = get_llm()
        out = _result_path(doc_id, llm)
        cached_path = _find_result(doc_id)
        if cached_path and not force:
            cached = _read_json(cached_path) or {}
            if not cached.get("partial"):
                job.update(status="done", stage="Loaded from cache", progress=1.0, ready=True)
                return

        async def progress(stage: str, frac: float):
            job.update(stage=stage, progress=round(frac, 3))

        job.update(stage="Reading the article" if kind == "html" else "Reading the paper", progress=0.02)
        doc, art = await asyncio.to_thread(_parse, kind, data)
        if len(doc.tokens) < 20:
            raise RuntimeError("Could not find text in this file. Scanned PDFs are not supported yet; upload page images instead.")
        t0 = time.time()
        write_lock = asyncio.Lock()

        async def on_update(result: dict):
            async with write_lock:
                payload = _decorate_result(result, doc_id, source, kind, art, doc.tokens)
                _write_json(out, payload)
            job.update(ready=True)

        result = await analyze.run(doc, llm, progress, on_update=on_update)
        result = _decorate_result(result, doc_id, source, kind, art, doc.tokens,
                                  elapsed_s=round(time.time() - t0, 1))
        result.pop("partial", None)
        _write_json(out, result)
        meta_p = _doc_dir(doc_id) / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        meta.update(title=result.get("title") or source, n_highlights=len(result["highlights"]), analyzed_at=time.time())
        meta_p.write_text(json.dumps(meta))
        job.update(status="done", stage="Done", progress=1.0, ready=True)
    except (LLMError, RuntimeError, HTTPException) as e:
        log.exception("job %s failed", doc_id)
        job.update(status="error", error=str(getattr(e, "detail", e)))
    except Exception as e:  # pragma: no cover
        log.exception("job %s failed", doc_id)
        job.update(status="error", error=f"Unexpected error: {e}")


def _start(data: bytes, ext: str, source: str, force: bool = False, label: str = "") -> dict:
    kind = "pdf" if ext == "pdf" else "html" if ext == "json" else "image"
    key = data
    if kind == "html":  # same article text means same document, whatever URL or page chrome it came with
        art = json.loads(data)
        key = json.dumps([art.get("title"), [b["text"] for b in art["blocks"]]], ensure_ascii=False).encode()
    doc_id = hashlib.sha256(key).hexdigest()[:16]
    d = DATA / doc_id
    d.mkdir(exist_ok=True)
    src = d / f"source.{ext}"
    if not src.exists():
        src.write_bytes(data)
    meta_p = d / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.update(kind=kind, ext=ext, source=source)
    meta.setdefault("title", label)
    meta_p.write_text(json.dumps(meta))
    running = JOBS.get(doc_id, {}).get("status") == "running"
    if not running:
        JOBS[doc_id] = {"status": "running", "stage": "Queued", "progress": 0.0, "ready": False}
        asyncio.create_task(_process(doc_id, data, kind, source, force))
    return {"doc_id": doc_id, "kind": kind}


async def _article_payload(html: bytes, url: str) -> bytes:
    try:
        art = await asyncio.to_thread(extract_article, html, url)
    except ArticleError as e:
        raise HTTPException(400, str(e))
    return json.dumps(art, ensure_ascii=False, sort_keys=True).encode()


@app.post("/api/analyze")
async def analyze_endpoint(url: str = Form(""), file: UploadFile | None = File(None), force: bool = Form(False)):
    if file is not None and file.filename:
        data = await file.read()
        hint, source = file.content_type or "", file.filename
    elif url.strip():
        data, hint, source = await _fetch(url)
    else:
        raise HTTPException(400, "Paste a link, or choose a PDF or image.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 60 MB.")
    ext = _sniff(data, hint)
    if not ext:
        raise HTTPException(400, "Unsupported file type. Use a PDF, an image (PNG, JPG, WEBP), or a link to a web page.")
    if ext == "html":
        data, ext = await _article_payload(data, source if source.startswith("http") else ""), "json"
    return _start(data, ext, source, force)


SAMPLES = {"attention": (ROOT / "samples" / "attention.pdf", "Attention Is All You Need")}


@app.post("/api/sample/{name}")
async def sample(name: str):
    if name not in SAMPLES or not SAMPLES[name][0].exists():
        raise HTTPException(404, "No such sample.")
    path, title = SAMPLES[name]
    return _start(path.read_bytes(), "pdf", path.name, label=title)


@app.get("/api/recent")
async def recent(limit: int = 6):
    items = []
    if not DATA.exists():
        return []
    for d in DATA.iterdir():
        meta_p = d / "meta.json"
        if not (d.is_dir() and meta_p.exists() and re.fullmatch(r"[0-9a-f]{16}", d.name)):
            continue
        res = _find_result(d.name)
        if not res:
            continue
        data = _read_json(res) or {}
        if data.get("partial"):
            continue
        try:
            meta = json.loads(meta_p.read_text())
        except ValueError:
            continue
        items.append({"doc_id": d.name, "title": meta.get("title") or data.get("title") or meta.get("source", ""),
                      "kind": meta.get("kind") or data.get("kind"),
                      "source": meta.get("source", ""), "n_highlights": meta.get("n_highlights") or data.get("stats", {}).get("highlights"),
                      "time": meta.get("analyzed_at") or res.stat().st_mtime})
    items.sort(key=lambda x: -x["time"])
    return items[:max(1, min(limit, 20))]


@app.get("/api/status/{doc_id}")
async def status(doc_id: str):
    job = JOBS.get(doc_id)
    if job is not None:
        return job
    p = _find_result(doc_id)
    if p:
        data = _read_json(p) or {}
        return {"status": "done", "stage": "Loaded from cache", "progress": 1.0,
                "ready": True, "partial": bool(data.get("partial"))}
    raise HTTPException(404, "unknown document")


@app.get("/api/doc/{doc_id}")
async def get_doc(doc_id: str):
    p = _find_result(doc_id)
    if not p:
        raise HTTPException(404, "not analyzed yet")
    return Response(p.read_text(), media_type="application/json")


@app.get("/api/doc/{doc_id}/page/{n}.png")
async def page_image(doc_id: str, n: int):
    d = _doc_dir(doc_id)
    meta = json.loads((d / "meta.json").read_text())
    if meta["kind"] == "html":
        raise HTTPException(404, "web articles have no page images")
    if meta["kind"] == "image":
        return FileResponse(d / f"source.{meta['ext']}")
    out = d / f"page-{n}.png"
    if not out.exists():
        def render():
            pdf = pymupdf.open(d / "source.pdf")
            if not 0 <= n < pdf.page_count:
                raise HTTPException(404, "no such page")
            pdf[n].get_pixmap(matrix=pymupdf.Matrix(PAGE_ZOOM, PAGE_ZOOM)).save(out)
        await asyncio.to_thread(render)
    return FileResponse(out, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/doc/{doc_id}/export")
async def export_doc(doc_id: str, fmt: str = "md"):
    fmt = fmt.lower().strip()
    if fmt not in ("md", "txt", "pdf"):
        raise HTTPException(400, "Use fmt=pdf, md, or txt.")
    p = _find_result(doc_id)
    if not p:
        raise HTTPException(404, "not analyzed yet")
    doc = json.loads(p.read_text())
    name = ex.slug(doc.get("title") or "kami")
    if fmt == "md":
        body = ex.notes_markdown(doc).encode("utf-8")
        return Response(body, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.md"})
    if fmt == "txt":
        body = ex.notes_text(doc).encode("utf-8")
        return Response(body, media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.txt"})
    d = _doc_dir(doc_id)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    kind = doc.get("kind") or meta.get("kind") or "pdf"
    src = None
    if kind == "pdf":
        src = d / "source.pdf"
    elif kind == "image":
        ext = meta.get("ext") or "png"
        src = d / f"source.{ext}"
        if not src.exists():
            src = next(d.glob("source.*"), None)
    pdf_bytes = await asyncio.to_thread(ex.highlighted_pdf, src if src and src.exists() else None, kind,
                                        doc.get("highlights") or [], doc)
    return Response(pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.pdf"})


@app.get("/api/config")
async def config():
    try:
        llm = get_llm()
        return {"provider": llm.name, "model": llm.model, "mock": llm.is_mock, **provider_status()}
    except LLMError as e:
        return JSONResponse({"error": str(e), **provider_status()}, status_code=500)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
