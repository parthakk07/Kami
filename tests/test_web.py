"""Web article input, arXiv link handling and the HTTP API. No network needed."""
import asyncio
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

os.environ.setdefault("PAPER_DATA_DIR", tempfile.mkdtemp(prefix="ph-test-"))
os.environ["LLM_PROVIDER"] = "mock"

import app as webapp  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pipeline import analyze  # noqa: E402
from pipeline.article import ArticleError, build_document, extract_article  # noqa: E402
from pipeline.mock_llm import MockLLM  # noqa: E402

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
BLOG = (FIX / "blog_post.html").read_bytes()


def test_extract_article_keeps_structure_and_drops_chrome():
    art = extract_article(BLOG, "https://example.com/small-caches")
    assert art["title"] == "Small Caches, Big Wins"
    text = " ".join(b["text"] for b in art["blocks"])
    heads = [b["text"] for b in art["blocks"] if b["tag"] == "h"]
    assert heads[:3] == ["The workload", "The experiment", "What we would do differently"]
    assert any(b["tag"] == "li" for b in art["blocks"])
    assert any(b["tag"] == "pre" and "hit_rate(small)" in b["text"] for b in art["blocks"])
    for junk in ("Popular posts", "Cookie settings", "Sign up for the newsletter"):
        assert junk not in text
    assert all(b["level"] == 1 for b in art["blocks"] if b["tag"] == "h")


def test_article_pipeline_mock():
    art = extract_article(BLOG, "https://example.com/small-caches")
    doc = build_document(art)
    titles = [s.title for s in doc.sections]
    assert titles[0] == "Opening" and "The experiment" in titles
    assert next(s for s in doc.sections if s.title == "References").kind == "references"
    r = asyncio.run(analyze.run(doc, MockLLM()))
    assert r["claims"] and r["highlights"]
    assert r["stats"]["unmatched_quotes"] == 0
    starts = {b["start"] for b in art["blocks"] if b["end"] > b["start"]}
    for h in r["highlights"]:
        a, b = h["token_range"]
        assert not any(a < s < b for s in starts), f"highlight crosses a block boundary: {h['quote']}"
    assert {s["token_range"][0] for s in r["sections"]} >= {doc.sections[1].start}


def test_paywalled_or_empty_page_is_rejected():
    with pytest.raises(ArticleError, match="paywalled"):
        extract_article((FIX / "paywall.html").read_bytes())
    with pytest.raises(ArticleError):
        extract_article(b"<html><body><div id='root'></div><script>app()</script></body></html>")


@pytest.mark.parametrize("raw,expected", [
    ("https://arxiv.org/abs/1706.03762", "https://arxiv.org/pdf/1706.03762"),
    ("https://arxiv.org/abs/1706.03762v7", "https://arxiv.org/pdf/1706.03762v7"),
    ("arxiv.org/abs/1706.03762/", "https://arxiv.org/pdf/1706.03762"),
    ("http://www.arxiv.org/html/2401.01234v2", "https://arxiv.org/pdf/2401.01234v2"),
    ("https://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1706.03762"),
    ("https://arxiv.org/abs/1706.03762?context=cs", "https://arxiv.org/pdf/1706.03762"),
    ("https://arxiv.org/abs/hep-th/9901001", "https://arxiv.org/pdf/hep-th/9901001"),
    ("1706.03762", "https://arxiv.org/pdf/1706.03762"),
    ("arXiv:1706.03762", "https://arxiv.org/pdf/1706.03762"),
    ("https://openreview.net/forum?id=abc123", "https://openreview.net/pdf?id=abc123"),
    ("lilianweng.github.io/posts/2023-06-23-agent/", "https://lilianweng.github.io/posts/2023-06-23-agent/"),
])
def test_normalize_url(raw, expected):
    assert webapp._normalize_url(raw) == expected


class _Site(BaseHTTPRequestHandler):
    PDF = (HERE / "attention.pdf").read_bytes()
    seen_ua: list = []

    def do_GET(self):
        self.seen_ua.append(self.headers.get("User-Agent", ""))
        routes = {
            "/post": (200, "text/html; charset=utf-8", BLOG),
            "/landing": (200, "text/html", (FIX / "landing_page.html").read_bytes()),
            "/files/paper.pdf": (200, "application/pdf", self.PDF),
            "/locked": (403, "text/html", b"<html>Forbidden</html>"),
            "/paywall": (200, "text/html", (FIX / "paywall.html").read_bytes()),
        }
        code, ctype, body = routes.get(self.path, (404, "text/html", b"missing"))
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def site():
    srv = HTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_fetch_html_landing_and_errors(site):
    data, ctype, final = asyncio.run(webapp._fetch(site + "/post"))
    assert ctype == "text/html" and b"Small Caches" in data and final.endswith("/post")
    assert "Mozilla/5.0" in _Site.seen_ua[-1] and "Chrome" in _Site.seen_ua[-1]
    data, ctype, final = asyncio.run(webapp._fetch(site + "/landing"))
    assert data[:5] == b"%PDF-" and final.endswith("/files/paper.pdf")
    with pytest.raises(HTTPException, match="paywalled"):
        asyncio.run(webapp._fetch(site + "/locked"))
    with pytest.raises(HTTPException, match="404"):
        asyncio.run(webapp._fetch(site + "/nope"))


def _wait(client, doc_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/api/status/{doc_id}").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.2)
    raise AssertionError("timed out")


def test_api_blog_link_sample_and_recent(site):
    with TestClient(webapp.app) as c:
        r = c.post("/api/analyze", data={"url": site + "/post"})
        assert r.status_code == 200, r.text
        doc_id = r.json()["doc_id"]
        assert r.json()["kind"] == "html"
        assert _wait(c, doc_id)["status"] == "done"
        d = c.get(f"/api/doc/{doc_id}").json()
        assert d["kind"] == "html" and d["title"] == "Small Caches, Big Wins"
        assert d["tokens"] and d["blocks"] and d["highlights"] and "page_image" not in d
        assert all("rects" not in h for h in d["highlights"])
        assert d["article"]["url"].endswith("/post")
        assert any(s["bucket"] for s in d["sections"])

        bad = c.post("/api/analyze", data={"url": site + "/paywall"})
        assert bad.status_code == 400 and "paywalled" in bad.json()["detail"]
        assert c.post("/api/analyze", data={"url": site + "/locked"}).status_code == 400
        assert c.post("/api/analyze", data={}).status_code == 400

        up = c.post("/api/analyze", files={"file": ("post.html", BLOG, "text/html")})
        assert up.status_code == 200 and up.json()["doc_id"] == doc_id  # same text, same cache entry

        s = c.post("/api/sample/attention")
        assert s.status_code == 200
        sid = s.json()["doc_id"]
        assert _wait(c, sid)["status"] == "done"
        assert c.post("/api/sample/nope").status_code == 404

        recent = c.get("/api/recent").json()
        assert {x["doc_id"] for x in recent} >= {doc_id, sid}
        assert next(x for x in recent if x["doc_id"] == sid)["title"] == "Attention Is All You Need"
        assert c.get(f"/api/doc/{doc_id}/page/0.png").status_code == 404
        home = c.get("/")
        assert home.status_code == 200 and "Kami" in home.text


def test_long_headingless_essay_in_layout_table_is_split():
    para = ("Good work usually starts with a question that the author cannot stop thinking about, "
            "and it grows because the author keeps asking it in new ways over many years. ")
    body = "<br><br>".join(f"Paragraph {i}. " + para * 3 for i in range(80))
    html = f"<html><head><title>An Essay</title></head><body><table><tr><td><font>{body}</font></td></tr></table></body></html>"
    art = extract_article(html.encode(), "https://example.com/essay")
    assert len([b for b in art["blocks"] if b["tag"] == "p"]) >= 60
    doc = build_document(art)
    assert len(doc.sections) >= 3 and all(s.synthetic for s in doc.sections)
    assert doc.sections[0].start == 0 and doc.sections[-1].end == len(doc.tokens)
