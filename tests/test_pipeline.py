"""End-to-end tests. Run: .venv/bin/pytest -q tests"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pymupdf
import pytest

from pipeline import analyze
from pipeline.llm import OpenAICompatLLM
from pipeline.mock_llm import MockLLM
from pipeline.parse import parse_image, parse_pdf
from pipeline.schemas import ClaimsOut
from pipeline.spans import SpanIndex

PDF = Path(__file__).parent / "attention.pdf"


@pytest.fixture(scope="module")
def doc():
    return parse_pdf(PDF.read_bytes())


def test_sections(doc):
    titles = [s.title for s in doc.sections]
    assert doc.title == "Attention Is All You Need"
    for t in ["Abstract", "1 Introduction", "3.2.1 Scaled Dot-Product Attention", "7 Conclusion", "References"]:
        assert t in titles


def test_span_exact_and_fuzzy(doc):
    idx = SpanIndex(doc)
    q = "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms"
    a, b, conf = idx.find(q)
    assert conf == 1.0 and doc.text(a, b).startswith("We propose")
    # paraphrased edges and a dropped word still land on the same place
    q2 = "we propose a new network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence"
    a2, b2, conf2 = idx.find(q2)
    assert abs(a2 - a) <= 1 and conf2 >= 0.6
    assert idx.rects(a, b)[0]["page"] == 0


def test_full_pipeline_mock(doc):
    r = asyncio.run(analyze.run(doc, MockLLM()))
    assert r["claims"] and r["highlights"]
    assert r["stats"]["unmatched_quotes"] == 0
    assert {o["bucket"] for o in r["plan"]["order"]} <= {"study", "read", "skim", "skip"}
    for h in r["highlights"]:
        assert h["rects"] and all(x["x1"] > x["x0"] for x in h["rects"])


def test_pipeline_emits_partial_updates(doc):
    snaps = []

    async def on_update(result):
        snaps.append(result)

    r = asyncio.run(analyze.run(doc, MockLLM(), on_update=on_update))
    assert len(snaps) >= 3
    assert snaps[0].get("partial") is True
    assert "partial" not in r
    assert any(s["highlights"] and s is not snaps[0] for s in snaps)


class FlakyLLM(MockLLM):
    """Returns garbage on the first call of each kind, then valid JSON."""
    def __init__(self):
        self.seen = set()
    async def complete(self, prompt, kind, ctx):
        if kind not in self.seen:
            self.seen.add(kind)
            return "Sure! Here is the answer: {not json"
        return await super().complete(prompt, kind, ctx)


def test_retry_once_on_bad_json(doc):
    llm = FlakyLLM()
    out = asyncio.run(analyze.call_validated(llm, "x", "claims", {"text": doc.section_text(doc.sections[1])}, ClaimsOut))
    assert out.claims


def test_openai_compatible_provider(doc):
    """Drive the real HTTP client against a local fake OpenAI-compatible server."""
    mock = MockLLM()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers["Authorization"] == "Bearer test-key"
            assert body["max_tokens"] == 2048
            prompt = body["messages"][-1]["content"]
            # answer in a markdown fence to exercise JSON extraction
            content = "```json\n" + json.dumps({"claims": [{"id": "C1", "claim": "Transformers use only attention.",
                       "quote": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                       "evidence_hint": "experiment"}]}) + "\n```" if "main claims" in prompt else "{}"
            data = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(data)
        def log_message(self, *a): pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    llm = OpenAICompatLLM("openai", "test-key", f"http://127.0.0.1:{srv.server_port}", "fake")
    out = asyncio.run(analyze.call_validated(llm, analyze.render("claims", text="..."), "claims", {}, ClaimsOut))
    srv.shutdown()
    assert out.claims[0].id == "C1"


def test_image_input(tmp_path):
    pdf = pymupdf.open(PDF)
    png = pdf[2].get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
    d = parse_image(png)
    assert len(d.tokens) > 200
    assert any("Encoder and Decoder" in s.title for s in d.sections)
    r = asyncio.run(analyze.run(d, MockLLM()))
    assert r["highlights"] and r["pages"][0]["width"] == pdf[2].rect.width * 2
