"""The analysis pipeline: signals, claims, per-section analysis, reading plan, highlight spans."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import BaseModel, ValidationError

from . import signals as sig
from .llm import BaseLLM, LLMError, extract_json
from .parse import Document, Section
from .schemas import ClaimsOut, ReadingPlan, SectionAnalysis
from .spans import SpanIndex

log = logging.getLogger("paper.analyze")
PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
MAX_SECTION_CHARS = 14000
MIN_SECTION_TOKENS = 40
CONCURRENCY = 4
PRIORITY = {"claim": 0, "complex": 1, "key": 2, "definition": 3}

Progress = Callable[[str, float], Awaitable[None] | None]
Update = Callable[[dict], Awaitable[None] | None]


def render(name: str, **kw) -> str:
    t = (PROMPTS / f"{name}.txt").read_text()
    for k, v in kw.items():
        t = t.replace("{{" + k + "}}", str(v))
    return t


async def call_validated(llm: BaseLLM, prompt: str, kind: str, ctx: dict, model: type[BaseModel]) -> BaseModel:
    """Ask, parse, validate. On a bad answer retry once with the error attached."""
    last_err = None
    for attempt in range(2):
        p = prompt if attempt == 0 else (
            prompt + f"\n\nYour previous answer was invalid ({last_err}). Return only one valid JSON object matching the format exactly.")
        raw = await llm.complete(p, kind, ctx)
        try:
            return model.model_validate(extract_json(raw))
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = str(e).splitlines()[0][:200]
            log.warning("invalid %s output (attempt %d): %s", kind, attempt + 1, last_err)
    raise LLMError(f"{kind}: model returned invalid JSON twice ({last_err})")


def _claims_text(doc: Document) -> str:
    parts = []
    for s in doc.sections:
        t = s.title.lower()
        if any(k in t for k in ("abstract", "introduction", "conclusion", "opening", "summary", "takeaway",
                                "final thoughts", "tl;dr", "tldr")):
            parts.append(doc.section_text(s))
    if sum(len(p) for p in parts) < 1500:  # no abstract-like sections (blog posts, single pages)
        parts = [doc.text(0, min(len(doc.tokens), 1800))]
    text = "\n\n".join(parts)
    return text[:16000]


async def _maybe(cb, *args):
    if cb:
        r = cb(*args)
        if asyncio.iscoroutine(r):
            await r


def _highlights(doc: Document, idx: SpanIndex, claims_list: list, analyses: dict) -> tuple[list, list, int]:
    sec_by_id = {s.id: s for s in doc.sections}
    cand = []
    abstract = next((s for s in doc.sections if "abstract" in s.title.lower()), None)
    for c in claims_list:
        cand.append(("claim", abstract.id if abstract else None, c["quote"], f"Key claim {c['id']}", c["claim"],
                     {"claim_id": c["id"], "evidence_hint": c["evidence_hint"]}))
    for sid, a in analyses.items():
        for k in a["key_points"]:
            cand.append(("key", sid, k["quote"], "Important", k["note"], {}))
        for k in a["complex_arguments"]:
            cand.append(("complex", sid, k["original"], "Complex argument", k["plain"], {}))
        for k in a["definitions"]:
            cand.append(("definition", sid, k["quote"], f"Definition: {k['term']}", k["plain"], {"term": k["term"]}))
    cand.sort(key=lambda c: PRIORITY[c[0]])

    highlights, taken, unmatched = [], [], []
    for typ, sid, quote, label, note, extra in cand:
        s = sec_by_id.get(sid) if sid else None
        hit = idx.find(quote, s.start, s.end) if s else idx.find(quote)
        if not hit:
            unmatched.append({"type": typ, "section_id": sid, "quote": quote})
            continue
        a, b, conf = hit
        if any(min(b, tb) - max(a, ta) > 0.5 * min(b - a, tb - ta) for ta, tb in taken):
            continue
        taken.append((a, b))
        owner = sid or next((x.id for x in doc.sections if x.start <= a < x.end), None)
        highlights.append({
            "id": f"H{len(highlights) + 1}", "type": typ, "section_id": owner, "label": label, "note": note,
            "quote": doc.text(a, b), "match_confidence": conf, "token_range": [a, b],
            "rects": idx.rects(a, b), **extra,
        })
    highlights.sort(key=lambda h: h["token_range"][0])
    return highlights, unmatched, len(cand)


def _sections_out(doc: Document, idx: SpanIndex, signals: dict, analyses: dict, plan: dict) -> list:
    bucket_of = {o["section_id"]: o for o in (plan or {}).get("order") or []}
    out = []
    for s in doc.sections:
        d = s.to_dict()
        d["signals"] = signals[s.id]
        d["analysis"] = analyses.get(s.id)
        p = bucket_of.get(s.id)
        d["bucket"] = p["bucket"] if p else ("skip" if s.kind in ("references", "front") else None)
        d["why"] = p["why"] if p else ""
        d["heading_rect"] = idx.line_rect(s.heading_line) if s.heading_line is not None else None
        first = doc.words[doc.tokens[s.start].words[0]] if s.end > s.start else None
        d["page"] = first.page if first else 0
        d["token_range"] = [s.start, s.end]
        out.append(d)
    return out


def pack(doc: Document, llm: BaseLLM, idx: SpanIndex, signals: dict, claims_list: list,
         analyses: dict, plan: dict, errors: list, analyzable_n: int, partial: bool = False) -> dict:
    highlights, unmatched, n_cand = _highlights(doc, idx, claims_list, analyses)
    result = {
        "title": doc.title,
        "kind": doc.kind,
        "provider": llm.name,
        "model": llm.model,
        "mock": llm.is_mock,
        "pages": [{"index": i, **p} for i, p in enumerate(doc.pages)],
        "claims": claims_list,
        "sections": _sections_out(doc, idx, signals, analyses, plan),
        "plan": plan,
        "highlights": highlights,
        "stats": {"highlights": len(highlights), "unmatched_quotes": len(unmatched),
                  "candidates": n_cand, "sections_analyzed": len(analyses),
                  "sections_total": analyzable_n},
        "unmatched": unmatched,
        "errors": errors,
    }
    if partial:
        result["partial"] = True
    return result


async def run(doc: Document, llm: BaseLLM, progress: Progress | None = None, level: str = "intermediate",
              on_update: Update | None = None) -> dict:
    idx = SpanIndex(doc)
    await _maybe(progress, "Computing section signals", 0.05)
    signals = {s.id: sig.compute(doc, s) for s in doc.sections}
    claims_list: list = []
    analyses: dict[str, dict] = {}
    errors: list[str] = []
    plan = {"summary": "", "order": [], "prerequisites": []}
    analyzable = [s for s in doc.sections if s.kind == "body" and s.end - s.start >= MIN_SECTION_TOKENS]
    emit_lock = asyncio.Lock()

    async def emit(stage: str, frac: float, *, partial: bool = True):
        await _maybe(progress, stage, frac)
        if on_update:
            result = pack(doc, llm, idx, signals, claims_list, analyses, plan, errors, len(analyzable), partial)
            await _maybe(on_update, result)

    await emit("Opening the document", 0.06)

    await _maybe(progress, "Finding the main claims", 0.1)
    ctext = _claims_text(doc)
    claims = await call_validated(llm, render("claims", text=ctext), "claims", {"text": ctext}, ClaimsOut)
    claims_list = [c.model_dump() for c in claims.claims]
    claims_brief = json.dumps([{"id": c["id"], "claim": c["claim"]} for c in claims_list])
    await emit("Found the main claims", 0.14)

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0

    async def one(s: Section):
        nonlocal done
        text = doc.section_text(s)[:MAX_SECTION_CHARS]
        sg = signals[s.id]
        prompt = render("section", title=doc.title, claims=claims_brief, section_title=s.title,
                        section_text=text, **sg)
        ctx = {"section_text": text, "section_title": s.title, "signals": sg, "claims_list": claims_list}
        async with sem:
            try:
                a = await call_validated(llm, prompt, "section", ctx, SectionAnalysis)
                analyses[s.id] = a.model_dump()
            except Exception as e:  # one bad section must not sink the document
                errors.append(f"{s.id} {s.title}: {e}")
                analyses[s.id] = SectionAnalysis(difficulty=1, difficulty_reason="analysis failed").model_dump()
        async with emit_lock:
            done += 1
            await emit(f"Analyzing {s.title}", 0.15 + 0.65 * done / max(1, len(analyzable)))

    await asyncio.gather(*(one(s) for s in analyzable))

    await _maybe(progress, "Building the reading plan", 0.85)
    plan_input = [{
        "id": s.id, "title": s.title, "n_words": signals[s.id]["n_words"],
        "difficulty": analyses[s.id]["difficulty"], "difficulty_reason": analyses[s.id]["difficulty_reason"],
        "supports_claims": analyses[s.id]["supports_claims"], "prerequisites": analyses[s.id]["prerequisites"],
    } for s in analyzable]
    try:
        plan = (await call_validated(llm, render("plan", sections=json.dumps(plan_input), level=level), "plan",
                                     {"sections": plan_input, "level": level}, ReadingPlan)).model_dump()
    except Exception as e:
        errors.append(f"plan: {e}")
        plan = {"summary": "", "order": [], "prerequisites": []}
    valid = {s.id for s in analyzable}
    plan["order"] = [o for o in plan["order"] if o["section_id"] in valid]
    seen = {o["section_id"] for o in plan["order"]}
    for s in analyzable:
        if s.id not in seen:
            a = analyses[s.id]
            plan["order"].append({"section_id": s.id, "bucket": "read" if a["supports_claims"] else "skim",
                                  "why": "Not ranked by the model; placed by default."})

    await emit("Mapping highlights onto the paper", 0.95, partial=False)
    return pack(doc, llm, idx, signals, claims_list, analyses, plan, errors, len(analyzable), partial=False)
