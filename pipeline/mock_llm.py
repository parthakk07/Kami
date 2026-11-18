"""Offline stand-in for an LLM.

It reads the same context the real prompts are built from and returns JSON with the same
shape, quoting sentences verbatim from the paper, so the full pipeline (validation, span
matching, highlighting, plan) runs end to end without an API key. Notes are heuristic and
labelled as demo output.
"""
from __future__ import annotations

import json
import re

from .llm import BaseLLM

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
CLAIM_WORDS = ["we propose", "we present", "we introduce", "we show", "we demonstrate", "achieves", "achieve",
               "outperform", "state-of-the-art", "state of the art", "new ", "our model", "our method", "superior",
               "significantly", "improv", "best", "first"]
KEY_WORDS = ["propose", "show", "result", "outperform", "key", "main", "important", "allows", "enables",
             "because", "therefore", "we use", "we found", "we observe", "in contrast", "unlike", "bleu", "accuracy"]
PREREQS = {
    "attention": "attention mechanism", "softmax": "softmax", "recurrent": "recurrent neural networks (RNNs)",
    "lstm": "LSTMs", "convolution": "convolutional networks", "embedding": "word embeddings",
    "gradient": "gradient descent", "backprop": "backpropagation", "dropout": "dropout",
    "layer normalization": "layer normalization", "residual": "residual connections", "bleu": "BLEU score",
    "encoder": "encoder-decoder models", "dot product": "dot products and matrix multiplication",
    "probability": "probability basics", "regularization": "regularization", "optimizer": "optimizers such as Adam",
    "adam": "optimizers such as Adam", "label smoothing": "label smoothing", "beam search": "beam search",
    "transformer": "transformer architecture", "loss": "loss functions", "variance": "mean and variance",
    "sinusoid": "sine and cosine functions", "kernel": "kernels", "bayes": "Bayes rule",
}
MATHCH = re.compile(r"[\u0370-\u03FF\u2200-\u22FF=+^_\u221A\u2211]")


def sentences(text: str) -> list[str]:
    out = []
    parts = [p for block in text.split("\n") for p in SENT.split(block)]
    for s in parts:
        s = s.strip()
        if 8 <= len(s.split()) <= 60:
            out.append(s)
    return out


def shorten(s: str, n: int = 22) -> str:
    w = s.split()
    return " ".join(w[:n]) + ("..." if len(w) > n else "")


def score(s: str, words: list[str]) -> int:
    low = s.lower()
    return sum(1 for w in words if w in low)


class MockLLM(BaseLLM):
    name = "mock"
    model = "heuristic-demo"
    is_mock = True

    async def complete(self, prompt: str, kind: str, ctx: dict) -> str:
        return json.dumps(getattr(self, f"_{kind}")(ctx))

    def _claims(self, ctx):
        sents = sentences(ctx["text"])
        ranked = sorted(sents, key=lambda s: -score(s, CLAIM_WORDS))
        picked = [s for s in ranked if score(s, CLAIM_WORDS) > 0][:4]
        if len(picked) < 2:  # essays rarely say "we propose": fall back to the most assertive sentences
            extra = sorted((s for s in sents if s not in picked), key=lambda s: -score(s, KEY_WORDS))
            picked += [s for s in extra if score(s, KEY_WORDS) > 0][:3 - len(picked)]
        picked.sort(key=sents.index)
        return {"claims": [{
            "id": f"C{i + 1}", "claim": "Demo note: " + shorten(s, 20),
            "quote": s, "evidence_hint": "experiment" if re.search(r"\d", s) else "argument",
        } for i, s in enumerate(picked)]}

    def _section(self, ctx):
        text = ctx["section_text"]
        sents = sentences(text)
        low = text.lower()
        sig = ctx["signals"]
        key = sorted(sents, key=lambda s: -score(s, KEY_WORDS))[:2]
        key = [s for s in key if score(s, KEY_WORDS) > 0]
        complex_ = sorted(sents, key=lambda s: -(len(MATHCH.findall(s)) * 3 + len(s.split()) / 10))
        complex_ = [s for s in complex_ if s not in key and (len(MATHCH.findall(s)) >= 2 or len(s.split()) > 32)][:2]
        defs = []
        for s in sents:
            m = re.search(r"\(([A-Z][A-Za-z\-]*[A-Z]s?)\)", s) or re.search(
                r"(?:called|termed|we call|referred to as|known as)\s+(?:the\s+)?\"?([\w\-]+(?: [\w\-]+)?)", s, re.I)
            if m and s not in key and s not in complex_:
                defs.append({"term": m.group(1), "quote": s, "plain": "Demo note: this sentence introduces " + m.group(1) + "."})
            if len(defs) >= 2:
                break
        claims = ctx.get("claims_list", [])
        supports = []
        words = set(re.findall(r"[a-z]{6,}", low))
        for c in claims:
            cw = set(re.findall(r"[a-z]{6,}", c["quote"].lower()))
            if len(cw & words) >= 3:
                supports.append(c["id"])
        d = 1 + min(2, sig["n_equations"] // 2) + (1 if sig["n_new_terms"] >= 2 else 0) + (1 if sig["citations_per_100_words"] > 2 else 0)
        prereqs = []
        for k, v in PREREQS.items():
            if k in low and v not in prereqs:
                prereqs.append(v)
        return {
            "supports_claims": supports[:3],
            "difficulty": max(1, min(5, d)),
            "difficulty_reason": f"Demo estimate from {sig['n_equations']} equation lines, {sig['n_new_terms']} new terms and {sig['n_citations']} citations.",
            "key_points": [{"quote": s, "note": "Demo note: a central statement of this section."} for s in key],
            "complex_arguments": [{"original": s, "plain": "Demo note: dense passage. In short: " + shorten(s, 14)} for s in complex_],
            "definitions": defs,
            "prerequisites": prereqs[:4],
            "self_check": [f"What is the main idea of '{ctx['section_title']}'?",
                           "How does this section support the paper's main claims?"],
        }

    def _plan(self, ctx):
        order, pre = [], []
        for s in ctx["sections"]:
            hard = s["difficulty"] >= 3
            carries = bool(s["supports_claims"])
            bucket = "study" if carries and hard else "read" if carries else "skim" if hard or s["n_words"] > 150 else "skip"
            order.append({"section_id": s["id"], "bucket": bucket,
                          "why": f"Demo plan: difficulty {s['difficulty']}, " + ("supports " + ", ".join(s["supports_claims"]) if carries else "no main claim") + "."})
            for p in s["prerequisites"]:
                if p not in pre:
                    pre.append(p)
        return {"summary": "Demo plan generated without an LLM. Start with the study sections, then read the rest in order.",
                "order": order, "prerequisites": pre[:8]}
