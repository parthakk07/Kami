"""Pluggable LLM client.

Select with LLM_PROVIDER = openai | anthropic | gemini | xai | groq | openrouter | mock.
If LLM_PROVIDER is unset, the first provider whose API key env var is present is used,
otherwise the app falls back to the offline mock (heuristic) analyzer.
LLM_MODEL overrides the default model of the chosen provider. LLM_BASE_URL overrides the
endpoint for any OpenAI-compatible provider (e.g. a local Ollama or vLLM server).
On OpenRouter, a 402/429 on one model tries cheaper models on the same key.
"""
from __future__ import annotations

from collections.abc import Sequence
import json
import logging
import os
import re

import httpx

log = logging.getLogger("paper.llm")

PROVIDERS: dict[str, dict] = {
    "openai": {"key": "OPENAI_API_KEY", "base": "https://api.openai.com/v1", "model": "gpt-4.1-mini"},
    "anthropic": {"key": "ANTHROPIC_API_KEY", "base": "https://api.anthropic.com/v1", "model": "claude-haiku-4-5"},
    "gemini": {"key": "GEMINI_API_KEY", "base": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"},
    "xai": {"key": "XAI_API_KEY", "base": "https://api.x.ai/v1", "model": "grok-3-mini"},
    "groq": {"key": "GROQ_API_KEY", "base": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "openrouter": {"key": "OPENROUTER_API_KEY", "base": "https://openrouter.ai/api/v1", "model": "google/gemini-2.5-flash"},
}

# Tried in order after the chosen OpenRouter model when that model is out of credits.
OPENROUTER_FALLBACK = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
]

SYSTEM = (
    "You are a careful research assistant. You always answer with a single valid JSON object and nothing else. "
    "When asked to quote the paper, you copy text exactly. You never use emojis."
)


class LLMError(RuntimeError):
    pass


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        return json.loads(text[a:b + 1])
    raise json.JSONDecodeError("no JSON object found", text, 0)


class BaseLLM:
    name = "base"
    model = ""
    is_mock = False

    async def complete(self, prompt: str, kind: str, ctx: dict) -> str:  # pragma: no cover
        raise NotImplementedError


class OpenAICompatLLM(BaseLLM):
    def __init__(self, name: str, key: str, base: str, model: str):
        self.name, self._key, self.base, self.model = name, key, base.rstrip("/"), model
        self.client = httpx.AsyncClient(timeout=120)

    async def complete(self, prompt: str, kind: str, ctx: dict) -> str:
        # OpenRouter credits are reserved against max_tokens. Unset = model max
        # (often 65536), which 402s a cheap key even when the JSON reply is small.
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS") or 1024)
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        r = await self.client.post(f"{self.base}/chat/completions", json=body,
                                   headers=self._headers())
        if r.status_code >= 400:
            raise LLMError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"] or ""

    def _headers(self) -> dict:
        h = {"Authorization": f"Bearer {self._key}"}
        if self.name == "openrouter":
            h["HTTP-Referer"] = "http://localhost:8765"
            h["X-Title"] = "Kami"
        return h


class AnthropicLLM(BaseLLM):
    def __init__(self, key: str, base: str, model: str):
        self.name, self._key, self.base, self.model = "anthropic", key, base.rstrip("/"), model
        self.client = httpx.AsyncClient(timeout=120)

    async def complete(self, prompt: str, kind: str, ctx: dict) -> str:
        body = {"model": self.model, "max_tokens": 4096, "system": SYSTEM, "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}]}
        r = await self.client.post(f"{self.base}/messages", json=body, headers={
            "x-api-key": self._key, "anthropic-version": "2023-06-01"})
        if r.status_code >= 400:
            raise LLMError(f"anthropic HTTP {r.status_code}: {r.text[:300]}")
        return "".join(p.get("text", "") for p in r.json().get("content", []))


class FallbackLLM(BaseLLM):
    """Try models in order; skip a model that is out of credits or rate-limited."""

    def __init__(self, clients: Sequence[BaseLLM]):
        self._clients = clients
        self.name = clients[0].name
        self.model = clients[0].model
        self.is_mock = False

    async def complete(self, prompt: str, kind: str, ctx: dict) -> str:
        last: LLMError | None = None
        for c in self._clients:
            try:
                out = await c.complete(prompt, kind, ctx)
                self.name, self.model = c.name, c.model
                return out
            except LLMError as e:
                last = e
                msg = str(e)
                if any(code in msg for code in (" HTTP 402", " HTTP 429", " HTTP 401", " HTTP 404")):
                    log.warning("skip %s/%s: %s", c.name, c.model, msg[:160])
                    continue
                raise
        raise last or LLMError("no LLM available")


def _openrouter_models(primary: str) -> list[str]:
    extra = [m.strip() for m in os.environ.get("LLM_MODELS", "").split(",") if m.strip()]
    out, seen = [], set()
    for m in [primary, *extra, *OPENROUTER_FALLBACK]:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def get_llm() -> BaseLLM:
    choice = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not choice:
        choice = next((n for n, p in PROVIDERS.items() if os.environ.get(p["key"])), "mock")
    if choice == "mock":
        from .mock_llm import MockLLM
        return MockLLM()
    if choice not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{choice}'. Use one of: {', '.join(PROVIDERS)}, mock")
    p = PROVIDERS[choice]
    key = os.environ.get(p["key"], "")
    if not key:
        raise LLMError(f"LLM_PROVIDER={choice} but {p['key']} is not set")
    model = os.environ.get("LLM_MODEL") or p["model"]
    base = os.environ.get("LLM_BASE_URL") or p["base"]
    if choice == "anthropic":
        return AnthropicLLM(key, base, model)
    if choice == "openrouter" and not os.environ.get("LLM_BASE_URL"):
        models = _openrouter_models(model)
        clients = [OpenAICompatLLM(choice, key, base, m) for m in models]
        return FallbackLLM(clients) if len(clients) > 1 else clients[0]
    return OpenAICompatLLM(choice, key, base, model)


def provider_status() -> dict:
    llm_choice = os.environ.get("LLM_PROVIDER", "").strip().lower()
    available = [n for n, p in PROVIDERS.items() if os.environ.get(p["key"])]
    return {"configured": llm_choice or None, "keys_present": available}
