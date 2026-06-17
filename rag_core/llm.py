"""
LLM generation helpers.

Provider is selected via LLM_PROVIDER in .env (or overridden at call time).
API keys are always read from .env — never from the UI.

Supported providers:
  ollama  → local Ollama server (no key required)
  openai  → OpenAI API            (OPENAI_API_KEY, OPENAI_MODEL)
  groq    → Groq API              (GROQ_API_KEY,   GROQ_MODEL)
  gemini  → Google Gemini SDK     (GEMINI_API_KEY, GEMINI_MODEL)
"""

import json
import os
import time
from typing import Generator

import httpx
from dotenv import load_dotenv

from rag_core.config import settings

# ---------------------------------------------------------------------------
# Provider tables
# ---------------------------------------------------------------------------

# OpenAI-compatible providers (httpx streaming)
_OPENAI_COMPAT = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1",       "OPENAI_MODEL", "gpt-4o-mini"),
    "groq":   ("GROQ_API_KEY",   "https://api.groq.com/openai/v1",  "GROQ_MODEL",   "llama3-70b-8192"),
}

_ALL_PROVIDERS = {"ollama", "openai", "groq", "gemini"}


def _resolve(provider: str) -> dict:
    """
    Read provider config fresh from .env at call time.
    Avoids stale cached values from the frozen settings dataclass.
    """
    load_dotenv(override=True)
    p = provider.lower().strip()

    if p not in _ALL_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{p}'. "
            f"Set LLM_PROVIDER in .env to one of: {', '.join(_ALL_PROVIDERS)}"
        )

    if p == "ollama":
        return {
            "api_key": "",
            "model": os.getenv("OLLAMA_CHAT_MODEL", settings.ollama_chat_model),
        }

    if p == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "No API key found for Gemini. "
                "Add GEMINI_API_KEY=your_key to your .env file and restart the app."
            )
        return {
            "api_key": key,
            "model": os.getenv("GEMINI_MODEL", "") or "gemini-1.5-flash",
        }

    # openai / groq
    key_var, base_url, model_var, fallback_model = _OPENAI_COMPAT[p]
    key = os.getenv(key_var, "")
    if not key:
        raise ValueError(
            f"No API key found for provider '{p}'. "
            f"Add {key_var}=your_key to your .env file and restart the app."
        )
    return {
        "api_key": key,
        "api_base_url": base_url,
        "model": os.getenv(model_var, "") or fallback_model,
    }


# ---------------------------------------------------------------------------
# Ollama streaming  (original logic — untouched)
# ---------------------------------------------------------------------------

def _stream_ollama(prompt: str, model: str) -> Generator[str, None, None]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 350},
    }
    with httpx.Client(timeout=settings.ollama_timeout_seconds) as client:
        with client.stream(
            "POST", f"{settings.ollama_base_url}/api/generate", json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break


# ---------------------------------------------------------------------------
# Google Gemini  (direct REST API — no SDK, no protobuf conflicts)
# ---------------------------------------------------------------------------

def _stream_gemini(prompt: str, api_key: str, model: str) -> Generator[str, None, None]:
    """
    Calls Gemini's streamGenerateContent REST endpoint with Server-Sent Events.
    Uses x-goog-api-key header (works with both AIza and AQ. format keys).
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:streamGenerateContent?alt=sse"
    )
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code in (429, 503):
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        response.raise_for_status()
                    response.raise_for_status()
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[len("data: "):]
                        if raw == "[DONE]":
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        try:
                            text = data["candidates"][0]["content"]["parts"][0]["text"]
                            if text:
                                yield text
                        except (KeyError, IndexError):
                            continue
                    return  # success
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 503) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


# ---------------------------------------------------------------------------
# OpenAI-compatible streaming  (OpenAI / Groq)
# ---------------------------------------------------------------------------

def _stream_openai_compat(
    prompt: str, api_key: str, api_base_url: str, model: str
) -> Generator[str, None, None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    url = api_base_url.rstrip("/") + "/chat/completions"

    # Retry up to 3 times on 429 / 503 with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code in (429, 503):
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        response.raise_for_status()
                    response.raise_for_status()
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: "):]
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        delta = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            yield delta
                    return  # success — exit retry loop
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 503) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer_stream(
    prompt: str,
    *,
    provider: str | None = None,
) -> Generator[str, None, None]:
    """
    Stream an answer for *prompt* using the configured provider.

    provider: override at runtime (defaults to LLM_PROVIDER in .env).
              One of: ollama | openai | groq | gemini
    """
    p = (provider or os.getenv("LLM_PROVIDER", settings.llm_provider)).lower().strip()
    cfg = _resolve(p)

    if p == "ollama":
        yield from _stream_ollama(prompt, cfg["model"])
    elif p == "gemini":
        yield from _stream_gemini(prompt, cfg["api_key"], cfg["model"])
    else:
        yield from _stream_openai_compat(
            prompt,
            api_key=cfg["api_key"],
            api_base_url=cfg["api_base_url"],
            model=cfg["model"],
        )


def generate_answer(prompt: str, *, provider: str | None = None) -> str:
    return "".join(generate_answer_stream(prompt, provider=provider)).strip()
