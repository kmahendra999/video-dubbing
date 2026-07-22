"""Minimal LLM client — Gemini and xAI Grok, over plain HTTP.

No vendor SDK on purpose: both APIs are a single POST, and the SDKs drag in
dependency trees that fight with torch/transformers in the ai image.
"""

import json
import logging
import time
import urllib.error
import urllib.request

from .config import settings

log = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
XAI_URL = "https://api.x.ai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def configured() -> tuple[str, str] | None:
    """Returns (provider, model) or None if no usable key is set."""
    p = (settings.llm_provider or "").lower()
    if p == "gemini" and settings.gemini_api_key:
        return "gemini", settings.llm_model or "gemini-2.5-flash"
    if p == "xai" and settings.xai_api_key:
        return "xai", settings.llm_model or "grok-3"
    # Fall back to whichever key exists.
    if settings.gemini_api_key:
        return "gemini", "gemini-2.5-flash"
    if settings.xai_api_key:
        return "xai", "grok-3"
    return None


def _post(url: str, payload: dict, headers: dict, retries: int = 4) -> dict:
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            last = f"HTTP {e.code}: {detail}"
            # 429/5xx are worth retrying; 4xx generally are not.
            if e.code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt * 3
                log.warning("llm %s — retrying in %ss", last, wait)
                time.sleep(wait)
                continue
            raise LLMError(last) from e
        except Exception as e:
            last = str(e)
            time.sleep(2 ** attempt)
    raise LLMError(f"LLM unreachable after {retries} attempts: {last}")


def complete_json(prompt: str, *, temperature: float = 0.3) -> object:
    """Ask the model for JSON and return it parsed."""
    cfg = configured()
    if not cfg:
        raise LLMError("no LLM key configured")
    provider, model = cfg

    if provider == "gemini":
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "maxOutputTokens": 65536,
            },
        }
        data = _post(
            GEMINI_URL.format(model=model),
            payload,
            {"content-type": "application/json", "x-goog-api-key": settings.gemini_api_key},
        )
        try:
            cand = data["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"].get("parts", []))
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected Gemini response: {json.dumps(data)[:300]}") from e
        if not text.strip():
            raise LLMError(f"empty Gemini reply (finishReason={cand.get('finishReason')})")
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        data = _post(
            XAI_URL,
            payload,
            {"content-type": "application/json",
             "authorization": f"Bearer {settings.xai_api_key}"},
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected xAI response: {json.dumps(data)[:300]}") from e

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"model did not return valid JSON: {text[:300]}") from e


def label() -> str:
    cfg = configured()
    return f"{cfg[0]}/{cfg[1]}" if cfg else "none"
