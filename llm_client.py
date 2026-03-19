"""
Minimal OpenAI-compatible chat client (LM Studio, vLLM, etc.).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ChatResult:
    content: str
    raw: dict[str, Any] | None = None


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.2,
    ) -> ChatResult:
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    obj = json.loads(body)
                choices = obj.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices in response: {obj!r}")
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                return ChatResult(content=str(content), raw=obj)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                # simple backoff
                time.sleep(min(2**attempt, 8))
                continue
        raise RuntimeError(f"LLM request failed after {self.max_retries} tries: {last_err!r}")


def client_from_env() -> LLMClient:
    base = os.environ.get("LLM_BASE_URL", "http://localhost:1234").rstrip("/")
    model = os.environ.get("LLM_MODEL", "google/gemma-3-27b")
    key = os.environ.get("LLM_API_KEY") or None
    timeout = float(os.environ.get("LLM_TIMEOUT_S", "120"))
    retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))
    return LLMClient(base_url=base, model=model, api_key=key, timeout_s=timeout, max_retries=retries)
