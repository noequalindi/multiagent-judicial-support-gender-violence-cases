from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from src.llm.registry import PROVIDER_SPECS


@dataclass
class OpenAICompatConfig:
    provider: str
    api_key: str
    model: str
    base_url: str

    @classmethod
    def from_provider(cls, provider: str, model: str | None = None) -> "OpenAICompatConfig | None":
        spec = PROVIDER_SPECS.get(provider)
        if not spec:
            return None
        key_prefix = spec.env_prefix
        api_key = os.getenv(f"{key_prefix}_API_KEY", "").strip()
        if not api_key:
            return None

        base_url = os.getenv(f"{key_prefix}_BASE_URL", "").strip() or spec.default_base_url
        resolved_model = model or os.getenv(f"{key_prefix}_MODEL", "").strip() or spec.default_model
        if not resolved_model or not base_url:
            return None
        return cls(provider=provider, api_key=api_key, model=resolved_model, base_url=base_url.rstrip("/"))


class OpenAICompatClient:
    def __init__(self, config: OpenAICompatConfig):
        self.config = config

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        thinking_budget_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = self._post_chat(payload, extra_headers=extra_headers)
        content = self._extract_content(data)
        if content is None:
            # Some providers/models reject response_format. Retry with prompt-only JSON instructions.
            fallback_payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if max_tokens is not None:
                fallback_payload["max_tokens"] = max_tokens
            data = self._post_chat(fallback_payload, extra_headers=extra_headers)
            content = self._extract_content(data)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Provider {self.config.provider} returned empty content.")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                raise RuntimeError(f"Provider {self.config.provider} did not return valid JSON.")
            return json.loads(match.group(0))

    def _post_chat(self, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    **(extra_headers or {}),
                },
                json=payload,
            )
        if response.is_error:
            try:
                detail = response.json()
            except Exception:
                detail = response.text

            # Allow caller to retry without response_format when model/provider rejects it.
            if response.status_code == 400 and "response_format" in str(detail):
                return {}
            if response.status_code == 400 and "temperature" in str(detail):
                retry_payload = dict(payload)
                retry_payload.pop("temperature", None)
                with httpx.Client(timeout=120.0) as client:
                    retry = client.post(
                        f"{self.config.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                            **(extra_headers or {}),
                        },
                        json=retry_payload,
                    )
                if retry.is_error:
                    try:
                        retry_detail = retry.json()
                    except Exception:
                        retry_detail = retry.text
                    raise RuntimeError(
                        f"{self.config.provider} API error {retry.status_code}: {retry_detail}"
                    )
                return retry.json()

            raise RuntimeError(
                f"{self.config.provider} API error {response.status_code}: {detail}"
            )
        return response.json()

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str | None:
        if not data:
            return None
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
