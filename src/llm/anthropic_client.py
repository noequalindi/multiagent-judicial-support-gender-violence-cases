from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from src.llm.registry import ProviderSpec


@dataclass
class AnthropicConfig:
    provider: str
    api_key: str
    model: str
    base_url: str
    api_version: str = "2023-06-01"

    @classmethod
    def from_spec(
        cls,
        spec: ProviderSpec,
        api_key: str,
        model: str,
        base_url: str,
    ) -> "AnthropicConfig":
        return cls(
            provider=spec.provider,
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
        )


class AnthropicClient:
    def __init__(self, config: AnthropicConfig):
        self.config = config

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        extra_headers: dict[str, str] | None = None,
        thinking_budget_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if thinking_budget_tokens:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }
        else:
            payload["temperature"] = temperature

        data = self._post_messages(payload, extra_headers=extra_headers)
        content = self._extract_text(data)
        if not content.strip():
            raise RuntimeError(f"Provider {self.config.provider} returned empty content.")
        return content

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        extra_headers: dict[str, str] | None = None,
        thinking_budget_tokens: int | None = None,
    ) -> dict[str, Any]:
        content = self.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        return self._parse_json_content(content)

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        cleaned = self._strip_code_fences(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as initial_error:
            extracted = self._extract_json_object(cleaned)
            if extracted:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass

            repaired = self._repair_json(cleaned)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(
                f"Provider {self.config.provider} did not return valid JSON: {initial_error}"
            ) from initial_error

    def _post_messages(self, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.config.base_url}/messages",
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": self.config.api_version,
                    "content-type": "application/json",
                    **(extra_headers or {}),
                },
                json=payload,
            )
        if response.is_error:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"{self.config.provider} API error {response.status_code}: {detail}"
            )
        return response.json()

    def _repair_json(self, raw_content: str) -> str | None:
        system_prompt = (
            "Convertí la salida recibida en un JSON válido. "
            "No agregues explicación. No inventes contenido. "
            "Preservá las mismas claves y valores en la medida de lo posible. "
            "Respondé solo el JSON corregido."
        )
        user_prompt = (
            "Salida a corregir:\n\n"
            f"{raw_content}"
        )
        try:
            repaired = self.chat_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception:
            return None
        cleaned = self._strip_code_fences(repaired)
        return self._extract_json_object(cleaned) or cleaned

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        stripped = content.strip()
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    @staticmethod
    def _extract_json_object(content: str) -> str | None:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return None
        return match.group(0).strip()

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        content = data.get("content", [])
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
