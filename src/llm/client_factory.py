from __future__ import annotations

import os

from src.llm.anthropic_client import AnthropicClient, AnthropicConfig
from src.llm.openai_compat_client import OpenAICompatClient, OpenAICompatConfig
from src.llm.registry import PROVIDER_SPECS


def build_json_llm_client(provider: str, model: str | None = None) -> tuple[object, object]:
    spec = PROVIDER_SPECS.get(provider)
    if not spec:
        raise RuntimeError(f"Unknown provider '{provider}'.")

    api_key = os.getenv(f"{spec.env_prefix}_API_KEY", "").strip()
    if provider != "ollama" and not api_key:
        raise RuntimeError(f"Provider '{provider}' is not configured in environment variables.")

    base_url = os.getenv(f"{spec.env_prefix}_BASE_URL", "").strip() or spec.default_base_url
    resolved_model = model or os.getenv(f"{spec.env_prefix}_MODEL", "").strip() or spec.default_model
    if not resolved_model or not base_url:
        raise RuntimeError(f"Provider '{provider}' is missing model/base URL configuration.")

    if provider == "anthropic":
        config = AnthropicConfig.from_spec(
            spec=spec,
            api_key=api_key,
            model=resolved_model,
            base_url=base_url,
        )
        return config, AnthropicClient(config)

    config = OpenAICompatConfig(
        provider=provider,
        api_key=api_key,
        model=resolved_model,
        base_url=base_url.rstrip("/"),
    )
    return config, OpenAICompatClient(config)
