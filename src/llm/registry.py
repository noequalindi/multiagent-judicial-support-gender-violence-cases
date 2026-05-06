from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    label: str
    env_prefix: str
    default_base_url: str
    default_model: str
    default_models: tuple[str, ...]


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        provider="anthropic",
        label="Anthropic Claude",
        env_prefix="ANTHROPIC",
        default_base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-4-6",
        default_models=("claude-sonnet-4-6", "claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest"),
    ),
    "openai": ProviderSpec(
        provider="openai",
        label="OpenAI",
        env_prefix="OPENAI",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-5-mini",
        default_models=("gpt-5-mini", "gpt-5.4"),
    ),
    "deepseek": ProviderSpec(
        provider="deepseek",
        label="DeepSeek",
        env_prefix="DEEPSEEK",
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        default_models=("deepseek-chat", "deepseek-reasoner"),
    ),
    "huggingface": ProviderSpec(
        provider="huggingface",
        label="Hugging Face Router",
        env_prefix="HUGGINGFACE",
        default_base_url="https://router.huggingface.co/v1",
        default_model="deepseek-ai/DeepSeek-V3",
        default_models=("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"),
    ),
    "ollama": ProviderSpec(
        provider="ollama",
        label="Ollama local",
        env_prefix="OLLAMA",
        default_base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b-instruct",
        default_models=("qwen2.5:7b-instruct", "qwen2.5:14b-instruct", "qwen3:8b"),
    ),
}


def _csv_env(name: str, defaults: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(defaults)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or list(defaults)


def provider_options() -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for key, spec in PROVIDER_SPECS.items():
        api_key = os.getenv(f"{spec.env_prefix}_API_KEY", "").strip()
        base_url = os.getenv(f"{spec.env_prefix}_BASE_URL", "").strip() or spec.default_base_url
        default_model = os.getenv(f"{spec.env_prefix}_MODEL", "").strip() or spec.default_model
        if key == "ollama":
            models = _csv_env(f"{spec.env_prefix}_AVAILABLE_MODELS", spec.default_models)
            configured = bool(base_url)
        else:
            models = _csv_env(f"{spec.env_prefix}_AVAILABLE_MODELS", spec.default_models)
            configured = bool(api_key)
        options.append(
            {
                "provider": key,
                "label": spec.label,
                "configured": configured,
                "base_url": base_url,
                "default_model": default_model,
                "models": models,
            }
        )
    return options


def ocr_options() -> dict[str, object]:
    ollama_models = _csv_env(
        "VISION_MODELS",
        tuple(_csv_env("OLLAMA_AVAILABLE_MODELS", ("minicpm-v:8b", "mistral-small3.1", "qwen2.5-vl:7b", "qwen3-vl:8b"))),
    )
    default_ollama_model = (
        os.getenv("VISION_OLLAMA_MODEL", "").strip()
        or os.getenv("OLLAMA_VISION_MODEL", "").strip()
        or (ollama_models[0] if ollama_models else "minicpm-v:8b")
    )
    return {
        "backends": [
            {"value": "tesseract", "label": "Tesseract"},
            {"value": "hybrid", "label": "Híbrido (Tesseract + Ollama fallback)"},
            {"value": "ollama", "label": "Modelo visual (Ollama)"},
        ],
        "tesseract_lang": os.getenv("TESSERACT_LANG", "spa").strip() or "spa",
        "ollama_models": ollama_models,
        "default_backend": os.getenv("OCR_BACKEND", "tesseract").strip() or "tesseract",
        "default_ollama_model": default_ollama_model,
    }
