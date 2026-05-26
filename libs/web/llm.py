"""Small helpers for optional dashboard LLM integrations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import requests


GOOGLE_API_KEY_ENV_VARS = ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
OPENAI_API_KEY_ENV_VARS = ("LLM_API_KEY", "OPENAI_API_KEY")
ANTHROPIC_API_KEY_ENV_VARS = ("LLM_API_KEY", "ANTHROPIC_API_KEY")
GROK_API_KEY_ENV_VARS = ("LLM_API_KEY", "XAI_API_KEY", "GROK_API_KEY")
LOCAL_API_KEY_ENV_VARS = ("LLM_API_KEY",)

DEFAULT_MODELS = {
    "google": "gemini-1.5-pro",
    "openai": "gpt-4o-mini",
    "codex": "gpt-5-codex",
    "anthropic": "claude-3-5-sonnet-latest",
    "grok": "grok-2-latest",
    "local": "llama3.1",
    "custom": "gpt-4o-mini",
}

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "codex": "https://api.openai.com/v1",
    "grok": "https://api.x.ai/v1",
    "local": "http://host.docker.internal:11434/v1",
}

PROVIDER_ALIASES = {
    "gemini": "google",
    "google": "google",
    "openai": "openai",
    "codex": "codex",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "grok": "grok",
    "xai": "grok",
    "local": "local",
    "ollama": "local",
    "lmstudio": "local",
    "lm-studio": "local",
    "custom": "custom",
    "openai-compatible": "custom",
}


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 90.0

    @property
    def needs_api_key(self) -> bool:
        return self.provider not in {"local", "custom"}

    @property
    def is_configured(self) -> bool:
        if not self.provider or not self.model:
            return False
        if self.needs_api_key and not self.api_key:
            return False
        if self.provider == "custom" and not self.base_url:
            return False
        return True


def google_api_key() -> str | None:
    for env_var in GOOGLE_API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value
    return None


def google_api_key_hint() -> str:
    return "Connect LLM_PROVIDER/LLM_MODEL plus an API key, or legacy GEMINI_API_KEY/GOOGLE_API_KEY"


def llm_config() -> LlmConfig | None:
    provider = _configured_provider()
    if not provider:
        return None

    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")
    base_url = os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URLS.get(provider)
    api_key = _provider_api_key(provider)
    timeout_seconds = _timeout_seconds()
    return LlmConfig(provider=provider, model=model, api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)


def llm_config_hint() -> str:
    return (
        "Configure LLM_PROVIDER and LLM_MODEL with LLM_API_KEY, provider keys like "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY, "
        "or set LLM_BASE_URL for a local/custom OpenAI-compatible endpoint."
    )


def llm_generate_text(prompt: dict[str, Any] | str, *, json_mode: bool = False) -> str:
    config = llm_config()
    if not config or not config.is_configured:
        raise RuntimeError(llm_config_hint())

    if config.provider == "google":
        return _generate_google(prompt, config)
    if config.provider == "anthropic":
        return _generate_anthropic(prompt, config, json_mode=json_mode)
    return _generate_openai_compatible(prompt, config, json_mode=json_mode)


def _configured_provider() -> str | None:
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return PROVIDER_ALIASES.get(explicit.strip().lower(), explicit.strip().lower())

    if _first_env(("GEMINI_API_KEY", "GOOGLE_API_KEY")):
        return "google"
    if _first_env(OPENAI_API_KEY_ENV_VARS):
        return "openai"
    if _first_env(ANTHROPIC_API_KEY_ENV_VARS):
        return "anthropic"
    if _first_env(GROK_API_KEY_ENV_VARS):
        return "grok"
    if os.getenv("LLM_BASE_URL"):
        return "custom"
    return None


def _provider_api_key(provider: str) -> str | None:
    if provider == "google":
        return _first_env(GOOGLE_API_KEY_ENV_VARS)
    if provider in {"openai", "codex"}:
        return _first_env(OPENAI_API_KEY_ENV_VARS)
    if provider == "anthropic":
        return _first_env(ANTHROPIC_API_KEY_ENV_VARS)
    if provider == "grok":
        return _first_env(GROK_API_KEY_ENV_VARS)
    return _first_env(LOCAL_API_KEY_ENV_VARS)


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _timeout_seconds() -> float:
    raw = os.getenv("LLM_TIMEOUT_SECONDS", "90")
    try:
        return float(raw)
    except ValueError:
        return 90.0


def _prompt_text(prompt: dict[str, Any] | str) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt)


def _generate_google(prompt: dict[str, Any] | str, config: LlmConfig) -> str:
    import google.generativeai as genai

    genai.configure(api_key=config.api_key)
    model = genai.GenerativeModel(config.model)
    response = model.generate_content(_prompt_text(prompt))
    return response.text


def _generate_openai_compatible(prompt: dict[str, Any] | str, config: LlmConfig, *, json_mode: bool) -> str:
    base_url = (config.base_url or DEFAULT_BASE_URLS["openai"]).rstrip("/")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "You support an MMA analytics dashboard. Be concise, factual, and respect read-only data constraints.",
            },
            {"role": "user", "content": _prompt_text(prompt)},
        ],
        "temperature": 0.2,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=config.timeout_seconds,
    )
    if getattr(response, "status_code", 0) >= 400 and json_mode and config.provider in {"local", "custom"}:
        # Some local OpenAI-compatible servers do not implement response_format.
        payload.pop("response_format", None)
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=config.timeout_seconds,
        )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _generate_anthropic(prompt: dict[str, Any] | str, config: LlmConfig, *, json_mode: bool) -> str:
    system_prompt = "You support an MMA analytics dashboard. Be concise, factual, and respect read-only data constraints."
    if json_mode:
        system_prompt += " Return strict JSON only."

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": config.api_key or "",
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": config.model,
            "max_tokens": 4096,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": _prompt_text(prompt)}],
        },
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    chunks = data.get("content") or []
    return "".join(chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text")
