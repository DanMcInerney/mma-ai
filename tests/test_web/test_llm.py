import copy

import pytest

from libs.web.llm import llm_config, llm_generate_text


def clear_llm_env(monkeypatch):
    for name in [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "PERPLEXITY_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_local_llm_config_does_not_require_api_key(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")

    config = llm_config()

    assert config is not None
    assert config.provider == "local"
    assert config.is_configured is True
    assert config.base_url == "http://host.docker.internal:11434/v1"


def test_custom_llm_config_accepts_keyless_openai_compatible_endpoint(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "custom")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")

    config = llm_config()

    assert config is not None
    assert config.provider == "custom"
    assert config.api_key is None
    assert config.is_configured is True
    assert config.base_url == "http://localhost:1234/v1"


def test_llm_base_url_without_provider_selects_keyless_custom_endpoint(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")

    config = llm_config()

    assert config is not None
    assert config.provider == "custom"
    assert config.api_key is None
    assert config.is_configured is True


@pytest.mark.parametrize(
    ("provider", "expected_provider", "expected_base_url", "requires_key"),
    [
        ("google", "google", None, True),
        ("gemini", "google", None, True),
        ("openai", "openai", "https://api.openai.com/v1", True),
        ("codex", "codex", "https://api.openai.com/v1", True),
        ("anthropic", "anthropic", None, True),
        ("claude", "anthropic", None, True),
        ("grok", "grok", "https://api.x.ai/v1", True),
        ("xai", "grok", "https://api.x.ai/v1", True),
        ("openrouter", "openrouter", "https://openrouter.ai/api/v1", True),
        ("open-router", "openrouter", "https://openrouter.ai/api/v1", True),
        ("deepseek", "deepseek", "https://api.deepseek.com", True),
        ("mistral", "mistral", "https://api.mistral.ai/v1", True),
        ("together", "together", "https://api.together.ai/v1", True),
        ("perplexity", "perplexity", "https://api.perplexity.ai", True),
        ("sonar", "perplexity", "https://api.perplexity.ai", True),
        ("local", "local", "http://host.docker.internal:11434/v1", False),
        ("ollama", "local", "http://host.docker.internal:11434/v1", False),
        ("lm-studio", "local", "http://host.docker.internal:11434/v1", False),
        ("custom", "custom", "http://localhost:1234/v1", False),
        ("openai-compatible", "custom", "http://localhost:1234/v1", False),
    ],
)
def test_setup_provider_choices_match_runtime_config(
    monkeypatch,
    provider,
    expected_provider,
    expected_base_url,
    requires_key,
):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "unit-test-model")
    if expected_provider == "custom":
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    if requires_key:
        monkeypatch.setenv("LLM_API_KEY", "unit-test-key")

    config = llm_config()

    assert config is not None
    assert config.provider == expected_provider
    assert config.model == "unit-test-model"
    assert config.is_configured is True
    assert config.needs_api_key is requires_key
    assert config.base_url == expected_base_url


@pytest.mark.parametrize(
    ("env_var", "expected_provider", "expected_model"),
    [
        ("OPENROUTER_API_KEY", "openrouter", "~openai/gpt-latest"),
        ("DEEPSEEK_API_KEY", "deepseek", "deepseek-chat"),
        ("MISTRAL_API_KEY", "mistral", "mistral-large-latest"),
        ("TOGETHER_API_KEY", "together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("PERPLEXITY_API_KEY", "perplexity", "sonar-pro"),
    ],
)
def test_hosted_openai_compatible_provider_keys_are_auto_detected(monkeypatch, env_var, expected_provider, expected_model):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv(env_var, "provider-key")

    config = llm_config()

    assert config is not None
    assert config.provider == expected_provider
    assert config.model == expected_model
    assert config.api_key == "provider-key"
    assert config.is_configured is True


def test_local_openai_compatible_json_mode_retries_without_response_format(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    calls = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("bad request")

        def json(self):
            return {"choices": [{"message": {"content": '{"answer":"ok","sql":"select 1"}'}}]}

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": copy.deepcopy(json), "timeout": timeout})
        return FakeResponse(400 if len(calls) == 1 else 200)

    monkeypatch.setattr("libs.web.llm.requests.post", fake_post)

    text = llm_generate_text({"task": "json please"}, json_mode=True)

    assert text == '{"answer":"ok","sql":"select 1"}'
    assert calls[0]["json"]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]["json"]
    assert "Authorization" not in calls[1]["headers"]


def test_openai_compatible_provider_error_body_raises_runtime_error(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "unit-test-model")
    monkeypatch.setenv("LLM_API_KEY", "unit-test-key")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"error": {"message": "No endpoints found for unit-test-model"}}

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setattr("libs.web.llm.requests.post", fake_post)

    with pytest.raises(RuntimeError, match="No endpoints found for unit-test-model"):
        llm_generate_text({"task": "json please"}, json_mode=True)


def test_openai_compatible_missing_choices_raises_runtime_error(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "unit-test-model")
    monkeypatch.setenv("LLM_API_KEY", "unit-test-key")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "chatcmpl-test", "object": "chat.completion"}

    monkeypatch.setattr("libs.web.llm.requests.post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="response did not include chat completion content"):
        llm_generate_text({"task": "json please"}, json_mode=True)
