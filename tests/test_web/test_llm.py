import copy

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
