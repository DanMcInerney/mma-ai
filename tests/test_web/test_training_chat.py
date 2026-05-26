import sys
from types import ModuleType, SimpleNamespace

from libs.web.training_chat import answer_training_question


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


def install_fake_gemini(monkeypatch, response_text: str) -> dict:
    captured = {}
    google_module = ModuleType("google")
    genai_module = ModuleType("google.generativeai")

    def configure(api_key):
        captured["api_key"] = api_key

    class FakeModel:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def generate_content(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(text=response_text)

    genai_module.configure = configure
    genai_module.GenerativeModel = FakeModel
    google_module.generativeai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai_module)
    return captured


def test_training_chat_fallback_uses_defaults(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))

    result = answer_training_question("How should I think about recency decay?")

    assert result["used_llm"] is False
    assert "decay_rate=0.15" in result["answer"]
    assert result["context"]["defaults"]["split_strategy"] == "timeseries_split"


def test_training_chat_uses_google_api_key_alias(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    captured = install_fake_gemini(monkeypatch, "Use walk-forward validation.")

    result = answer_training_question("How do I avoid leakage?")

    assert captured["api_key"] == "google-key"
    assert captured["model_name"] == "gemini-1.5-pro"
    assert result["used_llm"] is True
    assert result["answer"] == "Use walk-forward validation."


def test_training_chat_uses_openai_compatible_llm(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_MODEL", "coach-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Use time-aware splits."}}]}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("libs.web.llm.requests.post", fake_post)

    result = answer_training_question("How do I avoid leakage?")

    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer openai-key"
    assert captured["json"]["model"] == "coach-model"
    assert result["used_llm"] is True
    assert result["answer"] == "Use time-aware splits."
