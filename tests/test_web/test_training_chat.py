import sys
from types import ModuleType, SimpleNamespace

from libs.web.training_chat import answer_training_question


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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))

    result = answer_training_question("How should I think about recency decay?")

    assert result["used_llm"] is False
    assert "decay_rate=0.15" in result["answer"]
    assert result["context"]["defaults"]["split_strategy"] == "timeseries_split"


def test_training_chat_uses_google_api_key_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    captured = install_fake_gemini(monkeypatch, "Use walk-forward validation.")

    result = answer_training_question("How do I avoid leakage?")

    assert captured["api_key"] == "google-key"
    assert captured["model_name"] == "gemini-1.5-pro"
    assert result["used_llm"] is True
    assert result["answer"] == "Use walk-forward validation."
