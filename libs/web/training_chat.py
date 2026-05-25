"""Training-process chat support for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from libs.paths import PROJECT_ROOT
from libs.web.evaluations import summarize_model_evaluation
from libs.web.llm import google_api_key
from libs.web.services import get_dashboard_defaults


def answer_training_question(question: str, model_path: str | None = None) -> dict[str, Any]:
    context = _training_context(model_path)
    api_key = google_api_key()
    if api_key:
        return _ask_gemini(question, context, api_key)
    return {
        "answer": _fallback_answer(question, context),
        "used_llm": False,
        "context": context,
    }


def _training_context(model_path: str | None = None) -> dict[str, Any]:
    docs = {}
    for name in ("AGENTS.md", "CLAUDE.md", "TRAIN_ARCHITECTURE.md"):
        path = PROJECT_ROOT / name
        if path.exists():
            docs[name] = path.read_text(encoding="utf-8", errors="replace")[:8000]

    evaluation = None
    try:
        evaluation = summarize_model_evaluation(model_path)
    except Exception as exc:
        evaluation = {"available": False, "message": str(exc)}

    return {
        "defaults": get_dashboard_defaults()["train"],
        "evaluation": evaluation,
        "docs": docs,
    }


def _ask_gemini(question: str, context: dict[str, Any], api_key: str) -> dict[str, Any]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = {
        "task": "Answer a user question about this MMA AI model training workflow. Be specific, concise, and preserve time-ordering and leakage cautions.",
        "question": question,
        "context": context,
    }
    response = model.generate_content(json.dumps(prompt))
    return {"answer": response.text, "used_llm": True, "context": {"evaluation": context.get("evaluation"), "defaults": context.get("defaults")}}


def _fallback_answer(question: str, context: dict[str, Any]) -> str:
    defaults = context["defaults"]
    evaluation = context.get("evaluation") or {}
    metric_bits = []
    holdout = ((evaluation.get("metrics") or {}).get("holdout_predictions") or {})
    if holdout:
        metric_bits.append(f"holdout accuracy {holdout.get('accuracy', 'N/A')}")
        metric_bits.append(f"log loss {holdout.get('log_loss', 'N/A')}")
        metric_bits.append(f"Brier score {holdout.get('brier_score', 'N/A')}")

    lower_question = question.lower()
    if "feature" in lower_question:
        feature_note = "Top feature importance is available in evals.txt when training ran with calculate_importance enabled."
    elif "split" in lower_question or "leak" in lower_question:
        feature_note = "Use time-series or walk-forward splits for realistic validation; never let a fight use features derived from later event dates."
    elif "recency" in lower_question or "decay" in lower_question:
        feature_note = f"The default recency weighting is enabled with decay_rate={defaults['decay_rate']}."
    else:
        feature_note = "The default training path mirrors libs/modeling/train.py and keeps advanced knobs collapsed unless you need to experiment."

    metric_sentence = f" Current loaded metrics: {', '.join(metric_bits)}." if metric_bits else " No holdout prediction metrics are loaded yet."
    return (
        f"{feature_note} Defaults: target={defaults['model_type']}, preset={defaults['preset']}, "
        f"split={defaults['split_strategy']}, time_limit={defaults['time_limit']} seconds, "
        f"normalize={defaults['normalize']}, min_fights={defaults['num_fights']}.{metric_sentence}"
    )
