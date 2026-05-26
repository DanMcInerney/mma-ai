"""Model evaluation artifact summaries for the web dashboard."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from libs.paths import models_dir
from libs.web.path_safety import resolve_model_dir


METRIC_LINE = re.compile(
    r"^(?P<label>Training|Train|Validation|Val|Holdout|Test)\s+"
    r"(?P<metric>accuracy|log loss|brier score):\s+(?P<value>-?\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
REPORT_JSON_NAME = "dashboard_evaluation_summary.json"
REPORT_MARKDOWN_NAME = "dashboard_evaluation.md"


def summarize_model_evaluation(model_path: str | Path | None = None) -> dict[str, Any]:
    """Summarize evaluation files produced by the training pipeline."""
    path = _resolve_model_path(model_path)
    if path is None:
        return {
            "model_path": None,
            "available": False,
            "message": "No model directory found.",
            "metrics": {},
            "charts": {},
            "artifacts": [],
            "feature_importance": [],
            "best_practices": [],
        }

    artifacts = _collect_artifacts(path)
    eval_text = _read_text(path / "evals.txt")
    model_stats_text = _read_text(path / "model_stats.txt")
    report_text = _read_text(path / "report.txt")

    predictions = _load_predictions(path)
    prediction_summary = _summarize_predictions(predictions) if predictions is not None else {}

    metrics = {
        **_parse_evals_metrics(eval_text),
        **({"holdout_predictions": prediction_summary["metrics"]} if prediction_summary else {}),
        **_parse_model_stats(model_stats_text),
        **_parse_walkforward_report(report_text),
    }

    return {
        "model_path": str(path),
        "model_name": path.name,
        "available": True,
        "metrics": metrics,
        "charts": prediction_summary.get("charts", {}) if prediction_summary else {},
        "artifacts": artifacts,
        "feature_importance": _parse_feature_importance(eval_text),
        "model_weights": _parse_model_weights(eval_text),
        "configuration": _parse_configuration(eval_text) or _parse_configuration(report_text),
        "notes": _build_notes(path, artifacts, predictions),
        "best_practices": _build_best_practice_checks(metrics, artifacts, predictions),
    }


def write_model_evaluation_report(summary: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, str]:
    """Write dashboard-friendly evaluation artifacts beside a trained model."""
    if not summary.get("available"):
        return {}

    target_value = output_dir or summary.get("model_path")
    if not target_value:
        return {}

    target_dir = Path(target_value)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_paths = {
        "json": str(target_dir / REPORT_JSON_NAME),
        "markdown": str(target_dir / REPORT_MARKDOWN_NAME),
    }
    report_summary = {**summary, "report_paths": report_paths}
    Path(report_paths["json"]).write_text(json.dumps(report_summary, indent=2) + "\n", encoding="utf-8")
    Path(report_paths["markdown"]).write_text(_format_markdown_summary(report_summary) + "\n", encoding="utf-8")
    return report_paths


def _resolve_model_path(model_path: str | Path | None) -> Path | None:
    if model_path:
        return resolve_model_dir(model_path)

    root = models_dir()
    if not root.exists():
        return None
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and any((path / marker).exists() for marker in ("feats.txt", "evals.txt", "model_stats.txt", "test_predictions.csv", "all_predictions.csv"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _collect_artifacts(model_path: Path) -> list[dict[str, Any]]:
    artifact_names = [
        "evals.txt",
        "model_stats.txt",
        "report.txt",
        "test_predictions.csv",
        "all_predictions.csv",
        "fold_results.csv",
        "calibration_curve.png",
        "feats.txt",
        "holdout_fight_ids.txt",
        REPORT_JSON_NAME,
        REPORT_MARKDOWN_NAME,
    ]
    artifacts = []
    for name in artifact_names:
        path = model_path / name
        if path.exists():
            artifacts.append({"name": name, "path": str(path), "size_bytes": path.stat().st_size})

    for child in sorted(model_path.glob("window_*/test_predictions.csv")):
        artifacts.append({"name": f"{child.parent.name}/test_predictions.csv", "path": str(child), "size_bytes": child.stat().st_size})
    return artifacts


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_evals_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw_line in text.splitlines():
        match = METRIC_LINE.match(raw_line.strip())
        if not match:
            continue
        split_name = _normalize_split_name(match.group("label"))
        metric_name = match.group("metric").lower().replace(" ", "_")
        metrics.setdefault(split_name, {})[metric_name] = float(match.group("value"))
    best_model = _find_after_label(text, "Best Model")
    if best_model:
        metrics["best_model"] = best_model
    return metrics


def _normalize_split_name(label: str) -> str:
    normalized = label.lower()
    if normalized == "train":
        return "training"
    if normalized == "val":
        return "validation"
    if normalized == "test":
        return "holdout"
    return normalized


def _parse_model_stats(text: str) -> dict[str, Any]:
    if not text:
        return {}
    marker = "Comparison:"
    if marker not in text:
        return {}
    start = text.find("{", text.find(marker))
    if start < 0:
        return {}
    depth = 0
    end = None
    for index, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        return {}
    try:
        return {"comparison": json.loads(text[start:end])}
    except json.JSONDecodeError:
        return {}


def _parse_walkforward_report(text: str) -> dict[str, Any]:
    if not text:
        return {}
    parsed: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("Mean Accuracy:"):
            parsed["walkforward_mean_accuracy"] = _first_float(line)
        elif line.startswith("Mean Log Loss:"):
            parsed["walkforward_mean_log_loss"] = _first_float(line)
        elif line.startswith("Folds:"):
            parsed["walkforward_folds"] = int(_first_float(line) or 0)
    return parsed


def _parse_feature_importance(text: str, limit: int = 25) -> list[dict[str, Any]]:
    if "Top Most Important Features:" not in text:
        return []
    lines = text.split("Top Most Important Features:", 1)[1].split("Configuration:", 1)[0].splitlines()
    features = []
    for line in lines:
        match = re.match(r"\s*\d+\.\s+(.+):\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            features.append({"feature": match.group(1), "importance": float(match.group(2))})
        if len(features) >= limit:
            break
    return features


def _parse_model_weights(text: str) -> list[dict[str, Any]]:
    if "Model weights:" not in text:
        return []
    section = text.split("Model weights:", 1)[1].split("Top Most Important Features:", 1)[0].split("Configuration:", 1)[0]
    weights = []
    for line in section.splitlines():
        if "N/A" in line or ":" not in line:
            continue
        name, value = line.rsplit(":", 1)
        try:
            weights.append({"model": name.strip(), "weight": float(value.strip())})
        except ValueError:
            continue
    return weights


def _parse_configuration(text: str) -> dict[str, str]:
    if "Configuration:" not in text:
        return {}
    config = {}
    for line in text.split("Configuration:", 1)[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = value.strip()
    return config


def _find_after_label(text: str, label: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}:\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _first_float(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _load_predictions(model_path: Path) -> pd.DataFrame | None:
    candidates = [
        model_path / "test_predictions.csv",
        model_path / "all_predictions.csv",
        model_path / "fold_results.csv",
    ]
    candidates.extend(sorted(model_path.glob("window_*/test_predictions.csv")))
    frames = []
    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if {"y_true", "y_pred_proba"}.issubset(df.columns):
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _summarize_predictions(df: pd.DataFrame) -> dict[str, Any]:
    clean = df[["y_true", "y_pred_proba"] + [col for col in ["y_pred", "event_date"] if col in df.columns]].copy()
    clean["y_true"] = pd.to_numeric(clean["y_true"], errors="coerce")
    clean["y_pred_proba"] = pd.to_numeric(clean["y_pred_proba"], errors="coerce").clip(0.000001, 0.999999)
    clean = clean.dropna(subset=["y_true", "y_pred_proba"])
    if clean.empty:
        return {}

    if "y_pred" in clean.columns:
        y_pred = pd.to_numeric(clean["y_pred"], errors="coerce").fillna((clean["y_pred_proba"] >= 0.5).astype(int))
    else:
        y_pred = (clean["y_pred_proba"] >= 0.5).astype(int)

    y_true = clean["y_true"].astype(int)
    probs = clean["y_pred_proba"]
    metrics = {
        "samples": int(len(clean)),
        "accuracy": float((y_pred.astype(int) == y_true).mean()),
        "log_loss": float((-(y_true * probs.map(math.log) + (1 - y_true) * (1 - probs).map(math.log))).mean()),
        "brier_score": float(((probs - y_true) ** 2).mean()),
        "base_rate": float(y_true.mean()),
        "mean_probability": float(probs.mean()),
    }
    metrics["roc_auc"] = _roc_auc(y_true.tolist(), probs.tolist())

    bins = _calibration_bins(clean)
    return {
        "metrics": metrics,
        "charts": {
            "calibration": _calibration_chart(bins),
            "confidence": _confidence_chart(probs),
            "outcomes": _outcomes_chart(y_true, y_pred),
        },
    }


def _roc_auc(y_true: list[int], probs: list[float]) -> float | None:
    positives = [(prob, idx) for idx, (label, prob) in enumerate(zip(y_true, probs)) if label == 1]
    negatives = [(prob, idx) for idx, (label, prob) in enumerate(zip(y_true, probs)) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos_prob, _pos_idx in positives:
        for neg_prob, _neg_idx in negatives:
            if pos_prob > neg_prob:
                wins += 1.0
            elif pos_prob == neg_prob:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _calibration_bins(df: pd.DataFrame, bins: int = 10) -> list[dict[str, Any]]:
    rows = []
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        if bin_index == bins - 1:
            mask = (df["y_pred_proba"] >= low) & (df["y_pred_proba"] <= high)
        else:
            mask = (df["y_pred_proba"] >= low) & (df["y_pred_proba"] < high)
        bucket = df[mask]
        if bucket.empty:
            continue
        rows.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "mean_probability": float(bucket["y_pred_proba"].mean()),
                "actual_rate": float(bucket["y_true"].mean()),
                "count": int(len(bucket)),
            }
        )
    return rows


def _calibration_chart(bins: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": [
            {"x": [0, 1], "y": [0, 1], "type": "scatter", "mode": "lines", "name": "Perfect", "line": {"dash": "dash", "color": "#94a3b8"}},
            {
                "x": [row["mean_probability"] for row in bins],
                "y": [row["actual_rate"] for row in bins],
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Model",
                "line": {"color": "#0f766e"},
            },
        ],
        "layout": {"title": "Calibration", "xaxis": {"title": "Predicted probability"}, "yaxis": {"title": "Actual win rate"}, "template": "plotly_white"},
    }


def _confidence_chart(probs: pd.Series) -> dict[str, Any]:
    return {
        "data": [{"x": probs.round(3).tolist(), "type": "histogram", "name": "Predictions", "marker": {"color": "#0f766e"}}],
        "layout": {"title": "Prediction Confidence Distribution", "xaxis": {"title": "Predicted probability"}, "yaxis": {"title": "Count"}, "template": "plotly_white"},
    }


def _outcomes_chart(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    correct = int((y_true.astype(int) == y_pred.astype(int)).sum())
    incorrect = int(len(y_true) - correct)
    return {
        "data": [{"x": ["Correct", "Incorrect"], "y": [correct, incorrect], "type": "bar", "marker": {"color": ["#0f766e", "#b42318"]}}],
        "layout": {"title": "Holdout Prediction Outcomes", "yaxis": {"title": "Fights"}, "template": "plotly_white"},
    }


def _build_notes(model_path: Path, artifacts: list[dict[str, Any]], predictions: pd.DataFrame | None) -> list[str]:
    names = {artifact["name"] for artifact in artifacts}
    notes = []
    if "evals.txt" not in names:
        notes.append("No evals.txt found for this model.")
    if predictions is None:
        notes.append("No holdout prediction CSV found, so charted metrics are unavailable.")
    if "calibration_curve.png" not in names:
        notes.append("No saved calibration_curve.png found.")
    if not notes:
        notes.append(f"Loaded evaluation artifacts from {model_path.name}.")
    return notes


def _build_best_practice_checks(metrics: dict[str, Any], artifacts: list[dict[str, Any]], predictions: pd.DataFrame | None) -> list[dict[str, str]]:
    names = {artifact["name"] for artifact in artifacts}
    holdout = metrics.get("holdout_predictions") or {}
    train = metrics.get("training") or {}
    validation = metrics.get("validation") or {}
    holdout_scores = metrics.get("holdout") or {}
    checks = []

    samples = holdout.get("samples")
    if samples:
        checks.append(_check("Holdout coverage", "pass", f"{samples} prediction rows are available for post-training evaluation."))
    else:
        checks.append(_check("Holdout coverage", "warn", "No holdout prediction rows were found; accuracy, calibration, and confidence charts are incomplete."))

    train_accuracy = train.get("accuracy")
    eval_accuracy = holdout.get("accuracy", holdout_scores.get("accuracy", validation.get("accuracy")))
    if train_accuracy is not None and eval_accuracy is not None:
        gap = float(train_accuracy) - float(eval_accuracy)
        status = "pass" if gap <= 0.15 else "warn"
        checks.append(_check("Generalization gap", status, f"Train accuracy exceeds evaluation accuracy by {gap:.3f}."))
    else:
        checks.append(_check("Generalization gap", "info", "Training and evaluation accuracy were not both available."))

    brier_score = holdout.get("brier_score")
    if brier_score is not None:
        status = "pass" if float(brier_score) <= 0.25 else "warn"
        checks.append(_check("Brier score", status, f"Holdout Brier score is {float(brier_score):.3f}; lower means better probability calibration."))
    else:
        checks.append(_check("Brier score", "info", "No probability CSV was available for Brier score calculation."))

    roc_auc = holdout.get("roc_auc")
    if roc_auc is not None:
        status = "pass" if float(roc_auc) >= 0.55 else "warn"
        checks.append(_check("Rank ordering", status, f"Holdout ROC AUC is {float(roc_auc):.3f}."))
    else:
        checks.append(_check("Rank ordering", "info", "ROC AUC needs both winning and losing examples in the holdout predictions."))

    base_rate = holdout.get("base_rate")
    mean_probability = holdout.get("mean_probability")
    if base_rate is not None and mean_probability is not None:
        gap = abs(float(mean_probability) - float(base_rate))
        status = "pass" if gap <= 0.10 else "warn"
        checks.append(_check("Probability base rate", status, f"Mean predicted probability differs from the holdout base rate by {gap:.3f}."))

    comparison = metrics.get("comparison") or {}
    mma_accuracy = (comparison.get("mma_ai_performance") or {}).get("accuracy")
    vegas_accuracy = (comparison.get("vegas_odds_performance") or {}).get("accuracy")
    if mma_accuracy is not None and vegas_accuracy is not None:
        status = "pass" if float(mma_accuracy) >= float(vegas_accuracy) else "warn"
        checks.append(_check("Market benchmark", status, f"Model accuracy {float(mma_accuracy):.3f} vs Vegas accuracy {float(vegas_accuracy):.3f}."))

    required_artifacts = {"evals.txt", "feats.txt"}
    missing_required = sorted(required_artifacts - names)
    if predictions is None:
        missing_required.append("test_predictions.csv or all_predictions.csv")
    if missing_required:
        checks.append(_check("Artifact completeness", "warn", f"Missing: {', '.join(missing_required)}."))
    else:
        checks.append(_check("Artifact completeness", "pass", "Core evaluation, feature, and prediction artifacts are present."))

    return checks


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _summary_metric(summary: dict[str, Any], *path: str) -> Any:
    current: Any = summary
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if abs(value) <= 1 else f"{value:.2f}"
    return str(value)


def _format_markdown_summary(summary: dict[str, Any]) -> str:
    metrics = [
        ("Samples", _summary_metric(summary, "metrics", "holdout_predictions", "samples")),
        ("Holdout Accuracy", _first_present(_summary_metric(summary, "metrics", "holdout_predictions", "accuracy"), _summary_metric(summary, "metrics", "holdout", "accuracy"))),
        ("Holdout Log Loss", _first_present(_summary_metric(summary, "metrics", "holdout_predictions", "log_loss"), _summary_metric(summary, "metrics", "holdout", "log_loss"))),
        ("Brier Score", _summary_metric(summary, "metrics", "holdout_predictions", "brier_score")),
        ("ROC AUC", _summary_metric(summary, "metrics", "holdout_predictions", "roc_auc")),
        ("Train Accuracy", _summary_metric(summary, "metrics", "training", "accuracy")),
        ("Validation Accuracy", _summary_metric(summary, "metrics", "validation", "accuracy")),
        ("Mean Probability", _summary_metric(summary, "metrics", "holdout_predictions", "mean_probability")),
    ]
    lines = [
        f"# Model Evaluation Report: {summary.get('model_name') or 'Unknown Model'}",
        "",
        f"Model path: `{summary.get('model_path') or 'N/A'}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {label} | {_format_value(value)} |" for label, value in metrics)

    checks = summary.get("best_practices") or []
    lines.extend(["", "## Best-Practice Checks", ""])
    if checks:
        lines.extend(f"- **{check['name']}** ({check['status']}): {check['detail']}" for check in checks)
    else:
        lines.append("- No best-practice checks were available.")

    features = (summary.get("feature_importance") or [])[:10]
    lines.extend(["", "## Top Features", ""])
    if features:
        lines.extend(f"{index}. `{row['feature']}`: {_format_value(row['importance'])}" for index, row in enumerate(features, start=1))
    else:
        lines.append("No feature importance artifact was found.")

    artifacts = summary.get("artifacts") or []
    lines.extend(["", "## Artifacts", ""])
    if artifacts:
        lines.extend(f"- `{artifact['name']}` ({artifact['size_bytes']} bytes)" for artifact in artifacts)
    else:
        lines.append("No evaluation artifacts were found.")

    return "\n".join(lines)


def _format_text_summary(summary: dict[str, Any]) -> str:
    markdown = _format_markdown_summary(summary)
    return re.sub(r"`([^`]+)`", r"\1", markdown.replace("# ", "").replace("## ", ""))


def cli(argv: list[str] | None = None) -> int:
    """Write a portable model-evaluation summary for scripts and setup checks."""
    import argparse

    parser = argparse.ArgumentParser(description="Summarize MMA AI model evaluation artifacts.")
    parser.add_argument("--model-path", help="Model directory to evaluate. Defaults to the latest model.")
    parser.add_argument("--output-json", help="Optional path to write the evaluation summary JSON.")
    parser.add_argument("--output-markdown", help="Optional path to write a Markdown evaluation report.")
    parser.add_argument("--write-report", action="store_true", help="Write dashboard_evaluation artifacts into the model directory.")
    parser.add_argument("--format", choices=["json", "text"], default="json", help="Console output format.")
    args = parser.parse_args(argv)

    summary = summarize_model_evaluation(args.model_path)
    if args.write_report:
        summary["report_paths"] = write_model_evaluation_report(summary)
    output = json.dumps(summary, indent=2)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(output + "\n", encoding="utf-8")
    if args.output_markdown:
        Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_markdown).write_text(_format_markdown_summary(summary) + "\n", encoding="utf-8")
    print(output if args.format == "json" else _format_text_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
