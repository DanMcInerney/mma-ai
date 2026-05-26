from pathlib import Path

import pandas as pd

from libs.web.evaluations import cli, summarize_model_evaluation, write_model_evaluation_report
from scripts.evaluate_model import main as evaluate_model_main


def write_model_artifacts(model_dir: Path):
    model_dir.mkdir(parents=True)
    (model_dir / "feats.txt").write_text("feature_a\nfeature_b\n", encoding="utf-8")
    (model_dir / "evals.txt").write_text(
        "Model Performance:\n"
        "Training accuracy: 0.8123\n"
        "Training log loss: 0.4321\n"
        "Validation accuracy: 0.7010\n"
        "Validation log loss: 0.6120\n"
        "Holdout accuracy: 0.6667\n"
        "Holdout log loss: 0.6500\n"
        "\nBest Model: WeightedEnsemble_L2\n"
        "\nModel weights:\n"
        "LightGBM: 0.700\n"
        "CatBoost: 0.300\n"
        "\nTop Most Important Features:\n"
        "1. feature_a: 0.4100\n"
        "2. feature_b: 0.1200\n"
        "\nConfiguration:\n"
        "Model Type: win\n"
        "Split Strategy: timeseries_split\n",
        encoding="utf-8",
    )
    (model_dir / "model_stats.txt").write_text(
        'Comparison:\n{"mma_ai_performance": {"accuracy": 0.67}, "vegas_odds_performance": {"accuracy": 0.55}}\n\n',
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "fighter1_name": ["a", "b", "c", "d"],
            "fighter2_name": ["e", "f", "g", "h"],
            "y_pred_proba": [0.8, 0.7, 0.2, 0.4],
            "y_pred": [1, 1, 0, 0],
            "y_true": [1, 0, 0, 1],
            "event_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        }
    ).to_csv(model_dir / "test_predictions.csv", index=False)


def test_summarize_model_evaluation_parses_metrics_and_charts(monkeypatch, tmp_path):
    models_root = tmp_path / "models"
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_root))
    model_dir = models_root / "ag-test-win"
    write_model_artifacts(model_dir)

    summary = summarize_model_evaluation(model_dir)

    assert summary["available"] is True
    assert summary["metrics"]["training"]["accuracy"] == 0.8123
    assert summary["metrics"]["holdout_predictions"]["samples"] == 4
    assert summary["metrics"]["holdout_predictions"]["accuracy"] == 0.5
    assert summary["metrics"]["best_model"] == "WeightedEnsemble_L2"
    assert summary["feature_importance"][0]["feature"] == "feature_a"
    assert summary["model_weights"][0]["model"] == "LightGBM"
    assert set(summary["charts"]) == {"calibration", "confidence", "outcomes"}
    checks = {check["name"]: check for check in summary["best_practices"]}
    assert checks["Holdout coverage"]["status"] == "pass"
    assert checks["Generalization gap"]["status"] == "warn"
    assert checks["Brier score"]["status"] == "pass"
    assert checks["Artifact completeness"]["status"] == "pass"


def test_summarize_model_evaluation_warns_when_core_artifacts_missing(monkeypatch, tmp_path):
    models_root = tmp_path / "models"
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_root))
    model_dir = models_root / "ag-partial"
    model_dir.mkdir(parents=True)
    (model_dir / "feats.txt").write_text("feature_a\n", encoding="utf-8")

    summary = summarize_model_evaluation(model_dir)

    checks = {check["name"]: check for check in summary["best_practices"]}
    assert checks["Holdout coverage"]["status"] == "warn"
    assert checks["Artifact completeness"]["status"] == "warn"
    assert "evals.txt" in checks["Artifact completeness"]["detail"]


def test_summarize_model_evaluation_normalizes_test_and_val_metrics(monkeypatch, tmp_path):
    models_root = tmp_path / "models"
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_root))
    model_dir = models_root / "ag-test-labels"
    model_dir.mkdir(parents=True)
    (model_dir / "feats.txt").write_text("feature_a\n", encoding="utf-8")
    (model_dir / "evals.txt").write_text(
        "Train accuracy: 0.8000\n"
        "Train log loss: 0.4000\n"
        "Val accuracy: 0.7000\n"
        "Val log loss: 0.5500\n"
        "Test accuracy: 0.6500\n"
        "Test log loss: 0.6200\n"
        "Test brier score: 0.2100\n",
        encoding="utf-8",
    )

    summary = summarize_model_evaluation(model_dir)

    assert summary["metrics"]["training"]["accuracy"] == 0.8
    assert summary["metrics"]["validation"]["log_loss"] == 0.55
    assert summary["metrics"]["holdout"]["accuracy"] == 0.65
    assert summary["metrics"]["holdout"]["brier_score"] == 0.21


def test_summarize_model_evaluation_uses_latest_model_from_env(monkeypatch, tmp_path):
    models_root = tmp_path / "models"
    older = models_root / "ag-older"
    newer = models_root / "ag-newer"
    write_model_artifacts(older)
    write_model_artifacts(newer)
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_root))

    summary = summarize_model_evaluation()

    assert summary["model_name"] in {"ag-older", "ag-newer"}
    assert summary["available"] is True


def test_evaluation_cli_writes_json_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    model_dir = tmp_path / "models" / "ag-cli"
    output_path = tmp_path / "eval-summary.json"
    write_model_artifacts(model_dir)

    exit_code = cli(["--model-path", str(model_dir), "--output-json", str(output_path)])

    assert exit_code == 0
    assert '"model_name": "ag-cli"' in output_path.read_text(encoding="utf-8")
    assert '"Holdout coverage"' in capsys.readouterr().out


def test_write_model_evaluation_report_creates_readable_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    model_dir = tmp_path / "models" / "ag-report"
    write_model_artifacts(model_dir)
    summary = summarize_model_evaluation(model_dir)

    report_paths = write_model_evaluation_report(summary)

    json_report = Path(report_paths["json"])
    markdown_report = Path(report_paths["markdown"])
    assert json_report.exists()
    assert markdown_report.exists()
    assert "dashboard_evaluation_summary.json" == json_report.name
    markdown = markdown_report.read_text(encoding="utf-8")
    assert "Model Evaluation Report: ag-report" in markdown
    assert "Holdout Accuracy" in markdown
    assert "Holdout Log Loss" in markdown
    assert "Brier Score" in markdown


def test_evaluation_cli_can_write_markdown_and_text_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    model_dir = tmp_path / "models" / "ag-cli-report"
    markdown_path = tmp_path / "eval.md"
    write_model_artifacts(model_dir)

    exit_code = cli([
        "--model-path",
        str(model_dir),
        "--output-markdown",
        str(markdown_path),
        "--write-report",
        "--format",
        "text",
    ])

    assert exit_code == 0
    assert "Model Evaluation Report: ag-cli-report" in markdown_path.read_text(encoding="utf-8")
    assert "Best-Practice Checks" in capsys.readouterr().out
    assert (model_dir / "dashboard_evaluation_summary.json").exists()
    assert (model_dir / "dashboard_evaluation.md").exists()


def test_evaluate_model_script_wrapper_matches_console_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    model_dir = tmp_path / "models" / "ag-script-wrapper"
    write_model_artifacts(model_dir)

    exit_code = evaluate_model_main(["--model-path", str(model_dir), "--format", "text"])

    assert exit_code == 0
    assert "Model Evaluation Report: ag-script-wrapper" in capsys.readouterr().out
