import importlib.util

import pytest

from libs.modeling import runtime_dependencies


def test_prediction_runtime_dependency_report_names_missing_imports(monkeypatch):
    def fake_find_spec(module):
        if module in {"loguru", "tabpfn"}:
            return None
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    report = runtime_dependencies.prediction_runtime_dependency_report()

    assert report["ok"] is False
    assert [item["module"] for item in report["missing"]] == ["loguru", "tabpfn"]
    assert report["missing"][0]["reason"] == "MITRA model loading"


def test_assert_prediction_runtime_dependencies_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(
        runtime_dependencies,
        "missing_prediction_runtime_imports",
        lambda: [
            runtime_dependencies.RuntimeImport("loguru", "loguru", "MITRA model loading"),
            runtime_dependencies.RuntimeImport("tabicl", "tabicl", "TABICL model family"),
        ],
    )

    with pytest.raises(RuntimeError, match="Missing modules: loguru, tabicl"):
        runtime_dependencies.assert_prediction_runtime_dependencies()
