"""Family 10 preregistration and serialized outcome-decomposition execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..hashing import canonical_sha256
from ..outcome_decomposition import OutcomeDecompositionError
from .semantic_portfolio import V8_FEATURES


EXPERIMENT_ID = "family-10-outcome-decomposition"
RUN_ALIAS = "family-10-outcome-decomposition"
RUN_PATH = "runs/family-10-outcome-decomposition"
ARTIFACT_PATH = "artifacts/11-family-10-outcome-decomposition"
PROFILE_PATH = "profiles/family-10-outcome-decomposition.json"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"

VARIANT_IDS = (
    "direct-incumbent-control",
    "three-component",
    "shrinkage-gated-three-component",
    "decision-finish-specialist-mixture",
    "support-trimmed-specialist-mixture",
    "constant-prior-fallback",
)
COMPONENT_IDS = ("decision", "decision-win", "finish-win")

_MODEL = {
    "library": "catboost",
    "class": "CatBoostClassifier",
    "iterations": 300,
    "depth": 4,
    "learning_rate": 0.03,
    "l2_leaf_reg": 8.0,
    "loss_function": "Logloss",
    "random_seed": 20260815,
    "task_type": "CPU",
    "thread_count": 6,
    "allow_writing_files": False,
    "verbose": False,
}


def _variant(identifier: str, **parameters: Any) -> dict[str, Any]:
    value = {"id": identifier, **parameters}
    value["profile_sha256"] = canonical_sha256(value)
    return value


def build_preregistered_profile() -> dict[str, Any]:
    """Materialize every fit, support, prior, clipping, and fallback default."""

    variants = [
        _variant("direct-incumbent-control", kind="control", component_ids=[]),
        _variant(
            "three-component",
            kind="law-of-total-probability",
            component_ids=list(COMPONENT_IDS),
            clipping=[0.02, 0.98],
        ),
        _variant(
            "shrinkage-gated-three-component",
            kind="law-of-total-probability-shrunk",
            component_ids=list(COMPONENT_IDS),
            shrinkage_strength=80.0,
            clipping=[0.02, 0.98],
        ),
        _variant(
            "decision-finish-specialist-mixture",
            kind="decomposition-control-mixture",
            component_ids=list(COMPONENT_IDS),
            decomposition_weight=0.75,
            clipping=[0.02, 0.98],
        ),
        _variant(
            "support-trimmed-specialist-mixture",
            kind="support-trimmed-mixture",
            component_ids=list(COMPONENT_IDS),
            minimum_conditional_support=120,
            decomposition_weight_supported=0.8,
            decomposition_weight_sparse=0.35,
            clipping=[0.02, 0.98],
        ),
        _variant(
            "constant-prior-fallback",
            kind="constant-prior-decomposition",
            component_ids=list(COMPONENT_IDS),
            clipping=[0.02, 0.98],
        ),
    ]
    return {
        "experiment_id": EXPERIMENT_ID,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "source": {
            "sha256": "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA",
            "development_safe_id_count": 3_089,
            "development_max_date": "2025-12-13",
            "retired_id_count": 178,
            "safe_ids_asserted_before_label_decode": True,
        },
        "outer_folds": [2022, 2023, 2024, 2025],
        "outer_row_count": 1_108,
        "features": list(V8_FEATURES),
        "components": [
            {"id": "decision", "label": "method-contains-dec", "training_subset": "all-prior"},
            {"id": "decision-win", "label": "fighter1-win", "training_subset": "prior-decisions"},
            {"id": "finish-win", "label": "fighter1-win", "training_subset": "prior-finishes"},
        ],
        "model": deepcopy(_MODEL),
        "sample_weight": {"kind": "exponential-recency", "annual_decay": 0.15, "normalize": True},
        "fallbacks": [{"id": "constant-prior", "registered": True, "prior": 0.5}],
        "support": {"minimum_specialist_rows": 120, "sparse_action": "registered-constant-prior"},
        "gate": {"kind": "fixed", "outer_label_reads": 0},
        "selection": {
            "metric": "log_loss",
            "outer_label_component_fit_count": 0,
            "promotion_requires_full_denominator": True,
        },
        "runtime": {
            "production_process_count": 1,
            "lease_count": 1,
            "serialized": True,
            "retry_count": 0,
            "refit_full": False,
            "database_access": False,
        },
        "variants": variants,
    }


def validate_preregistered_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact bounded menu and all contamination barriers."""

    if profile.get("experiment_id") != EXPERIMENT_ID:
        raise OutcomeDecompositionError("family 10 experiment identity differs")
    variants = list(profile.get("variants", ()))
    if len(variants) > 6:
        raise OutcomeDecompositionError("family 10 may preregister at most six variants")
    if tuple(item.get("id") for item in variants) != VARIANT_IDS:
        raise OutcomeDecompositionError("family 10 variant menu differs")
    if profile.get("outer_folds") != [2022, 2023, 2024, 2025] or profile.get("outer_row_count") != 1_108:
        raise OutcomeDecompositionError("family 10 full development denominator differs")
    if profile.get("source", {}).get("safe_ids_asserted_before_label_decode") is not True:
        raise OutcomeDecompositionError("safe IDs must be asserted before label decode")
    if profile.get("gate") != {"kind": "fixed", "outer_label_reads": 0}:
        raise OutcomeDecompositionError("learned gate lineage differs")
    fallbacks = list(profile.get("fallbacks", ()))
    if fallbacks != [{"id": "constant-prior", "registered": True, "prior": 0.5}]:
        raise OutcomeDecompositionError("exact registered fallback is required")
    if profile.get("runtime", {}).get("refit_full") is not False:
        raise OutcomeDecompositionError("refit_full is forbidden")
    for item in variants:
        core = {key: value for key, value in item.items() if key != "profile_sha256"}
        if item.get("profile_sha256") != canonical_sha256(core):
            raise OutcomeDecompositionError("variant hash does not cover its defaults")
    return {
        "variant_count": len(variants),
        "variant_ids": [item["id"] for item in variants],
        "variant_hashes": {item["id"]: item["profile_sha256"] for item in variants},
    }
