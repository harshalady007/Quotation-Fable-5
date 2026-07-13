"""Leakage-safe offline evaluation for V3 contextual price adjustments.

This module never participates in request-time pricing. It asks a narrower
question: within conservative exact-product cohorts, does a quantity,
supplier, quotation-date or location signal improve held-out unit-price
accuracy over the cohort median? Every target quotation lineage is removed,
and every fold trains only on strictly earlier quotations.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config
from context_readiness import (APPROVED_CONTEXT_ADJUSTMENTS, CONTEXT_FIELDS,
                               IMPLEMENTED_CONTEXT_ADJUSTMENTS,
                               build_context_evidence_frame,
                               build_context_readiness,
                               context_adjustment_id,
                               normalize_quotation_date)
from family_readiness import dataset_fingerprint, load_readiness_snapshot
from pricing_dataset import load_pricing_dataset
from production_pricing import (APPROVED_AUTO_FAMILIES,
                                PRICING_ENGINE_VERSION,
                                SUPPORTED_INPUT_FAMILIES)
from similarity_search import SimilaritySearcher


CONTEXT_MODEL_SCHEMA_VERSION = 1
CONTEXT_MODEL_ALGORITHM_VERSION = "cohort-residual-ridge-v1"
_EMPTY_TEXT = {"", "nan", "none", "null", "n/a", "na", "unknown"}


class ContextModelEvaluationError(RuntimeError):
    """Raised when the shadow evaluator or its snapshot is unsafe."""


def _usable_text(value) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = " ".join(str(value).strip().lower().split())
    return None if text in _EMPTY_TEXT else text


def _field_values(frame: pd.DataFrame, field: str) -> pd.Series:
    source = frame.get(field, pd.Series(index=frame.index, dtype=object))
    if field == "quantity":
        values = pd.to_numeric(source, errors="coerce")
        return values.where(np.isfinite(values) & (values > 0))
    if field == "quotation_date":
        normalized = source.map(normalize_quotation_date)
        return pd.to_datetime(normalized, errors="coerce")
    return source.map(_usable_text)


def build_adjustment_observations(
    evidence_frame: pd.DataFrame,
    family: str,
    field: str,
) -> pd.DataFrame:
    """Collapse duplicate line evidence to quote/cohort/context medians."""
    if family not in SUPPORTED_INPUT_FAMILIES:
        raise ContextModelEvaluationError(f"Unsupported family: {family}.")
    if field not in CONTEXT_FIELDS:
        raise ContextModelEvaluationError(f"Unsupported context field: {field}.")
    frame = evidence_frame[evidence_frame["_family"] == family].copy()
    frame["_value"] = _field_values(frame, field)
    frame["_quote_date"] = pd.to_datetime(
        frame.get(
            "quotation_date", pd.Series(index=frame.index, dtype=object)
        ).map(normalize_quotation_date),
        errors="coerce",
    )
    frame["_rate"] = pd.to_numeric(frame.get("rate"), errors="coerce")
    frame = frame[
        frame["_cohort"].notna()
        & frame["_source_group"].notna()
        & frame["_value"].notna()
        & frame["_rate"].map(lambda value: math.isfinite(value) and value > 0)
    ]
    if frame.empty:
        return pd.DataFrame(columns=[
            "source_group", "cohort", "value", "quote_date", "rate",
            "source_rows",
        ])
    observations = (
        frame.groupby(
            ["_source_group", "_cohort", "_value", "_quote_date"],
            as_index=False,
            dropna=False,
        )
        .agg(_rate=("_rate", "median"), _source_rows=("_rate", "size"))
        .rename(columns={
            "_source_group": "source_group",
            "_cohort": "cohort",
            "_value": "value",
            "_quote_date": "quote_date",
            "_rate": "rate",
            "_source_rows": "source_rows",
        })
    )
    return observations.sort_values(
        ["source_group", "cohort", "value", "quote_date"], kind="stable"
    ).reset_index(drop=True)


def _cohort_varies(group: pd.DataFrame, field: str) -> bool:
    if group["source_group"].nunique() < config.V3_MODEL_MIN_TRAIN_GROUPS_PER_COHORT:
        return False
    if group["value"].nunique() < 2:
        return False
    if field == "quotation_date":
        span = (group["value"].max() - group["value"].min()).days
        return span >= config.V3_DATE_MIN_COHORT_SPAN_DAYS
    return True


def _fit_fold(train: pd.DataFrame, field: str) -> dict:
    """Fit one fixed-hyperparameter residual model without target evidence."""
    if train["source_group"].nunique() < config.V3_MODEL_MIN_TRAIN_QUOTE_GROUPS:
        return {"error": "too few independent training quotation groups"}
    history_span = (train["quote_date"].max() - train["quote_date"].min()).days
    if history_span < config.V3_MODEL_MIN_TRAIN_HISTORY_DAYS:
        return {"error": "training history is shorter than the minimum span"}
    cohorts = [
        cohort for cohort, group in train.groupby("cohort", sort=False)
        if _cohort_varies(group, field)
    ]
    model_train = train[train["cohort"].isin(cohorts)].copy()
    supported_levels: set[str] | None = None
    if field in {"supplier", "location"} and not model_train.empty:
        level_groups = model_train.groupby("value")["source_group"].nunique()
        supported_levels = set(level_groups[
            level_groups >= config.V3_CONTEXT_MIN_GROUPS_PER_LEVEL
        ].index)
        model_train = model_train[model_train["value"].isin(supported_levels)]
        cohorts = [
            cohort for cohort, group in model_train.groupby("cohort", sort=False)
            if _cohort_varies(group, field)
        ]
        model_train = model_train[model_train["cohort"].isin(cohorts)].copy()
    if len(cohorts) < config.V3_MODEL_MIN_COHORTS:
        return {"error": "too few varying exact-product training cohorts"}
    if model_train["source_group"].nunique() < config.V3_MODEL_MIN_TRAIN_QUOTE_GROUPS:
        return {"error": "too few usable training quotation groups"}
    if field == "quantity" and model_train["value"].nunique() < 3:
        return {"error": "fewer than three training quantity values"}
    if field in {"supplier", "location"} and len(supported_levels or set()) < 2:
        return {"error": "fewer than two supported categorical levels"}
    if field == "quotation_date":
        history_span = (
            model_train["value"].max() - model_train["value"].min()
        ).days
        if history_span < config.V3_DATE_MIN_SPAN_DAYS:
            return {"error": "training history is shorter than the minimum span"}

    group_cohort_rates = (
        model_train.groupby(["cohort", "source_group"])["rate"].median()
    )
    centers = group_cohort_rates.groupby("cohort").median()
    model_train["_center"] = model_train["cohort"].map(centers)
    model_train["_residual"] = np.log(
        model_train["rate"] / model_train["_center"]
    )
    group_sizes = model_train.groupby("source_group")["rate"].transform("size")
    sample_weight = 1.0 / group_sizes.to_numpy(dtype=float)
    target = model_train["_residual"].to_numpy(dtype=float)

    # Imports remain inside the offline fit path; API snapshot reads never
    # construct or deserialize a model.
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    model = Ridge(alpha=config.V3_MODEL_RIDGE_ALPHA, fit_intercept=True)
    result = {
        "model": model,
        "centers": centers,
        "cohorts": set(cohorts),
        "train_groups": int(model_train["source_group"].nunique()),
        "train_cohorts": len(cohorts),
        "train_value_min": model_train["value"].min(),
        "train_value_max": model_train["value"].max(),
        "train_quote_date_max": model_train["quote_date"].max(),
        "supported_levels": supported_levels,
    }
    if field in {"quantity", "quotation_date"}:
        raw = (
            np.log(model_train["value"].to_numpy(dtype=float))
            if field == "quantity"
            else (
                model_train["value"] - model_train["value"].min()
            ).dt.days.to_numpy(dtype=float) / 365.25
        ).reshape(-1, 1)
        scaler = StandardScaler()
        features = scaler.fit_transform(raw)
        model.fit(features, target, sample_weight=sample_weight)
        result["scaler"] = scaler
        if field == "quotation_date":
            result["date_origin"] = model_train["value"].min()
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        features = encoder.fit_transform(
            model_train[["value"]].astype(str)
        )
        model.fit(features, target, sample_weight=sample_weight)
        result["encoder"] = encoder
    return result


def _predict_effect(fold: dict, field: str, value) -> float:
    if field == "quantity":
        raw = np.array([[math.log(float(value))]], dtype=float)
        features = fold["scaler"].transform(raw)
    elif field == "quotation_date":
        days = (value - fold["date_origin"]).days / 365.25
        features = fold["scaler"].transform(np.array([[days]], dtype=float))
    else:
        features = fold["encoder"].transform(pd.DataFrame({"value": [str(value)]}))
    return float(fold["model"].predict(features)[0])


def _prediction_metrics(predictions: list[dict], price_field: str) -> dict:
    if not predictions:
        return {
            "cases": 0,
            "quote_groups": 0,
            "cohorts": 0,
            "within_20": None,
            "within_50": None,
            "median_ape": None,
            "p90_ape": None,
            "factor2_errors": 0,
        }
    actual = np.array([item["actual"] for item in predictions], dtype=float)
    predicted = np.array([item[price_field] for item in predictions], dtype=float)
    ratios = predicted / actual
    ape = np.abs(ratios - 1.0)
    weights = _group_balanced_weights(predictions)
    return {
        "cases": len(predictions),
        "quote_groups": len({item["source_group"] for item in predictions}),
        "cohorts": len({item["cohort"] for item in predictions}),
        "within_20": round(float(np.average(ape <= 0.20, weights=weights)), 4),
        "within_50": round(float(np.average(ape <= 0.50, weights=weights)), 4),
        "median_ape": round(_weighted_quantile(ape, weights, 0.50), 4),
        "p90_ape": round(_weighted_quantile(ape, weights, 0.90), 4),
        "factor2_errors": int(np.sum((ratios < 0.50) | (ratios > 2.0))),
    }


def _group_balanced_weights(predictions: list[dict]) -> np.ndarray:
    counts = Counter(item["source_group"] for item in predictions)
    return np.array([
        1.0 / counts[item["source_group"]] for item in predictions
    ], dtype=float)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = quantile * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def context_model_release_gate(report: dict) -> tuple[bool, list[str]]:
    """Apply statistical and incremental-value gates to one adjustment."""
    failures = []
    if report.get("evidence_status") not in {"ready_for_modeling", "production"}:
        failures.append("context data-readiness gates have not passed")
        return False, failures
    model = report.get("model_metrics") or {}
    if model.get("cases", 0) < config.V3_MODEL_MIN_CASES:
        failures.append(f"needs at least {config.V3_MODEL_MIN_CASES} evaluated cases")
    if model.get("quote_groups", 0) < config.V3_MODEL_MIN_EVAL_QUOTE_GROUPS:
        failures.append(
            f"needs at least {config.V3_MODEL_MIN_EVAL_QUOTE_GROUPS} evaluated "
            "quotation groups"
        )
    if model.get("cohorts", 0) < config.V3_MODEL_MIN_COHORTS:
        failures.append(
            f"needs at least {config.V3_MODEL_MIN_COHORTS} evaluated cohorts"
        )
    if report.get("evaluation_coverage", 0) < config.V3_MODEL_MIN_COVERAGE:
        failures.append(
            f"evaluation coverage below {config.V3_MODEL_MIN_COVERAGE:.0%}"
        )
    within_20 = model.get("within_20")
    if within_20 is None or within_20 < config.V3_MODEL_MIN_WITHIN_20:
        failures.append(
            f"within ±20% below {config.V3_MODEL_MIN_WITHIN_20:.0%}"
        )
    median_ape = model.get("median_ape")
    if median_ape is None or median_ape > config.V3_MODEL_MAX_MEDIAN_APE:
        failures.append(
            f"median APE above {config.V3_MODEL_MAX_MEDIAN_APE:.0%}"
        )
    p90_ape = model.get("p90_ape")
    if p90_ape is None or p90_ape > config.V3_MODEL_MAX_P90_APE:
        failures.append(f"p90 APE above {config.V3_MODEL_MAX_P90_APE:.0%}")
    if model.get("factor2_errors", 0) > config.V3_MODEL_MAX_FACTOR2_ERRORS:
        failures.append("contains factor-of-two errors")
    if (report.get("median_ape_improvement") is None
            or report["median_ape_improvement"]
            < config.V3_MODEL_MIN_MEDIAN_APE_IMPROVEMENT):
        failures.append(
            "median APE improvement over baseline below "
            f"{config.V3_MODEL_MIN_MEDIAN_APE_IMPROVEMENT:.0%}"
        )
    if report.get("improved_case_share", 0) < config.V3_MODEL_MIN_IMPROVED_CASE_SHARE:
        failures.append(
            "share of cases improved over baseline below "
            f"{config.V3_MODEL_MIN_IMPROVED_CASE_SHARE:.0%}"
        )
    if report.get("clamp_share", 0) > config.V3_MODEL_MAX_CLAMP_SHARE:
        failures.append(
            f"factor clamp used in more than {config.V3_MODEL_MAX_CLAMP_SHARE:.0%} "
            "of cases"
        )
    if report.get("temporal_order_violations", 0):
        failures.append("date evaluation contains temporal-order leakage")
    return not failures, failures


def evaluate_context_adjustment(
    evidence_frame: pd.DataFrame,
    family: str,
    field: str,
    evidence: dict,
) -> dict:
    """Evaluate one family-scoped context adjustment in offline shadow mode."""
    adjustment_id = context_adjustment_id(family, field)
    observations = build_adjustment_observations(evidence_frame, family, field)
    result = {
        "adjustment_id": adjustment_id,
        "family": family,
        "field": field,
        "evaluation_mode": "rolling_origin_and_quotation_lineage_holdout",
        "evidence_status": evidence.get("status"),
        "evidence_blocking_reasons": list(evidence.get("blocking_reasons") or []),
        "observation_rows": len(observations),
        "observation_quote_groups": int(
            observations["source_group"].nunique()
        ) if not observations.empty else 0,
        "eligible_targets": len(observations),
        "evaluation_coverage": 0.0,
        "model_metrics": _prediction_metrics([], "predicted"),
        "baseline_metrics": _prediction_metrics([], "baseline"),
        "median_ape_improvement": None,
        "improved_case_share": 0.0,
        "clamped_cases": 0,
        "clamp_share": 0.0,
        "raw_factor_min": None,
        "raw_factor_max": None,
        "temporal_order_violations": 0,
        "top_skip_reasons": [],
    }
    if evidence.get("status") not in {"ready_for_modeling", "production"}:
        result["evaluation_status"] = "blocked_by_data"
        _, failures = context_model_release_gate(result)
        result["statistical_gate_passed"] = False
        result["statistical_gate_failures"] = failures
        return result

    predictions = []
    skip_reasons: Counter[str] = Counter()
    fold_cache = {}
    temporal_violations = 0
    for _, target in observations.iterrows():
        group = target["source_group"]
        value = target["value"]
        target_date = target["quote_date"]
        if pd.isna(target_date):
            skip_reasons["target quotation date is missing"] += 1
            continue
        train = observations[
            (observations["source_group"] != group)
            & observations["quote_date"].notna()
            & (observations["quote_date"] < target_date)
        ].copy()
        cache_key = (group, target_date.isoformat())
        temporal_violations += int(
            (train["quote_date"] >= target_date).sum()
        )
        if cache_key not in fold_cache:
            fold_cache[cache_key] = _fit_fold(train, field)
        fold = fold_cache[cache_key]
        if fold.get("error"):
            skip_reasons[fold["error"]] += 1
            continue
        if target["cohort"] not in fold["cohorts"]:
            skip_reasons["target cohort lacks independent training variation"] += 1
            continue
        if field == "quantity":
            if not (fold["train_value_min"] <= value <= fold["train_value_max"]):
                skip_reasons["quantity target requires extrapolation"] += 1
                continue
        elif field != "quotation_date" and value not in (
            fold.get("supported_levels") or set()
        ):
            skip_reasons["categorical target level lacks training support"] += 1
            continue
        horizon = (target_date - fold["train_quote_date_max"]).days
        if horizon <= 0:
            temporal_violations += 1
            skip_reasons["target is not after training evidence"] += 1
            continue
        if horizon > config.V3_MODEL_MAX_FORECAST_HORIZON_DAYS:
            skip_reasons["target exceeds the maximum forecast horizon"] += 1
            continue
        baseline = float(fold["centers"].get(target["cohort"], np.nan))
        if not math.isfinite(baseline) or baseline <= 0:
            skip_reasons["target cohort has no valid training baseline"] += 1
            continue
        effect = _predict_effect(fold, field, value)
        try:
            raw_factor = math.exp(effect)
        except OverflowError:
            skip_reasons["model produced a non-finite adjustment factor"] += 1
            continue
        if not math.isfinite(raw_factor) or raw_factor <= 0:
            skip_reasons["model produced a non-finite adjustment factor"] += 1
            continue
        factor = min(
            config.V3_MODEL_FACTOR_MAX,
            max(config.V3_MODEL_FACTOR_MIN, raw_factor),
        )
        predictions.append({
            "source_group": group,
            "cohort": target["cohort"],
            "actual": float(target["rate"]),
            "baseline": baseline,
            "predicted": baseline * factor,
            "raw_factor": raw_factor,
            "factor": factor,
            "clamped": not math.isclose(factor, raw_factor, rel_tol=1e-12),
        })

    model_metrics = _prediction_metrics(predictions, "predicted")
    baseline_metrics = _prediction_metrics(predictions, "baseline")
    result["model_metrics"] = model_metrics
    result["baseline_metrics"] = baseline_metrics
    result["evaluation_coverage"] = round(
        len(predictions) / len(observations), 4
    ) if len(observations) else 0.0
    if predictions:
        model_ape = np.array([
            abs(item["predicted"] / item["actual"] - 1.0)
            for item in predictions
        ])
        baseline_ape = np.array([
            abs(item["baseline"] / item["actual"] - 1.0)
            for item in predictions
        ])
        balanced_weights = _group_balanced_weights(predictions)
        result["median_ape_improvement"] = round(
            _weighted_quantile(baseline_ape, balanced_weights, 0.50)
            - _weighted_quantile(model_ape, balanced_weights, 0.50),
            4,
        )
        result["improved_case_share"] = round(
            float(np.average(model_ape < baseline_ape, weights=balanced_weights)),
            4,
        )
        clamped = int(sum(item["clamped"] for item in predictions))
        result["clamped_cases"] = clamped
        result["clamp_share"] = round(
            float(np.average(
                [item["clamped"] for item in predictions],
                weights=balanced_weights,
            )),
            4,
        )
        result["raw_factor_min"] = round(
            min(item["raw_factor"] for item in predictions), 4
        )
        result["raw_factor_max"] = round(
            max(item["raw_factor"] for item in predictions), 4
        )
    result["temporal_order_violations"] = temporal_violations
    result["top_skip_reasons"] = [
        {"reason": reason, "count": count}
        for reason, count in skip_reasons.most_common(5)
    ]
    passed, failures = context_model_release_gate(result)
    result["statistical_gate_passed"] = passed
    result["statistical_gate_failures"] = failures
    result["evaluation_status"] = (
        "statistically_ready" if passed else "rejected_by_shadow_gate"
    )
    return result


def evaluate_context_model_readiness(
    data_path: str | None = None,
    corrections_path: str | None = None,
    base_readiness: dict | None = None,
) -> dict:
    """Evaluate every family/field pair against current audited evidence."""
    dataset = load_pricing_dataset(data_path, corrections_path)
    fingerprint = dataset_fingerprint(dataset)
    searcher = SimilaritySearcher(dataset)
    context_readiness = build_context_readiness(dataset, searcher.item_attrs)
    evidence_frame = build_context_evidence_frame(dataset, searcher.item_attrs)
    base = base_readiness or load_readiness_snapshot()
    if base.get("dataset_fingerprint") != fingerprint:
        raise ContextModelEvaluationError(
            "Family readiness snapshot is stale; regenerate it before evaluating "
            "context models."
        )
    base_families = {
        item["family"]: item for item in (base.get("families") or [])
    }
    adjustments = []
    for family in sorted(SUPPORTED_INPUT_FAMILIES):
        base_family = base_families.get(family) or {}
        for field in CONTEXT_FIELDS:
            evidence = context_readiness["families"][family][field]
            result = evaluate_context_adjustment(
                evidence_frame, family, field, evidence
            )
            base_passed = bool(base_family.get("release_gate_passed"))
            production_failures = list(result["statistical_gate_failures"])
            if not base_passed:
                production_failures.append(
                    "base family has not passed its pricing release gate"
                )
            result["base_family_release_passed"] = base_passed
            result["base_family_approved_for_automatic_pricing"] = (
                family in APPROVED_AUTO_FAMILIES
            )
            result["production_gate_passed"] = bool(
                result["statistical_gate_passed"] and base_passed
            )
            result["production_gate_failures"] = production_failures
            adjustment_id = result["adjustment_id"]
            result["implemented_in_pricing_engine"] = (
                adjustment_id in IMPLEMENTED_CONTEXT_ADJUSTMENTS
            )
            result["approved_for_price_adjustment"] = (
                adjustment_id in APPROVED_CONTEXT_ADJUSTMENTS
            )
            if (result["production_gate_passed"]
                    and adjustment_id not in APPROVED_CONTEXT_ADJUSTMENTS):
                result["status"] = "ready_for_approval"
            elif (result["production_gate_passed"]
                  and adjustment_id in APPROVED_CONTEXT_ADJUSTMENTS
                  and family in APPROVED_AUTO_FAMILIES):
                result["status"] = "production"
            else:
                result["status"] = result["evaluation_status"]
            adjustments.append(result)

    return {
        "schema_version": CONTEXT_MODEL_SCHEMA_VERSION,
        "algorithm_version": CONTEXT_MODEL_ALGORITHM_VERSION,
        "pricing_version": PRICING_ENGINE_VERSION,
        "dataset_fingerprint": fingerprint,
        "mode": "offline_shadow",
        "approved_adjustments": sorted(APPROVED_CONTEXT_ADJUSTMENTS),
        "implemented_adjustments": sorted(IMPLEMENTED_CONTEXT_ADJUSTMENTS),
        "algorithm": {
            "baseline": "held-out exact-product cohort median",
            "candidate": "ridge model of log-rate residual by one context field",
            "lineage_holdout": True,
            "all_fields_use_strictly_earlier_training_rows": True,
            "hyperparameter_tuning_on_evaluation_data": False,
            "metric_weighting": "equal total weight per quotation group",
            "ridge_alpha": config.V3_MODEL_RIDGE_ALPHA,
            "minimum_training_history_days": (
                config.V3_MODEL_MIN_TRAIN_HISTORY_DAYS
            ),
            "maximum_forecast_horizon_days": (
                config.V3_MODEL_MAX_FORECAST_HORIZON_DAYS
            ),
            "factor_bounds": [
                config.V3_MODEL_FACTOR_MIN, config.V3_MODEL_FACTOR_MAX,
            ],
        },
        "gate": {
            "minimum_cases": config.V3_MODEL_MIN_CASES,
            "minimum_evaluated_quote_groups": (
                config.V3_MODEL_MIN_EVAL_QUOTE_GROUPS
            ),
            "minimum_cohorts": config.V3_MODEL_MIN_COHORTS,
            "minimum_training_groups_per_cohort": (
                config.V3_MODEL_MIN_TRAIN_GROUPS_PER_COHORT
            ),
            "minimum_evaluation_coverage": config.V3_MODEL_MIN_COVERAGE,
            "minimum_within_20": config.V3_MODEL_MIN_WITHIN_20,
            "maximum_median_ape": config.V3_MODEL_MAX_MEDIAN_APE,
            "maximum_p90_ape": config.V3_MODEL_MAX_P90_APE,
            "maximum_factor2_errors": config.V3_MODEL_MAX_FACTOR2_ERRORS,
            "minimum_median_ape_improvement": (
                config.V3_MODEL_MIN_MEDIAN_APE_IMPROVEMENT
            ),
            "minimum_improved_case_share": (
                config.V3_MODEL_MIN_IMPROVED_CASE_SHARE
            ),
            "maximum_clamp_share": config.V3_MODEL_MAX_CLAMP_SHARE,
        },
        "adjustments": adjustments,
    }


def load_context_model_snapshot(path: str | Path | None = None) -> dict:
    """Load and validate the deterministic offline model-readiness snapshot."""
    snapshot_path = Path(path or config.CONTEXT_MODEL_READINESS_PATH)
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextModelEvaluationError(
            f"Could not load context model readiness snapshot: {exc}"
        ) from exc
    if snapshot.get("schema_version") != CONTEXT_MODEL_SCHEMA_VERSION:
        raise ContextModelEvaluationError(
            "Context model snapshot schema_version must be "
            f"{CONTEXT_MODEL_SCHEMA_VERSION}."
        )
    if snapshot.get("algorithm_version") != CONTEXT_MODEL_ALGORITHM_VERSION:
        raise ContextModelEvaluationError(
            "Context model snapshot algorithm_version is unsupported."
        )
    if not str(snapshot.get("dataset_fingerprint") or "").strip():
        raise ContextModelEvaluationError(
            "Context model snapshot has no dataset_fingerprint."
        )
    adjustments = snapshot.get("adjustments")
    if not isinstance(adjustments, list):
        raise ContextModelEvaluationError(
            "Context model snapshot adjustments must be a list."
        )
    expected_ids = {
        context_adjustment_id(family, field)
        for family in SUPPORTED_INPUT_FAMILIES
        for field in CONTEXT_FIELDS
    }
    actual_ids = {
        item.get("adjustment_id") for item in adjustments
        if isinstance(item, dict)
    }
    if len(actual_ids) != len(adjustments) or actual_ids != expected_ids:
        raise ContextModelEvaluationError(
            "Context model snapshot must contain each family/field adjustment "
            "exactly once."
        )
    return snapshot


def compact_context_model_readiness(snapshot: dict) -> dict:
    """Return a health-check summary without per-adjustment metrics."""
    statuses = Counter(
        item.get("status", "unknown")
        for item in (snapshot.get("adjustments") or [])
    )
    return {
        "mode": snapshot.get("mode", "offline_shadow"),
        "algorithm_version": snapshot.get("algorithm_version"),
        "status_counts": dict(sorted(statuses.items())),
        "ready_for_approval": sorted(
            item["adjustment_id"] for item in snapshot.get("adjustments", [])
            if item.get("status") == "ready_for_approval"
        ),
        "production": sorted(
            item["adjustment_id"] for item in snapshot.get("adjustments", [])
            if item.get("status") == "production"
        ),
    }
