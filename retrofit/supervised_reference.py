"""Governed supervised-reference API for CatBoost, LightGBM, and XGBoost.

This module owns analytical semantics and portable native fitted state.  It has
no Analytics Workstation persistence, UI, or execution-governance concerns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


CONTRACT_ID = "supervised.reference@1.1.0"
SUPPORTED_ENGINES = ("catboost", "lightgbm", "xgboost")
SUPPORTED_TASKS = ("regression", "binary")


class SupervisedReferenceError(RuntimeError):
    """A typed analytical-contract failure."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class ImplementationDescriptor:
    capability: str
    contract: str
    runtime: str
    package: str
    engine: str
    task: str
    algorithm_family: str
    ensemble_regime: str
    implementation_version: str


def _package_version(name: str) -> str:
    from importlib.metadata import version

    try:
        return version(name)
    except Exception:
        return "unavailable"


def environment_identity() -> dict[str, Any]:
    packages = {name: _package_version(name) for name in (
        "retrofit", "catboost", "lightgbm", "xgboost", "polars", "numpy",
        "pyarrow", "scikit-learn",
    )}
    value = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_class": "isolated_capability_environment",
        "packages": packages,
    }
    value["fingerprint"] = _fingerprint(value)
    return value


def implementation_descriptor(engine: str, task: str,
                              ensemble_regime: str = "boosting") -> dict[str, Any]:
    _validate_engine_task(engine, task)
    regime = normalize_ensemble_regime(engine, ensemble_regime)
    return asdict(ImplementationDescriptor(
        capability=f"supervised.{task}", contract=CONTRACT_ID,
        runtime="python", package="RetroFit", engine=engine, task=task,
        algorithm_family="tree_ensemble", ensemble_regime=regime,
        implementation_version=_package_version("retrofit"),
    ))


def normalize_ensemble_regime(engine: str, regime: str) -> str:
    regime = str(regime or "boosting").lower()
    allowed = {
        "catboost": {"boosting", "stochastic_boosting"},
        "lightgbm": {"boosting", "stochastic_boosting", "independent_tree"},
        "xgboost": {"boosting", "stochastic_boosting", "independent_tree"},
    }
    if engine not in allowed or regime not in allowed[engine]:
        raise SupervisedReferenceError("invalid_parameter_combination",
            f"{engine} does not support ensemble regime {regime!r}.",
            {"engine": engine, "regime": regime,
             "allowed": sorted(allowed.get(engine, set()))})
    return regime


def supervised_reference_contract() -> dict[str, Any]:
    return {
        "contract": CONTRACT_ID,
        "tasks": list(SUPPORTED_TASKS),
        "engines": list(SUPPORTED_ENGINES),
        "probability_semantics": "positive_class_probability_separate_from_decision_policy",
        "observation_weight_semantics": "optional_nonnegative_frequency_or_importance_weight",
        "early_stopping_authority": "explicit_request_with_zero_or_none_to_disable",
        "partition_authority": "explicit_seeded_row_partition",
        "feature_identity": "name_order_type_and_categorical_levels",
        "algorithm_dimensions": ["algorithm_family", "engine", "ensemble_regime"],
        "evidence_dimensions": {
            "importance_cost": ["CORE", "DERIVED", "EXTENDED", "SPECIALIZED"],
            "semantic_availability": ["COMMON", "ENGINE_NATIVE", "DERIVABLE", "UNAVAILABLE"],
        },
        "fitted_state": "native_engine_payload_plus_canonical_manifest",
    }


def tuning_space(engine: str, task: str) -> dict[str, Any]:
    _validate_engine_task(engine, task)
    common = {
        "iterations": {"type": "integer", "minimum": 10, "maximum": 5000},
        "depth": {"type": "integer", "minimum": 1, "maximum": 16},
        "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 0.5},
        "row_sample": {"type": "number", "minimum": 0.1, "maximum": 1.0},
        "feature_sample": {"type": "number", "minimum": 0.1, "maximum": 1.0},
    }
    regimes = {
        "catboost": ["boosting", "stochastic_boosting"],
        "lightgbm": ["boosting", "stochastic_boosting", "independent_tree"],
        "xgboost": ["boosting", "stochastic_boosting", "independent_tree"],
    }
    return {"schema_version": "retrofit_tuning_space_v1", "engine": engine,
            "task": task, "parameters": common,
            "ensemble_regime": {"type": "enum", "values": regimes[engine]},
            "fit_controls": {
                "weight": {"type": "column", "required": False},
                "early_stopping_rounds": {"type": "integer_or_null", "minimum": 0},
            }}


def read_parquet(path: str | os.PathLike[str]) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if not isinstance(frame, pl.DataFrame):
        raise SupervisedReferenceError("transport_failed", "Parquet did not yield a table.")
    return frame


def fit_from_parquet(request: Mapping[str, Any], input_path: str | os.PathLike[str],
                     bundle_dir: str | os.PathLike[str]) -> dict[str, Any]:
    return fit(request, read_parquet(input_path), bundle_dir)


def fit(request: Mapping[str, Any], data: pl.DataFrame,
        bundle_dir: str | os.PathLike[str]) -> dict[str, Any]:
    started = time.perf_counter()
    spec = _validate_fit_request(request, data)
    engine, task = spec["engine"], spec["task"]
    target, features = spec["target"], spec["features"]
    train_idx, valid_idx = _partition(len(data), spec["train_fraction"], spec["seed"])
    x = data.select(features)
    y, target_contract = _target(data.get_column(target), task, spec.get("positive_class"))
    weights, weight_contract = _weights(data, spec.get("weight"))
    feature_contract = _feature_contract(x)
    x, prep = _prepare_features(x, feature_contract)
    prep["dataframe_authority"] = "polars"
    prep["algorithm_boundary"] = _algorithm_boundary(engine)
    prep["pandas_used"] = False
    model, history = _fit_engine(engine, task, x[train_idx], y[train_idx],
        x[valid_idx], y[valid_idx], feature_contract, spec,
        weights[train_idx] if weights is not None else None,
        weights[valid_idx] if weights is not None else None)
    raw = _predict_engine(engine, task, model, x[valid_idx], feature_contract)
    metrics = _metrics(task, y[valid_idx], raw, spec["threshold"],
                       weights[valid_idx] if weights is not None else None)
    evidence = _evidence_manifest(engine, task, spec["ensemble_regime"])
    output = Path(bundle_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_file = _save_model(engine, model, output)
    model_digest = sha256(model_file.read_bytes()).hexdigest()
    env = environment_identity()
    manifest = {
        "schema_version": "retrofit_fitted_state_v1",
        "contract": CONTRACT_ID,
        "implementation_descriptor": implementation_descriptor(
            engine, task, spec["ensemble_regime"]),
        "environment": env,
        "engine_version": env["packages"][engine],
        "model_file": model_file.name,
        "model_payload_digest": model_digest,
        "task": task, "engine": engine, "target": target,
        "features": features, "feature_contract": feature_contract,
        "target_contract": target_contract,
        "weight_contract": weight_contract,
        "partition": {"method": "seeded_random", "seed": spec["seed"],
                      "train_fraction": spec["train_fraction"],
                      "train_rows": train_idx.tolist(), "validation_rows": valid_idx.tolist()},
        "decision_policy": {"threshold": spec["threshold"],
                            "positive_class": target_contract.get("positive_class")},
        "engine_params": spec["engine_params"],
        "ensemble_regime": spec["ensemble_regime"],
        "preprocessing": prep,
        "training_history": history,
        "evidence_manifest": evidence,
        "metrics": metrics,
        "resource_evidence": {"device": "cpu", "thread_count": spec["thread_count"],
            "fit_seconds": time.perf_counter() - started,
            "input_rows": int(len(data)), "input_columns": int(len(features))},
    }
    manifest["feature_contract"]["fingerprint"] = _fingerprint(feature_contract)
    manifest["fingerprint"] = _fingerprint(manifest)
    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics,
            "evidence_manifest": evidence,
            "validation": _prediction_records(task, valid_idx, raw, spec["threshold"])}


def predict_from_parquet(bundle_dir: str | os.PathLike[str],
                         input_path: str | os.PathLike[str],
                         output_path: str | os.PathLike[str]) -> dict[str, Any]:
    result = predict(bundle_dir, read_parquet(input_path))
    pl.DataFrame(result["predictions"]).write_parquet(output_path)
    result["output_path"] = str(output_path)
    return result


def predict(bundle_dir: str | os.PathLike[str], data: pl.DataFrame) -> dict[str, Any]:
    root = Path(bundle_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    model_file = root / manifest["model_file"]
    if sha256(model_file.read_bytes()).hexdigest() != manifest["model_payload_digest"]:
        raise SupervisedReferenceError("fitted_payload_incompatible",
                                       "The fitted payload digest does not match its manifest.")
    missing = [name for name in manifest["features"] if name not in data.columns]
    if missing:
        raise SupervisedReferenceError("feature_contract_mismatch",
            "Application data is missing required features.", {"missing": missing})
    x, _ = _prepare_features(data.select(manifest["features"]),
                             manifest["feature_contract"], application=True)
    model = _load_model(manifest["engine"], manifest["task"], model_file)
    raw = _predict_engine(manifest["engine"], manifest["task"], model, x,
                          manifest["feature_contract"])
    threshold = float(manifest["decision_policy"]["threshold"])
    return {"schema_version": "retrofit_prediction_v1",
            "model_fingerprint": manifest["fingerprint"], "refit": False,
            "predictions": _prediction_records(manifest["task"],
                                                np.arange(len(data)), raw, threshold)}


def _validate_engine_task(engine: str, task: str) -> None:
    if engine not in SUPPORTED_ENGINES:
        raise SupervisedReferenceError("unsupported_engine", f"Unsupported engine: {engine}")
    if task not in SUPPORTED_TASKS:
        raise SupervisedReferenceError("unsupported_task", f"Unsupported task: {task}")


def _validate_fit_request(request: Mapping[str, Any], data: pl.DataFrame) -> dict[str, Any]:
    engine, task = str(request.get("engine", "")), str(request.get("task", ""))
    _validate_engine_task(engine, task)
    target = str(request.get("target", ""))
    features = [str(value) for value in request.get("features", [])]
    if not target or not features or target in features:
        raise SupervisedReferenceError("invalid_features", "Target and distinct features are required.")
    missing = [name for name in [target, *features] if name not in data.columns]
    if missing:
        raise SupervisedReferenceError("feature_contract_mismatch",
                                       "Requested fields are unavailable.", {"missing": missing})
    fraction = float(request.get("train_fraction", 0.8))
    if not 0.1 <= fraction <= 0.95:
        raise SupervisedReferenceError("invalid_partition", "train_fraction must be in [0.1, 0.95].")
    seed = int(request.get("seed", 1))
    threshold = float(request.get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise SupervisedReferenceError("invalid_decision_policy", "threshold must be in [0, 1].")
    regime = normalize_ensemble_regime(engine, str(request.get("ensemble_regime", "boosting")))
    params = dict(request.get("engine_params", {}))
    threads = max(1, int(request.get("thread_count", params.get("thread_count", 1))))
    weight = request.get("weight")
    if weight is not None:
        weight = str(weight)
        if weight not in data.columns:
            raise SupervisedReferenceError("weight_contract_mismatch",
                "Requested observation-weight column is unavailable.", {"weight": weight})
        if weight == target or weight in features:
            raise SupervisedReferenceError("weight_contract_mismatch",
                "Observation weights must be distinct from target and feature columns.")
    early_stopping_rounds = request.get("early_stopping_rounds", 20)
    if early_stopping_rounds is not None:
        early_stopping_rounds = int(early_stopping_rounds)
        if early_stopping_rounds < 0:
            raise SupervisedReferenceError("invalid_parameter_combination",
                "early_stopping_rounds must be non-negative or None.")
    return {"engine": engine, "task": task, "target": target, "features": features,
            "train_fraction": fraction, "seed": seed, "threshold": threshold,
            "positive_class": request.get("positive_class"),
            "ensemble_regime": regime, "engine_params": params,
            "thread_count": threads, "weight": weight,
            "early_stopping_rounds": early_stopping_rounds}


def _partition(rows: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if rows < 4:
        raise SupervisedReferenceError("invalid_partition", "At least four rows are required.")
    order = np.random.default_rng(seed).permutation(rows)
    cut = max(1, min(rows - 1, int(np.floor(rows * fraction))))
    return np.sort(order[:cut]), np.sort(order[cut:])


def _target(series: pl.Series, task: str, positive: Any) -> tuple[np.ndarray, dict[str, Any]]:
    has_nan = bool(series.is_nan().fill_null(False).any()) if series.dtype.is_float() else False
    if series.null_count() or has_nan:
        raise SupervisedReferenceError("target_missing", "Target values may not be missing.")
    if task == "regression":
        values = series.cast(pl.Float64, strict=False)
        array = values.to_numpy()
        if values.null_count() or not np.isfinite(array).all():
            raise SupervisedReferenceError("target_type_invalid", "Regression target must be finite numeric.")
        return array.astype(float, copy=False), {"type": "numeric"}
    levels = series.unique(maintain_order=True).to_list()
    if len(levels) != 2:
        raise SupervisedReferenceError("target_type_invalid", "Binary target must have exactly two classes.")
    positive = levels[-1] if positive is None else positive
    if positive not in levels:
        raise SupervisedReferenceError("positive_class_invalid", "Positive class is not present in the target.")
    negative = next(value for value in levels if value != positive)
    return (series == positive).cast(pl.Int8).to_numpy(), {
        "type": "binary", "positive_class": str(positive), "negative_class": str(negative)}


def _weights(data: pl.DataFrame, column: str | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    if column is None:
        return None, {"enabled": False}
    series = data.get_column(column).cast(pl.Float64, strict=False)
    values = series.to_numpy()
    if series.null_count() or not np.isfinite(values).all() or np.any(values < 0):
        raise SupervisedReferenceError("weight_contract_mismatch",
            "Observation weights must be finite, non-missing, and non-negative.",
            {"weight": column})
    if not np.any(values > 0):
        raise SupervisedReferenceError("weight_contract_mismatch",
            "At least one observation weight must be positive.", {"weight": column})
    return values.astype(float, copy=False), {
        "enabled": True, "column": column, "semantics": "frequency_or_importance_weight",
        "sum": float(values.sum()), "positive_rows": int(np.count_nonzero(values > 0))}


def _feature_contract(frame: pl.DataFrame) -> dict[str, Any]:
    fields = []
    for name in frame.columns:
        series = frame.get_column(name)
        categorical = not series.dtype.is_numeric() or series.dtype == pl.Boolean
        fields.append({"name": str(name), "logical_type": "categorical" if categorical else "numeric",
                       "nullable": bool(series.null_count()),
                       "levels": sorted(str(value) for value in series.drop_nulls().unique())
                       if categorical else []})
    return {"schema_version": "supervised_feature_contract_v1",
            "features": [field["name"] for field in fields], "fields": fields}


def _prepare_features(frame: pl.DataFrame, contract: Mapping[str, Any],
                      application: bool = False) -> tuple[pl.DataFrame, dict[str, Any]]:
    expressions = []
    categorical = []
    for field in contract["fields"]:
        name = field["name"]
        if field["logical_type"] == "categorical":
            categorical.append(name)
            values = frame.get_column(name).cast(pl.String).fill_null("__RETROFIT_MISSING__")
            levels = list(field.get("levels", []))
            if "__RETROFIT_MISSING__" not in levels:
                levels.append("__RETROFIT_MISSING__")
            if application:
                unseen = sorted(set(values.to_list()) - set(levels))
                if unseen:
                    raise SupervisedReferenceError("feature_contract_mismatch",
                        f"Feature {name} contains unseen categorical levels.",
                        {"feature": name, "unseen": unseen[:20]})
            expressions.append(values.alias(name))
        else:
            expressions.append(pl.col(name).cast(pl.Float64, strict=False).alias(name))
    output = frame.select(expressions)
    return output, {"categorical_features": categorical,
                    "missing_numeric": "engine_native",
                    "missing_categorical": "explicit_sentinel"}


def _fit_engine(engine: str, task: str, x_train: pl.DataFrame, y_train: np.ndarray,
                x_valid: pl.DataFrame, y_valid: np.ndarray,
                contract: Mapping[str, Any], spec: Mapping[str, Any],
                train_weight: np.ndarray | None = None,
                valid_weight: np.ndarray | None = None):
    params = dict(spec["engine_params"])
    # Runtime/governance parameters are carried beside engine parameters by
    # some hosts. They must never be forwarded twice to a native constructor.
    params.pop("thread_count", None)
    params.pop("num_threads", None)
    params.pop("nthread", None)
    params.pop("ensemble_regime", None)
    iterations = int(params.pop("iterations", params.pop("n_estimators", 100)))
    depth = int(params.pop("depth", params.pop("max_depth", 6)))
    lr = float(params.pop("learning_rate", 0.05))
    threads = int(spec["thread_count"])
    regime = spec["ensemble_regime"]
    early_stopping_rounds = spec["early_stopping_rounds"]
    cat_indices = [index for index, field in enumerate(contract["fields"])
                   if field["logical_type"] == "categorical"]
    train_matrix = _engine_matrix(x_train, contract, engine)
    valid_matrix = _engine_matrix(x_valid, contract, engine)
    if engine == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        cls = CatBoostClassifier if task == "binary" else CatBoostRegressor
        base = dict(iterations=iterations, depth=depth, learning_rate=lr,
                    random_seed=spec["seed"], thread_count=threads, verbose=False,
                    allow_writing_files=False)
        if regime == "stochastic_boosting":
            base.update(bagging_temperature=float(params.pop("bagging_temperature", 1.0)),
                        rsm=float(params.pop("feature_sample", 0.8)))
        model = cls(**base, **params)
        train_pool = Pool(train_matrix, y_train, cat_features=cat_indices,
                          weight=train_weight)
        valid_pool = Pool(valid_matrix, y_valid, cat_features=cat_indices,
                          weight=valid_weight)
        fit_args = {"eval_set": valid_pool, "verbose": False}
        if early_stopping_rounds:
            fit_args["early_stopping_rounds"] = early_stopping_rounds
        model.fit(train_pool, **fit_args)
        return model, model.get_evals_result()
    if engine == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
        cls = LGBMClassifier if task == "binary" else LGBMRegressor
        base = dict(n_estimators=iterations, max_depth=depth, learning_rate=lr,
                    random_state=spec["seed"], n_jobs=threads, verbosity=-1)
        if regime == "independent_tree":
            base.update(boosting_type="rf", bagging_freq=1,
                        bagging_fraction=float(params.pop("row_sample", 0.8)),
                        feature_fraction=float(params.pop("feature_sample", 0.8)))
        elif regime == "stochastic_boosting":
            base.update(subsample=float(params.pop("row_sample", 0.8)), subsample_freq=1,
                        colsample_bytree=float(params.pop("feature_sample", 0.8)))
        model = cls(**base, **params)
        callbacks = [log_evaluation(0)]
        if early_stopping_rounds:
            callbacks.insert(0, early_stopping(early_stopping_rounds))
        model.fit(train_matrix, y_train, sample_weight=train_weight,
                  categorical_feature=cat_indices, eval_set=[(valid_matrix, y_valid)],
                  eval_sample_weight=[valid_weight] if valid_weight is not None else None,
                  callbacks=callbacks)
        return model, model.evals_result_
    from xgboost import XGBClassifier, XGBRegressor
    cls = XGBClassifier if task == "binary" else XGBRegressor
    base = dict(n_estimators=iterations, max_depth=depth, learning_rate=lr,
                random_state=spec["seed"], n_jobs=threads, tree_method="hist",
                enable_categorical=True,
                feature_types=["c" if field["logical_type"] == "categorical" else "q"
                               for field in contract["fields"]])
    if early_stopping_rounds:
        base["early_stopping_rounds"] = early_stopping_rounds
    if regime == "independent_tree":
        base.update(n_estimators=1, num_parallel_tree=iterations,
                    subsample=float(params.pop("row_sample", 0.8)),
                    colsample_bynode=float(params.pop("feature_sample", 0.8)))
    elif regime == "stochastic_boosting":
        base.update(subsample=float(params.pop("row_sample", 0.8)),
                    colsample_bytree=float(params.pop("feature_sample", 0.8)))
    model = cls(**base, **params)
    model.fit(train_matrix, y_train, sample_weight=train_weight,
              eval_set=[(valid_matrix, y_valid)],
              sample_weight_eval_set=[valid_weight] if valid_weight is not None else None,
              verbose=False)
    return model, model.evals_result()


def _predict_engine(engine: str, task: str, model: Any, frame: pl.DataFrame,
                    contract: Mapping[str, Any]) -> np.ndarray:
    matrix = _engine_matrix(frame, contract, engine)
    if engine == "catboost":
        from catboost import Pool
        cats = [index for index, field in enumerate(contract["fields"])
                if field["logical_type"] == "categorical"]
        matrix = Pool(matrix, cat_features=cats)
    if task == "binary":
        result = model.predict_proba(matrix)
        return np.asarray(result)[:, 1]
    return np.asarray(model.predict(matrix), dtype=float)


def _engine_matrix(frame: pl.DataFrame, contract: Mapping[str, Any],
                   engine: str) -> np.ndarray:
    """Create the narrow native algorithm boundary from canonical Polars data."""
    expressions = []
    for field in contract["fields"]:
        name = field["name"]
        if field["logical_type"] != "categorical":
            expressions.append(pl.col(name).cast(pl.Float64).alias(name))
            continue
        if engine == "catboost":
            expressions.append(pl.col(name).cast(pl.String).alias(name))
            continue
        levels = list(field.get("levels", []))
        mapping = {level: index for index, level in enumerate(levels)}
        expressions.append(
            pl.col(name).cast(pl.String).replace_strict(mapping, default=None)
            .cast(pl.Float64).alias(name)
        )
    prepared = frame.select(expressions)
    if engine == "catboost":
        return prepared.to_numpy().astype(object, copy=False)
    return prepared.to_numpy()


def _algorithm_boundary(engine: str) -> dict[str, Any]:
    """Describe the only materialization between Polars and the native engine."""
    if engine == "catboost":
        return {"input": "numpy_object_matrix", "native_wrapper": "catboost.Pool",
                "columns": "features_only", "copy_scope": "partition_only"}
    return {"input": "numpy_numeric_matrix", "native_wrapper": None,
            "columns": "features_only", "copy_scope": "partition_only"}


def _metrics(task: str, truth: np.ndarray, raw: np.ndarray, threshold: float,
             sample_weight: np.ndarray | None = None) -> dict[str, float]:
    if task == "regression":
        return {"rmse": float(mean_squared_error(truth, raw, sample_weight=sample_weight) ** 0.5),
                "mae": float(mean_absolute_error(truth, raw, sample_weight=sample_weight))}
    values = {"log_loss": float(log_loss(truth, raw, labels=[0, 1],
                                          sample_weight=sample_weight)),
              "accuracy": float(accuracy_score(truth, raw >= threshold,
                                                 sample_weight=sample_weight))}
    try:
        values["roc_auc"] = float(roc_auc_score(truth, raw, sample_weight=sample_weight))
    except ValueError:
        values["roc_auc"] = None
    return values


def _prediction_records(task: str, rows: np.ndarray, raw: np.ndarray,
                        threshold: float) -> list[dict[str, Any]]:
    if task == "binary":
        return [{"row_index": int(row), "positive_probability": float(value),
                 "decision": int(value >= threshold)} for row, value in zip(rows, raw)]
    return [{"row_index": int(row), "prediction": float(value)}
            for row, value in zip(rows, raw)]


def _evidence_manifest(engine: str, task: str, regime: str) -> list[dict[str, Any]]:
    common = [
        ("metrics", "CORE", "COMMON"),
        ("predictions", "CORE", "COMMON"),
        ("feature_importance", "CORE", "COMMON"),
        ("training_history", "DERIVED", "ENGINE_NATIVE"),
        ("shap_contributions", "EXTENDED", "ENGINE_NATIVE"),
    ]
    interactions = "ENGINE_NATIVE" if engine in ("catboost", "xgboost") else "UNAVAILABLE"
    common.append(("interaction_contributions", "SPECIALIZED", interactions))
    common.append(("out_of_bag", "DERIVED",
        "UNAVAILABLE" if regime != "independent_tree" or engine == "xgboost" else "DERIVABLE"))
    return [{"evidence": name, "importance_cost": cost,
             "semantic_availability": availability,
             "materialization": "on_demand" if cost in ("EXTENDED", "SPECIALIZED") else "eager"}
            for name, cost, availability in common]


def _save_model(engine: str, model: Any, root: Path) -> Path:
    suffix = {"catboost": "cbm", "lightgbm": "txt", "xgboost": "json"}[engine]
    path = root / f"model.{suffix}"
    if engine == "catboost":
        model.save_model(str(path))
    elif engine == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.get_booster().save_model(str(path))
    return path


def _load_model(engine: str, task: str, path: Path) -> Any:
    if engine == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        model = CatBoostClassifier() if task == "binary" else CatBoostRegressor()
        model.load_model(str(path))
        return model
    if engine == "lightgbm":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(path))
        class Wrapper:
            def predict(self, frame):
                return booster.predict(frame)
            def predict_proba(self, frame):
                p = np.asarray(booster.predict(frame))
                return np.column_stack([1 - p, p])
        return Wrapper()
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    class Wrapper:
        def predict(self, frame):
            return booster.predict(xgb.DMatrix(frame))
        def predict_proba(self, frame):
            p = np.asarray(booster.predict(xgb.DMatrix(frame)))
            return np.column_stack([1 - p, p])
    return Wrapper()


def _fingerprint(value: Any) -> str:
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
