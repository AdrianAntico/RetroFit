import json

import numpy as np
import polars as pl
import pytest

from retrofit.supervised_reference import (
    SupervisedReferenceError,
    fit,
    implementation_descriptor,
    normalize_ensemble_regime,
    predict,
    supervised_reference_contract,
)


def fixture(task="regression", rows=100):
    rng = np.random.default_rng(17)
    data = pl.DataFrame({
        "numeric": rng.normal(size=rows),
        "category": np.where(np.arange(rows) % 3, "a", "b"),
    })
    if task == "regression":
        data = data.with_columns((2 * pl.col("numeric") +
            (pl.col("category") == "a").cast(pl.Int8) + rng.normal(0, .05, rows)).alias("target"))
    else:
        data = data.with_columns(
            pl.when(pl.col("numeric") + (pl.col("category") == "a").cast(pl.Int8) > .4)
            .then(pl.lit("yes")).otherwise(pl.lit("no")).alias("target"))
    return data


def request(engine, task="regression", regime="boosting"):
    return {"engine": engine, "task": task, "target": "target",
            "features": ["numeric", "category"], "seed": 9,
            "train_fraction": .8, "threshold": .5,
            "positive_class": "yes" if task == "binary" else None,
            "ensemble_regime": regime, "thread_count": 1,
            "engine_params": {"iterations": 20, "depth": 3}}


def test_contract_separates_family_engine_and_regime():
    contract = supervised_reference_contract()
    assert contract["algorithm_dimensions"] == ["algorithm_family", "engine", "ensemble_regime"]
    assert implementation_descriptor("catboost", "binary")["runtime"] == "python"


def test_regime_contract_is_truthful():
    assert normalize_ensemble_regime("lightgbm", "independent_tree") == "independent_tree"
    assert normalize_ensemble_regime("xgboost", "independent_tree") == "independent_tree"
    with pytest.raises(SupervisedReferenceError) as failure:
        normalize_ensemble_regime("catboost", "independent_tree")
    assert failure.value.code == "invalid_parameter_combination"


@pytest.mark.parametrize("engine", ["catboost", "lightgbm", "xgboost"])
@pytest.mark.parametrize("task", ["regression", "binary"])
def test_fit_persist_reload_score_without_refit(tmp_path, engine, task):
    data = fixture(task)
    root = tmp_path / f"{engine}-{task}"
    result = fit(request(engine, task), data, root)
    scored = predict(root, data.drop("target"))
    assert result["manifest"]["engine"] == engine
    assert result["manifest"]["task"] == task
    assert result["manifest"]["preprocessing"]["dataframe_authority"] == "polars"
    assert result["manifest"]["preprocessing"]["pandas_used"] is False
    assert result["manifest"]["preprocessing"]["algorithm_boundary"]["columns"] == "features_only"
    assert scored["refit"] is False
    assert len(scored["predictions"]) == len(data)
    if task == "binary":
        assert all(0 <= row["positive_probability"] <= 1 for row in scored["predictions"])
    json.dumps(result["manifest"])


def test_payload_digest_is_enforced(tmp_path):
    data = fixture()
    root = tmp_path / "bundle"
    result = fit(request("catboost"), data, root)
    model = root / result["manifest"]["model_file"]
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(SupervisedReferenceError) as failure:
        predict(root, data.drop("target"))
    assert failure.value.code == "fitted_payload_incompatible"


def test_qualified_path_is_polars_first_and_has_no_pandas_boundary():
    import inspect
    import retrofit.supervised_reference as subject

    source = inspect.getsource(subject)
    assert "import pandas" not in source
    assert ".to_pandas(" not in source
    assert subject.read_parquet.__annotations__["return"] == "pl.DataFrame"


@pytest.mark.parametrize("engine", ["catboost", "lightgbm", "xgboost"])
def test_weighted_fit_persists_weight_contract_and_scores(tmp_path, engine):
    data = fixture("regression").with_columns(
        pl.when(pl.col("numeric") > 0).then(10.0).otherwise(0.25).alias("exposure")
    )
    spec = request(engine, "regression")
    spec.update({"weight": "exposure", "early_stopping_rounds": 0})

    result = fit(spec, data, tmp_path / engine)
    scored = predict(tmp_path / engine, data.drop("target", "exposure"))

    assert result["manifest"]["weight_contract"]["enabled"] is True
    assert result["manifest"]["weight_contract"]["column"] == "exposure"
    assert result["manifest"]["weight_contract"]["positive_rows"] == data.height
    assert len(scored["predictions"]) == data.height
    assert scored["refit"] is False


@pytest.mark.parametrize("values", [
    [-1.0] * 100,
    [0.0] * 100,
    [float("nan")] * 100,
])
def test_invalid_observation_weights_fail_closed(tmp_path, values):
    data = fixture("regression").with_columns(pl.Series("exposure", values))
    spec = request("catboost", "regression")
    spec["weight"] = "exposure"

    with pytest.raises(SupervisedReferenceError) as error:
        fit(spec, data, tmp_path / "invalid")

    assert error.value.code == "weight_contract_mismatch"
