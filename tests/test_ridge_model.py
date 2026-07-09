from datetime import datetime, timedelta

import joblib
import numpy as np
import polars as pl

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model.models.ridge_model import RidgeModel


def make_dataset() -> AlphaDataset:
    rows = []
    start = datetime(2024, 1, 1)
    symbols = ["000001.SZ", "000002.SZ"]

    for day_index in range(8):
        dt = start + timedelta(days=day_index)
        for symbol_index, vt_symbol in enumerate(symbols):
            feature_1 = float(day_index + symbol_index)
            feature_2 = float(day_index * 2 - symbol_index)
            label = 0.5 * feature_1 - 0.25 * feature_2 + symbol_index
            rows.append(
                {
                    "datetime": dt,
                    "vt_symbol": vt_symbol,
                    "alpha_001": feature_1,
                    "alpha_002": feature_2,
                    "sample_weight": 1.0 + symbol_index,
                    "label": label,
                }
            )

    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.when((pl.col("datetime") == start + timedelta(days=1)) & (pl.col("vt_symbol") == "000001.SZ"))
        .then(None)
        .otherwise(pl.col("alpha_001"))
        .alias("alpha_001"),
        pl.when((pl.col("datetime") == start + timedelta(days=6)) & (pl.col("vt_symbol") == "000002.SZ"))
        .then(float("inf"))
        .otherwise(pl.col("alpha_002"))
        .alias("alpha_002"),
    )

    dataset = AlphaDataset(
        df=df.select(["datetime", "vt_symbol"]).unique(),
        train_period=("2024-01-01", "2024-01-04"),
        valid_period=("2024-01-05", "2024-01-05"),
        test_period=("2024-01-06", "2024-01-08"),
    )
    dataset.result_df = df
    dataset.raw_df = df
    dataset.infer_df = df
    dataset.learn_df = df
    return dataset


def test_ridge_model_fit_predict_detail_and_pickle(tmp_path):
    dataset = make_dataset()
    model = RidgeModel(
        alpha=1.0,
        impute_strategy="constant",
        impute_fill_value=0.0,
        sample_weight_col="sample_weight",
    )

    model.fit(dataset)
    pred = model.predict(dataset, Segment.TEST)
    detail = model.detail()

    assert model.feature_names == ["alpha_001", "alpha_002"]
    assert model.input_size == 2
    assert model.params["impute_strategy"] == "constant"
    assert model.params["impute_fill_value"] == 0.0
    assert pred.shape == (6,)
    assert np.isfinite(pred).all()
    assert detail is not None
    assert list(detail.columns) == ["Feature", "Coef", "Abs_Coef"]
    assert set(detail["Feature"]) == {"alpha_001", "alpha_002"}

    model_path = tmp_path / "ridge_model.pkl"
    joblib.dump(model, model_path)
    loaded = joblib.load(model_path)

    loaded_pred = loaded.predict(dataset, Segment.TEST)
    np.testing.assert_allclose(pred, loaded_pred)


def test_ridge_model_cv_selects_alpha():
    dataset = make_dataset()
    model = RidgeModel(alphas=[0.1, 1.0, 10.0], cv_folds=3)

    model.fit(dataset)

    assert model.selected_alpha in {0.1, 1.0, 10.0}
