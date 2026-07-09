import os
from collections.abc import Callable, Iterator
from typing import Any, Literal, cast

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as st

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel

EvalMetric = Literal[
    "mae",
    "rank_ic",
    "rank_ic_ir",
    "simulated_sharpe",
    "weighted_rank_ic",
    "weighted_rank_ic_ir",
]
WeightSource = Literal["turnover"]
WeightTransform = Literal["sqrt"]

_WEIGHT_BASE_COL = "__eval_weight_base"
_WEIGHT_COL = "__eval_weight"
_SEGMENT_COL = "__segment_idx"


def _compute_group_sizes(df: pl.DataFrame) -> np.ndarray:
    """按 datetime 生成每个横截面的样本数。"""
    return (
        df.group_by("datetime", maintain_order=True)
        .len()
        .select("len")
        .to_numpy()
        .astype(np.int32)
        .flatten()
    )


def _iter_group_slices(group_sizes: np.ndarray) -> Iterator[tuple[int, int]]:
    """根据 group size 顺序生成切片区间。"""
    start = 0
    for size in group_sizes:
        end = start + int(size)
        yield start, end
        start = end


def _daily_rank_ic_series(preds: np.ndarray, labels: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
    """按横截面计算每日 Spearman Rank IC 序列。"""
    return _daily_rank_ic_series_weighted_mask(preds, labels, group_sizes, None)


def _daily_rank_ic_series_weighted_mask(
    preds: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
    sample_weights: np.ndarray | None,
) -> np.ndarray:
    """按横截面计算每日 Spearman Rank IC 序列，可用样本权重过滤样本。"""
    scores: list[float] = []

    for start, end in _iter_group_slices(group_sizes):
        group_preds = preds[start:end]
        group_labels = labels[start:end]
        mask = np.isfinite(group_preds) & np.isfinite(group_labels)
        if sample_weights is not None and sample_weights.size == labels.size:
            group_weights = sample_weights[start:end]
            mask &= np.isfinite(group_weights) & (group_weights > 0)

        if np.count_nonzero(mask) < 2:
            continue

        group_preds = group_preds[mask]
        group_labels = group_labels[mask]

        if np.allclose(group_preds, group_preds[0]) or np.allclose(group_labels, group_labels[0]):
            continue

        corr, _ = st.spearmanr(group_preds, group_labels)
        if np.isfinite(corr):
            scores.append(float(corr))

    return np.asarray(scores, dtype=np.float64)


def _weighted_rank_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """计算带权重的 Rank Correlation。"""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")

    x_rank = np.asarray(st.rankdata(x[mask], method="average"), dtype=np.float64)
    y_rank = np.asarray(st.rankdata(y[mask], method="average"), dtype=np.float64)
    weight_values = np.asarray(weights[mask], dtype=np.float64)

    weight_sum = float(weight_values.sum())
    if not weight_sum > 0:
        return float("nan")

    weight_values = weight_values / weight_sum
    x_center = x_rank - np.sum(weight_values * x_rank)
    y_center = y_rank - np.sum(weight_values * y_rank)

    cov = float(np.sum(weight_values * x_center * y_center))
    var_x = float(np.sum(weight_values * x_center * x_center))
    var_y = float(np.sum(weight_values * y_center * y_center))

    if not (var_x > 0 and var_y > 0):
        return float("nan")

    return float(cov / np.sqrt(var_x * var_y))


def _daily_weighted_rank_ic_series(
    preds: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    """按横截面计算每日带权 Rank IC 序列。"""
    if weights is None or weights.size != labels.size:
        return np.asarray([], dtype=np.float64)

    scores: list[float] = []

    for start, end in _iter_group_slices(group_sizes):
        corr = _weighted_rank_corr(preds[start:end], labels[start:end], weights[start:end])
        if np.isfinite(corr):
            scores.append(float(corr))

    return np.asarray(scores, dtype=np.float64)


def _daily_topk_return_series(
    preds: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
    top_k: int,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """按横截面构造每日 Top-K 多头收益序列。"""
    scores: list[float] = []

    for start, end in _iter_group_slices(group_sizes):
        group_preds = preds[start:end]
        group_labels = labels[start:end]
        mask = np.isfinite(group_preds) & np.isfinite(group_labels)
        if sample_weights is not None and sample_weights.size == labels.size:
            group_weights = sample_weights[start:end]
            mask &= np.isfinite(group_weights) & (group_weights > 0)

        valid_count = int(np.count_nonzero(mask))
        if valid_count <= 0:
            continue

        group_preds = group_preds[mask]
        group_labels = group_labels[mask]
        take_k = min(int(top_k), valid_count)
        if take_k <= 0:
            continue

        order = np.argsort(group_preds)[::-1]
        scores.append(float(np.mean(group_labels[order[:take_k]])))

    return np.asarray(scores, dtype=np.float64)


def _mean_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def _ir_or_zero(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0

    mean_value = float(np.nanmean(values))
    std_value = float(np.nanstd(values, ddof=1))

    if not (np.isfinite(mean_value) and np.isfinite(std_value) and std_value > 0):
        return 0.0

    return float(mean_value / std_value)


def _annualized_sharpe_or_zero(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0

    mean_value = float(np.nanmean(values))
    std_value = float(np.nanstd(values, ddof=1))

    if not (np.isfinite(mean_value) and np.isfinite(std_value) and std_value > 0):
        return 0.0

    return float(np.sqrt(252.0) * mean_value / std_value)


class LgbModel(AlphaModel):
    """
    量化 LightGBM：
    - regression: 直接拟合连续收益率
    - eval_metric: 支持 rank_ic / rank_ic_ir / simulated_sharpe / weighted_rank_ic / weighted_rank_ic_ir
    """

    def __init__(
        self,
        objective: str = "regression",
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        feature_fraction: float = 0.6,
        bagging_fraction: float = 0.7,
        bagging_freq: int = 1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        min_data_in_leaf: int | None = None,
        min_data_in_leaf_daily_fraction: float | None = 0.05,
        log_evaluation_period: int = 20,
        seed: int = 42,
        use_gpu: bool = True,
        gpu_platform_id: int | None = None,
        gpu_device_id: int | None = None,
        num_threads: int = -1,
        eval_metric: EvalMetric = "rank_ic",
        top_k: int = 50,
        weight_source: WeightSource = "turnover",
        weight_window: int = 20,
        weight_lag: int = 1,
        weight_transform: WeightTransform = "sqrt",
        weight_multiplier_col: str | None = None,
        extra_trees: bool = False,
        feature_fraction_bynode: float | None = None,
    ) -> None:
        if objective != "regression":
            raise ValueError("lgb_model_mod.py 当前仅保留 'regression' 功能")
        if min_data_in_leaf is not None and int(min_data_in_leaf) <= 0:
            raise ValueError("min_data_in_leaf 必须 > 0")
        if min_data_in_leaf_daily_fraction is not None:
            fraction = float(min_data_in_leaf_daily_fraction)
            if not (0 < fraction <= 1):
                raise ValueError("min_data_in_leaf_daily_fraction 必须在 (0, 1] 之间，或为 None")
        if int(top_k) <= 0:
            raise ValueError("top_k 必须 > 0")
        if int(weight_window) <= 0:
            raise ValueError("weight_window 必须 > 0")
        if int(weight_lag) < 0:
            raise ValueError("weight_lag 必须 >= 0")
        if int(num_threads) == 0 or int(num_threads) < -1:
            raise ValueError("num_threads 必须为正整数或 -1")
        if weight_multiplier_col is not None and not str(weight_multiplier_col).strip():
            raise ValueError("weight_multiplier_col 不能为空字符串")

        valid_metrics = {
            "mae",
            "rank_ic",
            "rank_ic_ir",
            "simulated_sharpe",
            "weighted_rank_ic",
            "weighted_rank_ic_ir",
        }
        normalized_metric = str(eval_metric).strip().lower()
        if normalized_metric not in valid_metrics:
            raise ValueError(f"不支持的 eval_metric: {eval_metric}")

        if weight_source != "turnover":
            raise ValueError("当前 weighted 指标仅支持 weight_source='turnover'")
        if weight_transform != "sqrt":
            raise ValueError("当前仅支持 weight_transform='sqrt'")

        self.objective: str = objective
        self.eval_metric: EvalMetric = cast(EvalMetric, normalized_metric)
        self.top_k: int = int(top_k)
        self.weight_source: WeightSource = weight_source
        self.weight_window: int = int(weight_window)
        self.weight_lag: int = int(weight_lag)
        self.weight_transform: WeightTransform = weight_transform
        self.weight_multiplier_col: str | None = (
            None if weight_multiplier_col is None else str(weight_multiplier_col)
        )
        self.min_data_in_leaf_daily_fraction: float | None = (
            None if min_data_in_leaf_daily_fraction is None else float(min_data_in_leaf_daily_fraction)
        )

        self.params: dict[str, Any] = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": reg_alpha,
            "lambda_l2": reg_lambda,
            "verbosity": -1,
            "seed": seed,
            "num_threads": int(num_threads),
        }
        if self.eval_metric == "mae":
            self.params["metric"] = ["mse", "mae"]
        else:
            self.params["metric"] = "None"
        if min_data_in_leaf is not None:
            self.params["min_data_in_leaf"] = int(min_data_in_leaf)
        if extra_trees:
            self.params["extra_trees"] = True
        if feature_fraction_bynode is not None:
            self.params["feature_fraction_bynode"] = float(feature_fraction_bynode)

        if use_gpu:
            self.params["device_type"] = "gpu"
            if gpu_platform_id is not None:
                self.params["gpu_platform_id"] = int(gpu_platform_id)
            if gpu_device_id is not None:
                self.params["gpu_device_id"] = int(gpu_device_id)
        else:
            self.params["device_type"] = "cpu"

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds
        self.log_evaluation_period: int = log_evaluation_period

        self.model: lgb.Booster | None = None
        self.resolved_min_data_in_leaf: int | None = None
        self.feature_names: list[str] = []

    def _resolve_min_data_in_leaf(self, train_data: lgb.Dataset) -> int | None:
        if "min_data_in_leaf" in self.params:
            return int(self.params["min_data_in_leaf"])
        if self.min_data_in_leaf_daily_fraction is None:
            return None

        group_sizes = np.asarray(train_data.get_group(), dtype=np.float64)
        group_sizes = group_sizes[np.isfinite(group_sizes) & (group_sizes > 0)]
        if group_sizes.size == 0:
            return None

        # LightGBM 的 min_data_in_leaf 是全局树参数，不能按交易日动态变化；
        # 这里用训练期每日截面样本数的中位数 * 比例，避免误用全训练集总样本数。
        return max(1, int(round(float(np.median(group_sizes)) * self.min_data_in_leaf_daily_fraction)))

    def _uses_weighted_metric(self) -> bool:
        return self.eval_metric in {"weighted_rank_ic", "weighted_rank_ic_ir"}

    def _build_eval_weight(self, df: pl.DataFrame) -> pl.DataFrame:
        """构造按 vt_symbol 回看的滞后流动性权重。"""
        if self.weight_source not in df.columns:
            raise ValueError(f"{self.eval_metric} 需要列: {self.weight_source}")

        weighted_df = df.sort(["vt_symbol", "datetime"]).with_columns(
            pl.col(self.weight_source)
            .cast(pl.Float64, strict=False)
            .shift(self.weight_lag)
            .rolling_mean(window_size=self.weight_window, min_samples=1)
            .over("vt_symbol")
            .alias(_WEIGHT_BASE_COL)
        )

        return weighted_df.with_columns(
            pl.when(pl.col(_WEIGHT_BASE_COL) > 0)
            .then(pl.col(_WEIGHT_BASE_COL).sqrt())
            .otherwise(None)
            .alias(_WEIGHT_COL)
        )

    def _prepare_segment_dataset(self, df: pl.DataFrame) -> tuple[lgb.Dataset, np.ndarray | None]:
        """把单个分段数据转成 LightGBM Dataset。"""
        df = df.sort(["datetime", "vt_symbol"])

        data = df.select(self.feature_names).to_numpy()
        labels = np.asarray(df["label"], dtype=np.float64)
        group_sizes = _compute_group_sizes(df)

        if int(group_sizes.sum()) != int(labels.size):
            raise ValueError("group size 与标签长度不一致，请检查输入数据是否按日分组完整")

        eval_weight: np.ndarray | None = None
        sample_weight: np.ndarray | None = None
        if self.weight_multiplier_col is not None:
            if self.weight_multiplier_col not in df.columns:
                raise ValueError(f"缺少 weight_multiplier_col: {self.weight_multiplier_col}")
            sample_weight = np.asarray(df[self.weight_multiplier_col], dtype=np.float64)
            sample_weight = np.where(np.isfinite(sample_weight) & (sample_weight > 0), sample_weight, 0.0)

        if self._uses_weighted_metric():
            if _WEIGHT_COL not in df.columns:
                raise ValueError(f"{self.eval_metric} 缺少辅助权重列，请检查数据准备流程")
            eval_weight = np.asarray(df[_WEIGHT_COL], dtype=np.float64)
            if sample_weight is not None:
                eval_weight = eval_weight * sample_weight

        lgb_data = lgb.Dataset(
            data,
            label=labels,
            weight=sample_weight,
            group=group_sizes,
            feature_name=self.feature_names,
        )
        try:
            setattr(lgb_data, "_vnpy_eval_weight", eval_weight)
            setattr(lgb_data, "_vnpy_sample_weight", sample_weight)
        except Exception:
            pass

        return lgb_data, eval_weight

    def _prepare_data(self, dataset: AlphaDataset) -> tuple[list[lgb.Dataset], list[np.ndarray | None]]:
        datasets: list[lgb.Dataset] = []
        eval_weights: list[np.ndarray | None] = []

        sample_df = dataset.fetch_learn(Segment.TRAIN)
        exclude = {"datetime", "vt_symbol", "label"}
        if self.weight_multiplier_col is not None:
            exclude.add(self.weight_multiplier_col)
        self.feature_names = [c for c in sample_df.columns if c not in exclude]

        segment_frames: list[pl.DataFrame] = []
        for segment_idx, segment in enumerate([Segment.TRAIN, Segment.VALID]):
            df = dataset.fetch_learn(segment).sort(["datetime", "vt_symbol"])
            if self._uses_weighted_metric():
                df = df.with_columns(pl.lit(segment_idx).alias(_SEGMENT_COL))
            segment_frames.append(df)

        if self._uses_weighted_metric():
            weighted_frame = self._build_eval_weight(pl.concat(segment_frames))
            for segment_idx in range(len(segment_frames)):
                segment_df = (
                    weighted_frame.filter(pl.col(_SEGMENT_COL) == segment_idx)
                    .drop([_SEGMENT_COL, _WEIGHT_BASE_COL])
                    .sort(["datetime", "vt_symbol"])
                )
                lgb_data, eval_weight = self._prepare_segment_dataset(segment_df)
                datasets.append(lgb_data)
                eval_weights.append(eval_weight)
        else:
            for df in segment_frames:
                lgb_data, eval_weight = self._prepare_segment_dataset(df)
                datasets.append(lgb_data)
                eval_weights.append(eval_weight)

        return datasets, eval_weights

    def _build_eval_func(
        self,
        datasets: list[lgb.Dataset],
        eval_weights: list[np.ndarray | None],
    ) -> Callable[[np.ndarray, lgb.Dataset], tuple[str, float, bool]]:
        metric_name = self.eval_metric
        weight_map = {id(ds): weights for ds, weights in zip(datasets, eval_weights)}

        def _eval(preds: np.ndarray, train_data: lgb.Dataset) -> tuple[str, float, bool]:
            labels = np.asarray(train_data.get_label(), dtype=np.float64)
            group_sizes = np.asarray(train_data.get_group(), dtype=np.int32)
            pred_values = np.asarray(preds, dtype=np.float64)
            sample_weights = cast(
                np.ndarray | None,
                getattr(train_data, "_vnpy_sample_weight", train_data.get_weight()),
            )

            if metric_name == "rank_ic":
                score = _mean_or_zero(
                    _daily_rank_ic_series_weighted_mask(pred_values, labels, group_sizes, sample_weights)
                )
            elif metric_name == "rank_ic_ir":
                score = _ir_or_zero(
                    _daily_rank_ic_series_weighted_mask(pred_values, labels, group_sizes, sample_weights)
                )
            elif metric_name == "simulated_sharpe":
                score = _annualized_sharpe_or_zero(
                    _daily_topk_return_series(
                        pred_values,
                        labels,
                        group_sizes,
                        top_k=self.top_k,
                        sample_weights=sample_weights,
                    )
                )
            elif metric_name == "weighted_rank_ic":
                weights = cast(
                    np.ndarray | None,
                    getattr(train_data, "_vnpy_eval_weight", weight_map.get(id(train_data))),
                )
                score = _mean_or_zero(
                    _daily_weighted_rank_ic_series(pred_values, labels, group_sizes, weights)
                )
            elif metric_name == "weighted_rank_ic_ir":
                weights = cast(
                    np.ndarray | None,
                    getattr(train_data, "_vnpy_eval_weight", weight_map.get(id(train_data))),
                )
                score = _ir_or_zero(
                    _daily_weighted_rank_ic_series(pred_values, labels, group_sizes, weights)
                )
            else:
                raise RuntimeError(f"未知 eval_metric: {metric_name}")

            return metric_name, score, True

        return _eval

    def fit(self, dataset: AlphaDataset) -> None:
        datasets, eval_weights = self._prepare_data(dataset)
        params = dict(self.params)
        resolved_min_data_in_leaf = self._resolve_min_data_in_leaf(datasets[0])
        self.resolved_min_data_in_leaf = resolved_min_data_in_leaf
        if resolved_min_data_in_leaf is not None:
            params["min_data_in_leaf"] = resolved_min_data_in_leaf

        callbacks = []
        if self.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
        if self.log_evaluation_period > 0:
            callbacks.append(lgb.log_evaluation(self.log_evaluation_period))

        feval = None if self.eval_metric == "mae" else self._build_eval_func(datasets, eval_weights)

        self.model = lgb.train(
            params,
            datasets[0],
            num_boost_round=self.num_boost_round,
            valid_sets=datasets,
            valid_names=["train", "valid"],
            feval=feval,
            callbacks=callbacks,
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型尚未训练！")

        df = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])
        data = df.select(self.feature_names).to_numpy()

        result = cast(np.ndarray, self.model.predict(data))
        return result

    def detail(
        self,
        *,
        shap: bool = False,
        dataset: AlphaDataset | None = None,
        sample_size: int = 5000,
        plot_shap: bool = False,
    ) -> pd.DataFrame | None:
        """返回特征重要性，按需追加验证集 SHAP 重要性。"""
        if not self.model:
            return None

        importance_split = self.model.feature_importance(importance_type="split")
        importance_gain = self.model.feature_importance(importance_type="gain")
        feature_name = self.model.feature_name()

        df = pd.DataFrame(
            {
                "Feature": feature_name,
                "Importance_Split": importance_split,
                "Importance_Gain": importance_gain,
            }
        )
        df = df.sort_values("Importance_Gain", ascending=False).reset_index(drop=True)

        if os.environ.get("VNPY_LGBM_PLOT_IMPORTANCE", "").strip() == "1":
            try:
                lgb.plot_importance(
                    self.model,
                    max_num_features=20,
                    importance_type="gain",
                    title="Feature Importance (Gain)",
                )
                plt.show()
            except Exception:
                pass

        if plot_shap:
            shap = True

        if shap:
            if dataset is None:
                raise ValueError("计算 SHAP 需要传入 dataset")
            if int(sample_size) <= 0:
                raise ValueError("sample_size 必须 > 0")

            shap_feature_names = self.feature_names or list(feature_name)
            valid_df = dataset.fetch_infer(Segment.VALID).sort(["datetime", "vt_symbol"])

            missing_features = [name for name in shap_feature_names if name not in valid_df.columns]
            if missing_features:
                preview = ", ".join(missing_features[:5])
                raise ValueError(f"验证集缺少模型特征列: {preview}")

            feature_df = valid_df.select(shap_feature_names)
            if feature_df.is_empty():
                raise ValueError("验证集没有可用于 SHAP 的样本")

            sample_count = min(int(sample_size), feature_df.height)
            if feature_df.height > sample_count:
                feature_df = feature_df.sample(n=sample_count, shuffle=True, seed=42)

            sample_data = feature_df.to_numpy()
            shap_values = cast(
                np.ndarray,
                self.model.predict(sample_data, pred_contrib=True),
            )
            if shap_values.ndim != 2 or shap_values.shape[1] != len(shap_feature_names) + 1:
                raise ValueError("LightGBM SHAP 输出维度与特征数量不一致")

            shap_importance = np.abs(shap_values[:, :-1]).mean(axis=0)
            shap_df = pd.DataFrame(
                {
                    "Feature": shap_feature_names,
                    "Importance_SHAP": shap_importance,
                }
            )
            df = df.merge(shap_df, on="Feature", how="left")
            df = df.sort_values("Importance_SHAP", ascending=False).reset_index(drop=True)

            if plot_shap:
                try:
                    import shap as shap_lib

                    sample_frame = pd.DataFrame(sample_data, columns=shap_feature_names)
                    shap_lib.summary_plot(shap_values[:, :-1], sample_frame, show=False)
                    plt.title("SHAP Summary Plot (VALID)")
                    plt.show()
                except ImportError:
                    print(
                        "未检测到 shap 库，已跳过 SHAP 绘图。"
                        "请使用 pip install shap 安装。"
                    )
                except Exception as exc:
                    print(f"SHAP 绘图失败，已跳过绘图: {exc}")

        return df
