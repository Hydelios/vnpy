import os
from collections.abc import Callable, Iterator
from datetime import datetime
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
WeightSource = Literal["volume", "turnover"]
WeightTransform = Literal["sqrt", "none"]
WeightNormalize = Literal["daily_mean", "none"]
TimeDecayMode = Literal["multiply", "replace"]
TimeDecayNormalize = Literal["global_mean", "none"]

_WEIGHT_BASE_COL = "__eval_weight_base"
_WEIGHT_COL = "__eval_weight"
_WEIGHT_DAILY_MEDIAN_COL = "__eval_weight_daily_median"
_WEIGHT_DAILY_MEAN_COL = "__eval_weight_daily_mean"
_WEIGHT_SEGMENT_MEAN_COL = "__eval_weight_segment_mean"
_WEIGHT_MULTIPLIER_COL = "__eval_weight_multiplier"
_TIME_DECAY_COL = "__eval_time_decay_weight"
_TIME_DECAY_DAYS_AGO_COL = "__eval_time_decay_days_ago"
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
    scores: list[float] = []

    for start, end in _iter_group_slices(group_sizes):
        group_preds = preds[start:end]
        group_labels = labels[start:end]
        mask = np.isfinite(group_preds) & np.isfinite(group_labels)

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
) -> np.ndarray:
    """按横截面构造每日 Top-K 多头收益序列。"""
    scores: list[float] = []

    for start, end in _iter_group_slices(group_sizes):
        group_preds = preds[start:end]
        group_labels = labels[start:end]
        mask = np.isfinite(group_preds) & np.isfinite(group_labels)

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
        learning_rate: float = 0.01,
        num_leaves: int = 31,
        max_depth: int = -1,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        feature_fraction: float = 0.6,
        bagging_fraction: float = 0.7,
        bagging_freq: int = 1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        min_data_in_leaf: int | None = 150,
        min_data_in_leaf_daily_fraction: float | None = 0.05,
        min_sum_hessian_in_leaf: float | None = 1e-3,
        max_bin: int = 255,
        log_evaluation_period: int = 20,
        seed: int = 42,
        use_gpu: bool = True,
        gpu_platform_id: int | None = None,
        gpu_device_id: int | None = None,
        num_threads: int = 6,
        eval_metric: EvalMetric = "mae",
        top_k: int = 50,
        seed_list: list[int] | tuple[int, ...] | None = None,
        use_sample_weight: bool = True,
        weight_source: WeightSource = "volume",
        weight_window: int = 1,
        weight_lag: int = 0,
        weight_transform: WeightTransform = "sqrt",
        weight_clip_median_multiple: float | None = 5.0,
        weight_normalize: WeightNormalize = "daily_mean",
        keep_weight_source_feature: bool = False,
        time_decay_half_life_days: float | None = None,
        time_decay_mode: TimeDecayMode = "multiply",
        time_decay_normalize: TimeDecayNormalize = "global_mean",
        refit_with_valid: bool = False,
        extra_trees: bool = False,
        feature_fraction_bynode: float | None = None,
        weight_multiplier_col: str | None = None,
        use_cv: bool = False,
        cv_n_splits: int = 5,
        cv_purge_days: int = 0,
    ) -> None:
        if objective != "regression":
            raise ValueError("lgb_model_sp.py 当前仅保留 'regression' 功能")
        if min_data_in_leaf is not None and int(min_data_in_leaf) <= 0:
            raise ValueError("min_data_in_leaf 必须 > 0")
        if min_data_in_leaf_daily_fraction is not None:
            fraction = float(min_data_in_leaf_daily_fraction)
            if not (0 < fraction <= 1):
                raise ValueError("min_data_in_leaf_daily_fraction 必须在 (0, 1] 之间，或为 None")
        if min_sum_hessian_in_leaf is not None and float(min_sum_hessian_in_leaf) < 0:
            raise ValueError("min_sum_hessian_in_leaf 必须 >= 0")
        if int(max_bin) <= 1:
            raise ValueError("max_bin 必须 > 1")
        if int(top_k) <= 0:
            raise ValueError("top_k 必须 > 0")
        if int(weight_window) <= 0:
            raise ValueError("weight_window 必须 > 0")
        if int(weight_lag) < 0:
            raise ValueError("weight_lag 必须 >= 0")
        if int(cv_n_splits) < 2:
            raise ValueError("cv_n_splits 必须 >= 2")
        if int(cv_purge_days) < 0:
            raise ValueError("cv_purge_days 必须 >= 0")
        if int(num_threads) == 0 or int(num_threads) < -1:
            raise ValueError("num_threads 必须为正整数或 -1")

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
        if seed_list is not None and len(seed_list) == 0:
            raise ValueError("seed_list 不能为空")

        if weight_source not in {"volume", "turnover"}:
            raise ValueError("weight_source 仅支持 'volume' 或 'turnover'")
        if weight_transform not in {"sqrt", "none"}:
            raise ValueError("weight_transform 仅支持 'sqrt' 或 'none'")
        if weight_normalize not in {"daily_mean", "none"}:
            raise ValueError("weight_normalize 仅支持 'daily_mean' 或 'none'")
        if weight_clip_median_multiple is not None and float(weight_clip_median_multiple) <= 0:
            raise ValueError("weight_clip_median_multiple 必须 > 0 或为 None")
        if time_decay_half_life_days is not None and float(time_decay_half_life_days) <= 0:
            raise ValueError("time_decay_half_life_days 必须 > 0 或为 None")
        if time_decay_mode not in {"multiply", "replace"}:
            raise ValueError("time_decay_mode 仅支持 'multiply' 或 'replace'")
        if time_decay_normalize not in {"global_mean", "none"}:
            raise ValueError("time_decay_normalize 仅支持 'global_mean' 或 'none'")
        if weight_multiplier_col is not None and not str(weight_multiplier_col).strip():
            raise ValueError("weight_multiplier_col 不能为空字符串")

        self.objective: str = objective
        self.eval_metric: EvalMetric = cast(EvalMetric, normalized_metric)
        self.top_k: int = int(top_k)
        self.seeds: list[int] = [int(item) for item in seed_list] if seed_list is not None else [int(seed)]
        self.use_sample_weight: bool = bool(use_sample_weight)
        self.weight_source: WeightSource = weight_source
        self.weight_window: int = int(weight_window)
        self.weight_lag: int = int(weight_lag)
        self.weight_transform: WeightTransform = weight_transform
        self.weight_clip_median_multiple: float | None = (
            None if weight_clip_median_multiple is None else float(weight_clip_median_multiple)
        )
        self.weight_normalize: WeightNormalize = weight_normalize
        self.keep_weight_source_feature: bool = bool(keep_weight_source_feature)
        self.time_decay_half_life_days: float | None = (
            None if time_decay_half_life_days is None else float(time_decay_half_life_days)
        )
        self.time_decay_mode: TimeDecayMode = time_decay_mode
        self.time_decay_normalize: TimeDecayNormalize = time_decay_normalize
        self.refit_with_valid: bool = bool(refit_with_valid)
        self.weight_multiplier_col: str | None = weight_multiplier_col
        self.min_data_in_leaf_daily_fraction: float | None = (
            None if min_data_in_leaf_daily_fraction is None else float(min_data_in_leaf_daily_fraction)
        )
        self.use_cv: bool = bool(use_cv)
        self.cv_n_splits: int = int(cv_n_splits)
        self.cv_purge_days: int = int(cv_purge_days)

        if self.use_cv:
            if len(self.seeds) != 1:
                raise ValueError("use_cv=True 时暂不支持 seed_list 多模型，请仅保留一个 seed")
            if self.refit_with_valid:
                raise ValueError("use_cv=True 时不支持 refit_with_valid")

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
            "max_bin": int(max_bin),
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
            
        if min_sum_hessian_in_leaf is not None:
            self.params["min_sum_hessian_in_leaf"] = float(min_sum_hessian_in_leaf)
            
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

        self.model: lgb.Booster | list[lgb.Booster] | None = None
        self.models: list[lgb.Booster] = []
        self.resolved_min_data_in_leaf: int | None = None
        self.feature_names: list[str] = []
        self.early_stopped_best_iterations: list[int] = []
        self.early_stopped_best_scores: list[dict[str, Any]] = []
        self.cv_fold_summaries: list[dict[str, Any]] = []
        self.cv_best_iterations: list[int] = []
        self.cv_best_scores: list[dict[str, Any]] = []

    def _infer_feature_names(self, sample_df: pl.DataFrame) -> None:
        exclude = {
            "datetime",
            "vt_symbol",
            "label",
            _WEIGHT_BASE_COL,
            _WEIGHT_COL,
            _WEIGHT_DAILY_MEDIAN_COL,
            _WEIGHT_DAILY_MEAN_COL,
            _WEIGHT_SEGMENT_MEAN_COL,
            _WEIGHT_MULTIPLIER_COL,
            _TIME_DECAY_COL,
            _TIME_DECAY_DAYS_AGO_COL,
            _SEGMENT_COL,
        }
        if self.weight_multiplier_col is not None:
            exclude.add(self.weight_multiplier_col)
        if not self.keep_weight_source_feature:
            exclude.add(self.weight_source)
        self.feature_names = [c for c in sample_df.columns if c not in exclude]

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

    def _uses_time_decay_weight(self) -> bool:
        return self.time_decay_half_life_days is not None

    def _uses_weight_column(self) -> bool:
        return (
            self.use_sample_weight
            or self._uses_weighted_metric()
            or self._uses_time_decay_weight()
            or self.weight_multiplier_col is not None
        )

    def _uses_liquidity_weight(self) -> bool:
        return self.use_sample_weight or self._uses_weighted_metric()

    def _build_weight_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """构造 LightGBM sample_weight / weighted metric 共用权重。

        流动性权重先做截面归一化；时间衰减必须在截面归一化之后再乘，
        否则同一天所有样本的 EMA 权重会被 daily_mean normalization 抹掉。
        """
        uses_liquidity_weight = self._uses_liquidity_weight()
        if uses_liquidity_weight and self.weight_source not in df.columns:
            raise ValueError(f"sample weight 或 {self.eval_metric} 需要列: {self.weight_source}")
        if self.weight_multiplier_col is not None and self.weight_multiplier_col not in df.columns:
            raise ValueError(f"缺少 weight_multiplier_col: {self.weight_multiplier_col}")

        weighted_df = df.sort(["vt_symbol", "datetime"])

        if uses_liquidity_weight:
            weighted_df = weighted_df.with_columns(
                pl.col(self.weight_source)
                .cast(pl.Float64, strict=False)
                .shift(self.weight_lag)
                .rolling_mean(window_size=self.weight_window, min_samples=1)
                .over("vt_symbol")
                .alias(_WEIGHT_BASE_COL)
            )

            if self.weight_transform == "sqrt":
                weight_expr = pl.col(_WEIGHT_BASE_COL).sqrt()
            else:
                weight_expr = pl.col(_WEIGHT_BASE_COL)

            weighted_df = weighted_df.with_columns(
                pl.when(pl.col(_WEIGHT_BASE_COL) > 0)
                .then(weight_expr)
                .otherwise(None)
                .alias(_WEIGHT_COL)
            )
        else:
            weighted_df = weighted_df.with_columns(pl.lit(1.0).alias(_WEIGHT_COL))

        if uses_liquidity_weight:
            weighted_df = (
                weighted_df.with_columns(
                    pl.col(_WEIGHT_COL).median().over("datetime").alias(_WEIGHT_DAILY_MEDIAN_COL)
                )
                .with_columns(
                    pl.when(pl.col(_WEIGHT_DAILY_MEDIAN_COL).is_finite() & (pl.col(_WEIGHT_DAILY_MEDIAN_COL) > 0))
                    .then(pl.col(_WEIGHT_DAILY_MEDIAN_COL))
                    .otherwise(1.0)
                    .alias(_WEIGHT_DAILY_MEDIAN_COL)
                )
                .with_columns(
                    pl.when(pl.col(_WEIGHT_COL).is_finite() & (pl.col(_WEIGHT_COL) > 0))
                    .then(pl.col(_WEIGHT_COL))
                    .otherwise(pl.col(_WEIGHT_DAILY_MEDIAN_COL))
                    .alias(_WEIGHT_COL)
                )
            )

            if self.weight_clip_median_multiple is not None:
                weighted_df = weighted_df.with_columns(
                    pl.min_horizontal(
                        pl.col(_WEIGHT_COL),
                        pl.col(_WEIGHT_DAILY_MEDIAN_COL) * float(self.weight_clip_median_multiple),
                    ).alias(_WEIGHT_COL)
                )

            if self.weight_normalize == "daily_mean":
                weighted_df = (
                    weighted_df.with_columns(
                        pl.col(_WEIGHT_COL).mean().over("datetime").alias(_WEIGHT_DAILY_MEAN_COL)
                    )
                    .with_columns(
                        pl.when(pl.col(_WEIGHT_DAILY_MEAN_COL).is_finite() & (pl.col(_WEIGHT_DAILY_MEAN_COL) > 0))
                        .then(pl.col(_WEIGHT_COL) / pl.col(_WEIGHT_DAILY_MEAN_COL))
                        .otherwise(1.0)
                        .alias(_WEIGHT_COL)
                    )
                )

        if self._uses_time_decay_weight():
            has_segment_col = _SEGMENT_COL in weighted_df.columns
            days_ago_expr = pl.col("datetime").rank("dense", descending=True)
            if has_segment_col:
                days_ago_expr = days_ago_expr.over(_SEGMENT_COL)
            weighted_df = weighted_df.with_columns(
                (days_ago_expr - 1.0)
                .cast(pl.Float64)
                .alias(_TIME_DECAY_DAYS_AGO_COL)
            )
            weighted_df = weighted_df.with_columns(
                (0.5 ** (pl.col(_TIME_DECAY_DAYS_AGO_COL) / float(self.time_decay_half_life_days)))
                .alias(_TIME_DECAY_COL)
            )
            if self.time_decay_mode == "replace":
                weighted_df = weighted_df.with_columns(pl.col(_TIME_DECAY_COL).alias(_WEIGHT_COL))
            else:
                weighted_df = weighted_df.with_columns((pl.col(_WEIGHT_COL) * pl.col(_TIME_DECAY_COL)).alias(_WEIGHT_COL))

            if self.time_decay_normalize == "global_mean":
                mean_expr = pl.col(_WEIGHT_COL).mean()
                if has_segment_col:
                    mean_expr = mean_expr.over(_SEGMENT_COL)
                weighted_df = (
                    weighted_df.with_columns(
                        mean_expr.alias(_WEIGHT_SEGMENT_MEAN_COL)
                    )
                    .with_columns(
                        pl.when(pl.col(_WEIGHT_SEGMENT_MEAN_COL).is_finite() & (pl.col(_WEIGHT_SEGMENT_MEAN_COL) > 0))
                        .then(pl.col(_WEIGHT_COL) / pl.col(_WEIGHT_SEGMENT_MEAN_COL))
                        .otherwise(1.0)
                        .alias(_WEIGHT_COL)
                    )
                )

        if self.weight_multiplier_col is not None:
            weighted_df = (
                weighted_df.with_columns(
                    pl.col(self.weight_multiplier_col)
                    .cast(pl.Float64, strict=False)
                    .alias(_WEIGHT_MULTIPLIER_COL)
                )
                .with_columns(
                    pl.when(pl.col(_WEIGHT_MULTIPLIER_COL).is_finite() & (pl.col(_WEIGHT_MULTIPLIER_COL) > 0))
                    .then(pl.col(_WEIGHT_MULTIPLIER_COL))
                    .otherwise(0.0)
                    .alias(_WEIGHT_MULTIPLIER_COL)
                )
                .with_columns((pl.col(_WEIGHT_COL) * pl.col(_WEIGHT_MULTIPLIER_COL)).alias(_WEIGHT_COL))
            )

        return weighted_df

    def _prepare_segment_dataset(self, df: pl.DataFrame) -> tuple[lgb.Dataset, np.ndarray | None]:
        """把单个分段数据转成 LightGBM Dataset。"""
        df = df.sort(["datetime", "vt_symbol"])

        data = df.select(self.feature_names).to_numpy()
        labels = np.asarray(df["label"], dtype=np.float64)
        group_sizes = _compute_group_sizes(df)

        if int(group_sizes.sum()) != int(labels.size):
            raise ValueError("group size 与标签长度不一致，请检查输入数据是否按日分组完整")

        sample_weight: np.ndarray | None = None
        if self._uses_weight_column():
            if _WEIGHT_COL not in df.columns:
                raise ValueError("缺少辅助权重列，请检查数据准备流程")
            sample_weight = np.asarray(df[_WEIGHT_COL], dtype=np.float64)

        lgb_data = lgb.Dataset(
            data,
            label=labels,
            group=group_sizes,
            feature_name=self.feature_names,
            weight=sample_weight,
        )
        try:
            setattr(lgb_data, "_vnpy_eval_weight", sample_weight if self._uses_weighted_metric() else None)
        except Exception:
            pass

        return lgb_data, sample_weight if self._uses_weighted_metric() else None

    def _prepare_data(self, dataset: AlphaDataset) -> tuple[list[lgb.Dataset], list[np.ndarray | None]]:
        datasets: list[lgb.Dataset] = []
        eval_weights: list[np.ndarray | None] = []

        sample_df = dataset.fetch_learn(Segment.TRAIN)
        self._infer_feature_names(sample_df)

        segment_frames: list[pl.DataFrame] = []
        for segment_idx, segment in enumerate([Segment.TRAIN, Segment.VALID]):
            df = dataset.fetch_learn(segment).sort(["datetime", "vt_symbol"])
            if self._uses_weight_column():
                df = df.with_columns(pl.lit(segment_idx).alias(_SEGMENT_COL))
            segment_frames.append(df)

        if self._uses_weight_column():
            weighted_frame = self._build_weight_columns(pl.concat(segment_frames))
            for segment_idx in range(len(segment_frames)):
                segment_df = (
                    weighted_frame.filter(pl.col(_SEGMENT_COL) == segment_idx)
                    .drop(
                        [
                            _SEGMENT_COL,
                            _WEIGHT_BASE_COL,
                            _WEIGHT_DAILY_MEDIAN_COL,
                            _WEIGHT_DAILY_MEAN_COL,
                            _WEIGHT_SEGMENT_MEAN_COL,
                            _WEIGHT_MULTIPLIER_COL,
                            _TIME_DECAY_COL,
                            _TIME_DECAY_DAYS_AGO_COL,
                        ],
                        strict=False,
                    )
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

    def _prepare_frame_pair(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
    ) -> tuple[list[lgb.Dataset], list[np.ndarray | None]]:
        train_df = train_df.sort(["datetime", "vt_symbol"])
        valid_df = valid_df.sort(["datetime", "vt_symbol"])

        self._infer_feature_names(train_df if not train_df.is_empty() else valid_df)

        if self._uses_weight_column():
            weighted_frame = self._build_weight_columns(
                pl.concat(
                    [
                        train_df.with_columns(pl.lit(0).alias(_SEGMENT_COL)),
                        valid_df.with_columns(pl.lit(1).alias(_SEGMENT_COL)),
                    ],
                    how="vertical",
                )
            )
            datasets: list[lgb.Dataset] = []
            eval_weights: list[np.ndarray | None] = []
            for segment_idx in (0, 1):
                segment_df = (
                    weighted_frame.filter(pl.col(_SEGMENT_COL) == segment_idx)
                    .drop(
                        [
                            _SEGMENT_COL,
                            _WEIGHT_BASE_COL,
                            _WEIGHT_DAILY_MEDIAN_COL,
                            _WEIGHT_DAILY_MEAN_COL,
                            _WEIGHT_SEGMENT_MEAN_COL,
                            _WEIGHT_MULTIPLIER_COL,
                            _TIME_DECAY_COL,
                            _TIME_DECAY_DAYS_AGO_COL,
                        ],
                        strict=False,
                    )
                    .sort(["datetime", "vt_symbol"])
                )
                lgb_data, eval_weight = self._prepare_segment_dataset(segment_df)
                datasets.append(lgb_data)
                eval_weights.append(eval_weight)
            return datasets, eval_weights

        train_data, train_eval_weight = self._prepare_segment_dataset(train_df)
        valid_data, valid_eval_weight = self._prepare_segment_dataset(valid_df)
        return [train_data, valid_data], [train_eval_weight, valid_eval_weight]

    def _prepare_refit_dataset(self, dataset: AlphaDataset) -> lgb.Dataset:
        train_df = dataset.fetch_learn(Segment.TRAIN).sort(["datetime", "vt_symbol"])
        valid_df = dataset.fetch_learn(Segment.VALID).sort(["datetime", "vt_symbol"])
        combined = pl.concat([train_df, valid_df], how="vertical").sort(["datetime", "vt_symbol"])
        if self._uses_weight_column():
            combined = (
                combined.with_columns(pl.lit(0).alias(_SEGMENT_COL))
                .pipe(self._build_weight_columns)
                .drop(
                    [
                        _SEGMENT_COL,
                        _WEIGHT_BASE_COL,
                        _WEIGHT_DAILY_MEDIAN_COL,
                        _WEIGHT_DAILY_MEAN_COL,
                        _WEIGHT_SEGMENT_MEAN_COL,
                        _WEIGHT_MULTIPLIER_COL,
                        _TIME_DECAY_COL,
                        _TIME_DECAY_DAYS_AGO_COL,
                    ],
                    strict=False,
                )
                .sort(["datetime", "vt_symbol"])
            )
        refit_data, _ = self._prepare_segment_dataset(combined)
        return refit_data

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

            if metric_name == "rank_ic":
                score = _mean_or_zero(_daily_rank_ic_series(pred_values, labels, group_sizes))
            elif metric_name == "rank_ic_ir":
                score = _ir_or_zero(_daily_rank_ic_series(pred_values, labels, group_sizes))
            elif metric_name == "simulated_sharpe":
                score = _annualized_sharpe_or_zero(
                    _daily_topk_return_series(pred_values, labels, group_sizes, top_k=self.top_k)
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

    def _build_cv_folds(
        self,
        pool_df: pl.DataFrame,
    ) -> list[tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]]:
        unique_dates = (
            pool_df.select("datetime")
            .unique()
            .sort("datetime")
            .get_column("datetime")
            .to_list()
        )
        if len(unique_dates) < self.cv_n_splits:
            raise ValueError(
                f"唯一交易日数量 {len(unique_dates)} 小于 cv_n_splits={self.cv_n_splits}"
            )

        date_array = np.asarray(unique_dates, dtype=object)
        blocks = np.array_split(date_array, self.cv_n_splits)
        folds: list[tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]] = []

        for fold_id, block in enumerate(blocks):
            if block.size == 0:
                raise ValueError(f"第 {fold_id} 折验证日期为空，请减小 cv_n_splits")

            start_idx = sum(prev.size for prev in blocks[:fold_id])
            end_idx = start_idx + block.size - 1
            excluded_idx = set(range(start_idx, end_idx + 1))
            if self.cv_purge_days > 0:
                left = max(0, start_idx - self.cv_purge_days)
                right = min(len(date_array) - 1, end_idx + self.cv_purge_days)
                excluded_idx.update(range(left, right + 1))

            valid_dates = list(block.tolist())
            train_dates = [date_array[idx] for idx in range(len(date_array)) if idx not in excluded_idx]
            if not train_dates:
                raise ValueError(f"第 {fold_id} 折训练日期为空，请减小 cv_purge_days 或 cv_n_splits")

            train_df = pool_df.filter(pl.col("datetime").is_in(train_dates)).sort(["datetime", "vt_symbol"])
            valid_df = pool_df.filter(pl.col("datetime").is_in(valid_dates)).sort(["datetime", "vt_symbol"])

            folds.append(
                (
                    train_df,
                    valid_df,
                    {
                        "fold_id": int(fold_id),
                        "train_start": cast(datetime, train_df["datetime"].min()).isoformat(),
                        "train_end": cast(datetime, train_df["datetime"].max()).isoformat(),
                        "valid_start": cast(datetime, valid_df["datetime"].min()).isoformat(),
                        "valid_end": cast(datetime, valid_df["datetime"].max()).isoformat(),
                        "train_days": int(len(train_dates)),
                        "valid_days": int(len(valid_dates)),
                        "train_rows": int(train_df.height),
                        "valid_rows": int(valid_df.height),
                    },
                )
            )

        return folds

    def _fit_cv(self, dataset: AlphaDataset) -> None:
        pool_df = pl.concat(
            [
                dataset.fetch_learn(Segment.TRAIN).sort(["datetime", "vt_symbol"]),
                dataset.fetch_learn(Segment.VALID).sort(["datetime", "vt_symbol"]),
            ],
            how="vertical",
        )
        if pool_df.is_empty():
            raise ValueError("CV 数据池为空，无法训练")

        self.models = []
        self.cv_fold_summaries = []
        self.cv_best_iterations = []
        self.cv_best_scores = []
        self.early_stopped_best_iterations = []
        self.early_stopped_best_scores = []

        for train_df, valid_df, fold_summary in self._build_cv_folds(pool_df):
            datasets, eval_weights = self._prepare_frame_pair(train_df, valid_df)
            resolved_min_data_in_leaf = self._resolve_min_data_in_leaf(datasets[0])
            self.resolved_min_data_in_leaf = resolved_min_data_in_leaf
            feval = None if self.eval_metric == "mae" else self._build_eval_func(datasets, eval_weights)

            callbacks = []
            if self.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            if self.log_evaluation_period > 0:
                callbacks.append(lgb.log_evaluation(self.log_evaluation_period))

            params = dict(self.params)
            if resolved_min_data_in_leaf is not None:
                params["min_data_in_leaf"] = resolved_min_data_in_leaf

            model = lgb.train(
                params,
                datasets[0],
                num_boost_round=self.num_boost_round,
                valid_sets=datasets,
                valid_names=["train", "valid"],
                feval=feval,
                callbacks=callbacks,
            )

            best_iteration = int(model.best_iteration or model.current_iteration())
            best_score = dict(model.best_score)
            fold_summary["best_iteration"] = best_iteration
            fold_summary["best_score"] = best_score

            self.models.append(model)
            self.cv_fold_summaries.append(fold_summary)
            self.cv_best_iterations.append(best_iteration)
            self.cv_best_scores.append(best_score)
            self.early_stopped_best_iterations.append(best_iteration)
            self.early_stopped_best_scores.append(best_score)

        self.model = list(self.models)

    def fit(self, dataset: AlphaDataset) -> None:
        if self.use_cv:
            self._fit_cv(dataset)
            return

        datasets, eval_weights = self._prepare_data(dataset)
        resolved_min_data_in_leaf = self._resolve_min_data_in_leaf(datasets[0])
        self.resolved_min_data_in_leaf = resolved_min_data_in_leaf

        feval = None if self.eval_metric == "mae" else self._build_eval_func(datasets, eval_weights)

        self.models = []
        self.early_stopped_best_iterations = []
        self.early_stopped_best_scores = []
        for seed in self.seeds:
            callbacks = []
            if self.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            if self.log_evaluation_period > 0:
                callbacks.append(lgb.log_evaluation(self.log_evaluation_period))

            params = dict(self.params)
            if resolved_min_data_in_leaf is not None:
                params["min_data_in_leaf"] = resolved_min_data_in_leaf
            params["seed"] = int(seed)
            params["feature_fraction_seed"] = int(seed)
            params["bagging_seed"] = int(seed)
            params["drop_seed"] = int(seed)
            params["extra_seed"] = int(seed)

            model = lgb.train(
                params,
                datasets[0],
                num_boost_round=self.num_boost_round,
                valid_sets=datasets,
                valid_names=["train", "valid"],
                feval=feval,
                callbacks=callbacks,
            )
            early_best_iteration = int(model.best_iteration or model.current_iteration())
            self.early_stopped_best_iterations.append(early_best_iteration)
            self.early_stopped_best_scores.append(dict(model.best_score))

            if self.refit_with_valid:
                refit_data = self._prepare_refit_dataset(dataset)
                refit_callbacks = []
                if self.log_evaluation_period > 0:
                    refit_callbacks.append(lgb.log_evaluation(self.log_evaluation_period))
                model = lgb.train(
                    params,
                    refit_data,
                    num_boost_round=early_best_iteration,
                    valid_sets=[refit_data],
                    valid_names=["train_valid"],
                    feval=None if self.eval_metric == "mae" else self._build_eval_func([refit_data], [None]),
                    callbacks=refit_callbacks,
                )
            self.models.append(model)

        self.model = self.models[0] if self.models else None

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        models = list(getattr(self, "models", []) or [])
        if not models and isinstance(self.model, list):
            models = list(self.model)
            self.models = models
        elif not models and self.model is not None:
            models = [self.model]
            self.models = models
        if not models:
            raise ValueError("模型尚未训练！")

        df = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])
        data = df.select(self.feature_names).to_numpy()

        predictions = [
            cast(np.ndarray, model.predict(data))
            for model in models
        ]
        if len(predictions) == 1:
            return predictions[0]
        return np.mean(np.vstack(predictions), axis=0)

    def detail(
        self,
        *,
        shap: bool = False,
        dataset: AlphaDataset | None = None,
        sample_size: int = 5000,
        plot_shap: bool = False,
    ) -> pd.DataFrame | None:
        """返回特征重要性，按需追加验证集 SHAP 重要性。"""
        models = list(getattr(self, "models", []) or [])
        if not models and isinstance(self.model, list):
            models = list(self.model)
            self.models = models
        elif not models and self.model is not None:
            models = [self.model]
            self.models = models
        if not models:
            return None

        feature_name = models[0].feature_name()
        importance_split = np.mean(
            [model.feature_importance(importance_type="split") for model in models],
            axis=0,
        )
        importance_gain = np.mean(
            [model.feature_importance(importance_type="gain") for model in models],
            axis=0,
        )

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
                    models[0],
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
            shap_values_by_model = [
                cast(np.ndarray, model.predict(sample_data, pred_contrib=True))
                for model in models
            ]
            shap_values = cast(
                np.ndarray,
                np.mean(np.stack(shap_values_by_model, axis=0), axis=0),
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
