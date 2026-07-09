"""
Additional Operators (Time-series and Cross-section)

Implemented with Polars expressions following the style of ts_function/cs_function.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import polars as pl

from .utility import DataProxy


# -------------------- Time-series extras --------------------

def ts_decay_linear(feature: DataProxy, window: int | float) -> DataProxy:
    """Linear weighted moving average over a rolling window.

    Weights: 1..window (recent heavier). First window-1 positions are null.
    """
    # 兼容浮点窗口：四舍五入为整数
    window = int(round(float(window)))
    window = max(1, window)
    weights = np.arange(1, window + 1, dtype=float)
    sum_w = float(weights.sum())

    # 确保数值类型：若输入为布尔，先转为 Float32，避免 rolling_map 对 bool 报错
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").cast(pl.Float32).rolling_map(
            lambda s: float(np.dot(s.to_numpy(), weights)) / sum_w,
            window
        ).over("vt_symbol")
    )
    return DataProxy(df)


# 极值敏感 需要配合clip使用，而且配合滚动的协方差能衡量feature1和feature2是否一起波动
def ts_cov(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
    """Rolling covariance with ddof=0 (population covariance).

    cov = E[XY] - E[X]E[Y] over the rolling window. Handles NaN by ignoring pairs.
    """
    df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"], how="inner")

    x = pl.col("data")
    y = pl.col("data_right")
    n = (x.is_not_nan() & y.is_not_nan()).cast(pl.Int32).rolling_sum(window, min_samples=1).over("vt_symbol")
    sum_x = x.fill_nan(0).rolling_sum(window, min_samples=1).over("vt_symbol")
    sum_y = y.fill_nan(0).rolling_sum(window, min_samples=1).over("vt_symbol")
    sum_xy = (x.fill_nan(0) * y.fill_nan(0)).rolling_sum(window, min_samples=1).over("vt_symbol")

    df: pl.DataFrame = df_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(n > 0)
         .then((sum_xy / n) - (sum_x / n) * (sum_y / n))
         .otherwise(None)
         .alias("data")
    )
    return DataProxy(df)

# # 补充ts_corr,协方差衡量的是，是否一起变动，ts_corr是在协方差的基础上做了标准化，把feature1和feature2的量纲统一，这个算子泄漏了未来数据

# def ts_corr(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
#     df = feature1.data.join(feature2.data, on=["datetime", "vt_symbol"], how="inner")
#     df = df.with_columns([
#         pl.pearson_corr(
#             pl.col("value_left"), 
#             pl.col("value_right")
#         ).over("vt_symbol").alias("value")
#     ])
#     return DataProxy(df.select(["datetime", "vt_symbol", "value"]))


def ts_prod(feature: DataProxy, window: int) -> DataProxy:
    """Rolling product over a window."""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").cast(pl.Float32).rolling_map(
            lambda s: float(np.prod(s.to_numpy())),
            window
        ).over("vt_symbol")
    )
    return DataProxy(df)


def ts_highday(feature: DataProxy, window: int) -> DataProxy:
    """Rolling 'highday' over window: len(window) - argmax(window).

    返回值范围 1..window：
    - 若最高值出现在当前（窗口末尾），返回 1；
    - 若出现在窗口起点，返回 window；
    - 若窗口内全 NaN，返回 NaN。
    """
    def _highday(s: pl.Series) -> float:
        a = s.to_numpy()
        # 处理全 NaN 的情况
        if np.isnan(a).all():
            return np.nan
        # nanargmax 会忽略 NaN
        idx = int(np.nanargmax(a))
        return float(len(a) - idx)

    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").cast(pl.Float32).rolling_map(_highday, window).over("vt_symbol").alias("data")
    )
    return DataProxy(df)


def ts_lowday(feature: DataProxy, window: int) -> DataProxy:
    """Rolling 'lowday' over window: len(window) - argmin(window).

    返回值范围 1..window：
    - 若最低值出现在当前（窗口末尾），返回 1；
    - 若出现在窗口起点，返回 window；
    - 若窗口内全 NaN，返回 NaN。
    """
    def _lowday(s: pl.Series) -> float:
        a = s.to_numpy()
        if np.isnan(a).all():
            return np.nan
        idx = int(np.nanargmin(a))
        return float(len(a) - idx)

    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").cast(pl.Float32).rolling_map(_lowday, window).over("vt_symbol").alias("data")
    )
    return DataProxy(df)


def ts_delta(feature: DataProxy, window: int) -> DataProxy:
    """Time-series delta: x_t - x_{t-window}.

    - 与表达式中的 delta(x, n) 语义一致（滚动差分）。
    - 注意 window 必须为正整数。
    """
    from .ts_function import ts_delay  # 局部导入避免循环
    delayed = ts_delay(feature, window)
    df = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("data") - delayed.df["data"]).alias("data")
    )
    return DataProxy(df)

# 为了解决ts_prod连续乘 出现极小极大值的数值问题，chatgpt给出了ts_log_prod+winsorize的方案
def ts_log_prod(feature: DataProxy,
                window: int,
                mode: str = "strict",
                eps: float = 1e-12,
                by: str = "vt_symbol",
                min_samples: int = 1) -> DataProxy:
    """
    更稳健的滚动乘积：在 log 域求和再 exp 回来。
    支持三种容错模式：
      - strict : 窗口内存在 (x <= 0) 或 NaN → 结果 NaN
      - clip   : 将 (x <= 0) 或 NaN 替换为 eps，再求 log-sum-exp
      - skipna : 仅对 x > 0 的项求和；若窗口全无有效项 → NaN

    参数
    ----
    feature      : DataProxy(["datetime", "vt_symbol", "data"])
    window       : int, 滚动窗口
    mode         : {"strict","clip","skipna"}
    eps          : float, 在 clip 模式下用于下限裁剪
    by           : 分组列，默认 "vt_symbol"
    min_samples  : 最小有效样本数，默认 1

    返回
    ----
    DataProxy(["datetime", "vt_symbol", "data"])
    """
    df = feature.df.with_columns(pl.col("data").cast(pl.Float64).alias("data"))
    invalid = pl.col("data").is_null() | pl.col("data").is_nan() | (pl.col("data") <= 0)

    if mode == "strict":
        # 标记非正/NaN
        flags = (
            df.with_columns([
                pl.when(invalid)
                  .then(1).otherwise(0).alias("_bad")
            ])
            .with_columns([
                pl.col("_bad").rolling_sum(window_size=window, min_samples=min_samples).over(by).alias("_bad_win"),
                pl.col("data").log().rolling_sum(window_size=window, min_samples=min_samples).over(by).alias("_logsum")
            ])
        )
        res = (
            flags.with_columns([
                pl.when(pl.col("_bad_win") > 0)
                  .then(None)                      # 严格：窗口内有坏值 → NaN
                  .otherwise(pl.col("_logsum").exp())
                  .alias("data")
            ])
            .select(["datetime", "vt_symbol", "data"])
        )

    elif mode == "clip":
        # 非正或NaN 用 eps 替换，再求 log-sum-exp
        safe = df.with_columns([
            pl.when(invalid)
              .then(eps).otherwise(pl.col("data")).alias("_v")
        ])
        res = (
            safe.with_columns([
                pl.col("_v").log().rolling_sum(window_size=window, min_samples=min_samples).over(by).alias("_logsum")
            ])
            .with_columns([
                pl.col("_logsum").exp().alias("data")
            ])
            .select(["datetime", "vt_symbol", "data"])
        )

    elif mode == "skipna":
        # 仅对 x>0 的项求和，并统计有效个数；若窗口内有效计数=0 → NaN
        safe = df.with_columns([
            pl.when(pl.col("data").is_not_null() & ~pl.col("data").is_nan() & (pl.col("data") > 0))
              .then(pl.col("data").log())
              .otherwise(None)
              .alias("_logv"),
            pl.when(pl.col("data").is_not_null() & ~pl.col("data").is_nan() & (pl.col("data") > 0))
              .then(1).otherwise(0)
              .alias("_cnt")
        ])
        res = (
            safe.with_columns([
                pl.col("_logv").rolling_sum(window_size=window, min_samples=min_samples).over(by).alias("_logsum"),
                pl.col("_cnt").rolling_sum(window_size=window, min_samples=min_samples).over(by).alias("_cntsum"),
            ])
            .with_columns([
                pl.when(pl.col("_cntsum") > 0)
                  .then(pl.col("_logsum").exp())
                  .otherwise(None)
                  .alias("data")
            ])
            .select(["datetime", "vt_symbol", "data"])
        )

    else:
        raise ValueError("mode must be one of {'strict','clip','skipna'}")

    return DataProxy(res)

# 平滑算子的响应问题，有三种平滑算子 滑动平均 线性权重 指数权重，这3种在t时刻数据突变的情况下，在接下来的t+n个时刻，会逐步弥补这个突变，不同的平滑算子的响应速度是不一样的。

def ts_ewm(feature: DataProxy, alpha: float, adjust: bool = False) -> DataProxy:
    """Exponential weighted mean with coefficient alpha in (0,1]."""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        # 确保数值类型，避免在旧版本上对布尔/整数出错
        pl.col("data").cast(pl.Float32).ewm_mean(alpha=alpha, adjust=adjust).over("vt_symbol").alias("data")
    )
    return DataProxy(df)


def ts_ema(feature: DataProxy, span: int, adjust: bool = False) -> DataProxy:
    """EMA defined by span (alpha = 2/(span+1))."""
    alpha = 2.0 / (span + 1.0)
    return ts_ewm(feature, alpha=alpha, adjust=adjust)


def ts_direction(feature: DataProxy, by: str = "vt_symbol") -> DataProxy:
    """
    时间序列方向变化算子：
      - 对每个标的（by 分组）计算一阶差分方向
      - 输出 sign(x_t - x_{t-1})
      - NaN 与首期差分自动处理为 None

    参数
    ----
    feature : DataProxy
        包含 ["datetime", "vt_symbol", "data"] 的时间序列因子
    by : str, 默认 "vt_symbol"
        分组列名，一般是股票代码或标的ID

    返回
    ----
    DataProxy : 与输入结构相同，data 为 {-1, 0, +1}
    """
    df = feature.df

    result = (
        df.with_columns([
            # 一阶差分（按标的分组）
            (pl.col("data").cast(pl.Float64) - pl.col("data").cast(pl.Float64).shift(1).over(by)).alias("_diff")
        ])
        .with_columns([
            # 取符号
            pl.when(pl.col("_diff").is_null() | pl.col("_diff").is_nan())
              .then(None)
              .when(pl.col("_diff") > 0)
              .then(1)
              .when(pl.col("_diff") < 0)
              .then(-1)
              .otherwise(0)
              .alias("data")
        ])
        .select(["datetime", "vt_symbol", "data"])
    )

    return DataProxy(result)



def ts_sign(feature: DataProxy) -> DataProxy:
    """Sign of series: -1, 0, 1"""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(pl.col("data") > 0).then(1)
          .when(pl.col("data") < 0).then(-1)
          .otherwise(0).alias("data")
    )
    return DataProxy(df)


def ts_where(cond: DataProxy, x: Union[DataProxy, float, int], y: Union[DataProxy, float, int]) -> DataProxy:
    """Element-wise selection: cond ? x : y"""
    base = cond.df.rename({"data": "cond"})

    if isinstance(x, DataProxy):
        base = base.join(x.df.rename({"data": "x"}), on=["datetime", "vt_symbol"], how="inner")
    else:
        base = base.with_columns(pl.lit(float(x)).alias("x"))

    if isinstance(y, DataProxy):
        base = base.join(y.df.rename({"data": "y"}), on=["datetime", "vt_symbol"], how="inner")
    else:
        base = base.with_columns(pl.lit(float(y)).alias("y"))

    df: pl.DataFrame = base.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(pl.col("cond")).then(pl.col("x")).otherwise(pl.col("y")).alias("data")
    )
    return DataProxy(df)


def cs_where(cond: DataProxy, x: Union[DataProxy, float, int], y: Union[DataProxy, float, int]) -> DataProxy:
    """Cross-section where: 语义等同于 ts_where（按行条件选择）。"""
    return ts_where(cond, x, y)


def ts_clip(feature: DataProxy, lower: float | None = None, upper: float | None = None) -> DataProxy:
    """Clip values into [lower, upper].

    为兼容较旧的 Polars 版本（Expr 无 clip_min/clip_max），
    使用条件表达式实现：
        if lower: data = max(data, lower)
        if upper: data = min(data, upper)
    """
    expr: pl.Expr = pl.col("data")
    if lower is not None:
        expr = pl.when(expr < pl.lit(float(lower))).then(pl.lit(float(lower))).otherwise(expr)
    if upper is not None:
        expr = pl.when(expr > pl.lit(float(upper))).then(pl.lit(float(upper))).otherwise(expr)

    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        expr.alias("data")
    )
    return DataProxy(df)


# -------------------- Cross-section extras --------------------

def cs_scale(feature: DataProxy, k: float = 1.0) -> DataProxy:
    """Scale within each date: x_scaled = k * x / sum(|x|).

    If denominator is 0 (all zeros), output zeros.
    """
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data"),
        pl.col("data").abs().sum().over("datetime").alias("sum_abs")
    ).select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(pl.col("sum_abs") > 0).then(pl.lit(float(k)) * pl.col("data") / pl.col("sum_abs")).otherwise(0.0).alias("data")
    )
    return DataProxy(df)


def cs_pct_rank(feature: DataProxy) -> DataProxy:
    """Cross-sectional percent rank within each date mapped to (0, 1].

    - 使用每日分组（按列 "datetime"）对非 NaN 值做平均名次 rank，
      然后按当日非 NaN 样本数 N 做缩放：(rank - 0.5) / N。
    - 原始值为 NaN 的位置输出 None。
    """
    # 计算每个日期的非 NaN 数量与平均名次
    n = pl.col("data").is_not_nan().cast(pl.Int32).sum().over("datetime")
    r = pl.col("data").rank(method="average").over("datetime")

    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(n > 0)
          .then((r - 0.5) / n)
          .otherwise(None)
          .alias("data")
    )
    return DataProxy(df)


def rq_rank(feature: DataProxy) -> DataProxy:
    """RiceQuant/WorldQuant cross-sectional rank in percentile scale."""
    data = pl.col("data").cast(pl.Float64, strict=False)
    valid = data.is_not_null() & data.is_finite()
    clean = pl.when(valid).then(data).otherwise(None)
    n = valid.cast(pl.Int32).sum().over("datetime")
    rank = clean.rank(method="average").over("datetime")

    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(n > 0).then(rank / n).otherwise(None).alias("data"),
    )
    return DataProxy(df)


def rq_ts_rank(feature: DataProxy, window: int | float) -> DataProxy:
    """RiceQuant/WorldQuant time-series rank.

    RQ defines TS_RANK as the current value's rank divided by the valid sample
    count in the rolling window. For example, if N=5 and the current value ranks
    third, the result is 3 / 5 = 0.6.
    """
    size = int(round(float(window)))
    size = max(1, size)

    def _pct_rank(series: pl.Series) -> float:
        values = series.to_numpy()
        current = values[-1]
        if current is None or not np.isfinite(current):
            return float("nan")
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return float("nan")
        less = float(np.sum(valid < current))
        equal = float(np.sum(valid == current))
        rank = less + (equal + 1.0) / 2.0
        return float(rank / len(valid))

    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data")
        .cast(pl.Float64, strict=False)
        .rolling_map(_pct_rank, size)
        .over("vt_symbol")
        .alias("data"),
    )
    return DataProxy(df)


def rq_ts_std(feature: DataProxy, window: int | float) -> DataProxy:
    """RiceQuant/TA-Lib style rolling STDDEV with population ddof=0."""
    size = int(round(float(window)))
    size = max(1, size)
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data")
        .cast(pl.Float64, strict=False)
        .rolling_std(window_size=size, min_samples=size, ddof=0)
        .over("vt_symbol")
        .alias("data"),
    )
    return DataProxy(df)


def rq_ts_argmax(feature: DataProxy, window: int | float) -> DataProxy:
    """RiceQuant/TA-Lib style rolling MAXINDEX.

    Returns the 0-based absolute row index of the maximum value within the
    rolling window for each symbol, matching TA-Lib MAXINDEX semantics after
    the warmup period.
    """
    size = int(round(float(window)))
    size = max(1, size)

    def _argmax(series: pl.Series) -> float:
        values = series.to_numpy()
        if not np.isfinite(values).any():
            return float("nan")
        return float(np.nanargmax(values))

    work = feature.df.with_columns(pl.int_range(0, pl.len()).over("vt_symbol").alias("_rq_index"))
    df: pl.DataFrame = work.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (
            pl.col("data")
            .cast(pl.Float64, strict=False)
            .rolling_map(_argmax, size)
            .over("vt_symbol")
            + pl.col("_rq_index")
            - size
            + 1
        ).alias("data"),
    )
    return DataProxy(df)


def rq_ts_argmin(feature: DataProxy, window: int | float) -> DataProxy:
    """RiceQuant/TA-Lib style rolling MININDEX.

    Returns the 0-based absolute row index of the minimum value within the
    rolling window for each symbol, matching TA-Lib MININDEX semantics after
    the warmup period.
    """
    size = int(round(float(window)))
    size = max(1, size)

    def _argmin(series: pl.Series) -> float:
        values = series.to_numpy()
        if not np.isfinite(values).any():
            return float("nan")
        return float(np.nanargmin(values))

    work = feature.df.with_columns(pl.int_range(0, pl.len()).over("vt_symbol").alias("_rq_index"))
    df: pl.DataFrame = work.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (
            pl.col("data")
            .cast(pl.Float64, strict=False)
            .rolling_map(_argmin, size)
            .over("vt_symbol")
            + pl.col("_rq_index")
            - size
            + 1
        ).alias("data"),
    )
    return DataProxy(df)


def rq_min(feature1: DataProxy, feature2: DataProxy | float | int) -> DataProxy:
    """Strict element-wise minimum matching RQ MIN.

    Unlike FMIN, if either side is missing/NaN the result is missing.
    """
    if isinstance(feature2, DataProxy):
        base = feature1.df.join(feature2.df.rename({"data": "data_right"}), on=["datetime", "vt_symbol"], how="inner")
    else:
        base = feature1.df.with_columns(pl.lit(float(feature2)).alias("data_right"))

    left = pl.col("data").cast(pl.Float64, strict=False)
    right = pl.col("data_right").cast(pl.Float64, strict=False)
    invalid = left.is_null() | right.is_null() | left.is_nan() | right.is_nan()
    df: pl.DataFrame = base.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(invalid).then(None).otherwise(pl.min_horizontal(left, right)).alias("data"),
    )
    return DataProxy(df)


def rq_max(feature1: DataProxy, feature2: DataProxy | float | int) -> DataProxy:
    """Strict element-wise maximum matching RQ MAX.

    Unlike FMAX, if either side is missing/NaN the result is missing.
    """
    if isinstance(feature2, DataProxy):
        base = feature1.df.join(feature2.df.rename({"data": "data_right"}), on=["datetime", "vt_symbol"], how="inner")
    else:
        base = feature1.df.with_columns(pl.lit(float(feature2)).alias("data_right"))

    left = pl.col("data").cast(pl.Float64, strict=False)
    right = pl.col("data_right").cast(pl.Float64, strict=False)
    invalid = left.is_null() | right.is_null() | left.is_nan() | right.is_nan()
    df: pl.DataFrame = base.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(invalid).then(None).otherwise(pl.max_horizontal(left, right)).alias("data"),
    )
    return DataProxy(df)


def rq_as_float(feature: DataProxy) -> DataProxy:
    """Cast boolean/numeric values to float for formulas using AS_FLOAT."""
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").cast(pl.Float64, strict=False).alias("data"),
    )
    return DataProxy(df)


def rq_sign(feature: DataProxy) -> DataProxy:
    """NumPy-style sign preserving NaN/null values."""
    data = pl.col("data").cast(pl.Float64, strict=False)
    df: pl.DataFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when(data.is_null() | data.is_nan())
        .then(None)
        .when(data > 0)
        .then(1.0)
        .when(data < 0)
        .then(-1.0)
        .otherwise(0.0)
        .alias("data"),
    )
    return DataProxy(df)


def rq_indneutralize(feature: DataProxy, category: DataProxy | None = None) -> DataProxy:
    """Industry-neutralize a feature by demeaning within date/category.

    If no category is supplied, fall back to cross-sectional demeaning. The
    strict RiceQuant Alpha101 template passes an ``industry`` column when it is
    available.
    """
    from .neutralize_function import cs_demean, cs_demean_by_category

    if category is None:
        return cs_demean(feature)
    return cs_demean_by_category(feature, category)


def ts_winsor(
    feature: DataProxy,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    by: str = "vt_symbol",
) -> DataProxy:
    """时间序列 Winsor 截尾（按分组在整个时间轴上计算上下分位界）。

    - 以 `by` 分组（默认每个 vt_symbol），在该组的全时段上计算 data 的下/上分位数边界；
    - 将小于下边界的值替换为下边界，大于上边界的值替换为上边界；
    - NaN 不参与分位数计算；原位为 NaN 的输出保持 NaN。
    """
    assert 0.0 <= lower_q < upper_q <= 1.0, "lower_q < upper_q 且都在 [0,1] 内"

    df = feature.df
    if by not in df.columns:
        # 仅支持现有列名（一般为 vt_symbol 或 datetime）
        raise ValueError(f"ts_winsor: 分组列 {by!r} 不存在于数据列 {df.columns}")

    bounds = df.group_by(by).agg([
        pl.col("data").quantile(lower_q, interpolation="linear").alias("_L"),
        pl.col("data").quantile(upper_q, interpolation="linear").alias("_U"),
    ])

    clipped = (
        df.join(bounds, on=by, how="left")
          .select(
              pl.col("datetime"),
              pl.col("vt_symbol"),
              pl.col("data"),
              pl.col("_L"),
              pl.col("_U"),
          )
          .with_columns(
              pl.when(pl.col("data") < pl.col("_L")).then(pl.col("_L"))
                .when(pl.col("data") > pl.col("_U")).then(pl.col("_U"))
                .otherwise(pl.col("data")).alias("data")
          )
          .select(["datetime", "vt_symbol", "data"])
    )

    return DataProxy(clipped)
