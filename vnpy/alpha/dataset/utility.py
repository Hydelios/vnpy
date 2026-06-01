from datetime import datetime
from enum import Enum
from typing import Union

import polars as pl


class DataProxy:
    """Feature data proxy"""

    def __init__(self, df: pl.DataFrame, interval: str | None = None) -> None:
        """Constructor"""
        self.name: str = df.columns[-1]
        self.df: pl.DataFrame = df.rename({self.name: "data"})
        self.interval: str | None = interval

        # Note that for numerical expressions, variables should be placed before numbers. e.g. a * 2

    def result(self, s: pl.Series) -> "DataProxy":
        """Convert series data to feature object"""
        result: pl.DataFrame = self.df[["datetime", "vt_symbol"]]
        result = result.with_columns(other=s)

        return DataProxy(result, interval=self.interval)

    def __add__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Addition operation"""
        left = self.df["data"]
        if left.dtype == pl.Boolean:
            left = left.cast(pl.Float32)
        if isinstance(other, DataProxy):
            right = other.df["data"]
            if right.dtype == pl.Boolean:
                right = right.cast(pl.Float32)
            s: pl.Series = left + right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = left + o
        return self.result(s)

    # 右加：支持常量/特征在左、DataProxy 在右的写法（如 1 + x）
    def __radd__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        right = self.df["data"]
        if right.dtype == pl.Boolean:
            right = right.cast(pl.Float32)
        if isinstance(other, DataProxy):
            left = other.df["data"]
            if left.dtype == pl.Boolean:
                left = left.cast(pl.Float32)
            s: pl.Series = left + right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = o + right
        return self.result(s)

    def __sub__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Subtraction operation"""
        left = self.df["data"]
        if left.dtype == pl.Boolean:
            left = left.cast(pl.Float32)
        if isinstance(other, DataProxy):
            right = other.df["data"]
            if right.dtype == pl.Boolean:
                right = right.cast(pl.Float32)
            s: pl.Series = left - right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = left - o
        return self.result(s)

    # 右减：支持常量 - DataProxy（如 1 - x）
    def __rsub__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        right = self.df["data"]
        if right.dtype == pl.Boolean:
            right = right.cast(pl.Float32)
        if isinstance(other, DataProxy):
            left = other.df["data"]
            if left.dtype == pl.Boolean:
                left = left.cast(pl.Float32)
            s: pl.Series = left - right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = o - right
        return self.result(s)

    def __mul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Multiplication operation"""
        left = self.df["data"]
        if left.dtype == pl.Boolean:
            left = left.cast(pl.Float32)
        if isinstance(other, DataProxy):
            right = other.df["data"]
            if right.dtype == pl.Boolean:
                right = right.cast(pl.Float32)
            s: pl.Series = left * right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = left * o
        return self.result(s)

    def __rmul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Right multiplication operation"""
        right = self.df["data"]
        if right.dtype == pl.Boolean:
            right = right.cast(pl.Float32)
        if isinstance(other, DataProxy):
            left = other.df["data"]
            if left.dtype == pl.Boolean:
                left = left.cast(pl.Float32)
            s: pl.Series = left * right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = o * right
        return self.result(s)

    def __truediv__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Division operation"""
        left = self.df["data"]
        if left.dtype == pl.Boolean:
            left = left.cast(pl.Float32)
        if isinstance(other, DataProxy):
            right = other.df["data"]
            if right.dtype == pl.Boolean:
                right = right.cast(pl.Float32)
            s: pl.Series = left / right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = left / o
        return self.result(s)

    # 右除：支持常量 / DataProxy（如 1 / x）
    def __rtruediv__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        right = self.df["data"]
        if right.dtype == pl.Boolean:
            right = right.cast(pl.Float32)
        if isinstance(other, DataProxy):
            left = other.df["data"]
            if left.dtype == pl.Boolean:
                left = left.cast(pl.Float32)
            s: pl.Series = left / right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = o / right
        return self.result(s)

    def __pow__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Power operation"""
        left = self.df["data"]
        if left.dtype == pl.Boolean:
            left = left.cast(pl.Float32)
        if isinstance(other, DataProxy):
            right = other.df["data"]
            if right.dtype == pl.Boolean:
                right = right.cast(pl.Float32)
            s: pl.Series = left ** right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = left ** o
        return self.result(s)

    # 右侧幂：支持常量 ** DataProxy（如 2 ** x）
    def __rpow__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        right = self.df["data"]
        if right.dtype == pl.Boolean:
            right = right.cast(pl.Float32)
        if isinstance(other, DataProxy):
            left = other.df["data"]
            if left.dtype == pl.Boolean:
                left = left.cast(pl.Float32)
            s: pl.Series = left ** right
        else:
            o = 1.0 if isinstance(other, bool) else other
            s = o ** right
        return self.result(s)

    # 一元负号：支持 -x
    def __neg__(self) -> "DataProxy":
        s: pl.Series = -self.df["data"]
        return self.result(s)

    def __abs__(self) -> "DataProxy":
        """Get absolute value"""
        s: pl.Series = self.df["data"].abs()
        return self.result(s)

    def __gt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Greater than comparison"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] > other.df["data"]
        else:
            s = self.df["data"] > other
        return self.result(s)

    def __ge__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Greater than or equal comparison"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] >= other.df["data"]
        else:
            s = self.df["data"] >= other
        return self.result(s)

    def __lt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Less than comparison"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] < other.df["data"]
        else:
            s = self.df["data"] < other
        return self.result(s)

    def __le__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        """Less than or equal comparison"""
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] <= other.df["data"]
        else:
            s = self.df["data"] <= other
        return self.result(s)

    def __eq__(self, other: Union["DataProxy", int, float]) -> "DataProxy":    # type: ignore
        """Equal comparison"""
        if isinstance(other, DataProxy):
            s = self.df["data"] == other.df["data"]
        else:
            s = self.df["data"] == other
        return self.result(s)

    # ---- Boolean logic operators (for building complex conditions) ----
    def __and__(self, other: Union["DataProxy", bool]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"].cast(pl.Boolean) & other.df["data"].cast(pl.Boolean)
        else:
            # Element-wise AND with scalar bool
            tmp = self.df.select((pl.col("data").cast(pl.Boolean) & pl.lit(bool(other))).alias("__and"))
            s = tmp["__and"]
        return self.result(s)

    def __rand__(self, other: Union["DataProxy", bool]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = other.df["data"].cast(pl.Boolean) & self.df["data"].cast(pl.Boolean)
        else:
            tmp = self.df.select((pl.lit(bool(other)) & pl.col("data").cast(pl.Boolean)).alias("__rand"))
            s = tmp["__rand"]
        return self.result(s)

    def __or__(self, other: Union["DataProxy", bool]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"].cast(pl.Boolean) | other.df["data"].cast(pl.Boolean)
        else:
            tmp = self.df.select((pl.col("data").cast(pl.Boolean) | pl.lit(bool(other))).alias("__or"))
            s = tmp["__or"]
        return self.result(s)

    def __ror__(self, other: Union["DataProxy", bool]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = other.df["data"].cast(pl.Boolean) | self.df["data"].cast(pl.Boolean)
        else:
            tmp = self.df.select((pl.lit(bool(other)) | pl.col("data").cast(pl.Boolean)).alias("__ror"))
            s = tmp["__ror"]
        return self.result(s)

    def __invert__(self) -> "DataProxy":
        s: pl.Series = (~self.df["data"].cast(pl.Boolean))
        return self.result(s)


def calculate_by_expression(
    df: pl.DataFrame,
    expression: str,
    interval: str | None = None,
) -> pl.DataFrame:
    """Execute calculation based on expression"""
    # Import operators locally to avoid polluting global namespace
    from .ts_function import (              # noqa
        ts_delay,
        ts_min, ts_max,
        ts_argmax, ts_argmin,
        ts_rank, ts_sum,
        ts_mean, ts_std,
        ts_slope, ts_quantile,
        ts_rsquare, ts_resi,
        ts_corr,
        ts_less, ts_greater,
        ts_log, ts_abs
    )
    from .cs_function import (              # noqa
        cs_rank,
        cs_mean,
        cs_std
    )
    from .ta_function import (              # noqa
        ta_rsi,
        ta_atr,
        ta_sma,
        ta_linearreg_slope,
    )
    from .ta_ewm_function import (          # noqa
        ewmMean,
        ts_ewm_var,
        ts_ewm_std,
        ts_ewm_cov,
        ts_ewm_corr,
        ta_wma,
        ta_dema,
        ta_tema,
        ta_trima,
        ta_wilder,
        ta_gema,
        ta_ma,
        ta_kama,
        ta_t3,
    )
    # Extra operators (time-series and cross-section)
    from .extra_function import (           # noqa
        ts_decay_linear,
        ts_cov,
        ts_prod,
        ts_log_prod,
        ts_delta,
        ts_ema,
        ts_ewm,
        ts_direction,
        ts_sign,
        ts_where,
        cs_where,
        ts_clip,
        ts_winsor,
        cs_pct_rank,
        cs_scale,
        ts_highday,
        ts_lowday,
    )
    from .stateful_function import (        # noqa
        ts_count,
        ts_var,
        ts_varp,
        ts_stdp,
        ts_skew,
        ts_kurtosis,
        ts_median,
        ts_first,
        ts_last,
        ts_wsum,
        ts_wavg,
        ts_beta,
        ts_ffill,
        ts_ratio,
        ts_pct_change,
        ts_prev_state,
        ts_cumsum,
        ts_cumprod,
        ts_cumcount,
        ts_cummin,
        ts_cummax,
        ts_cummean,
        ts_cumvar,
        ts_cumstd,
        ts_cumvarp,
        ts_cumstdp,
        ts_cumcov,
        ts_cumcorr,
        ts_cumbeta,
        ts_cumwsum,
        ts_cumwavg,
        ts_cumfirst_not,
        ts_cumlast_not,
        ts_cummedian,
        ts_cumpercentile,
        ts_cumnunique,
        ts_cum_positive_streak,
        ts_time_sum,
        ts_time_sum2,
        ts_time_mean,
        ts_time_count,
        ts_time_min,
        ts_time_max,
        ts_time_first,
        ts_time_last,
        ts_time_std,
        ts_time_stdp,
        ts_time_var,
        ts_time_varp,
        ts_time_prod,
        ts_time_skew,
        ts_time_kurtosis,
        ts_time_median,
        ts_time_rank,
        ts_time_percentile,
        ts_time_cov,
        ts_time_corr,
        ts_time_beta,
        ts_time_wsum,
        ts_time_wavg,
        ts_time_sum_topn,
        ts_time_mean_topn,
        ts_time_var_topn,
        ts_time_varp_topn,
        ts_time_std_topn,
        ts_time_stdp_topn,
        ts_time_cov_topn,
        ts_time_corr_topn,
        ts_time_beta_topn,
        ts_time_wsum_topn,
        ts_sum_topn,
        ts_mean_topn,
        ts_var_topn,
        ts_varp_topn,
        ts_std_topn,
        ts_stdp_topn,
        ts_skew_topn,
        ts_kurtosis_topn,
        ts_wsum_topn,
        ts_wavg_topn,
        ts_cov_topn,
        ts_beta_topn,
        ts_corr_topn,
        ts_cumsum_topn,
        ts_cummean_topn,
        ts_cumvar_topn,
        ts_cumvarp_topn,
        ts_cumstd_topn,
        ts_cumstdp_topn,
        ts_cumcov_topn,
        ts_cumcorr_topn,
        ts_cumbeta_topn,
        ts_cumwsum_topn,
        ts_state_iterate,
        ts_conditional_iterate,
    )
    from .stateful_ext_function import (  # noqa
        ts_rolling_ols_residual,
        ts_rolling_ols_residual_corr,
        ts_corr_quantile_mask,
    )
    from .neutralize_function import (      # noqa
        cs_demean,
        cs_zscore,
        cs_demean_by_category,
        cs_zscore_by_category,
        cs_neutralize_ols1,
    )
    from .dolphindb_compat_function import (    # noqa
        iif,
        sign,
        move,
        ratios,
        rowRank,
        msum,
        mavg,
        mstd,
        mcorr,
        mcovar,
        mcov,
        mbeta,
        mmin,
        mmax,
        mcount,
        mprod,
        mfirst,
        mrank,
        mimin,
        mimax,
        elem_min,
        elem_max,
        ddb_min,
        ddb_max,
        rowMax,
        rowMin,
        rollingOlsResidual3,
        rolling_ols_residual3,
        conditionalCumprod,
        conditional_cumprod,
        signedPower,
        signed_power,
        linearTimeTrend,
        linear_time_trend,
    )

    # Backward-compatibility aliases for expression strings
    # 一些历史模板中使用 log(...)，这里兼容为 ts_log
    d: dict = locals()
    d["log"] = ts_log  # type: ignore
    d["true"] = True
    d["false"] = False
    d["NULL"] = float("nan")
    d["null"] = float("nan")
    d["ewmMean"] = ewmMean
    d["ewmVar"] = ts_ewm_var
    d["ewmStd"] = ts_ewm_std
    d["ewmCov"] = ts_ewm_cov
    d["ewmCorr"] = ts_ewm_corr
    d["wma"] = ta_wma
    d["dema"] = ta_dema
    d["tema"] = ta_tema
    d["trima"] = ta_trima
    d["wilder"] = ta_wilder
    d["gema"] = ta_gema
    d["ma"] = ta_ma
    d["kama"] = ta_kama
    d["t3"] = ta_t3
    d["tmsum"] = ts_time_sum
    d["tmsum2"] = ts_time_sum2
    d["tmprod"] = ts_time_prod
    d["tmavg"] = ts_time_mean
    d["tmmean"] = ts_time_mean
    d["tmcount"] = ts_time_count
    d["tmmin"] = ts_time_min
    d["tmmax"] = ts_time_max
    d["tmfirst"] = ts_time_first
    d["tmlast"] = ts_time_last
    d["tmstd"] = ts_time_std
    d["tmstdp"] = ts_time_stdp
    d["tmvar"] = ts_time_var
    d["tmvarp"] = ts_time_varp
    d["tmskew"] = ts_time_skew
    d["tmkurtosis"] = ts_time_kurtosis
    d["tmmed"] = ts_time_median
    d["tmrank"] = ts_time_rank
    d["tmpercentile"] = ts_time_percentile
    d["tmcovar"] = ts_time_cov
    d["tmcov"] = ts_time_cov
    d["tmcorr"] = ts_time_corr
    d["tmbeta"] = ts_time_beta
    d["tmwsum"] = ts_time_wsum
    d["tmwavg"] = ts_time_wavg
    d["tmsumTopN"] = ts_time_sum_topn
    d["tmavgTopN"] = ts_time_mean_topn
    d["tmmeanTopN"] = ts_time_mean_topn
    d["tmstdTopN"] = ts_time_std_topn
    d["tmstdpTopN"] = ts_time_stdp_topn
    d["tmvarTopN"] = ts_time_var_topn
    d["tmvarpTopN"] = ts_time_varp_topn
    d["tmcorrTopN"] = ts_time_corr_topn
    d["tmcovarTopN"] = ts_time_cov_topn
    d["tmcovTopN"] = ts_time_cov_topn
    d["tmbetaTopN"] = ts_time_beta_topn
    d["tmwsumTopN"] = ts_time_wsum_topn
    d["cumsum"] = ts_cumsum
    d["cumprod"] = ts_cumprod
    d["cumcount"] = ts_cumcount
    d["cummin"] = ts_cummin
    d["cummax"] = ts_cummax
    d["cumavg"] = ts_cummean
    d["cummean"] = ts_cummean
    d["cumvar"] = ts_cumvar
    d["cumstd"] = ts_cumstd
    d["cumvarp"] = ts_cumvarp
    d["cumstdp"] = ts_cumstdp
    d["cumcovar"] = ts_cumcov
    d["cumcov"] = ts_cumcov
    d["cumcorr"] = ts_cumcorr
    d["cumbeta"] = ts_cumbeta
    d["cumwsum"] = ts_cumwsum
    d["cumwavg"] = ts_cumwavg
    d["cumfirstNot"] = ts_cumfirst_not
    d["cumlastNot"] = ts_cumlast_not
    d["cummed"] = ts_cummedian
    d["cumpercentile"] = ts_cumpercentile
    d["cumnunique"] = ts_cumnunique
    d["cumPositiveStreak"] = ts_cum_positive_streak
    d["msumTopN"] = ts_sum_topn
    d["mavgTopN"] = ts_mean_topn
    d["mmeanTopN"] = ts_mean_topn
    d["mvarTopN"] = ts_var_topn
    d["mvarpTopN"] = ts_varp_topn
    d["mstdTopN"] = ts_std_topn
    d["mstdpTopN"] = ts_stdp_topn
    d["mskewTopN"] = ts_skew_topn
    d["mkurtosisTopN"] = ts_kurtosis_topn
    d["mwsumTopN"] = ts_wsum_topn
    d["mwavgTopN"] = ts_wavg_topn
    d["mcovarTopN"] = ts_cov_topn
    d["mcovTopN"] = ts_cov_topn
    d["mbetaTopN"] = ts_beta_topn
    d["mcorrTopN"] = ts_corr_topn
    d["cumsumTopN"] = ts_cumsum_topn
    d["cumavgTopN"] = ts_cummean_topn
    d["cummeanTopN"] = ts_cummean_topn
    d["cumvarTopN"] = ts_cumvar_topn
    d["cumvarpTopN"] = ts_cumvarp_topn
    d["cumstdTopN"] = ts_cumstd_topn
    d["cumstdpTopN"] = ts_cumstdp_topn
    d["cumcovarTopN"] = ts_cumcov_topn
    d["cumcovTopN"] = ts_cumcov_topn
    d["cumcorrTopN"] = ts_cumcorr_topn
    d["cumbetaTopN"] = ts_cumbeta_topn
    d["cumwsumTopN"] = ts_cumwsum_topn

    # Extract feature objects to local space

    for column in df.columns:
        # Filter index columns
        if column == "vt_symbol":
            continue

        # Cache feature df
        if column == "datetime":
            column_df = df.select(
                pl.col("datetime"),
                pl.col("vt_symbol"),
                pl.col("datetime").alias("_datetime_data"),
            )
        else:
            column_df = df[["datetime", "vt_symbol", column]]
        d[column] = DataProxy(column_df, interval=interval)

    # Use eval to execute calculation
    other: DataProxy = eval(expression, {}, d)

    # Return result DataFrame
    return other.df


def calculate_by_polars(df: pl.DataFrame, expression: pl.expr.expr.Expr) -> pl.DataFrame:
    """Execute calculation based on Polars expression"""
    # 在 with_columns 上下文应用窗口/分组表达式更安全，随后再选择需要的列
    df2 = df.with_columns(expression.alias("data"))
    return df2.select(["datetime", "vt_symbol", "data"])


def to_datetime(arg: datetime | str) -> datetime:
    """Convert time data type to tz-naive datetime.

    - 字符串：按 YYYY-MM-DD 或 YYYYMMDD 解析，返回无时区 datetime。
    - datetime：若含时区信息，移除 tzinfo 确保与数据列（tz-naive）对齐。
    """
    if isinstance(arg, str):
        if "-" in arg:
            fmt: str = "%Y-%m-%d"
        else:
            fmt = "%Y%m%d"
        return datetime.strptime(arg, fmt)
    else:
        # 统一转为无时区，避免与 Polars 列 dtype 不匹配
        if getattr(arg, "tzinfo", None) is not None:
            return arg.replace(tzinfo=None)
        return arg


class Segment(Enum):
    """Data segment enumeration values"""

    TRAIN = 1
    VALID = 2
    TEST = 3
