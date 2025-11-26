from datetime import datetime
from enum import Enum
from typing import Union

import polars as pl


class DataProxy:
    """Feature data proxy"""

    def __init__(self, df: pl.DataFrame) -> None:
        """Constructor"""
        self.name: str = df.columns[-1]
        self.df: pl.DataFrame = df.rename({self.name: "data"})

        # Note that for numerical expressions, variables should be placed before numbers. e.g. a * 2

    def result(self, s: pl.Series) -> "DataProxy":
        """Convert series data to feature object"""
        result: pl.DataFrame = self.df[["datetime", "vt_symbol"]]
        result = result.with_columns(other=s)

        return DataProxy(result)

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


def calculate_by_expression(df: pl.DataFrame, expression: str) -> pl.DataFrame:
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
    # Extra operators (time-series and cross-section)
    from .extra_function import (           # noqa
        ts_decay_linear,
        ts_cov,
        ts_prod,
        ts_delta,
        ts_ema,
        ts_ewm,
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
    from .neutralize_function import (      # noqa
        cs_demean,
        cs_zscore,
        cs_demean_by_category,
        cs_zscore_by_category,
        cs_neutralize_ols1,
    )

    # Backward-compatibility aliases for expression strings
    # 一些历史模板中使用 log(...)，这里兼容为 ts_log
    d: dict = locals()
    d["log"] = ts_log  # type: ignore

    # Extract feature objects to local space

    for column in df.columns:
        # Filter index columns
        if column in {"datetime", "vt_symbol"}:
            continue

        # Cache feature df
        column_df = df[["datetime", "vt_symbol", column]]
        d[column] = DataProxy(column_df)

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
