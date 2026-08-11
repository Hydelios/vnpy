from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"不支持的信号格式: {path.suffix}")
    if isinstance(data, pd.DataFrame):
        return data.copy()
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        if isinstance(frame, pd.DataFrame):
            return frame
    raise TypeError(f"不支持的信号数据类型: {type(data)!r}")


def to_rq_order_book_id(symbol: str) -> str:
    value = str(symbol).strip()
    if value.endswith(".SSE"):
        return value[:-4] + ".XSHG"
    if value.endswith(".SZSE"):
        return value[:-5] + ".XSHE"
    if value.endswith(".SH"):
        return value[:-3] + ".XSHG"
    if value.endswith(".SZ"):
        return value[:-3] + ".XSHE"
    return value


def normalize_signal(signal: Any) -> pd.DataFrame:
    frame = _to_pandas(signal)
    required = {"datetime", "vt_symbol", "signal"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"信号缺少必要列: {missing}")

    columns = ["datetime", "vt_symbol", "signal"]
    if "weight" in frame.columns:
        columns.append("weight")
    out = frame.loc[:, columns].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
    out["order_book_id"] = out["vt_symbol"].map(to_rq_order_book_id)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce")
    if "weight" in out.columns:
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["datetime", "order_book_id", "signal"])
    out = out.drop_duplicates(["datetime", "order_book_id"], keep="last")
    return out.sort_values(
        ["datetime", "signal", "order_book_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def prepare_ranked_signals(
    signal: Any,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Normalize a signal and attach deterministic daily ranks."""

    frame = normalize_signal(signal)
    if start_date is not None:
        frame = frame[frame["datetime"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["datetime"] <= pd.Timestamp(end_date)]
    if frame.empty:
        raise ValueError("日期过滤后的信号为空")

    frame = frame.copy()
    frame["rank"] = frame.groupby("datetime", sort=True).cumcount() + 1
    columns = ["datetime", "order_book_id", "signal", "rank"]
    if "weight" in frame.columns:
        columns.append("weight")
    return frame.loc[:, columns].reset_index(drop=True)
