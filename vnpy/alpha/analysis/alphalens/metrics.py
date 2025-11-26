from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


def _select_period_label(clean_data: pd.DataFrame, preferred: Optional[int] = None) -> tuple[str, int]:
    from alphalens.utils import get_forward_returns_columns

    cols = list(get_forward_returns_columns(clean_data.columns))
    if not cols:
        raise ValueError("clean_data 中未找到 forward returns 列")

    def _days(label: str) -> int:
        try:
            return int(str(label).upper().rstrip("D").replace("D", "").replace(" ", ""))
        except Exception:
            return 1

    if preferred is not None:
        target = f"{int(preferred)}D"
        for c in cols:
            if str(c).upper() == target.upper():
                return c, int(preferred)

    cols_sorted = sorted(cols, key=lambda x: _days(x))
    sel = cols_sorted[0]
    return sel, _days(sel)


def _annualize_from_period_mean(mean_r: float, period_days: int) -> float:
    try:
        return (1.0 + float(mean_r)) ** (252.0 / max(1, period_days)) - 1.0
    except Exception:
        return float("nan")


def _annualized_sharpe(mean_r: float, std_r: float, period_days: int) -> float:
    import math
    if std_r == 0 or not (std_r > 0):
        return 0.0
    try:
        return (mean_r / std_r) * math.sqrt(252.0 / max(1, period_days))
    except Exception:
        return float("nan")


def compute_summary_metrics(
    clean_data: pd.DataFrame,
    *,
    period: Optional[int] = None,
    quantile_order: str = "asc",
) -> dict:
    """从 clean_data 计算单因子汇总指标。

    输出包含：IC Mean, IC IR, Annual TOP Return, Annual TOP Sharpe, Annual T-B Return。
    - period: 使用的 forward 天数（None 则取最小 forward 列）
    - quantile_order: asc/desc（若 desc，则 1 档为 Top；否则 N 档为 Top）
    """
    fwd_col, pdays = _select_period_label(clean_data, preferred=period)

    def _ic_per_date(df: pd.DataFrame) -> float:
        try:
            return df["factor"].corr(df[fwd_col], method="spearman")
        except Exception:
            return np.nan

    ic_daily = clean_data[["factor", fwd_col]].groupby(level=0).apply(_ic_per_date)
    ic_mean = float(np.nanmean(ic_daily)) if len(ic_daily) else float("nan")
    ic_std = float(np.nanstd(ic_daily, ddof=1)) if len(ic_daily) > 1 else float("nan")
    ic_ir = float(ic_mean / ic_std) if (ic_std and ic_std == ic_std and ic_std > 0) else float("nan")

    try:
        q_vals = pd.to_numeric(clean_data["factor_quantile"], errors="coerce")
        q_max = int(q_vals.max()) if q_vals.size else 10
    except Exception:
        q_max = 10
    top_label = 1 if quantile_order.lower() == "desc" else q_max
    bottom_label = q_max if quantile_order.lower() == "desc" else 1

    grp = clean_data[[fwd_col, "factor_quantile"]].reset_index().groupby("date")

    def _mean_q(df: pd.DataFrame, q: int) -> float:
        sub = df[df["factor_quantile"] == q]
        return float(sub[fwd_col].mean()) if not sub.empty else np.nan

    top_series = grp.apply(lambda x: _mean_q(x, top_label))
    bottom_series = grp.apply(lambda x: _mean_q(x, bottom_label))
    tb_series = top_series - bottom_series

    top_mean, top_std = float(np.nanmean(top_series)), float(np.nanstd(top_series, ddof=1))
    tb_mean = float(np.nanmean(tb_series))

    # Return T-test（对 Top-Bottom 序列做 t 检验，零均值假设）
    t_stat = float("nan")
    try:
        from scipy import stats as _stats  # type: ignore
        vals = pd.to_numeric(tb_series, errors="coerce").dropna().values
        if vals.size > 1:
            t_stat = float(_stats.ttest_1samp(vals, 0, nan_policy="omit").statistic)
    except Exception:
        # 手算 t 值作为兜底
        vals = pd.to_numeric(tb_series, errors="coerce").dropna().values
        if vals.size > 1:
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1))
            n = float(vals.size)
            t_stat = float(m / (s / np.sqrt(n))) if (s and s > 0) else float("nan")

    ann_top_ret = _annualize_from_period_mean(top_mean, pdays)
    ann_top_sharpe = _annualized_sharpe(top_mean, top_std, pdays)
    ann_tb_ret = _annualize_from_period_mean(tb_mean, pdays)

    # Group IC（若存在 group 列，则对每个日期-组计算秩相关并取均值）
    group_ic_mean = float("nan")
    if "group" in clean_data.columns:
        try:
            def _ic_per_date_group(df: pd.DataFrame) -> float:
                try:
                    return df["factor"].corr(df[fwd_col], method="spearman")
                except Exception:
                    return np.nan

            grp2 = clean_data[["factor", fwd_col, "group"]].reset_index().groupby(["date", "group"], sort=False)
            ic_by_dg = grp2.apply(_ic_per_date_group)
            group_ic_mean = float(pd.to_numeric(ic_by_dg, errors="coerce").mean()) if len(ic_by_dg) else float("nan")
        except Exception:
            group_ic_mean = float("nan")

    return {
        "period": f"{pdays}D",
        "Return T-test": t_stat,
        "IC Mean": ic_mean,
        "IC IR": ic_ir,
        "Annual T-B Return": ann_tb_ret,
        "Annual TOP Return": ann_top_ret,
        "Group IC": group_ic_mean,
        "Annual TOP Sharpe": ann_top_sharpe,
    }


def generate_multi_factor_summary(
    all_factors_data: Dict[str, pd.DataFrame],
    period: str,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    计算并汇总多个因子的绩效指标。

    参数:
    - all_factors_data: {因子名: clean factor_data}，即通过 alphalens.utils.get_clean_factor_and_forward_returns 产生的 DataFrame。
    - period: 远期收益周期标签，如 '5D'、'10D'。
    - annualization_factor: 年化因子（按日度 252）。

    返回:
    - pd.DataFrame: 索引为因子名称，列为指标：
      ['Return T-test', 'IC Mean', 'IC IR', 'Annual T-B Return', 'Annual TOP Return', 'Group IC', 'Annual TOP Sharp']
    """
    try:
        from alphalens import performance as perf  # type: ignore
    except Exception as e:  # pragma: no cover - 运行环境可能缺少 alphalens
        raise ImportError(f"缺少 alphalens 依赖：{e}")

    # 尝试导入 utils 以健壮地识别 forward 列
    try:
        from alphalens.utils import get_forward_returns_columns  # type: ignore
    except Exception:
        get_forward_returns_columns = None  # type: ignore

    results_list: list[Dict[str, Any]] = []

    def _select_period_col(df: pd.DataFrame, period_label: str) -> str:
        # 优先使用 alphalens 提供的辅助函数
        if get_forward_returns_columns is not None:
            try:
                cols = list(get_forward_returns_columns(df.columns))
                for c in cols:
                    if str(c).upper() == str(period_label).upper():
                        return c
            except Exception:
                pass
        # 回退：在列名里匹配（忽略大小写）
        for c in df.columns:
            if str(c).upper() == str(period_label).upper():
                return c
        # 仍找不到则返回 period_label 本身，后续可能触发 KeyError -> 由调用处兜底
        return period_label

    for factor_name, factor_data in all_factors_data.items():
        stats_dict: Dict[str, Any] = {}

        # 识别远期收益列名
        fwd_col = _select_period_col(factor_data, period)

        # 1) IC 时间序列（对指定 period 取列）
        ic_series = None
        try:
            ic_res = perf.factor_information_coefficient(
                factor_data, group_adjust=False, by_group=False, method="spearman"
            )
            if isinstance(ic_res, pd.DataFrame) and fwd_col in ic_res.columns:
                ic_series = ic_res[fwd_col]
            elif isinstance(ic_res, pd.Series):
                ic_series = ic_res
        except Exception:
            ic_series = None

        # 2) Top-Bottom 日度收益序列（优先使用 perf.factor_returns；若不稳定则回退手算）
        tb_returns = None
        try:
            fr = perf.factor_returns(factor_data)
            if isinstance(fr, pd.DataFrame) and fwd_col in fr.columns:
                tb_returns = fr[fwd_col].squeeze()
            elif isinstance(fr, pd.Series):
                tb_returns = fr.squeeze()
        except Exception:
            tb_returns = None

        # 回退：用分位收益手动计算 T-B（Top - Bottom）
        if tb_returns is None:
            try:
                df = factor_data.reset_index()
                q = pd.to_numeric(df.get("factor_quantile"), errors="coerce")
                if q.notna().any() and fwd_col in df.columns and "date" in df.columns:
                    q_max = int(q.max()) if q.size else 10
                    top_label = q_max
                    bottom_label = 1
                    top_ts = (
                        df[df["factor_quantile"] == top_label]
                        .groupby("date")[fwd_col]
                        .mean()
                    )
                    bottom_ts = (
                        df[df["factor_quantile"] == bottom_label]
                        .groupby("date")[fwd_col]
                        .mean()
                    )
                    tb_returns = (top_ts - bottom_ts).dropna()
            except Exception:
                tb_returns = None

        # 3) 分位数收益时间序列（用于 Top Sharpe）——手动按分位聚合，避免版本差异
        top_ret_ts = None
        try:
            df = factor_data.reset_index()
            if fwd_col in df.columns and "date" in df.columns and "factor_quantile" in df.columns:
                q = pd.to_numeric(df["factor_quantile"], errors="coerce")
                if q.notna().any():
                    q_max = int(q.max()) if q.size else 10
                    top_label = q_max
                    top_ret_ts = (
                        df[df["factor_quantile"] == top_label]
                        .groupby("date")[fwd_col]
                        .mean()
                        .dropna()
                    )
        except Exception:
            top_ret_ts = None

        # 4) (可选) 分组 IC（若数据不含 group 信息则可能不可用）
        mean_group_ic = np.nan
        try:
            group_ic = perf.mean_information_coefficient_by_group(factor_data)
            if isinstance(group_ic, pd.DataFrame):
                # 该返回通常 index = 分组，columns = period 列
                col = fwd_col if fwd_col in group_ic.columns else None
                if col is None:
                    # 找不到指定列则用第一个
                    col = group_ic.columns[0] if len(group_ic.columns) else None
                if col is not None:
                    mean_group_ic = float(pd.to_numeric(group_ic[col], errors="coerce").mean())
        except Exception:
            pass

        # 5) 指标计算
        # Return T-test
        t_stat = np.nan
        if tb_returns is not None and len(tb_returns.dropna()) > 1:
            arr = pd.to_numeric(tb_returns, errors="coerce").dropna().values
            if arr.size > 1:
                try:
                    from scipy import stats as _stats  # type: ignore
                    t_stat = float(_stats.ttest_1samp(arr, 0, nan_policy="omit").statistic)
                except Exception:
                    # 手算 t 值
                    m = float(np.mean(arr))
                    s = float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
                    n = float(arr.size)
                    t_stat = float(m / (s / np.sqrt(n))) if (s and s > 0) else np.nan

        # IC Mean / IC IR
        ic_mean = np.nan
        ic_ir = np.nan
        if ic_series is not None:
            ser = pd.to_numeric(ic_series, errors="coerce").dropna()
            if not ser.empty:
                ic_mean = float(ser.mean())
                std = float(ser.std(ddof=1)) if ser.size > 1 else float("nan")
                ic_ir = float(ic_mean / std) if (std and std > 0) else float("nan")

        # Annual T-B Return
        annual_tb_return = np.nan
        if tb_returns is not None:
            v = pd.to_numeric(tb_returns, errors="coerce").dropna()
            if not v.empty:
                annual_tb_return = float(v.mean() * annualization_factor)

        # Annual TOP Return / Annual TOP Sharp
        annual_top_return = np.nan
        annual_top_sharp = np.nan
        if top_ret_ts is not None and not top_ret_ts.empty:
            m = float(pd.to_numeric(top_ret_ts, errors="coerce").dropna().mean())
            s = float(pd.to_numeric(top_ret_ts, errors="coerce").dropna().std(ddof=1))
            if m == m:
                annual_top_return = float(m * annualization_factor)
            if s and s > 0:
                annual_top_sharp = float((m * annualization_factor) / (s * np.sqrt(annualization_factor)))

        stats_dict["Return T-test"] = t_stat
        stats_dict["IC Mean"] = ic_mean
        stats_dict["IC IR"] = ic_ir
        stats_dict["Annual T-B Return"] = annual_tb_return
        stats_dict["Annual TOP Return"] = annual_top_return
        stats_dict["Group IC"] = mean_group_ic
        stats_dict["Annual TOP Sharp"] = annual_top_sharp

        results_list.append({"Factor": factor_name, **stats_dict})

    final_table = pd.DataFrame.from_records(results_list)
    if not final_table.empty and "Factor" in final_table.columns:
        final_table = final_table.set_index("Factor")
    final_table.index.name = "Factor"
    return final_table
