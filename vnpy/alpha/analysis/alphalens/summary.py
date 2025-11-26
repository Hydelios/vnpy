from __future__ import annotations

from typing import Optional, Iterable
import time
import pandas as pd
from pathlib import Path
from ...logger import logger

from .metrics import compute_summary_metrics
from .runner import run_factor_analysis
from multiprocessing import get_context
from .extract import collect_period_range, prepare_shared_views, build_factor_series

# 全局引用（在 fork 场景下由父进程设置，子进程继承，避免在任务参数中反复 pickle 大对象）
_G_DATASET = None
_G_SHARED_KEYS_PD = None
_G_SHARED_PRICE = None


def _best_context():
    try:
        return get_context("fork")
    except Exception:
        return get_context("spawn")


def _summarize_factor_worker(args) -> pd.Series:
    global _G_DATASET, _G_SHARED_KEYS_PD, _G_SHARED_PRICE
    dataset, name, cfg, interval, period, periods, shared_keys, shared_price = args
    try:
        # 若未显式传入，则回退使用全局引用（仅在 fork 场景生效）
        if dataset is None:
            dataset = _G_DATASET
        if shared_keys is None:
            shared_keys = _G_SHARED_KEYS_PD
        if shared_price is None:
            shared_price = _G_SHARED_PRICE
        auto_adj = False
        drop_thres = 0.8
        try:
            if isinstance(cfg, dict) and cfg.get("__auto_adj__"):
                auto_adj = True
                drop_thres = float(cfg.get("__drop_thres__", 0.8))
                # 清理内部标记，避免传入后续流程
                cfg.pop("__auto_adj__", None)
                cfg.pop("__drop_thres__", None)
        except Exception:
            pass
        return summarize_factor(
            dataset,
            name,
            cfg=cfg,
            interval=interval,
            period=period,
            periods=periods,
            shared_keys=shared_keys,
            shared_price=shared_price,
            _auto_adjust_quantiles=auto_adj,
            _drop_threshold=drop_thres,
        )
    except Exception:
        return pd.Series({}, name=name)


def _estimate_base_count(dataset, factor_name: str, *, cfg: dict, interval: str, shared_keys) -> int:
    """估算输入样本量：对齐 keys 后的非空因子条目数量。"""
    start, end = collect_period_range(dataset)
    values_from = str(cfg.get("alphalens_values_from", "result")).lower()
    # 将共享 keys 转换回 polars（若可能）
    try:
        import polars as pl  # type: ignore
        if shared_keys is not None and not isinstance(shared_keys, pl.DataFrame):
            try:
                shared_keys = pl.from_pandas(shared_keys)
            except Exception:
                shared_keys = None
    except Exception:
        shared_keys = None
    try:
        s = build_factor_series(
            dataset,
            factor_name,
            start,
            end,
            values_from=values_from,
            keys_df=shared_keys if shared_keys is not None else None,
        )
        return int(s.dropna().shape[0])
    except Exception:
        return 0


def _rename_with_period_suffix(metrics: dict) -> pd.Series:
    """将 metrics 的 period 合并到列名后缀，如 IC_5D、IC_IR_5D、Annual_TOP_Return_5D 等。"""
    period = str(metrics.get("period")) if metrics.get("period") is not None else ""
    suffix = f"_{period}" if period else ""
    mapping = {
        "Return T-test": "Return_T_test",
        "IC Mean": "IC",
        "IC IR": "IC_IR",
        "Annual T-B Return": "Annual_TB_Return",
        "Annual TOP Return": "Annual_TOP_Return",
        "Annual TOP Sharpe": "Annual_TOP_Sharpe",
        "Group IC": "Group_IC",
    }
    data = {}
    for k, v in metrics.items():
        if k == "period":
            continue
        new_key = mapping.get(k, k.replace(" ", "_").replace("-", "_")) + suffix
        data[new_key] = v
    return pd.Series(data)


def _infer_factor_names(dataset) -> list[str]:
    """当 dataset.feature_expressions 不可用时，从 result_df 推断因子列名。

    策略：排除基础行情列和常见标签列，余下视为可评估因子列。
    """
    try:
        df = getattr(dataset, "result_df")
        cols = list(df.columns)
    except Exception:
        return []

    exclude = {
        "datetime",
        "vt_symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "open_interest",
        "vwap",
        "label",
    }
    # 也排除可能出现的中间列名（保守）
    candidates = [c for c in cols if c not in exclude]
    return candidates


def summarize_factor(
    dataset,
    factor_name: str,
    *,
    cfg: Optional[dict] = None,
    interval: Optional[str] = None,
    period: Optional[int] = None,
    periods: Optional[Iterable[int]] = None,
    shared_keys=None,
    shared_price=None,
    _auto_adjust_quantiles: bool = False,
    _drop_threshold: float = 0.8,
) -> pd.Series:
    """对单个因子计算汇总指标（不展示图，不落盘）。"""
    fa = dict(cfg or {})
    itv = (interval or getattr(dataset, "interval", "1d")).lower()

    # 统一期数集合（兼容单 period 与 periods 列表）
    req_periods: list[int] = []
    if periods is not None:
        try:
            req_periods = [int(p) for p in periods]
        except Exception:
            req_periods = []
    elif period is not None:
        try:
            req_periods = [int(period)]
        except Exception:
            req_periods = []

    # 若请求多个周期：一次性清洗包含所有周期
    fa2 = dict(fa)
    # 汇总场景：强制不落盘
    fa2["alphalens_save_figs"] = False
    fa2["alphalens_save_analysis"] = False
    if req_periods:
        fa2["alphalens_periods"] = tuple(sorted(set(req_periods)))

    def _metrics_from_clean(clean_df: pd.DataFrame) -> pd.Series:
        q_order = str(fa.get("alphalens_quantile_order", "asc"))
        series_parts: list[pd.Series] = []
        if req_periods:
            for p in req_periods:
                m = compute_summary_metrics(clean_df, period=p, quantile_order=q_order)
                series_parts.append(_rename_with_period_suffix(m))
        else:
            m = compute_summary_metrics(clean_df, period=None, quantile_order=q_order)
            series_parts.append(_rename_with_period_suffix(m))
        s = pd.concat(series_parts) if len(series_parts) > 1 else series_parts[0]
        s.name = factor_name
        return s

    if not _auto_adjust_quantiles:
        clean = run_factor_analysis(
            dataset,
            factor_name,
            cfg=fa2,
            interval=itv,
            display=False,
            shared_keys=shared_keys,
            shared_price=shared_price,
            no_save=True,
        )
        return _metrics_from_clean(clean)

    # 自动量化分箱自适应：按 10→5→3 尝试，丢弃率低于阈值即接受
    base_count = _estimate_base_count(dataset, factor_name, cfg=fa2, interval=itv, shared_keys=shared_keys)
    candidates = [10, 5, 3]
    best = None  # (drop_ratio, clean_df, q)
    for q in candidates:
        attempt_cfg = dict(fa2)
        attempt_cfg["alphalens_quantiles"] = q
        try:
            clean = run_factor_analysis(
                dataset,
                factor_name,
                cfg=attempt_cfg,
                interval=itv,
                display=False,
                shared_keys=shared_keys,
                shared_price=shared_price,
                no_save=True,
            )
        except Exception:
            continue
        kept = int(getattr(clean, "shape", (0,))[0]) if hasattr(clean, "shape") else 0
        drop_ratio = 1.0 if base_count <= 0 else max(0.0, 1.0 - (kept / float(base_count)))
        logger.info(f"因子 {factor_name} | 量化分箱={q} | 样本保留={kept}/{base_count} | 丢弃率={drop_ratio:.1%}")
        if best is None or drop_ratio < best[0]:
            best = (drop_ratio, clean, q)
        if drop_ratio <= float(_drop_threshold):
            logger.info(f"因子 {factor_name} 采用量化分箱={q}（丢弃率<=阈值 {float(_drop_threshold):.0%}）。")
            return _metrics_from_clean(clean)

    # 若无候选满足阈值，使用丢弃率最低的候选
    if best is not None:
        logger.warning(f"因子 {factor_name} 未满足阈值，回退使用量化分箱={best[2]}，丢弃率={best[0]:.1%}。")
        return _metrics_from_clean(best[1])

    # 全部失败：返回空结果
    return pd.Series({}, name=factor_name)


def summarize_factors(
    dataset,
    factor_names: Optional[Iterable[str]] = None,
    *,
    cfg: Optional[dict] = None,
    interval: Optional[str] = None,
    period: Optional[int] = None,
    periods: Optional[Iterable[int]] = None,
    workers: Optional[int] = None,
    auto_adjust_quantiles: bool = False,
    drop_threshold: float = 0.8,
    save_csv: bool | None = None,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """批量计算多个因子的汇总指标表（不展示图，不落盘）。"""
    # 1) 优先使用显式传入
    names: list[str]
    if factor_names is not None:
        names = list(factor_names)
    else:
        # 2) 其次尝试 feature_expressions
        try:
            names = list(getattr(dataset, "feature_expressions").keys())
        except Exception:
            names = []
    # 3) 若仍为空，尝试从 result_df 推断
    if not names:
        names = _infer_factor_names(dataset)

    rows = []
    itv = (interval or getattr(dataset, "interval", "1d")).lower()
    try:
        logger.info(
            f"开始批量因子汇总：因子数={len(names)}, periods={list(periods) if periods is not None else ([period] if period is not None else [])}, "
            f"interval={itv}, workers={int(workers) if workers else 1}, auto_adjust={bool(auto_adjust_quantiles)}, drop_threshold={float(drop_threshold):.0%}"
        )
    except Exception:
        pass
    # 预构建共享视图（keys + 价格宽表），减少逐因子重复计算
    start, end = collect_period_range(dataset)
    keys_from = str((cfg or {}).get("alphalens_keys_from", "infer")).lower()
    shared_keys, shared_price = prepare_shared_views(dataset, start, end, keys_from=keys_from)
    try:
        # 打印共享视图规模
        keys_cnt = None
        try:
            keys_cnt = getattr(shared_keys, "height", None) or (shared_keys.shape[0] if hasattr(shared_keys, "shape") else None)
        except Exception:
            keys_cnt = None
        price_shape = None
        try:
            price_shape = getattr(shared_price, "shape", None)
        except Exception:
            price_shape = None
        logger.info(f"共享视图就绪：keys={keys_cnt}, price_df.shape={price_shape}")
    except Exception:
        pass
    # 为兼容 spawn，将 keys 转为 pandas 以便可序列化；子进程内再转回 polars
    try:
        shared_keys_pd = shared_keys.to_pandas()  # type: ignore[attr-defined]
    except Exception:
        shared_keys_pd = None

    n_workers = int(workers) if (workers and int(workers) > 1) else 1
    total = len(names)
    if n_workers > 1:
        ctx = _best_context()
        try:
            # 在 fork 场景下，将大对象放入全局，避免作为任务参数 pickle
            try:
                if ctx.get_start_method() == "fork":
                    global _G_DATASET, _G_SHARED_KEYS_PD, _G_SHARED_PRICE
                    _G_DATASET = dataset
                    _G_SHARED_KEYS_PD = shared_keys_pd
                    _G_SHARED_PRICE = shared_price
            except Exception:
                pass
            with ctx.Pool(n_workers) as pool:
                tasks = []
                for n in names:
                    if bool(auto_adjust_quantiles):
                        task_cfg = {**(cfg or {}), "__auto_adj__": True, "__drop_thres__": float(drop_threshold)}
                    else:
                        task_cfg = cfg
                    # 若为 fork，则传 None 值以减少参数传输体积
                    use_dataset = None
                    use_keys = None
                    use_price = None
                    try:
                        if ctx.get_start_method() != "fork":
                            use_dataset = dataset
                            use_keys = shared_keys_pd
                            use_price = shared_price
                    except Exception:
                        use_dataset = dataset
                        use_keys = shared_keys_pd
                        use_price = shared_price
                    args = (use_dataset, n, task_cfg, interval, period, periods, use_keys, use_price)
                    t = pool.apply_async(_summarize_factor_worker, (args,))
                    tasks.append((n, t))

                # 心跳进度：定期打印已完成数量
                done_last = -1
                t0 = time.time()
                while True:
                    done = sum(1 for _, t in tasks if t.ready())
                    if verbose and (done != done_last or (time.time() - t0) >= 10):
                        try:
                            logger.info(f"进度：{done}/{total} 完成")
                        except Exception:
                            pass
                        done_last = done
                        t0 = time.time()
                    if done >= total:
                        break
                    time.sleep(1)

                # 收集结果（按提交顺序）
                for i, (n, t) in enumerate(tasks, 1):
                    try:
                        s = t.get()
                    except Exception:
                        s = pd.Series({}, name=n)
                    rows.append(s)
                    if verbose:
                        try:
                            logger.info(f"完成：{i}/{total} -> {n}")
                        except Exception:
                            pass
        except Exception:
            # 回退串行
            rows = []
            for i, n in enumerate(names, 1):
                if bool(auto_adjust_quantiles):
                    s = summarize_factor(
                        dataset,
                        n,
                        cfg=cfg,
                        interval=interval,
                        period=period,
                        periods=periods,
                        shared_keys=shared_keys_pd,
                        shared_price=shared_price,
                        _auto_adjust_quantiles=True,
                        _drop_threshold=float(drop_threshold),
                    )
                else:
                    s = summarize_factor(
                        dataset,
                        n,
                        cfg=cfg,
                        interval=interval,
                        period=period,
                        periods=periods,
                        shared_keys=shared_keys_pd,
                        shared_price=shared_price,
                    )
                rows.append(s)
                if verbose:
                    try:
                        logger.info(f"完成：{i}/{total} -> {n}")
                    except Exception:
                        pass
    else:
        rows = []
        for i, n in enumerate(names, 1):
            if bool(auto_adjust_quantiles):
                s = summarize_factor(
                    dataset,
                    n,
                    cfg=cfg,
                    interval=interval,
                    period=period,
                    periods=periods,
                    shared_keys=shared_keys_pd,
                    shared_price=shared_price,
                    _auto_adjust_quantiles=True,
                    _drop_threshold=float(drop_threshold),
                )
            else:
                s = summarize_factor(
                    dataset,
                    n,
                    cfg=cfg,
                    interval=interval,
                    period=period,
                    periods=periods,
                    shared_keys=shared_keys_pd,
                    shared_price=shared_price,
                )
            rows.append(s)
            try:
                logger.info(f"进度：{i}/{total} 完成 -> {n}")
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 列顺序重排：同一指标按多个 period 并列，如 IC_5D, IC_10D, IC_20D, 然后 IC_IR_5D, ...
    period_list: list[int] = []
    try:
        if periods is not None:
            period_list = [int(p) for p in periods]
        elif period is not None:
            period_list = [int(period)]
    except Exception:
        period_list = []

    if period_list and not df.empty:
        metric_order = [
            "Return_T_test",
            "IC",
            "IC_IR",
            "Annual_TB_Return",
            "Annual_TOP_Return",
            "Group_IC",
            "Annual_TOP_Sharpe",
        ]
        ordered_cols: list[str] = []
        for m in metric_order:
            for p in period_list:
                col = f"{m}_{int(p)}D"
                if col in df.columns:
                    ordered_cols.append(col)
        # 将未涵盖的列追加在后（如部分 period 不存在或额外列）
        ordered_cols += [c for c in df.columns if c not in ordered_cols]
        df = df.reindex(columns=ordered_cols)

    # 可选：落盘 CSV（不依赖 JSON 配置，完全由入参控制；若 save_dir 为空则回退 cfg/当前目录）
    if bool(save_csv):
        itv = (interval or getattr(dataset, "interval", "1d")).lower()
        base = None
        if save_dir:
            base = Path(save_dir)
        else:
            try:
                base = Path(cfg.get("alphalens_analysis_dir")) if isinstance(cfg, dict) and cfg.get("alphalens_analysis_dir") else None
            except Exception:
                base = None
        if base is None:
            base = Path.cwd() / "analysis"
        out_dir = base / itv
        out_dir.mkdir(parents=True, exist_ok=True)

        # 默认文件名：summary 或附加周期后缀
        if filename:
            fname = filename
        else:
            if periods:
                try:
                    tags = "_".join(f"{int(p)}D" for p in periods)
                except Exception:
                    tags = "multi"
                fname = f"summary_{tags}.csv"
            elif period is not None:
                fname = f"summary_{int(period)}D.csv"
            else:
                fname = "summary.csv"

        out_path = out_dir / fname
        try:
            # 为通用 CSV 兼容性，不用多层索引，直接按当前索引写出
            df.to_csv(out_path, index=True, encoding="utf-8-sig")
            logger.info(f"汇总指标已保存：{out_path}")
        except Exception as e:
            logger.error(f"保存汇总 CSV 失败：{out_path} | {e}")

    try:
        logger.info(f"批量因子汇总完成：共 {len(names)} 个因子，输出表形状={df.shape}")
    except Exception:
        pass

    return df
