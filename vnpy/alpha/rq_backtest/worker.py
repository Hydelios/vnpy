#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from rqalpha.apis import get_positions, order_target_portfolio_smart, update_universe

from selection import (
    MarketState,
    build_fixed_capital_topk_target,
    collect_fixed_capital_market_states,
)


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "_asdict"):
        return _to_builtin(value._asdict())
    return str(value)


def _drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    nav = pd.concat([pd.Series([1.0]), nav.reset_index(drop=True)], ignore_index=True)
    return float((nav / nav.cummax() - 1.0).min()) if len(nav) > 1 else 0.0


def _valid_price(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _market_state(bar: Any) -> MarketState:
    """Translate an RQAlpha opening bar into directional constraints."""

    open_price = float(bar.open) if _valid_price(bar.open) else 0.0
    last_price = getattr(bar, "last", 0.0)
    valuation_price = (
        open_price
        if open_price > 0
        else float(last_price) if _valid_price(last_price) else 0.0
    )
    suspended = bool(getattr(bar, "suspended", False))
    if suspended or open_price <= 0:
        return MarketState(
            price=valuation_price,
            can_buy=False,
            can_sell=False,
            buy_block_reason="inactive",
            sell_block_reason="inactive",
        )

    limit_up = float(bar.limit_up) if _valid_price(bar.limit_up) else math.inf
    limit_down = float(bar.limit_down) if _valid_price(bar.limit_down) else -math.inf
    at_limit_up = open_price >= limit_up or math.isclose(open_price, limit_up, abs_tol=1e-7)
    at_limit_down = open_price <= limit_down or math.isclose(open_price, limit_down, abs_tol=1e-7)
    return MarketState(
        price=open_price,
        can_buy=not at_limit_up,
        can_sell=not at_limit_down,
        buy_block_reason="limit_up" if at_limit_up else None,
        sell_block_reason="limit_down" if at_limit_down else None,
    )


def _annual_metrics(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    trading_days: int = 240,
) -> pd.DataFrame:
    if {"total_value", "units"}.issubset(portfolio.columns):
        strategy_nav = portfolio["total_value"].astype(float) / portfolio["units"].astype(float)
    else:
        strategy_nav = portfolio["unit_net_value"].astype(float)
    benchmark_nav = benchmark["unit_net_value"].astype(float).reindex(strategy_nav.index).ffill()
    strategy_return = strategy_nav.pct_change()
    benchmark_return = benchmark_nav.pct_change()
    if not strategy_return.empty:
        strategy_return.iloc[0] = strategy_nav.iloc[0] - 1.0
        benchmark_return.iloc[0] = benchmark_nav.iloc[0] - 1.0

    frame = pd.DataFrame(
        {
            "strategy_return": strategy_return.fillna(0.0),
            "benchmark_return": benchmark_return.fillna(0.0),
        },
        index=strategy_nav.index,
    )
    frame["excess_return"] = frame["strategy_return"] - frame["benchmark_return"]
    frame["year"] = pd.to_datetime(frame.index).year

    rows: list[dict[str, float | int]] = []
    for year, group in frame.groupby("year", sort=True):
        strategy_total = float((1.0 + group["strategy_return"]).prod() - 1.0)
        benchmark_total = float((1.0 + group["benchmark_return"]).prod() - 1.0)
        excess_total = (1.0 + strategy_total) / (1.0 + benchmark_total) - 1.0
        excess_std = float(group["excess_return"].std(ddof=0))
        excess_sharpe = (
            float(group["excess_return"].mean() / excess_std * np.sqrt(trading_days))
            if excess_std > 0
            else 0.0
        )
        relative_nav = (
            (1.0 + group["strategy_return"]).cumprod()
            / (1.0 + group["benchmark_return"]).cumprod()
        )
        relative_nav = pd.concat(
            [pd.Series([1.0]), relative_nav.reset_index(drop=True)],
            ignore_index=True,
        )
        rows.append(
            {
                "year": int(year),
                "n_dates": int(len(group)),
                "return": strategy_total,
                "max_drawdown": _drawdown(group["strategy_return"]),
                "benchmark_return": benchmark_total,
                "excess_return": excess_total,
                "excess_sharpe": excess_sharpe,
                "excess_max_drawdown": float(
                    (relative_nav / relative_nav.cummax() - 1.0).min()
                ),
            }
        )
    return pd.DataFrame(rows)


def run(config_path: Path, rankings_path: Path) -> None:
    from rqalpha import run_func

    settings = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(settings["output_dir"])
    rankings_frame = pl.read_parquet(rankings_path).sort(["datetime", "rank"])
    has_signal_weights = "weight" in rankings_frame.columns
    rankings: dict[date, tuple[list[str], dict[str, float | None] | None]] = {}
    for (timestamp,), group in rankings_frame.group_by("datetime", maintain_order=True):
        symbols = group["order_book_id"].to_list()
        weights = None
        if has_signal_weights:
            weights = dict(zip(symbols, group["weight"].to_list(), strict=True))
        rankings[timestamp.date()] = (symbols, weights)
    universe = rankings_frame["order_book_id"].unique().sort().to_list()
    target_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    def init(context):
        context.rankings = rankings
        context.pending_ranking = None
        context.days_until_signal = 0
        update_universe(universe)

    def handle_bar(context, bar_dict):
        # Match FixedCapitalTopKStrategy.on_bars: sample one close signal every
        # rebalance_interval trading days.  A missing signal is retried next day.
        if context.days_until_signal > 0:
            context.days_until_signal -= 1
            return
        ranking = context.rankings.get(context.now.date())
        if ranking is None:
            return
        context.pending_ranking = (context.now.date(), *ranking)
        context.days_until_signal = int(settings["rebalance_interval"]) - 1

    def open_auction(context, bar_dict):
        # The pending T-close signal is consumed only at the T+1 open.
        pending = context.pending_ranking
        context.pending_ranking = None
        if pending is None:
            return

        signal_date, ranked, signal_weights = pending
        position_objects = [position for position in get_positions() if position.quantity > 0]
        positions = {
            position.order_book_id: float(position.quantity)
            for position in position_objects
        }

        def get_market_state(symbol: str) -> MarketState:
            try:
                return _market_state(bar_dict[symbol])
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                return MarketState(
                    price=0.0,
                    can_buy=False,
                    can_sell=False,
                    buy_block_reason="inactive",
                    sell_block_reason="inactive",
                )

        market_states = collect_fixed_capital_market_states(
            ranked,
            positions,
            top_k=int(settings["top_k"]),
            exit_rank_buffer=int(settings["exit_rank_buffer"]),
            get_state=get_market_state,
            signal_weights=signal_weights,
        )
        # A suspended holding may have no opening bar.  Preserve it using the
        # account valuation, exactly as the vn.py strategy uses its cached bar.
        for position in position_objects:
            state = market_states[position.order_book_id]
            if state.price > 0 or position.market_value <= 0:
                continue
            market_states[position.order_book_id] = MarketState(
                price=float(position.market_value / position.quantity),
                can_buy=state.can_buy,
                can_sell=state.can_sell,
                buy_block_reason=state.buy_block_reason,
                sell_block_reason=state.sell_block_reason,
            )

        decision = build_fixed_capital_topk_target(
            ranked,
            positions,
            market_states,
            cash=float(context.portfolio.cash),
            initial_capital=float(settings["capital"]),
            top_k=int(settings["top_k"]),
            cash_ratio=float(settings["cash_ratio"]),
            exit_rank_buffer=int(settings["exit_rank_buffer"]),
            min_volume=int(settings["min_volume"]),
            signal_weights=signal_weights,
        )

        valuation_prices = pd.Series(
            {
                symbol: market_states[symbol].price
                for symbol in set(decision.target_weights) | set(positions)
                if symbol in market_states and market_states[symbol].price > 0
            },
            dtype=float,
        )
        order_target_portfolio_smart(
            pd.Series(decision.target_weights, dtype=float),
            valuation_prices=valuation_prices,
        )

        trade_date = context.now.date()
        for symbol, quantity in sorted(decision.target_quantities.items()):
            target_rows.append(
                {
                    "datetime": trade_date,
                    "signal_datetime": signal_date,
                    "order_book_id": symbol,
                    "target_quantity": quantity,
                    "target_value": decision.target_values[symbol],
                    "weight": decision.target_weights[symbol],
                    "rank": decision.ranks[symbol],
                    "frozen": symbol in decision.frozen_symbols,
                    "buffered": symbol in decision.buffered_symbols,
                }
            )

        skipped_limit_up = sum(
            market_states[symbol].buy_block_reason == "limit_up"
            for symbol in decision.skipped_unbuyable
            if symbol in market_states
        )
        diagnostic_rows.append(
            {
                "datetime": trade_date,
                "signal_datetime": signal_date,
                "strategy": "FixedCapitalTopKStrategy",
                "frozen_count": len(decision.frozen_symbols),
                "buffered_count": len(decision.buffered_symbols),
                "core_count": len(decision.core_symbols),
                "skipped_limit_up_count": skipped_limit_up,
                "skipped_inactive_count": len(decision.skipped_unbuyable) - skipped_limit_up,
                "skipped_invalid_weight_count": len(decision.skipped_invalid_weight),
                "buy_count": len(decision.buy_symbols),
                "sell_count": len(decision.sell_symbols),
                "desired_count": len(decision.target_quantities),
                "fixed_budget": decision.fixed_budget,
                "reserved_value": decision.reserved_value,
                "target_value": sum(decision.target_values.values()),
                "target_weight_sum": sum(decision.target_weights.values()),
            }
        )

    rq_config = {
        "base": {
            "data_bundle_path": settings["bundle_path"],
            "start_date": settings["start_date"],
            "end_date": settings["end_date"],
            "frequency": settings["frequency"],
            "run_type": "b",
            "accounts": {"stock": settings["capital"]},
        },
        "extra": {"log_level": settings["log_level"]},
        "mod": {
            "sys_analyser": {
                "enabled": True,
                "record": True,
                "benchmark": settings["benchmark"],
                "strategy_name": "FixedCapitalTopKStrategy_RQAlpha",
                "plot": False,
                "output_file": str(output_dir / "rqalpha_result.pkl"),
                "report_save_path": str(output_dir / "rqalpha_report"),
            },
            "sys_simulation": {
                "enabled": True,
                "matching_type": "current_bar",
                "slippage_model": "PriceRatioSlippage",
                "slippage": 0.0,
                "volume_limit": False,
                "volume_percent": 0.25,
                "price_limit": True,
                "inactive_limit": settings["inactive_limit"],
            },
            "sys_transaction_cost": {
                "enabled": True,
                "stock_commission_multiplier": settings["stock_commission_multiplier"],
                "tax_multiplier": settings["tax_multiplier"],
                "stock_min_commission": settings["min_commission"],
                "pit_tax": False,
            },
        },
    }
    result = run_func(
        config=rq_config,
        init=init,
        handle_bar=handle_bar,
        open_auction=open_auction,
    )
    analysis = result["sys_analyser"]

    dataframes = {
        "portfolio": analysis.get("portfolio"),
        "benchmark_portfolio": analysis.get("benchmark_portfolio"),
        "trades": analysis.get("trades"),
        "stock_account": analysis.get("stock_account"),
        "stock_positions": analysis.get("stock_positions"),
        "positions_weight": analysis.get("positions_weight"),
    }
    for name, frame in dataframes.items():
        if isinstance(frame, pd.DataFrame):
            frame.rename_axis("date").reset_index().to_csv(
                output_dir / f"{name}.csv",
                index=False,
            )

    pd.DataFrame(
        target_rows,
        columns=[
            "datetime",
            "signal_datetime",
            "order_book_id",
            "target_quantity",
            "target_value",
            "weight",
            "rank",
            "frozen",
            "buffered",
        ],
    ).to_csv(output_dir / "target_weights.csv", index=False)
    pd.DataFrame(
        diagnostic_rows,
        columns=[
            "datetime",
            "signal_datetime",
            "strategy",
            "frozen_count",
            "buffered_count",
            "core_count",
            "skipped_limit_up_count",
            "skipped_inactive_count",
            "skipped_invalid_weight_count",
            "buy_count",
            "sell_count",
            "desired_count",
            "fixed_budget",
            "reserved_value",
            "target_value",
            "target_weight_sum",
        ],
    ).to_csv(output_dir / "rebalance_diagnostics.csv", index=False)

    portfolio = analysis["portfolio"]
    benchmark = analysis["benchmark_portfolio"]
    _annual_metrics(portfolio, benchmark).to_csv(output_dir / "annual_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(_to_builtin(analysis["summary"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta = {
        "strategy": "FixedCapitalTopKStrategy",
        "target_rows": int(len(target_rows)),
        "target_dates": int(len(diagnostic_rows)),
        "target_symbols": int(len({row["order_book_id"] for row in target_rows})),
        "start_date": settings["start_date"],
        "end_date": settings["end_date"],
        "benchmark": settings["benchmark"],
        "top_k": settings["top_k"],
        "cash_ratio": settings["cash_ratio"],
        "rebalance_interval": settings["rebalance_interval"],
        "exit_rank_buffer": settings["exit_rank_buffer"],
        "min_volume": settings["min_volume"],
        "execution_timing": "T+1_open_auction",
        "budget_basis": "initial_capital",
        "note": (
            "冻结及缓冲持仓占位；涨停/停牌新候选顺延；"
            "RQAlpha 开盘市价撮合不映射 vn.py price_add"
        ),
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FixedCapitalTopKStrategy logic with RQAlpha"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--rankings", required=True)
    args = parser.parse_args()
    run(Path(args.config), Path(args.rankings))


if __name__ == "__main__":
    main()
