from datetime import datetime

import polars as pl
import pytest

from vnpy.alpha.strategy.backtesting import BacktestingEngine
from vnpy.alpha.strategy.strategies.fixed_capital_topk_strategy import (
    FixedCapitalTopKStrategy,
)
from vnpy.trader.constant import Direction, Exchange, Interval
from vnpy.trader.object import BarData


SYMBOLS = ["000001.SSE", "000002.SSE", "000003.SSE"]


class DummyLab:
    def load_contract_setttings(self) -> dict:
        return {
            symbol: {
                "long_rate": 0.0005,
                "short_rate": 0.0015,
                "size": 1,
                "pricetick": 0.01,
            }
            for symbol in SYMBOLS
        }


def make_bar(
    symbol: str,
    dt: datetime,
    open_price: float,
    limit_up: float,
    limit_down: float,
) -> BarData:
    return BarData(
        symbol=symbol.split(".")[0],
        exchange=Exchange.SSE,
        datetime=dt,
        interval=Interval.DAILY,
        open_price=open_price,
        high_price=open_price,
        low_price=open_price,
        close_price=open_price,
        limit_up=limit_up,
        limit_down=limit_down,
        gateway_name="TEST",
    )


def make_engine() -> BacktestingEngine:
    d1 = datetime(2024, 1, 2)
    d2 = datetime(2024, 1, 3)
    d3 = datetime(2024, 1, 4)
    engine = BacktestingEngine(DummyLab())
    engine.set_parameters(
        vt_symbols=SYMBOLS,
        interval=Interval.DAILY,
        start=d1,
        end=d3,
        capital=100_000,
    )
    signal = pl.DataFrame(
        {
            "datetime": [d1, d1, d1, d2, d2, d2, d3, d3, d3],
            "vt_symbol": SYMBOLS * 3,
            "signal": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    engine.add_strategy(
        FixedCapitalTopKStrategy,
        {
            "top_k": 1,
            "cash_ratio": 0.9,
            "price_add": 0.05,
            "open_rate": 0.0005,
            "close_rate": 0.0015,
            "min_commission": 5,
            "min_volume": 100,
        },
        signal,
    )

    prices = {
        d1: [10.0, 20.0, 30.0],
        d2: [11.0, 20.0, 30.0],
        d3: [10.0, 18.0, 30.0],
    }
    for dt, day_prices in prices.items():
        for symbol, price in zip(SYMBOLS, day_prices, strict=True):
            limit_up = round(price * 1.1, 2)
            limit_down = round(price * 0.9, 2)
            if dt == d2 and symbol == SYMBOLS[0]:
                limit_up = price
            if dt == d3 and symbol == SYMBOLS[1]:
                limit_down = price
            bar = make_bar(symbol, dt, price, limit_up, limit_down)
            engine.history_data[(dt, symbol)] = bar
        engine.dts.add(dt)
    return engine


def test_t1_limit_adjustment_and_frozen_position() -> None:
    engine = make_engine()
    engine.run_backtesting()

    trades = engine.get_all_trades()
    assert len(trades) == 1
    assert trades[0].datetime == datetime(2024, 1, 3)
    assert trades[0].vt_symbol == SYMBOLS[1]
    assert engine.strategy.get_pos(SYMBOLS[1]) > 0
    assert engine.strategy.get_pos(SYMBOLS[2]) == 0
    assert engine.cash >= 0

    diagnostics = engine.strategy.rebalance_diagnostics
    assert diagnostics[-1]["frozen_count"] == 1
    assert diagnostics[-1]["core_count"] == 0


def test_repeated_run_and_result_are_idempotent() -> None:
    engine = make_engine()
    engine.run_backtesting()
    first_trades = [
        (trade.datetime, trade.vt_symbol, trade.price, trade.volume)
        for trade in engine.get_all_trades()
    ]
    first_result = engine.calculate_result()
    assert first_result is not None
    second_result = engine.calculate_result()
    assert second_result is not None
    assert first_result.equals(second_result)
    assert first_result["commission"].sum() == sum(engine.trade_commissions.values())

    engine.run_backtesting()
    second_trades = [
        (trade.datetime, trade.vt_symbol, trade.price, trade.volume)
        for trade in engine.get_all_trades()
    ]
    assert first_trades == second_trades


def test_statistics_use_fixed_initial_capital_returns() -> None:
    engine = make_engine()
    engine.run_backtesting()
    daily = engine.calculate_result()
    assert daily is not None
    statistics = engine.calculate_statistics()
    expected_total_return = daily["net_pnl"].sum() / engine.capital * 100
    assert statistics["total_return"] == pytest.approx(expected_total_return)
    assert engine.daily_df["return"].to_list() == (
        engine.daily_df["net_pnl"] / engine.capital
    ).to_list()
    repeated = engine.calculate_statistics()
    assert repeated["total_return"] == statistics["total_return"]


def test_buy_orders_are_scaled_proportionally_to_real_cash() -> None:
    engine = make_engine()
    engine.strategy_setting["top_k"] = 2
    engine.strategy_setting["cash_ratio"] = 1.0
    d2 = datetime(2024, 1, 3)
    symbol = SYMBOLS[2]
    engine.history_data[(d2, symbol)] = make_bar(
        symbol,
        d2,
        open_price=25.0,
        limit_up=27.5,
        limit_down=22.5,
    )

    engine.run_backtesting()
    day_two_buys = [
        trade
        for trade in engine.get_all_trades()
        if trade.datetime == d2
    ]
    assert {trade.vt_symbol for trade in day_two_buys} == {SYMBOLS[1], SYMBOLS[2]}
    assert all(trade.volume > 0 for trade in day_two_buys)
    assert engine.cash >= 0
    assert engine.cash < 10_000


def test_limit_up_holding_can_still_be_sold() -> None:
    engine = make_engine()
    d3 = datetime(2024, 1, 4)
    symbol = SYMBOLS[1]
    engine.history_data[(d3, symbol)] = make_bar(
        symbol,
        d3,
        open_price=22.0,
        limit_up=22.0,
        limit_down=18.0,
    )

    engine.run_backtesting()
    day_three = [trade for trade in engine.get_all_trades() if trade.datetime == d3]
    assert any(
        trade.vt_symbol == SYMBOLS[1] and trade.direction == Direction.SHORT
        for trade in day_three
    )
    assert any(
        trade.vt_symbol == SYMBOLS[2] and trade.direction == Direction.LONG
        for trade in day_three
    )


def test_alpha_statistics_accept_precomputed_performance() -> None:
    engine = make_engine()
    performance = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 2).date(), datetime(2024, 1, 3).date()],
            "net_daily_return": [0.01, -0.005],
            "benchmark_daily_return": [0.002, -0.001],
        }
    )
    statistics = engine.calculate_alpha_statistics(performance_df=performance)
    assert statistics["alpha_total_return"] == pytest.approx(0.4)
    assert statistics["alpha_daily_return"] == pytest.approx(0.2)
