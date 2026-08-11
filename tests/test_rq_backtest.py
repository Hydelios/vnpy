import pandas as pd
import pytest

from vnpy.alpha.rq_backtest import (
    RQBacktestConfig,
    RQBacktestResult,
    create_rqalpha_performance_chart,
    normalize_signal,
    to_rq_order_book_id,
)
from vnpy.alpha.rq_backtest.selection import (
    MarketState,
    build_fixed_capital_topk_target,
    collect_fixed_capital_market_states,
)
from vnpy.alpha.rq_backtest.signal import prepare_ranked_signals


def test_symbol_conversion_and_signal_normalization() -> None:
    signal = pd.DataFrame(
        {
            "datetime": ["2026-01-02"] * 4,
            "vt_symbol": ["600000.SSE", "000001.SZSE", "000001.SZ", "bad"],
            "signal": [1.0, 2.0, 3.0, float("nan")],
        }
    )

    normalized = normalize_signal(signal)

    assert to_rq_order_book_id("600000.SSE") == "600000.XSHG"
    assert to_rq_order_book_id("000001.SZSE") == "000001.XSHE"
    assert normalized[["order_book_id", "signal"]].to_dict("records") == [
        {"order_book_id": "000001.XSHE", "signal": 3.0},
        {"order_book_id": "600000.XSHG", "signal": 1.0},
    ]


def test_prepare_ranked_signals_is_deterministic_and_preserves_weight() -> None:
    signal = pd.DataFrame(
        {
            "datetime": ["2026-01-02"] * 3,
            "vt_symbol": ["B.SSE", "A.SSE", "C.SSE"],
            "signal": [3.0, 3.0, 1.0],
            "weight": [0.2, 0.7, 0.1],
        }
    )

    ranked = prepare_ranked_signals(signal)

    assert ranked["order_book_id"].tolist() == ["A.XSHG", "B.XSHG", "C.XSHG"]
    assert ranked["rank"].tolist() == [1, 2, 3]
    assert ranked["weight"].tolist() == [0.7, 0.2, 0.1]


def test_fixed_capital_budget_does_not_grow_with_portfolio_value() -> None:
    states = {
        "A": MarketState(10.0, can_buy=True, can_sell=True),
        "B": MarketState(10.0, can_buy=True, can_sell=True),
    }
    decision = build_fixed_capital_topk_target(
        ["A", "B"],
        {},
        states,
        cash=2_000.0,
        initial_capital=1_000.0,
        top_k=2,
        cash_ratio=0.9,
        exit_rank_buffer=0,
        min_volume=10,
    )

    assert decision.fixed_budget == pytest.approx(900.0)
    assert decision.target_quantities == {"A": 40.0, "B": 40.0}
    assert decision.target_values == {"A": 400.0, "B": 400.0}
    assert decision.target_weights == pytest.approx({"A": 0.2, "B": 0.2})


def test_frozen_holding_occupies_slot_and_limit_up_candidate_is_replaced() -> None:
    states = {
        "OLD": MarketState(10.0, can_buy=True, can_sell=False, sell_block_reason="limit_down"),
        "A": MarketState(10.0, can_buy=False, can_sell=True, buy_block_reason="limit_up"),
        "B": MarketState(10.0, can_buy=True, can_sell=True),
        "C": MarketState(10.0, can_buy=True, can_sell=True),
    }
    decision = build_fixed_capital_topk_target(
        ["A", "B", "C"],
        {"OLD": 10.0},
        states,
        cash=900.0,
        initial_capital=1_000.0,
        top_k=3,
        cash_ratio=0.9,
        exit_rank_buffer=0,
        min_volume=10,
    )

    assert decision.frozen_symbols == {"OLD"}
    assert decision.skipped_unbuyable == ("A",)
    assert decision.core_symbols == ("B", "C")
    assert decision.target_quantities == {"OLD": 10.0, "B": 40.0, "C": 40.0}


def test_exit_rank_buffer_preserves_quantity_and_consumes_budget() -> None:
    states = {
        symbol: MarketState(10.0, can_buy=True, can_sell=True)
        for symbol in ["A", "B", "BUF", "C"]
    }
    decision = build_fixed_capital_topk_target(
        ["A", "B", "BUF", "C"],
        {"BUF": 30.0},
        states,
        cash=700.0,
        initial_capital=1_000.0,
        top_k=2,
        cash_ratio=0.9,
        exit_rank_buffer=1,
        min_volume=10,
    )

    assert decision.buffered_symbols == {"BUF"}
    assert decision.reserved_value == pytest.approx(300.0)
    assert decision.core_symbols == ("A",)
    assert decision.target_quantities == {"BUF": 30.0, "A": 60.0}


def test_signal_weights_are_normalized_over_tradable_core() -> None:
    states = {
        symbol: MarketState(10.0, can_buy=True, can_sell=True)
        for symbol in ["A", "B", "C"]
    }
    decision = build_fixed_capital_topk_target(
        ["A", "B", "C"],
        {},
        states,
        cash=1_000.0,
        initial_capital=1_000.0,
        top_k=2,
        cash_ratio=0.9,
        exit_rank_buffer=0,
        min_volume=10,
        signal_weights={"A": None, "B": 1.0, "C": 2.0},
    )

    assert decision.skipped_invalid_weight == ("A",)
    assert decision.core_symbols == ("B", "C")
    assert decision.target_quantities == {"B": 30.0, "C": 60.0}


def test_market_state_scan_stops_after_fixed_topk_is_filled() -> None:
    states = {
        "LOCKED": MarketState(10.0, can_buy=True, can_sell=False),
        "A": MarketState(10.0, can_buy=False, can_sell=True),
        "B": MarketState(10.0, can_buy=True, can_sell=True),
        "C": MarketState(10.0, can_buy=True, can_sell=True),
        "D": MarketState(10.0, can_buy=True, can_sell=True),
    }
    calls: list[str] = []

    loaded = collect_fixed_capital_market_states(
        ["A", "B", "C", "D"],
        {"LOCKED": 10.0},
        top_k=3,
        exit_rank_buffer=0,
        get_state=lambda symbol: calls.append(symbol) or states[symbol],
    )

    assert calls == ["LOCKED", "A", "B", "C"]
    assert set(loaded) == {"LOCKED", "A", "B", "C"}


def test_fee_mapping_matches_vnpy_rates() -> None:
    config = RQBacktestConfig(open_rate=0.0005, close_rate=0.0015)

    assert config.stock_commission_multiplier == pytest.approx(0.625)
    assert config.tax_multiplier == pytest.approx(1.0)


def test_rq_config_only_accepts_fixed_capital_topk_settings() -> None:
    config = RQBacktestConfig()

    assert config.top_k == 300
    assert config.rebalance_interval == 5
    assert config.exit_rank_buffer == 0
    assert config.min_volume == 100
    assert not hasattr(config, "rebalance_mode")
    assert not hasattr(config, "execution_mode")

    shared = RQBacktestConfig.from_settings(
        portfolio={
            "top_k": 200,
            "cash_ratio": 0.95,
            "rebalance_interval": 10,
            "exit_rank_buffer": 20,
            "min_volume": 100,
        },
        cost={
            "buy_commission": 0.0002,
            "sell_commission": 0.0002,
            "sell_tax": 0.001,
            "min_commission": 3,
        },
        execution={
            "inactive_limit": True,
        },
    )
    assert shared.top_k == 200
    assert shared.rebalance_interval == 10
    assert shared.exit_rank_buffer == 20
    assert shared.open_rate == pytest.approx(0.0002)
    assert shared.close_rate == pytest.approx(0.0012)
    assert shared.inactive_limit is True
    assert not hasattr(shared, "slippage")

    with pytest.raises(ValueError, match="不再接受旧版或改变策略语义的参数"):
        RQBacktestConfig.from_settings(
            portfolio={"top_k": 200, "n_drop": 20},
            cost={
                "buy_commission": 0.0002,
                "sell_commission": 0.0002,
                "sell_tax": 0.001,
                "min_commission": 3,
            },
            execution={},
        )

    with pytest.raises(ValueError, match="不再接受旧版或改变策略语义的参数"):
        RQBacktestConfig.from_settings(
            portfolio={"top_k": 200},
            cost={
                "buy_commission": 0.0002,
                "sell_commission": 0.0002,
                "sell_tax": 0.001,
                "min_commission": 3,
            },
            execution={"slippage": 0.001},
        )


def test_create_rqalpha_performance_chart(tmp_path) -> None:
    portfolio = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "unit_net_value": [1.0, 1.1, 1.0],
            "benchmark_unit_net_value": [1.0, 1.02, 1.01],
        }
    )
    result = RQBacktestResult(
        output_dir=tmp_path,
        target_weights=pd.DataFrame(),
        rebalance_diagnostics=pd.DataFrame(),
        summary={},
        annual_metrics=pd.DataFrame(),
        portfolio=portfolio,
        trades=pd.DataFrame(),
        stdout="",
    )
    output_path = tmp_path / "performance.png"

    figure = create_rqalpha_performance_chart(result, output_path=output_path)

    assert len(figure.axes) == 3
    assert output_path.exists()
    figure.clear()
