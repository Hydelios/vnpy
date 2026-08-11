"""Fixed-initial-capital, long-only Top-K strategy."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from vnpy.alpha import AlphaStrategy
from vnpy.trader.object import BarData, TradeData


class FixedCapitalTopKStrategy(AlphaStrategy):
    """Execute the previous close's ranked signal at the next open."""

    top_k: int = 50
    cash_ratio: float = 0.95
    rebalance_interval: int = 1
    exit_rank_buffer: int = 0
    min_volume: int = 100
    price_add: float = 0.05
    open_rate: float | None = None
    close_rate: float | None = None
    min_commission: float | None = None

    def __init__(
        self,
        strategy_engine: Any,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict,
    ) -> None:
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self._validate_parameters()
        initial_capital = self.strategy_engine.capital
        if not isinstance(initial_capital, (int, float)) or not math.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("engine capital must be a positive finite number")
        self._initial_capital = float(initial_capital)
        self._days_until_signal = 0
        self._pending_ranked: list[dict[str, Any]] | None = None
        self._pending_signal_date: object | None = None
        self._rebalance_trade_count_before: int | None = None
        self.rebalance_diagnostics: list[dict[str, Any]] = []
        self.strategy_engine.configure_stock_costs(
            open_rate=self.open_rate,
            close_rate=self.close_rate,
            min_commission=self.min_commission,
            min_volume=self.min_volume,
        )

    def on_init(self) -> None:
        self._validate_parameters()
        self._days_until_signal = 0
        self._pending_ranked = None
        self._pending_signal_date = None
        self._rebalance_trade_count_before = None
        self.rebalance_diagnostics.clear()
        self.write_log("Fixed-capital Top-K strategy initialized")

    def on_trade(self, trade: TradeData) -> None:
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """Cache today's close signal; it cannot be used before the next open."""
        if self._days_until_signal > 0:
            self._days_until_signal -= 1
            return

        signal_df = self.get_signal()
        if signal_df.is_empty():
            self.write_log("Warning: no signal data; rebalance skipped")
            return
        self._pending_ranked = self._rank_signals(signal_df)
        self._pending_signal_date = self.strategy_engine.datetime
        self._days_until_signal = self.rebalance_interval - 1

    def on_open(self, bars: dict[str, BarData]) -> None:
        """Build a tradable Top-K from the pending signal and trade this open."""
        if self._pending_ranked is None:
            return

        self.cancel_all()
        self._rebalance_trade_count_before = self.strategy_engine.trade_count
        ranked = self._pending_ranked
        self._pending_ranked = None
        positions = {
            symbol: volume
            for symbol, volume in self.pos_data.items()
            if volume > 0
        }
        rank_index = {row["vt_symbol"]: index for index, row in enumerate(ranked)}

        frozen = {
            symbol: volume
            for symbol, volume in positions.items()
            if not self.strategy_engine.is_sellable_at_open(symbol, bars.get(symbol))
        }
        buffered = self._buffered_positions(rank_index, positions, frozen, bars)
        reserved = {**frozen, **buffered}
        available_slots = max(self.top_k - len(reserved), 0)
        core_rows = self._select_core_rows(ranked, reserved, positions, bars, available_slots)
        targets = self._build_targets(core_rows, reserved, positions, bars)

        self.target_data.clear()
        for symbol in sorted(targets):
            self.set_target(symbol, targets[symbol])
        self.execute_trading(bars, price_add=self.price_add, use_open=True)

        desired = {symbol for symbol, volume in targets.items() if volume > 0}
        self.rebalance_diagnostics.append(
            {
                "date": self.strategy_engine.datetime.date(),
                "signal_date": (
                    self._pending_signal_date.date()
                    if self._pending_signal_date is not None
                    else None
                ),
                "desired_count": len(desired),
                "frozen_count": len(frozen),
                "buffered_count": len(buffered),
                "core_count": len(core_rows),
                "cash_before": self.get_cash_available(),
                "target_value": self._target_value(targets, bars),
            }
        )
        self._pending_signal_date = None

    def on_after_open(self, bars: dict[str, BarData]) -> None:
        """Complete diagnostics with actual fills and cash."""
        if self._rebalance_trade_count_before is None or not self.rebalance_diagnostics:
            return
        diagnostic = self.rebalance_diagnostics[-1]
        diagnostic["trade_count"] = (
            self.strategy_engine.trade_count - self._rebalance_trade_count_before
        )
        diagnostic["actual_count"] = sum(
            1 for volume in self.pos_data.values() if volume > 0
        )
        diagnostic["cash_after"] = self.get_cash_available()
        self._rebalance_trade_count_before = None

    def get_rebalance_diagnostics(self) -> pl.DataFrame:
        """Return opening rebalance diagnostics as a stable dataframe."""
        return pl.DataFrame(self.rebalance_diagnostics)

    def _validate_parameters(self) -> None:
        positive_ints = {
            "top_k": self.top_k,
            "rebalance_interval": self.rebalance_interval,
            "min_volume": self.min_volume,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in positive_ints.values()):
            raise ValueError("top_k, rebalance_interval and min_volume must be positive integers")
        if not isinstance(self.exit_rank_buffer, int) or isinstance(self.exit_rank_buffer, bool) or self.exit_rank_buffer < 0:
            raise ValueError("exit_rank_buffer must be a non-negative integer")
        for name, value, allow_none in (
            ("cash_ratio", self.cash_ratio, False),
            ("price_add", self.price_add, False),
            ("open_rate", self.open_rate, True),
            ("close_rate", self.close_rate, True),
            ("min_commission", self.min_commission, True),
        ):
            if value is None and allow_none:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")
        if self.cash_ratio <= 0:
            raise ValueError("cash_ratio must be positive")
        if self.cash_ratio > 1:
            raise ValueError("cash_ratio must not exceed 1")
        if self.price_add >= 1:
            raise ValueError("price_add must be less than 1")

    def _rank_signals(self, signal_df: pl.DataFrame) -> list[dict[str, Any]]:
        required = {"vt_symbol", "signal"}
        if not required.issubset(signal_df.columns):
            raise ValueError("signal data must contain vt_symbol and signal columns")
        columns = ["vt_symbol", "signal"] + (["weight"] if "weight" in signal_df.columns else [])
        rows: list[dict[str, Any]] = []
        for row in signal_df.select(columns).iter_rows(named=True):
            symbol = row["vt_symbol"]
            signal = row["signal"]
            if (
                isinstance(symbol, str)
                and symbol
                and isinstance(signal, (int, float))
                and math.isfinite(signal)
            ):
                rows.append(row)
        symbols = [row["vt_symbol"] for row in rows]
        if len(symbols) != len(set(symbols)):
            raise ValueError("signal data contains duplicate vt_symbol")
        return sorted(rows, key=lambda row: (-float(row["signal"]), row["vt_symbol"]))

    def _buffered_positions(
        self,
        rank_index: dict[str, int],
        positions: dict[str, float],
        frozen: dict[str, float],
        bars: dict[str, BarData],
    ) -> dict[str, float]:
        if self.exit_rank_buffer == 0:
            return {}
        start = self.top_k
        stop = self.top_k + self.exit_rank_buffer
        return {
            symbol: volume
            for symbol, volume in sorted(positions.items())
            if symbol not in frozen
            and start <= rank_index.get(symbol, stop) < stop
            and self._valid_open_bar(bars.get(symbol))
        }

    def _select_core_rows(
        self,
        ranked: list[dict[str, Any]],
        reserved: dict[str, float],
        positions: dict[str, float],
        bars: dict[str, BarData],
        slots: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        has_weight = bool(ranked and "weight" in ranked[0])
        for row in ranked:
            if len(selected) >= slots:
                break
            symbol = row["vt_symbol"]
            if symbol in reserved:
                continue
            bar = bars.get(symbol)
            if not self._valid_open_bar(bar):
                continue
            if symbol not in positions and not self.strategy_engine.is_buyable_at_open(symbol, bar):
                continue
            if has_weight:
                weight = row.get("weight")
                if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                    continue
            selected.append(row)
        return selected

    def _build_targets(
        self,
        core_rows: list[dict[str, Any]],
        reserved: dict[str, float],
        positions: dict[str, float],
        bars: dict[str, BarData],
    ) -> dict[str, float]:
        budget = self._initial_capital * self.cash_ratio
        reserved_value = sum(
            volume * self._valuation_price(symbol, bars.get(symbol)) * self._size(symbol)
            for symbol, volume in reserved.items()
            if self._valuation_price(symbol, bars.get(symbol)) > 0
        )
        available_budget = max(budget - reserved_value, 0.0)
        targets = {symbol: 0.0 for symbol in positions}
        targets.update(reserved)
        weights = self._normalized_weights(core_rows)
        for row in core_rows:
            symbol = row["vt_symbol"]
            price = bars[symbol].open_price
            raw_volume = available_budget * weights[symbol] / (price * self._size(symbol))
            targets[symbol] = self._floor_volume(raw_volume)
        return targets

    @staticmethod
    def _normalized_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {}
        if "weight" not in rows[0]:
            return {row["vt_symbol"]: 1 / len(rows) for row in rows}
        total = sum(float(row["weight"]) for row in rows)
        return {row["vt_symbol"]: float(row["weight"]) / total for row in rows}

    def _target_value(self, targets: dict[str, float], bars: dict[str, BarData]) -> float:
        return sum(
            volume * self._valuation_price(symbol, bars.get(symbol)) * self._size(symbol)
            for symbol, volume in targets.items()
            if volume > 0 and self._valuation_price(symbol, bars.get(symbol)) > 0
        )

    def _valuation_price(self, symbol: str, bar: BarData | None) -> float:
        if self._valid_open_bar(bar):
            return float(bar.open_price)
        cached = self.strategy_engine.bars.get(symbol)
        if cached is not None:
            for price in (cached.close_price, cached.open_price):
                if isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
                    return float(price)
        return 0.0

    def _size(self, symbol: str) -> float:
        return float(self.strategy_engine.sizes.get(symbol, 1.0))

    @staticmethod
    def _valid_open_bar(bar: BarData | None) -> bool:
        return (
            bar is not None
            and isinstance(bar.open_price, (int, float))
            and math.isfinite(bar.open_price)
            and bar.open_price > 0
        )

    def _floor_volume(self, volume: float) -> float:
        return max(math.floor(volume / self.min_volume) * self.min_volume, 0)
