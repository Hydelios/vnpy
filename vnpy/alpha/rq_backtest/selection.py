from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class MarketState:
    """T+1 open-auction trading state for one stock."""

    price: float
    can_buy: bool
    can_sell: bool
    buy_block_reason: str | None = None
    sell_block_reason: str | None = None


@dataclass(frozen=True)
class FixedCapitalTopKDecision:
    """One fixed-initial-capital TopK rebalance decision."""

    target_quantities: dict[str, float]
    target_values: dict[str, float]
    target_weights: dict[str, float]
    ranks: dict[str, int | None]
    frozen_symbols: frozenset[str]
    buffered_symbols: frozenset[str]
    skipped_unbuyable: tuple[str, ...]
    skipped_invalid_weight: tuple[str, ...]
    core_symbols: tuple[str, ...]
    buy_symbols: tuple[str, ...]
    sell_symbols: tuple[str, ...]
    portfolio_value: float
    fixed_budget: float
    reserved_value: float


def collect_fixed_capital_market_states(
    ranked_symbols: Sequence[str],
    positions: Mapping[str, float],
    *,
    top_k: int,
    exit_rank_buffer: int,
    get_state: Callable[[str], MarketState],
    signal_weights: Mapping[str, float | None] | None = None,
) -> dict[str, MarketState]:
    """Load held states and enough candidates to fill the tradable TopK."""

    held = {symbol for symbol, quantity in positions.items() if quantity > 0}
    states = {symbol: get_state(symbol) for symbol in sorted(held)}
    frozen = {symbol for symbol in held if not states[symbol].can_sell}
    rank_index = {symbol: index for index, symbol in enumerate(ranked_symbols)}
    buffer_stop = top_k + exit_rank_buffer
    buffered = {
        symbol
        for symbol in held.difference(frozen)
        if top_k <= rank_index.get(symbol, buffer_stop) < buffer_stop
        and states[symbol].price > 0
    }
    reserved = frozen | buffered
    remaining_slots = max(top_k - len(reserved), 0)

    selected = 0
    for symbol in ranked_symbols:
        if selected >= remaining_slots:
            break
        if symbol in reserved:
            continue
        state = states.get(symbol)
        if state is None:
            state = get_state(symbol)
            states[symbol] = state
        if state.price <= 0:
            continue
        if symbol not in held and not state.can_buy:
            continue
        if signal_weights is not None:
            weight = signal_weights.get(symbol)
            if weight is None or not math.isfinite(weight) or weight <= 0:
                continue
        selected += 1
    return states


def build_fixed_capital_topk_target(
    ranked_symbols: Sequence[str],
    positions: Mapping[str, float],
    market_states: Mapping[str, MarketState],
    *,
    cash: float,
    initial_capital: float,
    top_k: int,
    cash_ratio: float,
    exit_rank_buffer: int,
    min_volume: int,
    signal_weights: Mapping[str, float | None] | None = None,
) -> FixedCapitalTopKDecision:
    """Mirror ``FixedCapitalTopKStrategy`` at the T+1 open.

    The investable budget is always ``initial_capital * cash_ratio``.  Frozen
    and exit-buffer holdings preserve their current quantities and consume
    both slots and budget.  Remaining core names share the residual budget,
    and target quantities are floored to ``min_volume`` lots.
    """

    if top_k <= 0 or min_volume <= 0:
        raise ValueError("top_k 和 min_volume 必须为正整数")
    if exit_rank_buffer < 0:
        raise ValueError("exit_rank_buffer 必须为非负整数")
    if initial_capital <= 0:
        raise ValueError("initial_capital 必须大于0")
    if not 0 < cash_ratio <= 1:
        raise ValueError("cash_ratio 必须在 (0, 1] 内")

    held = {symbol for symbol, quantity in positions.items() if quantity > 0}
    rank_index = {symbol: index for index, symbol in enumerate(ranked_symbols)}
    rank_map = {symbol: index + 1 for symbol, index in rank_index.items()}
    frozen = {
        symbol
        for symbol in held
        if symbol not in market_states or not market_states[symbol].can_sell
    }
    buffer_stop = top_k + exit_rank_buffer
    buffered = {
        symbol
        for symbol in held.difference(frozen)
        if top_k <= rank_index.get(symbol, buffer_stop) < buffer_stop
        and market_states.get(symbol) is not None
        and market_states[symbol].price > 0
    }
    reserved = frozen | buffered

    slots = max(top_k - len(reserved), 0)
    core: list[str] = []
    skipped_unbuyable: list[str] = []
    skipped_invalid_weight: list[str] = []
    for symbol in ranked_symbols:
        if len(core) >= slots:
            break
        if symbol in reserved:
            continue
        state = market_states.get(symbol)
        if state is None or state.price <= 0:
            skipped_unbuyable.append(symbol)
            continue
        if symbol not in held and not state.can_buy:
            skipped_unbuyable.append(symbol)
            continue
        if signal_weights is not None:
            weight = signal_weights.get(symbol)
            if weight is None or not math.isfinite(weight) or weight <= 0:
                skipped_invalid_weight.append(symbol)
                continue
        core.append(symbol)

    position_values = {
        symbol: max(
            float(positions[symbol])
            * market_states.get(symbol, MarketState(0, False, False)).price,
            0.0,
        )
        for symbol in held
    }
    portfolio_value = max(float(cash) + sum(position_values.values()), 0.0)
    fixed_budget = float(initial_capital) * float(cash_ratio)
    reserved_value = sum(position_values[symbol] for symbol in reserved)
    available_budget = max(fixed_budget - reserved_value, 0.0)

    target_quantities = {symbol: float(positions[symbol]) for symbol in sorted(reserved)}
    if core:
        if signal_weights is None:
            normalized = {symbol: 1.0 / len(core) for symbol in core}
        else:
            total_weight = sum(float(signal_weights[symbol]) for symbol in core)
            normalized = {
                symbol: float(signal_weights[symbol]) / total_weight
                for symbol in core
            }
        for symbol in core:
            price = market_states[symbol].price
            raw_quantity = available_budget * normalized[symbol] / price
            target_quantities[symbol] = float(
                max(math.floor(raw_quantity / min_volume) * min_volume, 0)
            )

    target_quantities = {
        symbol: quantity
        for symbol, quantity in target_quantities.items()
        if quantity > 0
    }
    target_values = {
        symbol: quantity * market_states[symbol].price
        for symbol, quantity in target_quantities.items()
    }
    target_weights = (
        {symbol: value / portfolio_value for symbol, value in target_values.items()}
        if portfolio_value > 0
        else {}
    )
    all_symbols = set(held) | set(target_quantities)
    buy_symbols = tuple(
        sorted(
            symbol
            for symbol in all_symbols
            if target_quantities.get(symbol, 0.0) > float(positions.get(symbol, 0.0))
        )
    )
    sell_symbols = tuple(
        sorted(
            symbol
            for symbol in all_symbols
            if target_quantities.get(symbol, 0.0) < float(positions.get(symbol, 0.0))
        )
    )
    return FixedCapitalTopKDecision(
        target_quantities=target_quantities,
        target_values=target_values,
        target_weights=target_weights,
        ranks={symbol: rank_map.get(symbol) for symbol in target_quantities},
        frozen_symbols=frozenset(frozen),
        buffered_symbols=frozenset(buffered),
        skipped_unbuyable=tuple(skipped_unbuyable),
        skipped_invalid_weight=tuple(skipped_invalid_weight),
        core_symbols=tuple(core),
        buy_symbols=buy_symbols,
        sell_symbols=sell_symbols,
        portfolio_value=portfolio_value,
        fixed_budget=fixed_budget,
        reserved_value=reserved_value,
    )
