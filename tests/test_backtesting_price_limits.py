from datetime import datetime, timedelta

import polars as pl

from vnpy.alpha.strategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData


SYMBOL = "600000.SSE"
DATE = datetime(2021, 1, 5)


class AdjustedPriceLab:
    def load_contract_setttings(self) -> dict:
        return {
            SYMBOL: {
                "long_rate": 0.0005,
                "short_rate": 0.0015,
                "size": 1,
                "pricetick": 0.01,
            }
        }

    def load_bar_data(self, *args, **kwargs) -> list[BarData]:
        return [
            BarData(
                symbol="600000",
                exchange=Exchange.SSE,
                datetime=DATE,
                interval=Interval.DAILY,
                open_price=250.0,
                high_price=255.0,
                low_price=245.0,
                close_price=252.0,
                gateway_name="TEST",
            )
        ]

    def load_price_limit_df(self, *args, **kwargs) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "datetime": [DATE],
                "vt_symbol": [SYMBOL],
                "open": [10.0],
                "limit_up": [11.0],
                "limit_down": [9.0],
            }
        )


def test_daily_price_limits_follow_adjusted_bar_scale() -> None:
    engine = BacktestingEngine(AdjustedPriceLab())
    engine.set_parameters(
        vt_symbols=[SYMBOL],
        interval=Interval.DAILY,
        start=DATE,
        end=DATE + timedelta(days=1),
    )

    engine.load_data()

    bar = engine.history_data[(DATE, SYMBOL)]
    assert bar.limit_up == 275.0
    assert bar.limit_down == 225.0
    assert engine.is_buyable_at_open(SYMBOL, bar)
    assert engine.is_sellable_at_open(SYMBOL, bar)
