import pandas as pd
import pytest

from vnpy.alpha.rq_optimizer import RQOptimizerConfig
from vnpy.alpha.rq_optimizer.runner import prepare_optimizer_input


def test_optimizer_config_validates_weight_feasibility() -> None:
    RQOptimizerConfig().validate()
    with pytest.raises(ValueError, match="可行组合"):
        RQOptimizerConfig(top_k=300, max_weight=0.003).validate()


def test_prepare_optimizer_input_samples_strategy_dates_and_topk() -> None:
    rows = []
    dates = pd.date_range("2026-01-05", periods=6, freq="B")
    for date in dates:
        for rank in range(4):
            rows.append(
                {
                    "datetime": date,
                    "vt_symbol": f"{rank:06d}.SZSE",
                    "signal": float(4 - rank),
                }
            )
    signal = pd.DataFrame(rows)
    config = RQOptimizerConfig(top_k=3, rebalance_interval=5, min_weight=0.2, max_weight=0.5)

    result = prepare_optimizer_input(signal, config)

    assert result["datetime"].drop_duplicates().tolist() == [dates[0], dates[5]]
    assert result.groupby("datetime").size().tolist() == [3, 3]
    assert result.groupby("datetime")["rank"].apply(list).tolist() == [[1, 2, 3], [1, 2, 3]]
    assert result["order_book_id"].str.endswith(".XSHE").all()
