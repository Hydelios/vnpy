from __future__ import annotations
import polars as pl
from .factors_template import FactorTemplate

# --- 示例 1：range_pos（pl.Expr 版本，价格在 rolling 通道中的位置，居中到 [-0.5, 0.5]）
class RangePos(FactorTemplate):
    factor_name = "range_pos"
    params = {"window": [100, 200, 300]}
    required_columns = ("close",)
    doc = "(close - min) / (max - min) - 0.5 with EWMA smoothing"

    def build_expr(self, window: int):
        # 为避免可能的嵌套窗口限制，这里改为字符串表达式，复用 ts_* 算子实现同等逻辑
        return (
            f"(close - ts_min(close, {window})) / "
            f"(ts_max(close, {window}) - ts_min(close, {window}) + 1e-12) - 0.5"
        )

# --- Alpha101：示例 1（Alpha#001）
class Alpha101_001(FactorTemplate):
    factor_name = "alpha101_001"
    # window_arg：Ts_ArgMax 的窗口；window_std：std(returns) 的窗口
    params = {"window_arg": [5], "window_std": [20]}
    required_columns = ("close",)
    doc = "rank(ts_argmax((returns<0?std(returns,window_std):close)^2, window_arg)) - 0.5"

    def build_expr(self, window_arg: int, window_std: int) -> str:
        # returns = close/ts_delay(close,1) - 1
        ret_expr = "close / ts_delay(close, 1) - 1"
        # inner = ts_where(returns<0, ts_std(returns, window_std), close)
        inner_expr = (
            f"ts_where(({ret_expr}) < 0, ts_std(({ret_expr}), {window_std}), close)"
        )
        # 因子：对 inner^2 做 ts_argmax(window_arg)，再做横截面排名并居中
        return f"cs_rank(ts_argmax((({inner_expr}) ** 2), {window_arg})) - 0.5"

# --- 示例 2：Alpha101_002（字符串表达式，复用 ts_* / cs_* 操作符实现）
class Alpha101_002(FactorTemplate):
    factor_name = "alpha101_002"
    params = {"window": [5,10,20]}  # 可改为 [5,6,10]
    required_columns = ("volume", "close", "open")
    doc = "-corr(rank(delta(log(volume),2)), rank((close-open)/open), window)"

    def build_expr(self, window: int) -> str:
        return (
            f"-1 * ts_corr(cs_rank(ts_log(volume) - ts_delay(ts_log(volume), 2)), "
            f"cs_rank((close - open) / open), {window})"
        )

# --- 示例 3：RSI（字符串表达式 + ta_function 封装）
class TA_RSI(FactorTemplate):
    factor_name = "ta_rsi"
    params = {"window": [5, 10]}
    required_columns = ("close",)
    doc = "talib RSI(close, window)"

    def build_expr(self, window: int) -> str:
        return f"ta_rsi(close, {window})"
