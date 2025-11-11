# alpha001_str.py
from __future__ import annotations
from .factors_template import FactorTemplate

class Alpha001(FactorTemplate):
    """
    Alpha#1:
      rank( Ts_ArgMax( ((returns < 0) ? stddev(returns, 20) : close) ^ 2 , 5 ) ) - 0.5

    其中：
      returns = close / delay(close, 1) - 1
      条件选择用 ts_where(cond, a, b)
      注意：ts_argmax 返回窗口内最大值的位置；这里显式 +1 与常见实现对齐
    """
    factor_name = "alpha001"
    params = {"w_std": [20], "w_arg": [5]}          # 可调窗口
    required_columns = ("close",)
    doc = "rank(ts_argmax( (ts_where(ret<0, ts_std(ret,w_std), close))^2 , w_arg ) + 1) - 0.5"
    expr_kind = "str"

    def build_expr(self, w_std: int, w_arg: int):
        ret = "(close / (ts_delay(close, 1) + 1e-12) - 1)"
        inner = f"ts_where({ret} < 0, ts_std({ret}, {w_std}), close)"
        # +1：若你的 ts_argmax 已经返回 1-based，可把下面的 + 1 去掉
        argmax_idx = f"(ts_argmax(({inner})**2, {w_arg}) + 1)"
        return f"cs_rank({argmax_idx}) - 0.5"