from __future__ import annotations

from dataclasses import dataclass, field

from .metadata import ImplementationStatus


@dataclass(frozen=True)
class ImplementationSpec:
    key: str
    formula: str
    required_fields: tuple[str, ...]
    input_frequency: str = "1d"
    warmup: int = 0
    params: dict[str, object] = field(default_factory=dict)
    status: ImplementationStatus = ImplementationStatus.EXACT
    notes: str = ""


def alias(field: str, *, notes: str = "供应商日频PIT衍生字段") -> ImplementationSpec:
    return ImplementationSpec(f"alias:{field}", field, (field,), notes=notes)


def liquidity_metric(metric: str, formula: str, *, window: int = 20, long_window: int = 250) -> ImplementationSpec:
    return ImplementationSpec(
        "daily_liquidity_metric",
        formula,
        ("close", "volume", "turnover", "a_share_market_val_in_circulation"),
        warmup=max(window, long_window if metric == "turnover_abnormal" else window) + 1,
        params={"metric": metric, "window": window, "long_window": long_window},
    )


def intraday_path(
    metric: str,
    formula: str,
    *,
    post_window: int | None = None,
    post_stat: str = "mean",
) -> ImplementationSpec:
    params: dict[str, object] = {"metric": metric}
    if post_window is not None:
        params.update(post_window=post_window, post_stat=post_stat)
    return ImplementationSpec(
        "intraday_path_metric",
        formula,
        ("open", "high", "low", "close", "volume", "turnover"),
        input_frequency="1m",
        warmup=post_window or 0,
        params=params,
    )


def calendar_intraday_technical(metric: str, formula: str) -> ImplementationSpec:
    """2025 high-frequency technical factor followed by the stated 20-day decay."""
    return ImplementationSpec(
        "intraday_path_metric",
        formula,
        ("open", "high", "low", "close", "volume", "turnover"),
        input_frequency="1m",
        warmup=20,
        params={
            "metric": metric,
            "post_window": 20,
            "post_stat": "linear_decay",
        },
    )


def rolling_risk(metric: str, formula: str, *, window: int = 252, min_samples: int = 126) -> ImplementationSpec:
    return ImplementationSpec(
        "daily_rolling_risk",
        formula,
        ("close", "market_ret"),
        warmup=window + 1,
        params={"metric": metric, "window": window, "min_samples": min_samples},
    )


def autocorrelation_composite(mode: str, formula: str) -> ImplementationSpec:
    return ImplementationSpec(
        "intraday_autocorrelation_composite", formula,
        ("close" if mode.startswith("price") else "volume", "a_share_market_val_in_circulation"),
        input_frequency="1m", warmup=20, params={"mode": mode, "window": 20},
    )


def chip_metric(metric: str, formula: str) -> ImplementationSpec:
    return ImplementationSpec(
        "chip_distribution_metric",
        formula,
        ("high", "low", "close", "volume", "a_share_market_val_in_circulation"),
        input_frequency="1m",
        params={"metric": metric},
        notes="按分钟成交量价格分布与日换手存活模型重建筹码分布，价格网格精度1e-4",
    )


# Calendar titles with a direct, traceable mapping to fields present in the lab.
FIELD_ALIASES: dict[str, str] = {
    "总市值": "market_cap_2",
    "自由流通股市值": "a_share_market_val_in_circulation",
    "流通市值": "a_share_market_val_in_circulation",
    "市盈率": "pe_ratio_ttm",
    "账面市值比": "book_to_market_ratio_ttm",
    "市净率": "pb_ratio_ttm",
    "销售收入市值比": "sp_ratio_ttm",
    "股息率": "dividend_yield_ttm",
    "企业价值": "ev_ttm",
    "企业价值倍数": "ev_to_ebitda_ttm",
    "每股资本公积金": "capital_reserve_per_share_ttm",
    "每股未分配利润": "undistributed_profit_per_share_ttm",
    "每股留存收益": "retained_earnings_per_share_ttm",
    "每股盈余公积金": "earned_reserve_per_share_ttm",
    "每股净资产": "book_value_per_share_ttm",
    "每股货币资金": "cash_equivalent_per_share_ttm",
    "每股收益（摊薄）": "diluted_earnings_per_share_ttm",
    "每股收益(摊薄)": "diluted_earnings_per_share_ttm",
    "每股收益（扣除+摊薄）": "adjusted_fully_diluted_earnings_per_share_ttm",
    "每股收益(扣除+摊薄)": "adjusted_fully_diluted_earnings_per_share_ttm",
    "每股营业收入": "operating_revenue_per_share_ttm",
    "每股营业总收入": "operating_total_revenue_per_share_ttm",
    "每股经营现金流": "operating_cash_flow_per_share_ttm",
    "每股企业自由现金流": "free_cash_flow_company_per_share_ttm",
    "净资产收益率": "return_on_equity_ttm",
    "净资产收益率 (ROE)": "return_on_equity_ttm",
    "摊薄净资产收益率": "return_on_equity_diluted_ttm",
    "摊薄净资产收益率 (Diluted ROE)": "return_on_equity_diluted_ttm",
    "扣非净资产收益率": "adjusted_return_on_equity_ttm",
    "摊薄扣非净资产收益率": "adjusted_return_on_equity_diluted_ttm",
    "总资产净利率": "return_on_asset_net_profit_ttm",
    "总资产报酬率": "return_on_asset_ttm",
    "投入资本回报率": "return_on_invested_capital_ttm",
    "投入资本回报率 (ROIC)": "return_on_invested_capital_ttm",
    "毛利率": "gross_profit_margin_ttm",
    "净利率": "net_profit_margin_ttm",
    "销售成本率": "cost_to_sales_ttm",
    "营业利润率": "profit_from_operation_to_revenue_ttm",
    "营业费用率": "expense_to_revenue_ttm",
    "所得税费用率": "income_tax_to_profit_before_tax_ttm",
    "应收账款周转率": "account_receivable_turnover_rate_ttm",
    "应收账款周转天数": "account_receivable_turnover_days_ttm",
    "应付账款周转率": "account_payable_turnover_rate_ttm",
    "应付账款周转天数": "account_payable_turnover_days_ttm",
    "存货周转率": "inventory_turnover_ttm",
    "流动资产周转率": "current_asset_turnover_ttm",
    "固定资产周转率": "fixed_asset_turnover_ttm",
    "总资产周转率": "total_asset_turnover_ttm",
    "股东权益周转率": "equity_turnover_ratio_ttm",
    "现金循环周期": "cash_conversion_cycle_ttm",
    "利息保障倍数": "time_interest_earned_ratio_ttm",
    "经营现金流量比率": "cash_flow_ratio_ttm",
    "现金流覆盖率": "ocf_to_debt_ttm",
    "资产负债率": "debt_to_asset_ratio_ttm",
    "权益乘数": "equity_multiplier_ttm",
    "负债权益比": "debt_to_equity_ratio_ttm",
    "债务权益比": "debt_to_equity_ratio_ttm",
    "股东权益比率": "equity_ratio_ttm",
    "流动比率": "current_ratio_ttm",
    "速动比率": "quick_ratio_ttm",
    "现金比率": "cash_ratio_ttm",
    "账面杠杆": "book_leverage_ttm",
    "市场杠杆": "market_leverage_ttm",
    "固定资产比例": "fixed_asset_ratio_ttm",
    "无形资产强度": "intangible_asset_ratio_ttm",
    "营业利润同比增长率": "operating_profit_growth_ratio_ttm",
    "净利润同比增长率": "net_profit_growth_ratio_ttm",
    "销售收入同比增长率": "operating_revenue_growth_ratio_ttm",
    "营业总收入同比增长率": "operating_revenue_growth_ratio_ttm",
    "归属母公司净利润同比增长率": "net_profit_parent_company_growth_ratio_ttm",
    "总资产增长率": "total_asset_growth_ratio_ttm",
    "净资产增长率": "net_asset_growth_ratio_ttm",
    "应收周转率": "account_receivable_turnover_rate_ttm",
    "流动资产占比": "current_asset_to_total_asset_lf",
    "流动负债率": "current_debt_to_total_debt_ttm",
    "长期负债比率": "non_current_debt_to_total_debt_ttm",
    "盈利市值比": "ep_ratio_ttm",
    "现金流市值比": "cfp_ratio_ttm",
    "市销率": "ps_ratio_ttm",
    "市现率": "pcf_ratio_ttm",
    "PEG": "peg_ratio_ttm",
    "EBIT": "ebit_ttm",
    "EBITDA": "ebitda_ttm",
    "有息负债": "interest_bearing_debt_ttm",
    "净债务": "net_debt_ttm",
    "营运资本": "working_capital_ttm",
    "净营运资本": "net_working_capital_ttm",
    "经营周期": "operating_cycle_ttm",
    "平均付款期": "average_payment_period_ttm",
}


TITLE_SPECS: dict[str, ImplementationSpec] = {
    "“朝没晨雾”因子": ImplementationSpec(
        "intraday_volume_diff_regression", "mean(std(tvalues(lag1..lag5) in return~1+volume_delta_lag0..5),20)",
        ("close", "volume"), input_frequency="1m", warmup=20, params={"metric": "morning"},
    ),
    '"朝没晨雾" 因子': ImplementationSpec(
        "intraday_volume_diff_regression", "mean(std(tvalues(lag1..lag5) in return~1+volume_delta_lag0..5),20)",
        ("close", "volume"), input_frequency="1m", warmup=20, params={"metric": "morning"},
    ),
    "“午蔽古木”因子": ImplementationSpec(
        "intraday_volume_diff_regression", "sign(F-all-cross_mean(F-all))*abs(t_intercept)",
        ("close", "volume"), input_frequency="1m", params={"metric": "noon"},
    ),
    "“夜眠霜路”因子": ImplementationSpec(
        "intraday_volume_diff_regression", "mean_peer(abs(corr(t_intercept_stock,t_intercept_peer,20)))",
        ("close", "volume"), input_frequency="1m", warmup=20, params={"metric": "night"},
    ),
    "个股收益最高时的价变共振": ImplementationSpec(
        "intraday_market_event_metric", "cs_rank(mean(market_return at top5 stock_return minutes))/mean(market_cross_std at same minutes)",
        ("close", "turnover"), input_frequency="1m", params={"metric": "resonance", "side": "high"},
    ),
    "个股收益最低时的价变共振": ImplementationSpec(
        "intraday_market_event_metric", "inverse_cs_rank(mean(market_return at bottom5 stock_return minutes))/mean(market_cross_std at same minutes)",
        ("close", "turnover"), input_frequency="1m", params={"metric": "resonance", "side": "low"},
    ),
    "个股收益最高时的领先成交量异动": ImplementationSpec(
        "intraday_market_event_metric", "sum(lag(stock_amount/market_amount) at top30 return minutes)/mean(stock_amount/market_amount)",
        ("close", "turnover"), input_frequency="1m", params={"metric": "leading_amount", "side": "high"},
    ),
    "个股收益最低时的领先成交量异动": ImplementationSpec(
        "intraday_market_event_metric", "sum(lag(stock_amount/market_amount) at bottom30 return minutes)/mean(stock_amount/market_amount)",
        ("close", "turnover"), input_frequency="1m", params={"metric": "leading_amount", "side": "low"},
    ),
    "趋势清晰度": ImplementationSpec(
        "daily_trend_clarity", "R**2(close~time,T-240:T-20)", ("close",), warmup=240,
        params={"metric": "clarity"},
    ),
    "趋势清晰度动量": ImplementationSpec(
        "daily_trend_clarity", "-abs(z(momentum(T-240:T-20))-z(trend_clarity))", ("close",), warmup=240,
        params={"metric": "momentum"},
    ),
    "基于收益规模的信息离散度": ImplementationSpec(
        "daily_magnitude_information_dispersion", "-sign(PRET)*mean(sign(return_i)*quintile_weight(abs(return_i)),T-252:T-21)",
        ("close",), warmup=252,
    ),
    "凸显性收益": ImplementationSpec(
        "daily_salience_return", "cov(normalize(0.7**rank(salience)),return),monthly; salience=abs(r-median)/(abs(r)+abs(median)+.1)",
        ("close",), warmup=21, params={"variant": "standard"},
    ),
    "极端收益反转": ImplementationSpec(
        "intraday_extreme_reversal", "cs_rank(mean(extreme_minute_return,20))+cs_rank(mean(pre_extreme_return,20))",
        ("close",), input_frequency="1m", warmup=20, params={"window": 20},
    ),
    "非流动性因子 Gamma": ImplementationSpec(
        "daily_gamma_illiquidity", "-abs(gamma from next_excess_return~1+return+sign(excess_return)*turnover,monthly)",
        ("close", "turnover", "market_ret"), warmup=21,
    ),
    "修正模糊价差": ImplementationSpec(
        "intraday_fuzzy_spread", "0.5*(mean(cross_section_rescaled_negative(fuzzy_amount-fuzzy_volume),20)+std(...,20))",
        ("open", "high", "low", "close", "volume", "turnover"), input_frequency="1m", warmup=29,
    ),
    "聪明钱因子": ImplementationSpec(
        "intraday_smart_money", "VWAP(top_S_until_20pct_volume,last10d)/VWAP(all,last10d)",
        ("close", "volume"), input_frequency="1m", warmup=10, params={"window": 10},
    ),
    "成交量“潮汐”的价格变动速率": intraday_path(
        "tide_rate", "mean(((close_ebb-close_rise)/close_rise)/(t_ebb-t_rise),20)",
        post_window=20, post_stat="mean",
    ),
    "模糊金额比": intraday_path(
        "fuzzy_amount", "0.5*(mean(mean(turnover|ambiguity>daily_mean)/mean(turnover),20)+std(...,20))",
        post_window=20, post_stat="mean_std",
    ),
    "模糊数量比": intraday_path(
        "fuzzy_volume", "0.5*(mean(mean(volume|ambiguity>daily_mean)/mean(volume),20)+std(...,20))",
        post_window=20, post_stat="mean_std",
    ),
    "正向日内逆转的频率": ImplementationSpec(
        "daily_reversal_frequency", "monthly_mean(overnight_return<0 and intraday_return>0)",
        ("open", "close"), warmup=21, params={"direction": "positive", "abnormal": False},
    ),
    "反向日内逆转的频率": ImplementationSpec(
        "daily_reversal_frequency", "monthly_mean(overnight_return>0 and intraday_return<0)",
        ("open", "close"), warmup=21, params={"direction": "negative", "abnormal": False},
    ),
    "正向日内逆转的异常频率": ImplementationSpec(
        "daily_reversal_frequency", "positive_reversal_frequency/mean(positive_reversal_frequency,12m)",
        ("open", "close"), warmup=252, params={"direction": "positive", "abnormal": True},
    ),
    "反向日内逆转的异常频率": ImplementationSpec(
        "daily_reversal_frequency", "negative_reversal_frequency/mean(negative_reversal_frequency,12m)",
        ("open", "close"), warmup=252, params={"direction": "negative", "abnormal": True},
    ),
    "惊恐因子": ImplementationSpec(
        "daily_panic_metric", "0.5*(mean(panic*return,20)+std(panic*return,20)); panic=abs(r-rm)/(abs(r)+abs(rm)+.1)",
        ("close", "market_ret"), warmup=21, params={"metric": "base"},
    ),
    "波动率加剧-惊恐因子": ImplementationSpec(
        "daily_panic_metric", "0.5*(mean(panic*return*intraday_vol,20)+std(...,20))",
        ("close", "market_ret"), input_frequency="1m", warmup=21, params={"metric": "volatility"},
    ),
    "注意力衰减-惊恐因子": ImplementationSpec(
        "daily_panic_metric", "0.5*(mean(max(panic-mean(lag1,lag2),0)*return,20)+std(...,20))",
        ("close", "market_ret"), warmup=23, params={"metric": "decay"},
    ),
    "流动性冲击": ImplementationSpec(
        "daily_liquidity_shock", "-(monthly_amihud-mean(lag(monthly_amihud,1:12)))",
        ("close", "turnover"), warmup=273, params={"metric": "simple"},
    ),
    "带条件的流动性冲击": ImplementationSpec(
        "daily_liquidity_shock", "-innovation(ARMA(1,1),monthly_amihud,rolling=60,min=24,MLE)",
        ("close", "turnover"), warmup=504, params={"metric": "conditional"},
    ),
    "资本收益过剩": ImplementationSpec(
        "daily_capital_gain_overhang", "(lag(close)-reference_price_from_turnover_survival)/lag(close)",
        ("close", "volume", "a_share_market_val_in_circulation"), warmup=1,
    ),
    "协偏度 (朱剑涛版)": rolling_risk(
        "coskew_zhu", "sum((stock_ret-mean)*(market_ret-mean)**2)/sum((market_ret-mean)**3)",
        window=20, min_samples=15,
    ),
    "有形资本回报率": ImplementationSpec(
        "legacy_expression", "ebit_ttm/(invested_capital_ttm+1e-12)",
        ("ebit_ttm", "invested_capital_ttm"),
        notes="供应商投入资本字段对应净营运资本与长期经营资本口径",
    ),
    "APM因子": ImplementationSpec(
        "intraday_apm", "residual(tstat(epsilon_am-epsilon_pm,20),return20)",
        ("close", "market_minute_ret"), input_frequency="1m", warmup=20, params={"window": 20},
    ),
    "APM 因子": ImplementationSpec(
        "intraday_apm", "residual(tstat(epsilon_am-epsilon_pm,20),return20)",
        ("close", "market_minute_ret"), input_frequency="1m", warmup=20, params={"window": 20},
    ),
    "Frazzini-Pedersen贝塔": ImplementationSpec(
        "daily_frazzini_pedersen_beta", "corr(stock_3d,market_3d,5y)*std(stock_return,12m)/std(market_return,12m)",
        ("close", "market_ret"), warmup=1260,
    ),
    "非流动性": intraday_path(
        "shortcut_illiquidity", "mean(sum((2*(high-low)-abs(close-open))/turnover),20)",
        post_window=20, post_stat="mean",
    ),
    "换手率分布均匀度因子": ImplementationSpec(
        "intraday_turnover_uniformity", "std(std(volume_1m/float_shares),20)/mean(std(volume_1m/float_shares),20)",
        ("close", "volume", "a_share_market_val_in_circulation"), input_frequency="1m", warmup=20,
        params={"window": 20},
    ),
    "换手率分布均匀度": ImplementationSpec(
        "intraday_turnover_uniformity", "std(std(volume_1m/float_shares),20)/mean(std(volume_1m/float_shares),20)",
        ("close", "volume", "a_share_market_val_in_circulation"), input_frequency="1m", warmup=20,
        params={"window": 20},
    ),
    "累计振动升降指标": ImplementationSpec(
        "daily_technical_metric", "sum(16*X/R*K,20)",
        ("open", "high", "low", "close", "volume"), warmup=20,
        params={"metric": "asi", "window": 20},
    ),
    "克林格成交量摆动指标": ImplementationSpec(
        "daily_technical_metric", "EMA(VF,34)-EMA(VF,55)",
        ("open", "high", "low", "close", "volume"), warmup=55,
        params={"metric": "kvo", "window": 55},
    ),
    "价量相关性综合因子": ImplementationSpec(
        "intraday_price_volume_composite", "z(PVcorravg)+z(PVcorrstd)",
        ("close", "volume", "a_share_market_val_in_circulation"), input_frequency="1m", warmup=20,
        params={"mode": "composite", "window": 20},
    ),
    "价量相关性因子": ImplementationSpec(
        "intraday_price_volume_composite", "z(residual(PVcorravg+PVcorrstd,return20))+z(PVcorrtrend)",
        ("close", "volume", "a_share_market_val_in_circulation"), input_frequency="1m", warmup=40,
        params={"mode": "cpv", "window": 20},
    ),
    "价格自相关性因子(单序列差分)": autocorrelation_composite("price_single", "z(neutralize(mean(corr(delta_price,lead_price)|sign,20),cap))_positive-z(...)_negative"),
    "成交量自相关性因子(单序列差分)": autocorrelation_composite("volume_single", "z(neutralize(mean(corr(delta_volume,lead_volume)|sign,20),cap))_positive-z(...)_negative"),
    "价格自相关性合成因子(双序列差分)": autocorrelation_composite("price_double", "z(mean(corr(delta_price,lead_delta_price)|both_positive,20))+z(...both_negative)"),
    "价格自相关性合成因子 (双序列差分)": autocorrelation_composite("price_double", "z(mean(corr(delta_price,lead_delta_price)|both_positive,20))+z(...both_negative)"),
    "价格自相关性合成因子": autocorrelation_composite("price_double", "z(mean(corr(delta_price,lead_delta_price)|both_positive,20))+z(...both_negative)"),
    "成交量自相关性合成因子(双序列差分)": autocorrelation_composite("volume_double", "z(mean(corr(delta_volume,lead_delta_volume)|both_positive,20))+z(...both_negative)"),
    "成交量自相关性合成因子（双序列差分）": autocorrelation_composite("volume_double", "z(mean(corr(delta_volume,lead_delta_volume)|both_positive,20))+z(...both_negative)"),
    "信息离散度": ImplementationSpec(
        "legacy_expression",
        "ts_sign(ts_delay(close,21)/(ts_delay(close,252)+1e-12)-1)*(ts_mean(ts_where(ts_delay(close/(ts_delay(close,1)+1e-12)-1,21)<0,1,0),231)-ts_mean(ts_where(ts_delay(close/(ts_delay(close,1)+1e-12)-1,21)>0,1,0),231))",
        ("close",), warmup=252,
    ),
    "上行跳跃波动率": intraday_path("up_jump", "max(up_semivariance-0.5*tripower_variation,0)"),
    "下行跳跃波动率": intraday_path("down_jump", "max(down_semivariance-0.5*tripower_variation,0)"),
    "大的上行跳跃波动率": intraday_path("large_up_jump", "min(up_jump,sum(r**2 where r>gamma))"),
    "大的下行跳跃波动率": intraday_path("large_down_jump", "min(down_jump,sum(r**2 where r<-gamma))"),
    "小的上行跳跃波动率": intraday_path("small_up_jump", "up_jump-large_up_jump"),
    "小的下行跳跃波动率": intraday_path("small_down_jump", "down_jump-large_down_jump"),
    "上下行跳跃波动的不对称性": intraday_path("jump_asymmetry", "up_jump-down_jump"),
    "大的上下行跳跃波动不对称性": intraday_path("large_jump_asymmetry", "large_up_jump-large_down_jump"),
    "小的上下行跳跃波动不对称性": intraday_path("small_jump_asymmetry", "small_up_jump-small_down_jump"),
    "SemiBeta（正市场收益&正资产收益）高频因子-波动跳跃类": ImplementationSpec(
        "intraday_semibeta", "sum(max(stock_ret,0)*max(market_ret,0))/sum(market_ret**2)",
        ("close", "market_minute_ret"), input_frequency="1m",
        params={"stock_side": "positive", "market_side": "positive"},
    ),
    "SemiBeta（负市场收益&正资产收益）高频因子-波动跳跃类": ImplementationSpec(
        "intraday_semibeta", "-sum(max(stock_ret,0)*min(market_ret,0))/sum(market_ret**2)",
        ("close", "market_minute_ret"), input_frequency="1m",
        params={"stock_side": "positive", "market_side": "negative"},
    ),
    "SemiBeta（正市场收益&负资产收益）高频因子-波动跳跃类": ImplementationSpec(
        "intraday_semibeta", "-sum(min(stock_ret,0)*max(market_ret,0))/sum(market_ret**2)",
        ("close", "market_minute_ret"), input_frequency="1m",
        params={"stock_side": "negative", "market_side": "positive"},
    ),
    "SemiBeta（负市场收益&负资产收益）高频因子-波动跳跃类": ImplementationSpec(
        "intraday_semibeta", "sum(min(stock_ret,0)*min(market_ret,0))/sum(market_ret**2)",
        ("close", "market_minute_ret"), input_frequency="1m",
        params={"stock_side": "negative", "market_side": "negative"},
    ),
    "筹码换手率": chip_metric("chip_turnover", "chip_share(low<=price<=high)/turnover_rate"),
    "筹码穿透率": chip_metric("chip_penetration", "(winner_ratio-lag(winner_ratio))/turnover_rate"),
    "当日新增筹码盈利占比": chip_metric("new_gain_ratio", "new_chip_gain/total_paper_gain"),
    "当日新增筹码亏损占比": chip_metric("new_loss_ratio", "new_chip_loss/total_paper_loss"),
    "营运资本同比增速": ImplementationSpec(
        "legacy_expression", "(working_capital_ttm-working_capital_lyr)/(ts_abs(working_capital_lyr)+1e-12)",
        ("working_capital_ttm", "working_capital_lyr"),
    ),
    "现金与总资产比": ImplementationSpec(
        "legacy_expression", "cash_equivalent_per_share_ttm*equity_ratio_ttm/(book_value_per_share_ttm+1e-12)",
        ("cash_equivalent_per_share_ttm", "equity_ratio_ttm", "book_value_per_share_ttm"),
    ),
    "全部资产现金回收率": ImplementationSpec(
        "legacy_expression",
        "operating_cash_flow_per_share_ttm/((book_value_per_share_ttm/(equity_ratio_ttm+1e-12)+book_value_per_share_lyr/(equity_ratio_lyr+1e-12))/2+1e-12)",
        ("operating_cash_flow_per_share_ttm", "book_value_per_share_ttm", "book_value_per_share_lyr", "equity_ratio_ttm", "equity_ratio_lyr"),
    ),
    "股票交易周转率": ImplementationSpec(
        "legacy_expression", "ts_sum(turnover,20)/((market_cap_2+ts_delay(market_cap_2,20))/2+1e-12)",
        ("turnover", "market_cap_2"), warmup=20,
    ),
    "市值调整换手率": ImplementationSpec(
        "legacy_expression", "neutralize(log(volume*close/float_market_cap),log(float_market_cap))",
        ("volume", "close", "a_share_market_val_in_circulation"),
        params={"base_expression": "ts_log(volume*close/(a_share_market_val_in_circulation+1e-12))", "post_neutralize_market_cap": True},
    ),
    "非线性市值": ImplementationSpec(
        "legacy_expression", "neutralize(log(market_cap)**3,log(market_cap))",
        ("market_cap_2", "a_share_market_val_in_circulation"),
        params={"base_expression": "ts_log(market_cap_2)**3", "post_neutralize_market_cap": True},
    ),
    "修正的日内反转": ImplementationSpec(
        "legacy_expression", "neutralize(ts_sum(close/open-1,20),log(float_market_cap))",
        ("open", "close", "a_share_market_val_in_circulation"), warmup=20,
        params={"base_expression": "ts_sum(close/(open+1e-12)-1,20)", "post_neutralize_market_cap": True},
    ),
    "修正的隔夜反转": ImplementationSpec(
        "legacy_expression", "neutralize(ts_sum(open/lag(close)-1,20),log(float_market_cap))",
        ("open", "close", "a_share_market_val_in_circulation"), warmup=21,
        params={"base_expression": "ts_sum(open/(ts_delay(close,1)+1e-12)-1,20)", "post_neutralize_market_cap": True},
    ),
    "资本固定化率": ImplementationSpec(
        "legacy_expression", "non_current_asset_to_total_asset_ttm/(equity_ratio_ttm+1e-12)",
        ("non_current_asset_to_total_asset_ttm", "equity_ratio_ttm"),
    ),
    "有形净值债务率": ImplementationSpec(
        "legacy_expression", "liabilities_per_share_ttm/(tangible_asset_per_share_ttm+1e-12)",
        ("liabilities_per_share_ttm", "tangible_asset_per_share_ttm"),
    ),
    "总资产营业利润": ImplementationSpec(
        "legacy_expression", "profit_from_operation_to_revenue_ttm*total_asset_turnover_ttm",
        ("profit_from_operation_to_revenue_ttm", "total_asset_turnover_ttm"),
    ),
    "总资产毛利率": ImplementationSpec(
        "legacy_expression", "gross_profit_margin_ttm*total_asset_turnover_ttm",
        ("gross_profit_margin_ttm", "total_asset_turnover_ttm"),
    ),
    "净资产营业利润": ImplementationSpec(
        "legacy_expression", "profit_from_operation_to_revenue_ttm*total_asset_turnover_ttm/(equity_ratio_ttm+1e-12)",
        ("profit_from_operation_to_revenue_ttm", "total_asset_turnover_ttm", "equity_ratio_ttm"),
    ),
    "成本费用利润率": ImplementationSpec(
        "legacy_expression", "net_profit_margin_ttm/(cost_to_sales_ttm+expense_to_revenue_ttm+1e-12)",
        ("net_profit_margin_ttm", "cost_to_sales_ttm", "expense_to_revenue_ttm"),
    ),
    "修正的振幅": ImplementationSpec(
        "legacy_expression", "(high-low)/(ts_delay(close,1)+1e-12)",
        ("high", "low", "close"), warmup=1,
    ),
    "上下影线/收盘价的标准差": ImplementationSpec(
        "legacy_expression",
        "ts_std((high-rq_max(open,close)+rq_min(open,close)-low)/(close+1e-12),20)",
        ("open", "high", "low", "close"), warmup=20,
    ),
    "绝对收益与调整后滞后成交量相关性": ImplementationSpec(
        "legacy_expression",
        "ts_corr(ts_abs(close/(ts_delay(close,1)+1e-12)-1),ts_delay((turnover-cs_mean(turnover))/(cs_std(turnover)+1e-12),1),20)",
        ("close", "turnover"), warmup=21,
    ),
    "滞后绝对收益与调整后成交量相关性": ImplementationSpec(
        "legacy_expression",
        "ts_corr(ts_delay(ts_abs(close/(ts_delay(close,1)+1e-12)-1),1),(turnover-cs_mean(turnover))/(cs_std(turnover)+1e-12),20)",
        ("close", "turnover"), warmup=21,
    ),
    "成交量分桶熵": intraday_path("volume_bucket_entropy", "entropy(histogram(volume_1m))"),
    "流通股市值": alias("a_share_market_val_in_circulation"),
    "Q棒指标": ImplementationSpec(
        "legacy_expression", "ts_mean(close-open,20)", ("open", "close"), warmup=20,
    ),
    "趋势分数指标": ImplementationSpec(
        "legacy_expression", "ts_sum(ts_where(close>=ts_delay(close,1),1,-1),20)",
        ("close",), warmup=20,
    ),
    "真实波动范围指标": ImplementationSpec(
        "legacy_expression",
        "rq_max(high-low,rq_max(ts_abs(high-ts_delay(close,1)),ts_abs(low-ts_delay(close,1))))",
        ("high", "low", "close"), warmup=1,
    ),
    "收集派发指标": ImplementationSpec(
        "legacy_expression",
        "ts_sum(ts_where(close==ts_delay(close,1),0,close-ts_where(close>ts_delay(close,1),rq_min(low,ts_delay(close,1)),rq_max(high,ts_delay(close,1)))),20)",
        ("high", "low", "close"), warmup=20,
    ),
    "估波指标": ImplementationSpec(
        "legacy_expression",
        "ta_wma(100*((close-ts_delay(close,14))/(ts_delay(close,14)+1e-12)+(close-ts_delay(close,11))/(ts_delay(close,11)+1e-12)),10)",
        ("close",), warmup=24,
    ),
    "随机相对强弱指标": ImplementationSpec(
        "legacy_expression",
        "100*(ta_rsi(close,14)-ts_max(ta_rsi(close,14),14))/(ts_max(ta_rsi(close,14),14)-ts_min(ta_rsi(close,14),14)+1e-12)",
        ("close",), warmup=28,
    ),
    "终极震荡指标": ImplementationSpec(
        "legacy_expression",
        "100*((ts_sum(close-rq_min(low,ts_delay(close,1)),5)/(ts_sum(rq_max(high,ts_delay(close,1))-rq_min(low,ts_delay(close,1)),5)+1e-12))*10*20+(ts_sum(close-rq_min(low,ts_delay(close,1)),10)/(ts_sum(rq_max(high,ts_delay(close,1))-rq_min(low,ts_delay(close,1)),10)+1e-12))*5*20+(ts_sum(close-rq_min(low,ts_delay(close,1)),20)/(ts_sum(rq_max(high,ts_delay(close,1))-rq_min(low,ts_delay(close,1)),20)+1e-12))*5*10)/(10*20+5*20+5*10)",
        ("high", "low", "close"), warmup=20,
    ),
    "正量指标": ImplementationSpec(
        "daily_technical_metric", "PVI[0]=1000; update when volume>lag(volume)",
        ("open", "high", "low", "close", "volume"), params={"metric": "pvi"},
    ),
    "负量指标": ImplementationSpec(
        "daily_technical_metric", "NVI[0]=1000; subtract return when volume<lag(volume)",
        ("open", "high", "low", "close", "volume"), params={"metric": "nvi"},
    ),
    "上下行波动率": ImplementationSpec(
        "daily_technical_metric", "log(((n_up-1)*downside_ss)/((n_down-1)*upside_ss))",
        ("open", "high", "low", "close", "volume"), warmup=20,
        params={"metric": "duvol", "window": 20},
    ),
    "阿隆指标": ImplementationSpec(
        "daily_technical_metric", "100*(argmax(high,20)-argmin(low,20))/20",
        ("open", "high", "low", "close", "volume"), warmup=20,
        params={"metric": "aroon", "window": 20},
    ),
    "趋势占比": intraday_path(
        "trend_ratio", "(last(close_1m)-first(close_1m))/sum(abs(delta(close_1m)))"
    ),
    "加权收盘价比因子": intraday_path(
        "weighted_close_ratio", "vwap(close_1m,volume_1m)/mean(close_1m)"
    ),
    "结构化反转": intraday_path(
        "structured_reversal", "volume_weighted_return(bottom10pct(volume_1m))"
    ),
    "日内黄金分割反转": intraday_path(
        "golden_reversal", "sum(log(close/price_10:00),20)", post_window=20, post_stat="sum"
    ),
    "时间加权平均的股票相对价格位置": intraday_path(
        "time_weighted_relative_price", "(mean((open+high+low+close)/4)-low)/(high-low)"
    ),
    "时间加权平均的股票相对价格位置  流动性因子": intraday_path(
        "time_weighted_relative_price", "(mean((open+high+low+close)/4)-low)/(high-low)"
    ),
    "价量相关性平均数因子": ImplementationSpec(
        "intraday_path_metric", "neutralize(mean(corr(close_1m,volume_1m),20),log(float_market_cap))",
        ("open", "high", "low", "close", "volume", "turnover", "a_share_market_val_in_circulation"), input_frequency="1m", warmup=20,
        params={"metric": "price_volume_corr", "post_window": 20, "post_stat": "mean", "post_neutralize_market_cap": True},
    ),
    "价量相关性波动性因子": ImplementationSpec(
        "intraday_path_metric", "neutralize(std(corr(close_1m,volume_1m),20),log(float_market_cap))",
        ("open", "high", "low", "close", "volume", "turnover", "a_share_market_val_in_circulation"), input_frequency="1m", warmup=20,
        params={"metric": "price_volume_corr", "post_window": 20, "post_stat": "std", "post_neutralize_market_cap": True},
    ),
    "价量相关性趋势因子": intraday_path(
        "price_volume_corr", "slope(corr(close_1m,volume_1m),20)", post_window=20, post_stat="slope"
    ),
    "隔夜跳空": ImplementationSpec(
        "legacy_expression", "ts_sum(ts_abs(ts_log(open/(ts_delay(close,1)+1e-12))),20)",
        ("open", "close"), warmup=21,
    ),
    "理想振幅因子": ImplementationSpec(
        "daily_ideal_amplitude", "mean(amplitude|top25pct(close),20)-mean(amplitude|bottom25pct(close),20)",
        ("high", "low", "close"), warmup=20, params={"window": 20, "fraction": 0.25},
    ),
    "平均换手率": liquidity_metric("turnover_mean", "mean(volume*close/float_market_cap,20)"),
    "最大异常日收益率": ImplementationSpec(
        "daily_attention_metric", "max(abs(daily_return),20)", ("close", "volume"),
        warmup=21, params={"metric": "max_abnormal_return", "window": 20, "baseline_window": 250},
    ),
    "每日最大异常成交量": ImplementationSpec(
        "daily_attention_metric", "max(volume/mean(volume,250),20)", ("close", "volume"),
        warmup=270, params={"metric": "max_abnormal_volume", "window": 20, "baseline_window": 250},
    ),
    "每日平均异常成交": ImplementationSpec(
        "daily_attention_metric", "mean(volume/mean(volume,250),20)", ("close", "volume"),
        warmup=270, params={"metric": "mean_abnormal_volume", "window": 20, "baseline_window": 250},
    ),
    "衰减加权月频成交量": ImplementationSpec(
        "daily_attention_metric", "linear_decay(volume,20)", ("close", "volume"),
        warmup=20, params={"metric": "decay_volume", "window": 20, "baseline_window": 250},
    ),
    "前景价值 TK": ImplementationSpec(
        "daily_prospect_value", "prospect_value(log_return,20,alpha=.88,lambda=2.25,gamma=.61,delta=.69)",
        ("close",), warmup=21, params={"source": "return", "window": 20},
    ),
    "基于 PB 的前景价值 TK": ImplementationSpec(
        "daily_prospect_value", "prospect_value(delta_log_pb,20,alpha=.88,lambda=2.25,gamma=.61,delta=.69)",
        ("pb_ratio_ttm",), warmup=21, params={"source": "pb", "window": 20},
    ),
    "筹码分布成本重心": chip_metric("reference_price", "sum(chip_price*chip_weight)"),
    "资本利得突出量": chip_metric(
        "capital_gain_overhang", "(close-reference_price)/close"
    ),
    "未实现盈利量": chip_metric("unrealized_profit", "(close-reference_price)/reference_price"),
    "活动筹码占比": chip_metric(
        "active_share", "sum(chip_weight where close*(1-limit)<=price<=close*(1+limit))"
    ),
    "筹码集中度": chip_metric("concentration", "2*(price95-price05)/(price95+price05)"),
    "筹码乖离率": chip_metric(
        "chip_bias", "winner_ratio*turnover+lag(chip_bias)*(1-turnover)"
    ),
    "筹码分布相对价位": chip_metric(
        "relative_position", "(close-chip_mean)/(chip_highest-chip_lowest)"
    ),
    "筹码分布变异系数": chip_metric("coefficient_of_variation", "chip_std/chip_mean"),
    "筹码分布成本宽带": chip_metric(
        "cost_bandwidth", "(chip_highest-chip_lowest)/chip_lowest"
    ),
    "筹码分布峰度(偏度)": chip_metric(
        "kurtosis", "sum(((price-chip_mean)/chip_std)**4*chip_weight)"
    ),
    "筹码分布峰度（偏度）": chip_metric(
        "kurtosis", "sum(((price-chip_mean)/chip_std)**4*chip_weight)"
    ),
    "盈利卖出倾向": chip_metric("gain_tendency", "sum(max((close-price)/close,0)*chip_weight)"),
    "亏损卖出倾向": chip_metric("loss_tendency", "sum(min((close-price)/close,0)*chip_weight)"),
    "已实现盈利比率": chip_metric(
        "realized_profit_ratio", "realized_gain/(realized_gain+paper_gain)"
    ),
    "已实现亏损比率": chip_metric(
        "realized_loss_ratio", "realized_loss/(realized_loss+paper_loss)"
    ),
    "基于筹码分布的处置效应": chip_metric(
        "disposition_effect", "realized_profit_ratio-realized_loss_ratio"
    ),
    "基于筹码分布的 V 型处置效应": chip_metric(
        "v_disposition_effect", "realized_profit_ratio+0.23*realized_loss_ratio"
    ),
    "V型处置效应": chip_metric(
        "v_tendency", "gain_tendency+0.23*abs(loss_tendency)"
    ),
    "成交量变动速率指标": ImplementationSpec(
        "legacy_expression", "100*(volume-ts_delay(volume,20))/(ts_delay(volume,20)+1e-12)",
        ("volume",), warmup=20,
    ),
    "成交量加权移动均线指标": ImplementationSpec(
        "legacy_expression", "ts_sum(close*volume,20)/(ts_sum(volume,20)+1e-12)",
        ("close", "volume"), warmup=20,
    ),
    "三角移动均线指标": ImplementationSpec(
        "legacy_expression", "ts_mean(ts_mean(close,11),10)", ("close",), warmup=20,
    ),
    "动量指标": ImplementationSpec(
        "legacy_expression", "close-ts_delay(close,20)", ("close",), warmup=20,
    ),
    "相对动量指标": ImplementationSpec(
        "legacy_expression",
        "100*ts_ema(ts_greater(close-ts_delay(close,5),0),14)/(ts_ema(ts_greater(close-ts_delay(close,5),0),14)+ts_ema(ts_greater(ts_delay(close,5)-close,0),14)+1e-12)",
        ("close",), warmup=19,
    ),
    "真实强度指数": ImplementationSpec(
        "legacy_expression",
        "100*ts_ema(ts_ema(close-ts_delay(close,1),25),13)/(ts_ema(ts_ema(ts_abs(close-ts_delay(close,1)),25),13)+1e-12)",
        ("close",), warmup=39,
    ),
    "优瑟指数": ImplementationSpec(
        "legacy_expression",
        "(ts_mean(((close-ts_max(close,20))/(ts_max(close,20)+1e-12)*100)**2,20))**0.5",
        ("close",), warmup=39,
    ),
    "动态动量指标": ImplementationSpec(
        "legacy_expression", "14*ts_mean(ts_std(close,5),10)/(ts_std(close,5)+1e-12)",
        ("close",), warmup=15,
    ),
    "前K个月动量": ImplementationSpec(
        "return", "close/ts_delay(close,126)-1", ("close",), warmup=126,
        params={"window": 126, "sign": 1.0},
    ),
    "动量加速度": ImplementationSpec(
        "momentum_acceleration_halves",
        "return(t-6m,t)-return(t-12m,t-6m)",
        ("close",), warmup=252, params={"window": 126},
    ),
    "加速度动量": ImplementationSpec(
        "daily_price_acceleration", "quadratic_coef(close~1+t+t**2,60)",
        ("close",), warmup=60, params={"window": 60},
    ),
    "市场Beta": rolling_risk("beta", "beta(log_return,market_return,252)"),
    "基于日度收益的市场Beta": rolling_risk("beta", "beta(log_return,market_return,252)"),
    "下行贝塔": rolling_risk("downside_beta", "beta(return,market_return where market_return<0,252)"),
    "尾部 Beta": rolling_risk("tail_beta", "beta(return,market_return where market_return<=q10,252)"),
    "尾部Beta": rolling_risk("tail_beta", "beta(return,market_return where market_return<=q10,252)"),
    "协偏度(系统偏度)": rolling_risk("coskew", "coskew(return,market_return,252)"),
    "协偏度 (系统偏度)": rolling_risk("coskew", "coskew(return,market_return,252)"),
    "协偏度（系统偏度）　　　　波动率因子": rolling_risk("coskew", "coskew(return,market_return,252)"),
    "特质波动率": rolling_risk("idio_vol", "std(residual(return~market_return),252)"),
    "特质偏度": rolling_risk("idio_skew", "skew(residual(return~market_return),252)"),
    "特异度": rolling_risk("idiosyncrasy", "var(residual)/var(return) = 1-R**2"),
    "在险价值": rolling_risk("var", "quantile(return,0.05,252)"),
    "条件风险价值": rolling_risk("cvar", "mean(return where return<=q05,252)"),
    "最大涨幅": intraday_path("max_rise", "product(1+top10pct(r_1m))"),
    "日内跳跃度": intraday_path(
        "jump_degree", "mean(2*(simple_return-log_return)-log_return**2)"
    ),
    "已实现双幂次变差": intraday_path(
        "bipower_variation", "mu_1**-2*sum(abs(r_t)*abs(r_t-1))"
    ),
    "已实现三幂次变差": intraday_path(
        "tripower_variation", "mu_2/3**-3*sum(abs(r_t*r_t-1*r_t-2)**(2/3))"
    ),
    "已实现跳跃波动率": intraday_path(
        "jump_variation", "max(realized_variance-tripower_variation,0)"
    ),
    "交易量变异系数": intraday_path("turnover_cv", "std(turnover_1m)/mean(turnover_1m)"),
    "价格弹性": intraday_path("price_elasticity", "mean((high_1m-low_1m)/turnover_1m)"),
    "成交额占比熵": intraday_path(
        "amount_entropy", "-sum((turnover_i/sum(turnover))*log(turnover_i/sum(turnover)))"
    ),
    "单一成交额占比熵": intraday_path(
        "single_amount_entropy", "-sum((volume_i/VOL*close_i/CLOSE)*log(volume_i/VOL*close_i/CLOSE))"
    ),
    "加权偏度": intraday_path(
        "weighted_skew", "sum(volume_weight*(close-mean(close))**3)/std(close)**3"
    ),
    "成交量比值": intraday_path(
        "volume_ratio", "sum(volume,09:30-10:00)/sum(volume,13:00-13:30)"
    ),
    "日内条件在险价值": intraday_path(
        "intraday_cvar", "CVaR_5pct(r_1m*volume_1m/sum(volume_1m))"
    ),
    "高频收益偏度": ImplementationSpec(
        "intraday_return_stat", "intraday_skew(log(close/lag(close)))", ("close",),
        input_frequency="1m", params={"stat": "skew"},
    ),
    "市盈率增长率": alias("peg_ratio_ttm", notes="日历定义即PE/盈利增速，映射供应商PEG TTM字段"),
    "EBIT/企业价值": ImplementationSpec(
        "legacy_expression", "ebit_ttm/(ev_ttm+1e-12)", ("ebit_ttm", "ev_ttm")
    ),
    "Sale/企业价值": ImplementationSpec(
        "legacy_expression",
        "operating_revenue_per_share_ttm*weighted_common_stock_ttm/(ev_ttm+1e-12)",
        ("operating_revenue_per_share_ttm", "weighted_common_stock_ttm", "ev_ttm"),
    ),
    "负债市值比": ImplementationSpec(
        "legacy_expression",
        "liabilities_per_share_ttm*weighted_common_stock_ttm/(market_cap_2+1e-12)",
        ("liabilities_per_share_ttm", "weighted_common_stock_ttm", "market_cap_2"),
    ),
    "市盈率增长率/股息率": ImplementationSpec(
        "legacy_expression", "peg_ratio_ttm/(dividend_yield_ttm+1e-12)",
        ("peg_ratio_ttm", "dividend_yield_ttm"),
    ),
    "盈利市值比60日变动": ImplementationSpec(
        "legacy_expression", "ep_ratio_ttm-ts_delay(ep_ratio_ttm,60)", ("ep_ratio_ttm",), warmup=60,
    ),
    "盈利现金比率": ImplementationSpec(
        "legacy_expression",
        "operating_cash_flow_per_share_ttm/(diluted_earnings_per_share_ttm+1e-12)",
        ("operating_cash_flow_per_share_ttm", "diluted_earnings_per_share_ttm"),
    ),
    "销售现金比率": ImplementationSpec(
        "legacy_expression",
        "operating_cash_flow_per_share_ttm/(operating_revenue_per_share_ttm+1e-12)",
        ("operating_cash_flow_per_share_ttm", "operating_revenue_per_share_ttm"),
    ),
    "毛利润与净利润之比": ImplementationSpec(
        "legacy_expression", "gross_profit_margin_ttm/(net_profit_margin_ttm+1e-12)",
        ("gross_profit_margin_ttm", "net_profit_margin_ttm"),
    ),
    "普通股回报率": alias("return_on_equity_ttm", notes="供应商ROE TTM字段对应普通股权益回报率"),
    "ROE增长减净资产增长": ImplementationSpec(
        "legacy_expression", "inc_return_on_equity_ttm-net_asset_growth_ratio_ttm",
        ("inc_return_on_equity_ttm", "net_asset_growth_ratio_ttm"),
    ),
    "月均换手率": liquidity_metric("turnover_mean", "mean(volume*close/float_market_cap,20)"),
    "换手波动": liquidity_metric("turnover_std", "std(volume*close/float_market_cap,20)"),
    "换手率变异系数": liquidity_metric(
        "turnover_cv", "std(turnover_rate,20)/mean(turnover_rate,20)"
    ),
    "月度异常换手率": liquidity_metric(
        "turnover_abnormal", "mean(turnover_rate,20)/mean(turnover_rate,250)"
    ),
    "成交额波动": liquidity_metric("amount_std", "std(turnover,20)"),
    "成交额波动系数": liquidity_metric("amount_mean_over_std", "mean(turnover,20)/std(turnover,20)"),
    "成交额变动": liquidity_metric("amount_std", "std(turnover,20)"),
    "Amihud 非流动性因子": liquidity_metric("amihud", "mean(abs(return)/turnover,20)"),
    "Amihud非流动性因子": liquidity_metric("amihud", "mean(abs(return)/turnover,20)"),
    "非流动性的变异系数": liquidity_metric(
        "amihud_cv", "std(abs(return)/turnover,20)/mean(abs(return)/turnover,20)"
    ),
    "负收益非流动性": liquidity_metric(
        "negative_amihud", "mean(abs(return)/turnover where return<0,20)"
    ),
    "动态买卖气指标": ImplementationSpec(
        "legacy_expression",
        "ts_where(ts_sum(ts_where(open<ts_delay(open,1),0,ts_greater(high-open,open-ts_delay(open,1))),23)>ts_sum(ts_where(open>=ts_delay(open,1),0,ts_greater(open-low,open-ts_delay(open,1))),23),(ts_sum(ts_where(open<ts_delay(open,1),0,ts_greater(high-open,open-ts_delay(open,1))),23)-ts_sum(ts_where(open>=ts_delay(open,1),0,ts_greater(open-low,open-ts_delay(open,1))),23))/(ts_sum(ts_where(open<ts_delay(open,1),0,ts_greater(high-open,open-ts_delay(open,1))),23)+1e-12),(ts_sum(ts_where(open<ts_delay(open,1),0,ts_greater(high-open,open-ts_delay(open,1))),23)-ts_sum(ts_where(open>=ts_delay(open,1),0,ts_greater(open-low,open-ts_delay(open,1))),23))/(ts_sum(ts_where(open>=ts_delay(open,1),0,ts_greater(open-low,open-ts_delay(open,1))),23)+1e-12))",
        ("open", "high", "low"), warmup=24,
    ),
    "货币流量指标": ImplementationSpec(
        "legacy_expression",
        "100-100/(1+ts_sum(ts_where((high+low+close)/3>ts_delay((high+low+close)/3,1),(high+low+close)/3*volume,0),20)/(ts_sum(ts_where((high+low+close)/3<=ts_delay((high+low+close)/3,1),(high+low+close)/3*volume,0),20)+1e-12))",
        ("high", "low", "close", "volume"), warmup=21,
    ),
    "方向标准离差指标": ImplementationSpec(
        "legacy_expression",
        "100*(ts_sum(ts_where(high+low<=ts_delay(high+low,1),0,ts_greater(ts_abs(high-ts_delay(high,1)),ts_abs(low-ts_delay(low,1)))),20)-ts_sum(ts_where(high+low>ts_delay(high+low,1),0,ts_greater(ts_abs(high-ts_delay(high,1)),ts_abs(low-ts_delay(low,1)))),20))/(ts_sum(ts_where(high+low<=ts_delay(high+low,1),0,ts_greater(ts_abs(high-ts_delay(high,1)),ts_abs(low-ts_delay(low,1)))),20)+ts_sum(ts_where(high+low>ts_delay(high+low,1),0,ts_greater(ts_abs(high-ts_delay(high,1)),ts_abs(low-ts_delay(low,1)))),20)+1e-12)",
        ("high", "low"), warmup=21,
    ),
    "梅斯线": ImplementationSpec(
        "legacy_expression",
        "ts_ema(high-low,9)/(ts_ema(ts_ema(high-low,9),9)+1e-12)",
        ("high", "low"), warmup=18,
    ),
    "随机动量指标": ImplementationSpec(
        "legacy_expression",
        "100*ts_ema(ts_ema(close-(ts_max(high,10)+ts_max(low,10))/2,3),3)/(ts_ema(ts_ema(ts_max(high,10)-ts_max(low,10),3),3)/2+1e-12)",
        ("high", "low", "close"), warmup=16,
    ),
    "日内动量指标": ImplementationSpec(
        "legacy_expression",
        "100*ts_sum(ts_where(close>open,close-open,0),20)/(ts_sum(ts_where(close>open,close-open,0),20)+ts_sum(ts_where(close<=open,open-close,0),20)+1e-12)",
        ("open", "close"), warmup=20,
    ),
    "简易波动指标": ImplementationSpec(
        "legacy_expression",
        "(((high-low)-ts_delay(high-low,1))/2)*(high-low)/(volume+1e-12)",
        ("high", "low", "volume"), warmup=2,
    ),
    "买卖意愿指标": ImplementationSpec(
        "legacy_expression",
        "100*ts_sum(ts_greater(high-ts_delay(close,1),0),20)/(ts_sum(ts_greater(ts_delay(close,1)-low,0),20)+1e-12)",
        ("high", "low", "close"), warmup=21,
    ),
    "佳庆离散指标": ImplementationSpec(
        "legacy_expression",
        "100*(ts_ema(high-low,10)-ts_delay(ts_ema(high-low,10),10))/(ts_delay(ts_ema(high-low,10),10)+1e-12)",
        ("high", "low"), warmup=20,
    ),
    "相对波动率指标": ImplementationSpec(
        "legacy_expression",
        "50*(ts_ema(ts_where(high>ts_delay(high,1),ts_std(high,10),0),20)/(ts_ema(ts_where(high>ts_delay(high,1),ts_std(high,10),0),20)+ts_ema(ts_where(high<ts_delay(high,1),ts_std(high,10),0),20)+1e-12)+ts_ema(ts_where(low>ts_delay(low,1),ts_std(low,10),0),20)/(ts_ema(ts_where(low>ts_delay(low,1),ts_std(low,10),0),20)+ts_ema(ts_where(low<ts_delay(low,1),ts_std(low,10),0),20)+1e-12))",
        ("high", "low"), warmup=30,
    ),
    "区域指数": ImplementationSpec(
        "legacy_expression",
        "ts_ema(100*(ts_where(close>ts_delay(close,1),ts_greater(ts_greater(high-low,ts_abs(ts_delay(close,1)-high)),ts_abs(ts_delay(close,1)-low))/(close-ts_delay(close,1)+1e-12),ts_greater(ts_greater(high-low,ts_abs(ts_delay(close,1)-high)),ts_abs(ts_delay(close,1)-low)))-ts_min(ts_where(close>ts_delay(close,1),ts_greater(ts_greater(high-low,ts_abs(ts_delay(close,1)-high)),ts_abs(ts_delay(close,1)-low))/(close-ts_delay(close,1)+1e-12),ts_greater(ts_greater(high-low,ts_abs(ts_delay(close,1)-high)),ts_abs(ts_delay(close,1)-low))),20))/(ts_max(ts_where(close>ts_delay(close,1),ts_greater(ts_greater(high-low,ts_abs(ts_delay(close,1)-high)),ts_abs(ts_delay(close,1)-low))/(close-ts_delay(close,1)+1e-12),ts_greater(ts_greater(high-low,ts_abs(ts_delay(close,1)-high)),ts_abs(ts_delay(close,1)-low))),20)-ts_min(ts_where(close>ts_delay(close,1),ts_greater(ts_greater(high-low,ts_abs(ts_delay(close,1)-high)),ts_abs(ts_delay(close,1)-low))/(close-ts_delay(close,1)+1e-12),ts_greater(ts_greater(high-low,ts_abs(ts_delay(close,1)-high)),ts_abs(ts_delay(close,1)-low))),20)+1e-12),5)",
        ("high", "low", "close"), warmup=26,
    ),
    "成交量平滑异同均线指标": ImplementationSpec(
        "legacy_expression",
        "(ts_ema(volume,12)-ts_ema(volume,26))-ts_ema(ts_ema(volume,12)-ts_ema(volume,26),9)",
        ("volume",), warmup=35,
    ),
    "蔡金货币流量指标": ImplementationSpec(
        "legacy_expression",
        "100*ts_sum(volume*((close-low)-(high-close))/(high-low+1e-12),20)/(ts_sum(volume,20)+1e-12)",
        ("high", "low", "close", "volume"), warmup=20,
    ),
    "佳庆指标": ImplementationSpec(
        "legacy_expression",
        "ts_ema(ts_cumsum(volume*(2*close-high-low)/(high+low+1e-12)),10)-ts_ema(ts_cumsum(volume*(2*close-high-low)/(high+low+1e-12)),3)",
        ("high", "low", "close", "volume"), warmup=10,
    ),
    "成交量相对强弱指标": ImplementationSpec(
        "legacy_expression",
        "100*ts_ema(ts_where(close>ts_delay(close,1),volume,ts_where(close==ts_delay(close,1),volume/2,0)),20)/(ts_ema(ts_where(close>ts_delay(close,1),volume,ts_where(close==ts_delay(close,1),volume/2,0)),20)+ts_ema(ts_where(close<ts_delay(close,1),volume,ts_where(close==ts_delay(close,1),volume/2,0)),20)+1e-12)",
        ("close", "volume"), warmup=21,
    ),
    "成交量比率指标": ImplementationSpec(
        "legacy_expression",
        "100*ts_sum(ts_where(close>ts_delay(close,1),volume,0),20)/(ts_sum(ts_where(close<ts_delay(close,1),volume,0),20)+1e-12)",
        ("close", "volume"), warmup=21,
    ),
    "随机指标": ImplementationSpec(
        "legacy_expression",
        "(close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9) + 1e-12) * 100",
        ("high", "low", "close"), warmup=9,
    ),
    "异同离差乖离率": ImplementationSpec(
        "legacy_expression",
        "ts_ema(((close-ts_mean(close,5))/(ts_mean(close,5)+1e-12))-ts_delay((close-ts_mean(close,5))/(ts_mean(close,5)+1e-12),16),17)",
        ("close",), warmup=38,
    ),
    "乖离率": ImplementationSpec(
        "legacy_expression", "100*(close-ts_mean(close,6))/(ts_mean(close,6)+1e-12)", ("close",), warmup=6,
    ),
    "变动速率": ImplementationSpec(
        "legacy_expression", "100*(close-ts_delay(close,12))/(ts_delay(close,12)+1e-12)", ("close",), warmup=12,
    ),
    "相对强弱指标": ImplementationSpec("legacy_expression", "ta_rsi(close,14)", ("close",), warmup=14),
    "商品通道指标": ImplementationSpec(
        "legacy_expression",
        "((high+low+close)/3-ts_mean((high+low+close)/3,14))/(0.015*ts_avedev((high+low+close)/3,14)+1e-12)",
        ("high", "low", "close"), warmup=14,
    ),
    "人气指数": ImplementationSpec(
        "legacy_expression", "100*ts_sum(high-open,26)/(ts_sum(open-low,26)+1e-12)",
        ("open", "high", "low"), warmup=26,
    ),
    "心理线": ImplementationSpec(
        "legacy_expression", "100*ts_mean(rq_as_float(close>ts_delay(close,1)),12)", ("close",), warmup=13,
    ),
    "钱德动量摆动指标": ImplementationSpec(
        "legacy_expression",
        "100*(ts_sum(ts_where(close-ts_delay(close,1)>0,close-ts_delay(close,1),0),14)-ts_sum(ts_where(close-ts_delay(close,1)<0,-1*(close-ts_delay(close,1)),0),14))/(ts_sum(ts_abs(close-ts_delay(close,1)),14)+1e-12)",
        ("close",), warmup=15,
    ),
    "阿隆指数": ImplementationSpec(
        "legacy_expression", "100*(ts_lowday(low,25)-ts_highday(high,25))/25", ("high", "low"), warmup=25,
    ),
    "平均真实波幅": ImplementationSpec("legacy_expression", "ta_atr(high,low,close,14)", ("high", "low", "close"), warmup=15),
    "蔡金波动率指标": ImplementationSpec(
        "legacy_expression", "100*((high-low)-ts_delay(high-low,10))/(ts_delay(high-low,10)+1e-12)",
        ("high", "low"), warmup=10,
    ),
    "多空指数": ImplementationSpec(
        "legacy_expression", "(ts_mean(close,3)+ts_mean(close,6)+ts_mean(close,12)+ts_mean(close,24))/4", ("close",), warmup=24,
    ),
    "垂直水平过滤指标": ImplementationSpec(
        "legacy_expression", "(ts_max(close,28)-ts_min(close,28))/(ts_sum(ts_abs(close-ts_delay(close,1)),28)+1e-12)",
        ("close",), warmup=29,
    ),
    "三重指数移动平均指标": ImplementationSpec(
        "legacy_expression", "100*(ta_tema(close,12)-ts_delay(ta_tema(close,12),1))/(ts_delay(ta_tema(close,12),1)+1e-12)",
        ("close",), warmup=36,
    ),
    "平滑异同均线指标": ImplementationSpec(
        "legacy_expression", "ts_ema(close,12)-ts_ema(close,26)", ("close",), warmup=26,
    ),
    "累积/派发线": ImplementationSpec(
        "legacy_expression", "ts_cumsum(((close-low)-(high-close))/(high-low+1e-12)*volume)",
        ("high", "low", "close", "volume"), warmup=1,
    ),
    "价量趋势指标": ImplementationSpec(
        "legacy_expression", "ts_cumsum((close/ts_delay(close,1)-1)*volume)", ("close", "volume"), warmup=2,
    ),
    "能量指标": ImplementationSpec(
        "legacy_expression", "ts_cumsum(ts_where(close>ts_delay(close,1),volume,ts_where(close<ts_delay(close,1),-1*volume,0)))",
        ("close", "volume"), warmup=2,
    ),
    "成交量摆动指标": ImplementationSpec(
        "legacy_expression", "100*(ts_mean(volume,12)-ts_mean(volume,26))/(ts_mean(volume,26)+1e-12)",
        ("volume",), warmup=26,
    ),
    "对数总市值": ImplementationSpec("log", "log(market_cap_2)", ("market_cap_2",)),
    "对数流通市值": ImplementationSpec(
        "log", "log(a_share_market_val_in_circulation)", ("a_share_market_val_in_circulation",)
    ),
    "换手率": ImplementationSpec(
        "turnover_rate", "volume*close/a_share_market_val_in_circulation",
        ("volume", "close", "a_share_market_val_in_circulation")
    ),
    "成交额": ImplementationSpec("alias:turnover", "turnover", ("turnover",)),
    "最近52周的最高价": ImplementationSpec(
        "rolling_high_ratio", "close / ts_max(close, 250)", ("close",), warmup=250, params={"window": 250}
    ),
    "短期反转": ImplementationSpec(
        "return", "-1 * (close / ts_delay(close, 20) - 1)", ("close",), warmup=20,
        params={"window": 20, "sign": -1.0},
    ),
    "长期反转": ImplementationSpec(
        "return", "-1 * (close / ts_delay(close, 250) - 1)", ("close",), warmup=250,
        params={"window": 250, "sign": -1.0},
    ),
    "时间序列动量": ImplementationSpec(
        "return", "close / ts_delay(close, 252) - 1", ("close",), warmup=252,
        params={"window": 252, "sign": 1.0},
    ),
    "总波动率": ImplementationSpec(
        "rolling_return_stat", "ts_std(log(close/ts_delay(close,1)),20)", ("close",), warmup=21,
        params={"window": 20, "stat": "std"},
    ),
    "月度特质波动率": ImplementationSpec(
        "market_residual_stat", "std(residual(stock_ret ~ market_ret),20)", ("close", "market_ret"),
        warmup=20, params={"window": 20, "stat": "std"},
    ),
    "最大收益率": ImplementationSpec(
        "rolling_return_stat", "ts_max(log(close/ts_delay(close,1)),20)", ("close",), warmup=21,
        params={"window": 20, "stat": "max"},
    ),
    "最小收益率": ImplementationSpec(
        "rolling_return_stat", "ts_min(log(close/ts_delay(close,1)),20)", ("close",), warmup=21,
        params={"window": 20, "stat": "min"},
    ),
    "总偏度": ImplementationSpec(
        "rolling_return_stat", "ts_skew(log(close/ts_delay(close,1)),20)", ("close",), warmup=21,
        params={"window": 20, "stat": "skew"},
    ),
    "负偏度系数": ImplementationSpec(
        "rolling_return_stat", "-ts_skew(log(close/ts_delay(close,1)),20)", ("close",), warmup=21,
        params={"window": 20, "stat": "skew", "sign": -1.0},
    ),
    "高频已实现偏度": ImplementationSpec(
        "intraday_return_stat", "intraday_skew(log(close/ts_delay(close,1)))", ("close",),
        input_frequency="1m", params={"stat": "skew"},
    ),
    "高频已实现峰度": ImplementationSpec(
        "intraday_return_stat", "intraday_kurtosis(log(close/ts_delay(close,1)))", ("close",),
        input_frequency="1m", params={"stat": "kurtosis"},
    ),
    "已实现波动率": ImplementationSpec(
        "intraday_return_measure", "sum(log(close/lag(close))**2)", ("close",),
        input_frequency="1m", params={"measure": "realized_variance"},
    ),
    "高频上行波动占比": ImplementationSpec(
        "intraday_return_measure", "sum(r_1m**2 where r_1m>0)/sum(r_1m**2)", ("close",),
        input_frequency="1m", params={"measure": "up_share"},
    ),
    "高频下行波动占比": ImplementationSpec(
        "intraday_return_measure", "sum(r_1m**2 where r_1m<0)/sum(r_1m**2)", ("close",),
        input_frequency="1m", params={"measure": "down_share"},
    ),
    "上下行波动率不对称性": ImplementationSpec(
        "intraday_return_measure", "(RV_up-RV_down)/RV", ("close",),
        input_frequency="1m", params={"measure": "asymmetry"},
    ),
    "上行已实现波动率": ImplementationSpec(
        "intraday_semivariance", "sum(r_1m**2 where r_1m>0)", ("close",), input_frequency="1m",
        params={"side": "up"},
    ),
    "下行已实现波动率": ImplementationSpec(
        "intraday_semivariance", "sum(r_1m**2 where r_1m<=0)", ("close",), input_frequency="1m",
        params={"side": "down"},
    ),
    "尾盘成交额占比": ImplementationSpec(
        "intraday_tail_share", "sum(turnover,last30m)/sum(turnover,day)", ("turnover",),
        input_frequency="1m", params={"field": "turnover", "minutes": 30},
    ),
    "尾盘成交占比": ImplementationSpec(
        "intraday_tail_share", "sum(volume,last30m)/sum(volume,day)", ("volume",),
        input_frequency="1m", params={"field": "volume", "minutes": 30},
    ),
    "分钟成交额方差": ImplementationSpec(
        "intraday_reduce", "intraday_var(turnover)", ("turnover",), input_frequency="1m",
        params={"field": "turnover", "stat": "var"},
    ),
    "日内收益率": ImplementationSpec(
        "daily_intraday_return", "close/open-1", ("open", "close"),
    ),
    "隔夜收益率": ImplementationSpec(
        "overnight_return", "open/lag(close,1)-1", ("open", "close"), warmup=2,
    ),
    "尾盘半小时收益率": ImplementationSpec(
        "intraday_period_return", "close(day_end)/close(day_end-30m)-1", ("close",),
        input_frequency="1m", params={"minutes": 30},
    ),
    "日内最大回撤": ImplementationSpec(
        "intraday_max_drawdown", "min(close/cummax(close)-1)", ("close",), input_frequency="1m",
    ),
    "成交量占比峰度": ImplementationSpec(
        "intraday_volume_share_stat", "kurtosis(volume/sum(volume))", ("volume",),
        input_frequency="1m", params={"stat": "kurtosis"},
    ),
    "成交量占比偏度": ImplementationSpec(
        "intraday_volume_share_stat", "skew(volume/sum(volume))", ("volume",),
        input_frequency="1m", params={"stat": "skew"},
    ),
    "成交量占比标准差": ImplementationSpec(
        "intraday_volume_share_stat", "std(volume/sum(volume))", ("volume",),
        input_frequency="1m", params={"stat": "std"},
    ),
    "分钟成交额自相关性": ImplementationSpec(
        "intraday_autocorr", "corr(turnover,lag(turnover,1))", ("turnover",),
        input_frequency="1m", params={"field": "turnover", "lag": 1},
    ),
    "绝对收益与成交量相关性": ImplementationSpec(
        "intraday_return_turnover_corr", "corr(abs(r_1m),turnover)", ("close", "turnover"),
        input_frequency="1m", params={"return_lag": 0, "turnover_lag": 0},
    ),
    "绝对收益与滞后成交量相关性": ImplementationSpec(
        "intraday_return_turnover_corr", "corr(abs(r_1m),lag(turnover,1))", ("close", "turnover"),
        input_frequency="1m", params={"return_lag": 0, "turnover_lag": 1},
    ),
    "滞后绝对收益与成交量相关性": ImplementationSpec(
        "intraday_return_turnover_corr", "corr(lag(abs(r_1m),1),turnover)", ("close", "turnover"),
        input_frequency="1m", params={"return_lag": 1, "turnover_lag": 0},
    ),
    "量谷相对加权价格": ImplementationSpec(
        "intraday_volume_valley_price",
        "mean(VWAP(volume<=mean20_same_slot+std20_same_slot)/VWAP(all),20)",
        ("close", "volume"), input_frequency="1m", warmup=40,
        notes="过去20日同时点阈值仅使用当前日之前的数据",
    ),
    "CSAD模型(基于时序回归变形)": ImplementationSpec(
        "daily_peer_csad",
        "-(std(CSAD,20)/std(r_focal,20))/(std(CSAD,120)/std(r_focal,120))",
        ("close",), warmup=120, params={"metric": "ratio"},
    ),
    "CSAD 模型(基于“小型”板块)": ImplementationSpec(
        "daily_peer_csad", "-mean(zscore(CSAD_peer_mean,120),20)",
        ("close",), warmup=120, params={"metric": "small"},
    ),
    "CSAD 模型(基于板块中地位)": ImplementationSpec(
        "daily_peer_csad", "-mean(zscore(CSAD_focal_center,120),20)",
        ("close",), warmup=120, params={"metric": "position"},
    ),
    "基于换手率的注意力溢出": ImplementationSpec(
        "daily_attention_spillover", "mean(peer_monthly_turnover)-monthly_turnover",
        ("close", "volume", "a_share_market_val_in_circulation", "first_industry_code"),
        warmup=20, params={"metric": "turnover"}, status=ImplementationStatus.PROXY,
        notes="代理实现：当前lab仅有申万一级行业，替代原定义的中信一级行业；行业内仍严格按3:4:3市值分组",
    ),
    "基于异常收益的注意力溢出": ImplementationSpec(
        "daily_attention_spillover", "mean(peer_monthly_squared_abnormal_return)-monthly_squared_abnormal_return",
        ("close", "volume", "a_share_market_val_in_circulation", "first_industry_code"),
        warmup=20, params={"metric": "abnormal_return"}, status=ImplementationStatus.PROXY,
        notes="代理实现：当前lab仅有申万一级行业，替代原定义的中信一级行业；行业内仍严格按3:4:3市值分组",
    ),
    "趋势资金相对均价因子": ImplementationSpec(
        "intraday_trend_fund_metric", "neutralize(mean(VWAP(trend)/VWAP(all)-1,20),cap,industry)",
        ("close", "volume", "turnover", "a_share_market_val_in_circulation", "first_industry_code"),
        input_frequency="1m", warmup=25, params={"metric": "relative_vwap"},
        status=ImplementationStatus.PROXY,
        notes="代理实现：量价计算为原式，行业中性化因当前lab仅有申万一级而替代中信一级",
    ),
    "净支撑成交量因子": ImplementationSpec(
        "intraday_trend_fund_metric", "neutralize(mean((support_volume-resistance_volume)/float_shares,20),cap,industry)",
        ("close", "volume", "turnover", "a_share_market_val_in_circulation", "first_industry_code"),
        input_frequency="1m", warmup=20, params={"metric": "net_support"},
        status=ImplementationStatus.PROXY,
        notes="代理实现：量价计算为原式，行业中性化因当前lab仅有申万一级而替代中信一级",
    ),
    "趋势资金净支撑量因子": ImplementationSpec(
        "intraday_trend_fund_metric", "neutralize(mean((trend_support-trend_resistance)/float_shares,20),cap,industry)",
        ("close", "volume", "turnover", "a_share_market_val_in_circulation", "first_industry_code"),
        input_frequency="1m", warmup=25, params={"metric": "trend_net_support"},
        status=ImplementationStatus.PROXY,
        notes="代理实现：量价计算为原式，行业中性化因当前lab仅有申万一级而替代中信一级",
    ),
    "极端跟随行为_比值因子": ImplementationSpec(
        "intraday_trend_fund_metric", "neutralize(mean(mean(follower_amount/trend_amount),20),cap,industry)",
        ("close", "volume", "turnover", "a_share_market_val_in_circulation", "first_industry_code"),
        input_frequency="1m", warmup=25, params={"metric": "follower_ratio"},
        status=ImplementationStatus.PROXY,
        notes="代理实现：当前lab以申万一级替代中信一级；且日历第4步的‘相关系数’与第3步比值定义冲突，按比值因子语义取每日比值均值后20日均值",
    ),
    "极端跟随行为_相关性因子": ImplementationSpec(
        "intraday_trend_fund_metric", "neutralize(mean(corr(trend_amount,follower_amount),20),cap,industry)",
        ("close", "volume", "turnover", "a_share_market_val_in_circulation", "first_industry_code"),
        input_frequency="1m", warmup=25, params={"metric": "follower_corr"},
        status=ImplementationStatus.PROXY,
        notes="代理实现：量价相关性计算为原式，行业中性化因当前lab仅有申万一级而替代中信一级",
    ),
}

CALENDAR_SPECS: dict[tuple[int, str], ImplementationSpec] = {
    (2025, "26"): calendar_intraday_technical(
        "calendar_cci",
        "linear_decay20((HLC_last-mean(HLC_1m,240))/(0.015*avedev(HLC_1m,240)))",
    ),
    (2025, "38"): ImplementationSpec(
        "daily_attention_metric",
        "mean((return-median_cross_section(return))**2,20)",
        ("close", "volume"),
        warmup=21,
        params={"metric": "mean_squared_abnormal_return", "window": 20},
    ),
    (2025, "58"): calendar_intraday_technical(
        "calendar_rsi",
        "linear_decay20(100*sum(max(delta(close_1m),0))/sum(abs(delta(close_1m))))",
    ),
    (2025, "88"): calendar_intraday_technical(
        "calendar_br",
        "linear_decay20(100*sum(max(high_t-close_t-1,0))/sum(max(close_t-1-low_t-1,0)))",
    ),
    (2025, "104"): calendar_intraday_technical(
        "calendar_elder",
        "linear_decay20(((high_last-EMA240(close))-(low_last-EMA240(close)))/close_last)",
    ),
    (2025, "118"): calendar_intraday_technical(
        "calendar_psy",
        "linear_decay20(mean(close_1m>lag(close_1m)))",
    ),
    (2025, "132"): calendar_intraday_technical(
        "calendar_volume_ratio",
        "linear_decay20(sum(volume where delta(close)>0)/sum(volume where delta(close)<=0))",
    ),
    (2025, "156"): calendar_intraday_technical(
        "calendar_chande",
        "linear_decay20(100*(sum(up_delta)-sum(down_delta))/(sum(up_delta)+sum(down_delta)))",
    ),
    (2025, "170"): calendar_intraday_technical(
        "calendar_srmi",
        "linear_decay20(mean(delta(close_1m)/max(close_t,close_t-1)))",
    ),
    (2025, "219"): calendar_intraday_technical(
        "calendar_hurst_rs",
        "linear_decay20((max(cumsum(close-mean(close)))-min(cumsum(close-mean(close))))/std(close))",
    ),
    (2025, "240"): calendar_intraday_technical(
        "calendar_price_mad",
        "linear_decay20(mean(abs(close_1m-mean(close_1m))))",
    ),
    (2025, "251"): TITLE_SPECS["CSAD模型(基于时序回归变形)"],
    (2025, "259"): calendar_intraday_technical(
        "calendar_mfi",
        "linear_decay20(100-100/(1+positive_typical_price_flow/negative_typical_price_flow))",
    ),
    (2025, "268"): calendar_intraday_technical(
        "calendar_money_flow",
        "linear_decay20(sum(((high_1m+low_1m+close_1m)/3)*volume_1m))",
    ),
    (2025, "297"): calendar_intraday_technical(
        "calendar_pvt",
        "linear_decay20(sum((close_1m/lag(close_1m)-1)*volume_1m))",
    ),
    (2025, "305"): calendar_intraday_technical(
        "calendar_emv",
        "linear_decay20(mean(delta((high_1m+low_1m)/2)/(volume_1m/(high_1m-low_1m))))",
    ),
    (2025, "358"): TITLE_SPECS["基于 PB 的前景价值 TK"],
    (2026, "1.1.12"): ImplementationSpec(
        "daily_panic_metric", "0.5*(mean(panic*return,20)+std(panic*return,20)); panic=(abs(r-rm)+.1)/(abs(r)+abs(rm))",
        ("close", "market_ret"), warmup=21,
        params={"metric": "base", "numerator_offset": True},
        notes="2026日历原式将0.1置于分子，按该年度文本单独实现",
    ),
    (2026, "1.1.11"): ImplementationSpec(
        "daily_salience_return", "cov(normalize(0.7**rank(salience)),return),monthly; salience=(abs(r)+abs(median)-r-median)/.1",
        ("close",), warmup=21, params={"variant": "2026"},
        notes="2026日历凸显性系数公式与2024版本不同，按年度文本单独实现",
    ),
}


for title, field in FIELD_ALIASES.items():
    TITLE_SPECS.setdefault(title, alias(field))


def find_spec(title: str, *, year: int | None = None, calendar_id: str = "") -> ImplementationSpec | None:
    if year is not None:
        specific = CALENDAR_SPECS.get((year, calendar_id))
        if specific is not None:
            return specific
    normalized = title.strip().replace("（", "(").replace("）", ")")
    for candidate in (title.strip(), normalized):
        if candidate in TITLE_SPECS:
            return TITLE_SPECS[candidate]
    return None
