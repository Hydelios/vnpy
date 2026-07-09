"""Curated cross-section supreme factor library.

This file records round5 relaxed data-dimension candidates from
``factorfarm/data_dimension_factor_archive_20260518``. These factors
passed the relaxed coverage>=0.70/no rolling20-min pre-screen; only a
subset survived corr<=0.80. They did not pass the full strict SOP yet.
"""

from __future__ import annotations

import polars as pl

from .baseAlphaStrategy import BaseAlphaStrategy


class AlphaCsSupremeFactors(BaseAlphaStrategy):
    """截面精选因子入口；当前包含 round5 放宽口径候选。"""

    CATEGORY = "AlphaCSSupreme"
    VERSION = "v20260518_round5_relaxed"

    SELECTION_NOTE = (
        "round5 relaxed pre-screen: coverage>=0.70, no rolling20_min hard gate; "
        "48 pre-corr candidates, 8 kept after corr<=0.80. Full strict SOP pass count is 0."
    )
    SOURCE_FILE = "playground/alpha_research/factorfarm/data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv"
    CORR_SELECTED_FILE = "playground/alpha_research/factorfarm/data_dimension_factor_archive_20260518/round5_selected_coverage070_no_rollmin_corr080.csv"
    ROUND5_RELAXED_PRESELECT_COUNT = 48
    ROUND5_RELAXED_CORR_KEEP_COUNT = 8
    SOURCE_COUNTS = {'cross_source_regime_switch': 36, 'cross_source_composite': 12}
    SUB_CATEGORY_COUNTS = {'CompositeNearMissGrowthGapSkewPullbackFlip': 5,
 'CompositeNearMissGrowthGapSkewRegimeFlip': 5,
 'CompositeNearMissGrowthGapSkew': 5,
 'CompositeNearMissGrowthGapSkewMeltupFlip': 5,
 'CompositeNearMissQualityDefensePullbackFlip': 7,
 'CompositeNearMissQualityDefense': 7,
 'CompositeNearMissQualityDefenseRegimeFlip': 7,
 'CompositeNearMissQualityDefenseMeltupFlip': 7}

    CANDIDATE_SPECS: list[dict[str, object]] = [{'base_name': 'ai_cs_combo_nearmiss_922_growth_gap_skew_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewPullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': True,
  'eval_robust_ic': 0.0168838463,
  'eval_robust_icir': 0.3724072095,
  'coverage_mean': 0.7573272538,
  'eval_rolling20_aligned_min': -0.0387736607,
  'rank_turnover_1d': 0.0158753544,
  'idea': '成长/效率 surprise、000922.SSE gap 结构和行业残差偏度反转共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_growth_gap_skew_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), ts_where(idx922_r20 < -0.08, '
              'ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': True,
  'eval_robust_ic': 0.0180728913,
  'eval_robust_icir': 0.3735191717,
  'coverage_mean': 0.7572911017,
  'eval_rolling20_aligned_min': -0.055373489,
  'rank_turnover_1d': 0.0212177653,
  'idea': '成长/效率 surprise、000300.SSE gap 结构和行业残差偏度反转共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_growth_gap_skew_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), ts_where(idx922_r20 < -0.08, '
              'ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0169041116,
  'eval_robust_icir': 0.3729163632,
  'coverage_mean': 0.7573272538,
  'eval_rolling20_aligned_min': -0.0480616947,
  'rank_turnover_1d': 0.0213344209,
  'idea': '成长/效率 surprise、000922.SSE gap 结构和行业残差偏度反转共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_growth_gap_skew_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewPullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0176905929,
  'eval_robust_icir': 0.3645543928,
  'coverage_mean': 0.7572911017,
  'eval_rolling20_aligned_min': -0.055373489,
  'rank_turnover_1d': 0.0157738719,
  'idea': '成长/效率 surprise、000300.SSE gap 结构和行业残差偏度反转共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_905_growth_gap_skew_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewPullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0165946527,
  'eval_robust_icir': 0.3508955036,
  'coverage_mean': 0.7566635356,
  'eval_rolling20_aligned_min': -0.045830588,
  'rank_turnover_1d': 0.0156787243,
  'idea': '成长/效率 surprise、000905.SSE gap 结构和行业残差偏度反转共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_growth_gap_skew_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkew',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0165115126,
  'eval_robust_icir': 0.3630972863,
  'coverage_mean': 0.7573272538,
  'eval_rolling20_aligned_min': -0.0592605813,
  'rank_turnover_1d': 0.0125917085,
  'idea': '训练期近似有效的成长 surprise、效率 surprise、000922.SSE 高开耗尽原方向和行业残差偏度反转的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_growth_gap_skew_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0178522199,
  'eval_robust_icir': 0.3683349435,
  'coverage_mean': 0.7572911017,
  'eval_rolling20_aligned_min': -0.055373489,
  'rank_turnover_1d': 0.0179432519,
  'idea': '成长/效率 surprise、000300.SSE gap 结构和行业残差偏度反转共振；只在指数 20 日极端上涨时翻转，处理 melt-up '
          '中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_905_growth_gap_skew_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), ts_where(idx922_r20 < -0.08, '
              'ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0167699126,
  'eval_robust_icir': 0.3550661719,
  'coverage_mean': 0.7566635356,
  'eval_rolling20_aligned_min': -0.0451335584,
  'rank_turnover_1d': 0.0211274233,
  'idea': '成长/效率 surprise、000905.SSE gap 结构和行业残差偏度反转共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_growth_gap_skew_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkew',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0174699215,
  'eval_robust_icir': 0.3594149286,
  'coverage_mean': 0.7572911017,
  'eval_rolling20_aligned_min': -0.055373489,
  'rank_turnover_1d': 0.0124993585,
  'idea': '训练期近似有效的成长 surprise、效率 surprise、000300.SSE 高开耗尽原方向和行业残差偏度反转的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_905_growth_gap_skew_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkew',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0165981108,
  'eval_robust_icir': 0.3509776398,
  'coverage_mean': 0.7566635356,
  'eval_rolling20_aligned_min': -0.045830588,
  'rank_turnover_1d': 0.0124024469,
  'idea': '训练期近似有效的成长 surprise、效率 surprise、000905.SSE 高开耗尽原方向和行业残差偏度反转的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_growth_gap_skew_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewPullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.016743968,
  'eval_robust_icir': 0.3562638302,
  'coverage_mean': 0.7571248099,
  'eval_rolling20_aligned_min': -0.0504441401,
  'rank_turnover_1d': 0.0157608911,
  'idea': '成长/效率 surprise、000906.SSE gap 结构和行业残差偏度反转共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_growth_gap_skew_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), ts_where(idx922_r20 < -0.08, '
              'ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0169641186,
  'eval_robust_icir': 0.3615562666,
  'coverage_mean': 0.7571248099,
  'eval_rolling20_aligned_min': -0.0504441401,
  'rank_turnover_1d': 0.021207314,
  'idea': '成长/效率 surprise、000906.SSE gap 结构和行业残差偏度反转共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_growth_gap_skew_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0165317779,
  'eval_robust_icir': 0.363601848,
  'coverage_mean': 0.7573272538,
  'eval_rolling20_aligned_min': -0.0592605813,
  'rank_turnover_1d': 0.0180507731,
  'idea': '成长/效率 surprise、000922.SSE gap 结构和行业残差偏度反转共振；只在指数 20 日极端上涨时翻转，处理 melt-up '
          '中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': True,
  'eval_robust_ic': 0.0162252066,
  'eval_robust_icir': 0.3713626574,
  'coverage_mean': 0.7589642159,
  'eval_rolling20_aligned_min': -0.0564649359,
  'rank_turnover_1d': 0.0163308084,
  'idea': '质量 surprise、000922.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_905_growth_gap_skew_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0167733707,
  'eval_robust_icir': 0.3551486316,
  'coverage_mean': 0.7566635356,
  'eval_rolling20_aligned_min': -0.0451335584,
  'rank_turnover_1d': 0.0178511459,
  'idea': '成长/效率 surprise、000905.SSE gap 结构和行业残差偏度反转共振；只在指数 20 日极端上涨时翻转，处理 melt-up '
          '中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefense',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0161473151,
  'eval_robust_icir': 0.3646117638,
  'coverage_mean': 0.7589642159,
  'eval_rolling20_aligned_min': -0.0627428778,
  'rank_turnover_1d': 0.0131004639,
  'idea': '训练期近似有效的质量 surprise、000922.SSE 高开耗尽原方向和指数下跌日抗跌残差的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_852_growth_gap_skew_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkew',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0166190044,
  'eval_robust_icir': 0.3399491332,
  'coverage_mean': 0.756896797,
  'eval_rolling20_aligned_min': -0.0582226108,
  'rank_turnover_1d': 0.0123167839,
  'idea': '训练期近似有效的成长 surprise、效率 surprise、000852.SSE 高开耗尽原方向和行业残差偏度反转的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_852_growth_gap_skew_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewPullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0165421722,
  'eval_robust_icir': 0.3381971486,
  'coverage_mean': 0.756896797,
  'eval_rolling20_aligned_min': -0.0582226108,
  'rank_turnover_1d': 0.0155849801,
  'idea': '成长/效率 surprise、000852.SSE gap 结构和行业残差偏度反转共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_852_growth_gap_skew_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0166813806,
  'eval_robust_icir': 0.3413735511,
  'coverage_mean': 0.756896797,
  'eval_rolling20_aligned_min': -0.0582625258,
  'rank_turnover_1d': 0.0177597459,
  'idea': '成长/效率 surprise、000852.SSE gap 结构和行业残差偏度反转共振；只在指数 20 日极端上涨时翻转，处理 melt-up '
          '中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_growth_gap_skew_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkew',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0164034599,
  'eval_robust_icir': 0.3481298781,
  'coverage_mean': 0.7571248099,
  'eval_rolling20_aligned_min': -0.0504441401,
  'rank_turnover_1d': 0.0124844275,
  'idea': '训练期近似有效的成长 surprise、效率 surprise、000906.SSE 高开耗尽原方向和行业残差偏度反转的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_growth_gap_skew_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0166236105,
  'eval_robust_icir': 0.3533816452,
  'coverage_mean': 0.7571248099,
  'eval_rolling20_aligned_min': -0.0504441401,
  'rank_turnover_1d': 0.0179308504,
  'idea': '成长/效率 surprise、000906.SSE gap 结构和行业残差偏度反转共振；只在指数 20 日极端上涨时翻转，处理 melt-up '
          '中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_pullback_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': True,
  'eval_robust_ic': 0.0169982032,
  'eval_robust_icir': 0.3277693532,
  'coverage_mean': 0.7125215655,
  'eval_rolling20_aligned_min': -0.0914939309,
  'rank_turnover_1d': 0.0244383477,
  'idea': '质量 surprise、000906.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 120))), ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0159342049,
  'eval_robust_icir': 0.3619007639,
  'coverage_mean': 0.7589642159,
  'eval_rolling20_aligned_min': -0.0715992962,
  'rank_turnover_1d': 0.0218019132,
  'idea': '质量 surprise、000922.SSE gap 结构和指数下跌日抗跌共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_852_growth_gap_skew_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), ts_where(idx922_r20 < -0.08, '
              'ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, '
              '120)) / (ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx852_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + (-1 * cs_rank(ts_skew((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissGrowthGapSkewRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0166045485,
  'eval_robust_icir': 0.3396192848,
  'coverage_mean': 0.756896797,
  'eval_rolling20_aligned_min': -0.0582625258,
  'rank_turnover_1d': 0.0210279431,
  'idea': '成长/效率 surprise、000852.SSE gap 结构和行业残差偏度反转共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_extreme_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 60))), ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.01665893,
  'eval_robust_icir': 0.3200376311,
  'coverage_mean': 0.7125215655,
  'eval_rolling20_aligned_min': -0.0914939309,
  'rank_turnover_1d': 0.028581297,
  'idea': '质量 surprise、000906.SSE gap 结构和指数下跌日抗跌共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_extreme_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 60))), ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 60))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': True,
  'eval_robust_ic': 0.0144461189,
  'eval_robust_icir': 0.3154004663,
  'coverage_mean': 0.7125088919,
  'eval_rolling20_aligned_min': -0.069363083,
  'rank_turnover_1d': 0.0289399102,
  'idea': '质量 surprise、000922.SSE gap 结构和指数下跌日抗跌共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 120))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 120)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': True,
  'eval_robust_ic': 0.0158563134,
  'eval_robust_icir': 0.3551966226,
  'coverage_mean': 0.7589642159,
  'eval_rolling20_aligned_min': -0.0715992962,
  'rank_turnover_1d': 0.0185715705,
  'idea': '质量 surprise、000922.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_quality_gap_defensive_idx922_extreme_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx300_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx300_r1, 0), 60))), ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 60))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.01681968,
  'eval_robust_icir': 0.3197684417,
  'coverage_mean': 0.7126455203,
  'eval_rolling20_aligned_min': -0.0912369098,
  'rank_turnover_1d': 0.0286519937,
  'idea': '质量 surprise、000300.SSE gap 结构和指数下跌日抗跌共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_quality_gap_defensive_idx922_pullback_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 60)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0169604881,
  'eval_robust_icir': 0.3235980958,
  'coverage_mean': 0.7126455203,
  'eval_rolling20_aligned_min': -0.0912369098,
  'rank_turnover_1d': 0.0245150942,
  'idea': '质量 surprise、000300.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_w60',
  'expr_tpl': 'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefense',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0166891385,
  'eval_robust_icir': 0.321095845,
  'coverage_mean': 0.7125215655,
  'eval_rolling20_aligned_min': -0.0914939309,
  'rank_turnover_1d': 0.0208036676,
  'idea': '训练期近似有效的质量 surprise、000906.SSE 高开耗尽原方向和指数下跌日抗跌残差的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_meltup_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 60))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 60)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0163498653,
  'eval_robust_icir': 0.3134205173,
  'coverage_mean': 0.7125215655,
  'eval_rolling20_aligned_min': -0.0914939309,
  'rank_turnover_1d': 0.024946617,
  'idea': '质量 surprise、000906.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_meltup_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 60))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 60)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0141230541,
  'eval_robust_icir': 0.3072517384,
  'coverage_mean': 0.7125088919,
  'eval_rolling20_aligned_min': -0.069363083,
  'rank_turnover_1d': 0.0253112409,
  'idea': '质量 surprise、000922.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_quality_gap_defensive_idx922_meltup_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx300_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx300_r1, 0), 60))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + '
              '1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), 60)) + '
              'ts_mean(ts_where(idx300_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx300_r1, 0), 60)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0165242276,
  'eval_robust_icir': 0.3137854779,
  'coverage_mean': 0.7126455203,
  'eval_rolling20_aligned_min': -0.0912369098,
  'rank_turnover_1d': 0.0250059925,
  'idea': '质量 surprise、000300.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_quality_gap_defensive_w60',
  'expr_tpl': 'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 60))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefense',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0166650358,
  'eval_robust_icir': 0.317583165,
  'coverage_mean': 0.7126455203,
  'eval_rolling20_aligned_min': -0.0912369098,
  'rank_turnover_1d': 0.020869093,
  'idea': '训练期近似有效的质量 surprise、000300.SSE 高开耗尽原方向和指数下跌日抗跌残差的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_pullback_flip_w60',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 60)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0141580613,
  'eval_robust_icir': 0.3085056691,
  'coverage_mean': 0.7125088919,
  'eval_rolling20_aligned_min': -0.0499020475,
  'rank_turnover_1d': 0.0248146299,
  'idea': '质量 surprise、000922.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 120)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0162311821,
  'eval_robust_icir': 0.346625846,
  'coverage_mean': 0.7587043503,
  'eval_rolling20_aligned_min': -0.056646846,
  'rank_turnover_1d': 0.0161167774,
  'idea': '质量 surprise、000906.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 120))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefense',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': True,
  'eval_robust_ic': 0.0159761174,
  'eval_robust_icir': 0.3436420869,
  'coverage_mean': 0.7587043503,
  'eval_rolling20_aligned_min': -0.056646846,
  'rank_turnover_1d': 0.012892087,
  'idea': '训练期近似有效的质量 surprise、000906.SSE 高开耗尽原方向和指数下跌日抗跌残差的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_quality_gap_defensive_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 120)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0162684744,
  'eval_robust_icir': 0.3374723007,
  'coverage_mean': 0.75896234,
  'eval_rolling20_aligned_min': -0.0699348194,
  'rank_turnover_1d': 0.0161417387,
  'idea': '质量 surprise、000300.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_quality_gap_defensive_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 120))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefense',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0162171455,
  'eval_robust_icir': 0.3369408352,
  'coverage_mean': 0.75896234,
  'eval_rolling20_aligned_min': -0.0699348194,
  'rank_turnover_1d': 0.0129145002,
  'idea': '训练期近似有效的质量 surprise、000300.SSE 高开耗尽原方向和指数下跌日抗跌残差的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 120))), ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx906_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0158165237,
  'eval_robust_icir': 0.3394551307,
  'coverage_mean': 0.7587043503,
  'eval_rolling20_aligned_min': -0.056646846,
  'rank_turnover_1d': 0.021573741,
  'idea': '质量 surprise、000906.SSE gap 结构和指数下跌日抗跌共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_w60',
  'expr_tpl': 'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx922_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 60))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefense',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': True,
  'eval_robust_ic': 0.0138349965,
  'eval_robust_icir': 0.3008204455,
  'coverage_mean': 0.7125088919,
  'eval_rolling20_aligned_min': -0.051812881,
  'rank_turnover_1d': 0.0211859606,
  'idea': '训练期近似有效的质量 surprise、000922.SSE 高开耗尽原方向和指数下跌日抗跌残差的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 120))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 120)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.015561459,
  'eval_robust_icir': 0.336211201,
  'coverage_mean': 0.7587043503,
  'eval_rolling20_aligned_min': -0.056646846,
  'rank_turnover_1d': 0.0183490496,
  'idea': '质量 surprise、000906.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_quality_gap_defensive_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx300_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx300_r1, 0), 120))), ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx300_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 0), 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0161064039,
  'eval_robust_icir': 0.3324849269,
  'coverage_mean': 0.75896234,
  'eval_rolling20_aligned_min': -0.0699348194,
  'rank_turnover_1d': 0.0215938445,
  'idea': '质量 surprise、000300.SSE gap 结构和指数下跌日抗跌共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_300_quality_gap_defensive_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx300_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx300_r1, 0), 120))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx300_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx300_r1, 0), 120)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.016055075,
  'eval_robust_icir': 0.3319504453,
  'coverage_mean': 0.75896234,
  'eval_rolling20_aligned_min': -0.0699348194,
  'rank_turnover_1d': 0.018366605,
  'idea': '质量 surprise、000300.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_905_quality_gap_defensive_idx922_meltup_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx905_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx905_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx905_r1, 0), 120))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx905_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx905_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx905_r1, 0), 120)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0149179618,
  'eval_robust_icir': 0.3124207898,
  'coverage_mean': 0.7582198182,
  'eval_rolling20_aligned_min': -0.0679545484,
  'rank_turnover_1d': 0.0182548016,
  'idea': '质量 surprise、000905.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。'},
 {'base_name': 'ai_cs_combo_nearmiss_905_quality_gap_defensive_w120',
  'expr_tpl': 'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx905_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx905_r1, 0), 120))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefense',
  'source': 'cross_source_composite',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0150422387,
  'eval_robust_icir': 0.3139903364,
  'coverage_mean': 0.7582198182,
  'eval_rolling20_aligned_min': -0.0679545484,
  'rank_turnover_1d': 0.0127906138,
  'idea': '训练期近似有效的质量 surprise、000905.SSE 高开耗尽原方向和指数下跌日抗跌残差的共振。'},
 {'base_name': 'ai_cs_combo_nearmiss_905_quality_gap_defensive_idx922_extreme_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) '
              '+ 1e-12)) + cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - '
              'idx905_r1) > 0, (high - close) / (high - low + 1e-12), 0), 120)) + '
              'ts_mean(ts_where(idx905_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx905_r1, 0), 120))), ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx905_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx905_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx905_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx905_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx905_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx905_r1, 0), 120))))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefenseRegimeFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0147570008,
  'eval_robust_icir': 0.3096373137,
  'coverage_mean': 0.7582198182,
  'eval_rolling20_aligned_min': -0.0679545484,
  'rank_turnover_1d': 0.0214768965,
  'idea': '质量 surprise、000905.SSE gap 结构和指数下跌日抗跌共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。'},
 {'base_name': 'ai_cs_combo_nearmiss_905_quality_gap_defensive_idx922_pullback_flip_w120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx905_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx905_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx905_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx905_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, '
              '120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx905_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - idx905_r1, 0), 120)))',
  'param_grid': {},
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'source': 'cross_source_regime_switch',
  'origin': 'data_dimension_factor_archive_20260518/round5_candidates_coverage070_no_rollmin_corr_candidates.csv',
  'selection_level': 'round5_relaxed_pre_corr_coverage070_no_rollmin',
  'keep_after_corr_080': False,
  'eval_robust_ic': 0.0148812777,
  'eval_robust_icir': 0.3112003121,
  'coverage_mean': 0.7582198182,
  'eval_rolling20_aligned_min': -0.0679545484,
  'rank_turnover_1d': 0.0160127077,
  'idea': '质量 surprise、000905.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。'}]

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        interval: str = "1d",
        enable_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            interval=interval,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
        self.register_all()

    def register_all(self) -> None:
        self._add_specs(self.CANDIDATE_SPECS)

    @staticmethod
    def _normalize_base_name(base_name: str) -> str:
        if base_name.startswith("alpha_cs_supreme_"):
            return base_name
        if base_name.startswith("alpha_cs_"):
            return f"alpha_cs_supreme_{base_name[9:]}"
        return f"alpha_cs_supreme_{base_name}"

    def _add_specs(self, specs: list[dict[str, object]]) -> None:
        for spec in specs:
            self.add_parametric_feature(
                base_name=self._normalize_base_name(str(spec["base_name"])),
                expr_tpl=str(spec["expr_tpl"]),
                param_grid=spec.get("param_grid") or {},
                category=self.CATEGORY,
                sub_category=str(spec["sub_category"]),
            )
