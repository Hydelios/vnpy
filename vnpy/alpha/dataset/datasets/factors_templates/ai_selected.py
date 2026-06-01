from __future__ import annotations

"""Selected AI cross-sectional factors.

This template contains the 286 corr-deduplicated AI CS factors used in the
latest model run, plus the previous strict 6 selected factors renamed from
``ai_`` to ``ai_cs_``. Specs that only differ by window are represented as one
parametric factor family with a wider ``param_grid``.
"""

import polars as pl

from .baseAlphaStrategy import BaseAlphaStrategy


class AiSelectedFactors(BaseAlphaStrategy):
    CATEGORY = "AiCsSelectedFactors"

    FACTOR_NAMES_CORR_SELECTED_286 = ['ai_cs_combo_nearmiss_922_growth_gap_skew_idx922_pullback_flip_120',
 'ai_cs_combo_nearmiss_300_growth_gap_skew_idx922_extreme_flip_120',
 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_pullback_flip_120',
 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_pullback_flip_60',
 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_meltup_flip_120',
 'ai_cs_daily_v3_liq_impact_range_turn_crowd_flip_10',
 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_meltup_flip_60',
 'ai_cs_combo_nearmiss_905_quality_gap_defensive_120',
 'ai_cs_r07_amihud_like_efficiency_w3',
 'ai_cs_daily_v4_beta_term_corr_drop_value_neglect_40',
 'ai_cs_daily_v5_regime_cross_stock_low_corr_dividend_20',
 'ai_cs_daily_v4_capacity_size_turn_capacity_jump_guard_10',
 'ai_cs_idx922_relative_gap_exhaustion_3',
 'ai_cs_r174_body_vwap_confirm_high_turn_clv_gate_w40',
 'ai_cs_idx905_relative_gap_exhaustion_3',
 'ai_cs_daily_v3_recognition_margin_growth_vol_compress_20',
 'ai_cs_idx906_relative_gap_exhaustion_3',
 'ai_cs_daily_v5_regime_cross_stock_low_corr_small_value_20',
 'ai_cs_r79_open_panic_recover_plain_downside_gate_w20',
 'ai_cs_r123_range_turnover_confirm_plain_body_confirm_w40',
 'ai_cs_daily_v4_beta_term_def_beta_cash_stable_20',
 'ai_cs_r40_range_expansion_risk_raw_level_x_ret_confirm_w40',
 'ai_cs_daily_v3_recognition_efficiency_growth_vol_compress_20',
 'ai_cs_daily_v4_beta_term_corr_rise_risk_neglect_120',
 'ai_cs_ind_residual_skew_reversal_20',
 'ai_cs_daily_v3_peer_market_cap_resid_accel_20',
 'ai_cs_daily_v5_regime_cross_stock_low_corr_capacity_cash_20',
 'ai_cs_r196_return_efficiency_low_turn_abs_turn_confirm_w20',
 'ai_cs_r234_body_push_quality_high_range_gap_shock_gate_w40',
 'ai_cs_fund_net_profit_growth_change_120',
 'ai_cs_fund_operating_revenue_growth_surprise_120',
 'ai_cs_daily_v4_capacity_capacity_range_discount_jump_guard_20',
 'ai_cs_r73_vol_regime_switch_range_adj_w10',
 'ai_cs_daily_v5_cash_beta_ocf_low_beta_low_idio_10',
 'ai_cs_combo_nearmiss_300_growth_lowvol_gap_idx922_pullback_flip_120',
 'ai_cs_fund_total_asset_turnover_surprise_240',
 'ai_cs_daily_v4_cash_defense_dividend_lowvol_jump_guard_20',
 'ai_cs_fund_operating_revenue_growth_change_120',
 'ai_cs_daily_v4_capacity_capacity_rebalance_demand_jump_guard_20',
 'ai_cs_b_pv_turnover_return_corr_neg_w40',
 'ai_cs_daily_v4_beta_term_corr_drop_value_stable_60',
 'ai_cs_r160_volume_absorb_high_turn_mean_w20',
 'ai_cs_fund_return_on_asset_surprise_120',
 'ai_cs_daily_v3_path_overnight_intraday_conflict_level_60',
 'ai_cs_ind_range_adjusted_relative_strength_3',
 'ai_cs_daily_v5_value_capacity_ep_size_capacity_low_idio_10',
 'ai_cs_daily_v5_impact_beta_ret_impact_value_low_idio_10',
 'ai_cs_daily_v4_cash_defense_dividend_lowvol_stable_20',
 'ai_cs_daily_v3_recognition_fcf_value_low_corr_10',
 'ai_cs_daily_v4_capacity_size_turn_capacity_stable_10',
 'ai_cs_daily_v4_capacity_value_impact_buffer_carry_10',
 'ai_cs_r129_open_panic_recover_high_turn_gap_confirm_w20',
 'ai_cs_combo_nearmiss_300_growth_lowvol_gap_idx922_meltup_flip_120',
 'ai_cs_daily_v3_peer_roe_resid_accel_120',
 'ai_cs_fund_gross_profit_margin_surprise_120',
 'ai_cs_fund_account_receivable_turnover_rate_surprise_240',
 'ai_cs_daily_v5_impact_beta_ret_impact_value_price_lag_20',
 'ai_cs_r22_turnover_without_price_stable_rank_20',
 'ai_cs_r175_gap_vwap_disagree_high_turn_abs_volume_confirm_w40',
 'ai_cs_fund_ep_change_120',
 'ai_cs_daily_v6_stable_qv_debt_cover_value_low_idio_10',
 'ai_cs_r110_volume_absorb_plain_mean_w40',
 'ai_cs_fund_net_profit_growth_surprise_120',
 'ai_cs_daily_v4_cash_defense_dividend_lowvol_price_lag_20',
 'ai_cs_r16_gap_abs_range_mean_rank_w20',
 'ai_cs_r83_vwap_turn_crowd_plain_body_range_confirm_w40',
 'ai_cs_combo_quality_growth_low_idio_vol_60',
 'ai_cs_idx852_residual_skew_reversal_40',
 'ai_cs_dv01_gap_follow_with_turnover_w40',
 'ai_cs_daily_v3_recognition_undervalued_growth_score_stability_10',
 'ai_cs_ind_residual_path_efficiency_3',
 'ai_cs_daily_v3_liq_turn_cluster_scaled_20',
 'ai_cs_daily_v3_recognition_undervalued_growth_vol_compress_10',
 'ai_cs_fund_return_on_equity_change_120',
 'ai_cs_daily_v3_path_vwap_stretch_fade_jump_damped_60',
 'ai_cs_daily_v3_regime_ind_up_vol_compress_40',
 'ai_cs_r160_volume_absorb_high_turn_abs_turn_confirm_w40',
 'ai_cs_daily_v3_liq_wet_exhaustion_crowd_flip_20',
 'ai_cs_daily_v3_path_gap_up_fade_level_60',
 'ai_cs_r169_winrate_memory_high_turn_body_confirm_w40',
 'ai_cs_ind_idio_vol_compression_3',
 'ai_cs_dv02_participation_volatility_penalty_40',
 'ai_cs_b_liq_turnover_compression_40',
 'ai_cs_fund_free_cash_flow_company_indz_price_confirm_240',
 'ai_cs_daily_v4_capacity_capacity_range_discount_stable_20',
 'ai_cs_fund_net_profit_margin_surprise_120',
 'ai_cs_fund_operating_cash_flow_indz_price_confirm_120',
 'ai_cs_daily_v3_peer_gross_margin_resid_accel_120',
 'ai_cs_daily_v3_regime_turn_hot_range_eff_120',
 'ai_cs_r09_trend_volume_confirmed_path_w40',
 'ai_cs_daily_v3_recognition_fcf_value_vwap_absorb_20',
 'ai_cs_daily_v3_recognition_efficiency_growth_score_delta_60',
 'ai_cs_r231_vwap_revert_high_range_range_adj_w40',
 'ai_cs_r83_vwap_turn_crowd_plain_lower_shadow_confirm_w40',
 'ai_cs_daily_v6_rotation_small_corr_drop_dividend_60',
 'ai_cs_r99_overextension_reversal_plain_raw_level_x_core_body_range_corr_w40',
 'ai_cs_fund_market_cap_indz_price_divergence_240',
 'ai_cs_r57_liquidity_replenish_raw_level_x_abs_vwap_confirm_w40',
 'ai_cs_ind_vwap_crowded_reversal_40',
 'ai_cs_r127_gap_follow_path_high_turn_abs_turn_confirm_w20',
 'ai_cs_r263_pv_diverge_volume_high_range_body_confirm_w40',
 'ai_cs_daily_v5_cash_beta_div_low_beta_low_idio_10',
 'ai_cs_fund_market_cap_indz_low_idio_vol_20',
 'ai_cs_daily_v4_beta_term_beta_stability_quality_change_60',
 'ai_cs_daily_v3_recognition_fcf_value_score_delta_10',
 'ai_cs_fund_operating_revenue_growth_indz_change_120',
 'ai_cs_daily_v3_path_gap_down_repair_shape_60',
 'ai_cs_r67_turnover_stability_dryup_confirm_10',
 'ai_cs_daily_v3_recognition_capital_efficiency_score_delta_60',
 'ai_cs_fund_pb_indz_change_20',
 'ai_cs_fund_account_receivable_turnover_rate_indz_change_120',
 'ai_cs_daily_v6_accounting_smooth_roe_growth_spread_stable_60',
 'ai_cs_r115_amihud_impact_plain_downside_gate_w40',
 'ai_cs_daily_v4_cap_alloc_roic_proxy_industry_disagree_20',
 'ai_cs_fund_gross_profit_growth_change_120',
 'ai_cs_daily_v3_path_volume_close_support_shape_40',
 'ai_cs_fund_ocf_to_debt_stable_level_60',
 'ai_cs_r171_clv_volume_confirm_high_turn_body_range_confirm_w40',
 'ai_cs_combo_fcf_value_low_idio_vol_60',
 'ai_cs_daily_v3_peer_dividend_industry_repair_10',
 'ai_cs_daily_v5_value_capacity_ep_size_capacity_corr_drop_40',
 'ai_cs_r30_vwap_premium_decay_raw_level_x_abs_turn_confirm_w20',
 'ai_cs_r17_turnover_stability_stable_rank_10',
 'ai_cs_daily_v3_liq_gap_per_turn_jump_damped_10',
 'ai_cs_ind_leader_follow_strength_10',
 'ai_cs_r83_vwap_turn_crowd_plain_max_gap_w20',
 'ai_cs_r64_crowded_profit_unwind_raw_level_x_abs_vwap_confirm_w20',
 'ai_cs_fund_ev_to_ebitda_change_60',
 'ai_cs_daily_v3_liq_upper_pressure_per_turn_scaled_60',
 'ai_cs_fund_free_cash_flow_equity_change_60',
 'ai_cs_fund_free_cash_flow_equity_surprise_120',
 'ai_cs_r22_turnover_volatility_discount_mean_raw_w10',
 'ai_cs_r129_open_panic_recover_high_turn_raw_level_x_abs_vwap_confirm_w20',
 'ai_cs_r160_volume_absorb_high_turn_range_adj_w40',
 'ai_cs_daily_v3_recognition_efficiency_growth_low_corr_10',
 'ai_cs_r129_open_panic_recover_high_turn_max_gap_w20',
 'ai_cs_fund_pe_change_60',
 'ai_cs_daily_v4_cash_defense_dividend_neglect_stable_20',
 'ai_cs_fund_sp_indz_change_60',
 'ai_cs_daily_v5_value_capacity_bp_float_capacity_price_lag_10',
 'ai_cs_r06_gap_low_reclaim_quality_w40',
 'ai_cs_idx905_ind_dual_residual_strength_10',
 'ai_cs_daily_v6_repair_low_corr_repair_div_10',
 'ai_cs_daily_v3_peer_dividend_scarce_quality_20',
 'ai_cs_fund_market_cap_indz_change_20',
 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_gap_body_spread_40',
 'ai_cs_daily_v3_peer_revenue_growth_resid_accel_120',
 'ai_cs_daily_v3_liq_turn_vs_volume_level_20',
 'ai_cs_daily_v4_capacity_ep_capacity_jump_guard_40',
 'ai_cs_daily_v3_recognition_fcf_value_score_stability_10',
 'ai_cs_daily_v4_beta_term_beta_stability_value_change_20',
 'ai_cs_r80_close_pressure_vwap_plain_lower_shadow_confirm_w20',
 'ai_cs_fund_free_cash_flow_company_change_60',
 'ai_cs_idx852_residual_illiq_penalty_10',
 'ai_cs_r175_gap_vwap_disagree_high_turn_abs_range_confirm_w40',
 'ai_cs_fund_net_profit_margin_change_60',
 'ai_cs_fund_ep_indz_change_60',
 'ai_cs_daily_v3_path_gap_up_fade_shape_120',
 'ai_cs_fund_ps_indz_change_20',
 'ai_cs_r160_volume_absorb_high_turn_body_confirm_w40',
 'ai_cs_fund_gross_profit_growth_surprise_240',
 'ai_cs_r252_new_low_reclaim_high_range_clv_gate_w40',
 'ai_cs_daily_v4_beta_term_def_beta_cash_change_40',
 'ai_cs_fund_inventory_turnover_indz_low_idio_vol_20',
 'ai_cs_r260_volume_absorb_high_range_lower_shadow_confirm_w40',
 'ai_cs_r175_gap_vwap_disagree_high_turn_max_gap_w20',
 'ai_cs_daily_v3_peer_dividend_resid_accel_10',
 'ai_cs_daily_v4_acc_cycle_inventory_margin_quality_unrecognized_20',
 'ai_cs_fund_total_asset_turnover_indz_change_120',
 'ai_cs_daily_v6_liq_sponsor_dry_repair_small_value_10',
 'ai_cs_r128_open_absorb_clv_high_turn_max_gap_w20',
 'ai_cs_fund_pe_indz_change_60',
 'ai_cs_daily_v4_acc_cycle_gross_net_gap_control_jump_guard_60',
 'ai_cs_daily_v5_impact_beta_ret_impact_value_ind_down_20',
 'ai_cs_daily_v3_path_overnight_intraday_conflict_shape_120',
 'ai_cs_daily_v3_regime_turn_hot_vol_compress_10',
 'ai_cs_daily_v3_path_volume_upper_pressure_jump_damped_60',
 'ai_cs_r179_open_panic_recover_low_turn_body_range_confirm_w20',
 'ai_cs_daily_v4_cash_defense_dividend_debt_cover_price_lag_20',
 'ai_cs_daily_v4_beta_term_def_beta_dividend_stable_20',
 'ai_cs_daily_v3_path_gap_up_fade_hot_flip_60',
 'ai_cs_daily_v3_path_weakday_vwap_recover_selfz_120',
 'ai_cs_r228_open_absorb_clv_high_range_abs_turn_confirm_w20',
 'ai_cs_daily_v3_liq_range_elasticity_crowd_flip_120',
 'ai_cs_daily_v5_impact_beta_gap_impact_dividend_corr_drop_60',
 'ai_cs_r17_turnover_stability_neg_mean_rank_20',
 'ai_cs_r229_open_panic_recover_high_range_abs_turn_confirm_w20',
 'ai_cs_daily_v4_cash_defense_ocf_debt_value_jump_guard_40',
 'ai_cs_r24_vwap_reward_risk_dev_rank_w40',
 'ai_cs_r22_turnover_without_price_dev_raw_40',
 'ai_cs_daily_v5_accounting_price_inventory_margin_price_lag_vwap_10',
 'ai_cs_daily_v5_cash_beta_fcf_roa_beta_low_idio_10',
 'ai_cs_daily_v6_accounting_smooth_rev_asset_light_stable_10',
 'ai_cs_fund_gross_profit_margin_change_60',
 'ai_cs_daily_v4_acc_cycle_receivable_inventory_stability_risk_persist_10',
 'ai_cs_fund_book_to_market_indz_change_20',
 'ai_cs_daily_v3_liq_price_elasticity_level_10',
 'ai_cs_daily_v3_liq_gap_liquidity_absorb_scaled_60',
 'ai_cs_r37_lower_shadow_absorb_mean_change_w40',
 'ai_cs_r58_turnover_spike_exhaust_raw_level_x_abs_turn_confirm_w20',
 'ai_cs_r141_range_expand_risk_high_turn_min_gap_w20',
 'ai_cs_daily_v6_stable_qv_turnover_quality_vwap_absorb_10',
 'ai_cs_ind_upper_shadow_relative_exhaustion_3',
 'ai_cs_daily_v4_capacity_capacity_beta_shadow_stable_10',
 'ai_cs_fund_ev_to_ebitda_indz_change_20',
 'ai_cs_fund_operating_cash_flow_surprise_240',
 'ai_cs_daily_v3_liq_gap_per_turn_scaled_10',
 'ai_cs_daily_v4_capacity_capacity_vwap_absorb_carry_20',
 'ai_cs_daily_v3_path_vwap_stretch_fade_shape_40',
 'ai_cs_daily_v5_value_capacity_ep_size_capacity_price_lag_10',
 'ai_cs_daily_v3_path_overnight_intraday_conflict_hot_flip_60',
 'ai_cs_fund_free_cash_flow_company_surprise_240',
 'ai_cs_fund_ev_to_ebitda_surprise_120',
 'ai_cs_daily_v6_stable_qv_debt_cover_value_price_lag_10',
 'ai_cs_r252_new_low_reclaim_high_range_abs_turn_confirm_w20',
 'ai_cs_fund_a_share_market_val_in_circulation_indz_level_240',
 'ai_cs_daily_v3_path_vwap_stretch_fade_hot_flip_40',
 'ai_cs_r137_upper_reject_high_turn_min_gap_w20',
 'ai_cs_r214_pv_diverge_turnover_low_turn_raw_level_x_core_ret_turn_confirm_w40',
 'ai_cs_daily_v6_accounting_smooth_sales_velocity_stable_40',
 'ai_cs_r110_volume_absorb_plain_body_confirm_w20',
 'ai_cs_r50_new_high_breakout_raw_level_x_abs_range_confirm_w20',
 'ai_cs_daily_v4_cash_defense_fcf_equity_quality_price_lag_20',
 'ai_cs_fund_inventory_turnover_indz_price_confirm_20',
 'ai_cs_fund_return_on_equity_indz_change_120',
 'ai_cs_combo_efficiency_growth_low_turnover_calm_60',
 'ai_cs_daily_v3_liq_turn_cluster_jump_damped_20',
 'ai_cs_r87_upper_reject_plain_min_gap_w40',
 'ai_cs_daily_v5_impact_beta_ret_impact_value_corr_drop_60',
 'ai_cs_daily_v5_impact_beta_body_impact_quality_corr_drop_60',
 'ai_cs_fund_return_on_asset_indz_price_divergence_240',
 'ai_cs_idx852_residual_vol_breakout_60',
 'ai_cs_daily_v4_acc_cycle_asset_growth_efficiency_neglect_20',
 'ai_cs_idx852_beta_instability_penalty_5',
 'ai_cs_r22_crowded_gap_fail_neg_mean_raw_w40',
 'ai_cs_daily_v3_liq_turn_accel_ind_stress_10',
 'ai_cs_daily_v4_cap_alloc_value_investment_gap_industry_disagree_20',
 'ai_cs_daily_v3_path_closepos_consistency_selfz_10',
 'ai_cs_daily_v5_accounting_price_inventory_margin_price_lag_lag_ind_10',
 'ai_cs_fund_pb_indz_price_divergence_240',
 'ai_cs_daily_v6_repair_low_corr_repair_book_20',
 'ai_cs_daily_v4_acc_cycle_inventory_margin_quality_risk_persist_10',
 'ai_cs_daily_v3_liq_upper_liquidity_mismatch_crowd_flip_120',
 'ai_cs_r266_turnover_stability_high_range_raw_level_20',
 'ai_cs_r06_gap_open_liquidity_penalty_w20',
 'ai_cs_daily_v4_capacity_capacity_rebalance_supply_jump_guard_20',
 'ai_cs_daily_v4_cash_defense_fcf_value_company_price_lag_20',
 'ai_cs_daily_v3_path_gap_down_repair_level_60',
 'ai_cs_r258_liquidity_replenish_high_range_gap_shock_gate_w20',
 'ai_cs_combo_undervalued_growth_low_turnover_calm_60',
 'ai_cs_r17_volume_stability_neg_mean_rank_w10',
 'ai_cs_combo_fcf_value_low_turnover_calm_60',
 'ai_cs_daily_v6_rotation_small_corr_drop_margin_value_40',
 'ai_cs_daily_v3_regime_idx_hot_intraday_follow_60',
 'ai_cs_idx906_residual_vol_compression_3',
 'ai_cs_r17_turnover_stability_dev_rank_20',
 'ai_cs_daily_v5_impact_beta_wet_exhaust_beta_beta_stable_60',
 'ai_cs_daily_v5_accounting_price_gross_net_control_price_lag_vwap_10',
 'ai_cs_daily_v4_beta_term_corr_drop_value_change_60',
 'ai_cs_daily_v3_path_volume_upper_pressure_selfz_20',
 'ai_cs_daily_v3_regime_idx_hot_vwap_recover_20',
 'ai_cs_daily_v3_liq_turn_cluster_crowd_flip_10',
 'ai_cs_daily_v3_path_body_range_tension_selfz_10',
 'ai_cs_r137_upper_reject_high_turn_abs_range_confirm_w20',
 'ai_cs_daily_v4_cap_alloc_funding_resilience_industry_disagree_20',
 'ai_cs_fund_pe_surprise_240',
 'ai_cs_daily_v3_path_overnight_intraday_conflict_selfz_10',
 'ai_cs_daily_v3_liq_gap_per_turn_level_10',
 'ai_cs_daily_v3_regime_range_quiet_vol_compress_60',
 'ai_cs_daily_v3_regime_idx_cold_gap_fade_60',
 'ai_cs_daily_v6_repair_idx_lag_growth_120',
 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_clv_mean_3',
 'ai_cs_daily_v3_regime_idx_hot_ind_resid_40',
 'ai_cs_daily_v4_capacity_quality_capacity_price_lag_10',
 'ai_cs_r135_body_failure_high_turn_body_range_confirm_w40',
 'ai_cs_r20_range_break_hold_stable_rank_w5',
 'ai_cs_daily_v5_regime_cross_industry_beats_small_small_value_10',
 'ai_cs_daily_v5_impact_beta_support_impact_value_ind_down_10',
 'ai_cs_daily_v3_path_vwap_stretch_fade_selfz_10',
 'ai_cs_r78_open_absorb_clv_plain_max_gap_w40',
 'ai_cs_idx300_down_market_resilience_20',
 'ai_cs_daily_v4_capacity_value_impact_buffer_jump_guard_20',
 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_body_range_turn_40',
 'ai_cs_daily_v3_path_volume_close_support_selfz_10',
 'ai_cs_daily_v3_regime_turn_hot_intraday_follow_40',
 'ai_cs_daily_v3_liq_turn_vs_volume_crowd_flip_60']

    FACTOR_NAMES_STRICT_SELECTED_6 = ['ai_cs_r160_volume_absorb_high_turn_body_range_confirm_w10',
 'ai_cs_r160_volume_absorb_high_turn_mean_w10',
 'ai_cs_r260_volume_absorb_high_range_body_range_confirm_w10',
 'ai_cs_r102_new_low_reclaim_plain_abs_range_confirm_w10',
 'ai_cs_r102_new_low_reclaim_plain_raw_level_x_abs_range_confirm_w10',
 'ai_cs_r174_body_vwap_confirm_high_turn_min_gap_w20']

    FACTOR_NAMES_SELECTED = ['ai_cs_combo_nearmiss_922_growth_gap_skew_idx922_pullback_flip_120',
 'ai_cs_combo_nearmiss_300_growth_gap_skew_idx922_extreme_flip_120',
 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_pullback_flip_120',
 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_pullback_flip_60',
 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_meltup_flip_120',
 'ai_cs_daily_v3_liq_impact_range_turn_crowd_flip_10',
 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_meltup_flip_60',
 'ai_cs_combo_nearmiss_905_quality_gap_defensive_120',
 'ai_cs_r07_amihud_like_efficiency_w3',
 'ai_cs_daily_v4_beta_term_corr_drop_value_neglect_40',
 'ai_cs_daily_v5_regime_cross_stock_low_corr_dividend_20',
 'ai_cs_daily_v4_capacity_size_turn_capacity_jump_guard_10',
 'ai_cs_idx922_relative_gap_exhaustion_3',
 'ai_cs_r174_body_vwap_confirm_high_turn_clv_gate_w40',
 'ai_cs_idx905_relative_gap_exhaustion_3',
 'ai_cs_daily_v3_recognition_margin_growth_vol_compress_20',
 'ai_cs_idx906_relative_gap_exhaustion_3',
 'ai_cs_daily_v5_regime_cross_stock_low_corr_small_value_20',
 'ai_cs_r79_open_panic_recover_plain_downside_gate_w20',
 'ai_cs_r123_range_turnover_confirm_plain_body_confirm_w40',
 'ai_cs_daily_v4_beta_term_def_beta_cash_stable_20',
 'ai_cs_r40_range_expansion_risk_raw_level_x_ret_confirm_w40',
 'ai_cs_daily_v3_recognition_efficiency_growth_vol_compress_20',
 'ai_cs_daily_v4_beta_term_corr_rise_risk_neglect_120',
 'ai_cs_ind_residual_skew_reversal_20',
 'ai_cs_daily_v3_peer_market_cap_resid_accel_20',
 'ai_cs_daily_v5_regime_cross_stock_low_corr_capacity_cash_20',
 'ai_cs_r196_return_efficiency_low_turn_abs_turn_confirm_w20',
 'ai_cs_r234_body_push_quality_high_range_gap_shock_gate_w40',
 'ai_cs_fund_net_profit_growth_change_120',
 'ai_cs_fund_operating_revenue_growth_surprise_120',
 'ai_cs_daily_v4_capacity_capacity_range_discount_jump_guard_20',
 'ai_cs_r73_vol_regime_switch_range_adj_w10',
 'ai_cs_daily_v5_cash_beta_ocf_low_beta_low_idio_10',
 'ai_cs_combo_nearmiss_300_growth_lowvol_gap_idx922_pullback_flip_120',
 'ai_cs_fund_total_asset_turnover_surprise_240',
 'ai_cs_daily_v4_cash_defense_dividend_lowvol_jump_guard_20',
 'ai_cs_fund_operating_revenue_growth_change_120',
 'ai_cs_daily_v4_capacity_capacity_rebalance_demand_jump_guard_20',
 'ai_cs_b_pv_turnover_return_corr_neg_w40',
 'ai_cs_daily_v4_beta_term_corr_drop_value_stable_60',
 'ai_cs_r160_volume_absorb_high_turn_mean_w20',
 'ai_cs_fund_return_on_asset_surprise_120',
 'ai_cs_daily_v3_path_overnight_intraday_conflict_level_60',
 'ai_cs_ind_range_adjusted_relative_strength_3',
 'ai_cs_daily_v5_value_capacity_ep_size_capacity_low_idio_10',
 'ai_cs_daily_v5_impact_beta_ret_impact_value_low_idio_10',
 'ai_cs_daily_v4_cash_defense_dividend_lowvol_stable_20',
 'ai_cs_daily_v3_recognition_fcf_value_low_corr_10',
 'ai_cs_daily_v4_capacity_size_turn_capacity_stable_10',
 'ai_cs_daily_v4_capacity_value_impact_buffer_carry_10',
 'ai_cs_r129_open_panic_recover_high_turn_gap_confirm_w20',
 'ai_cs_combo_nearmiss_300_growth_lowvol_gap_idx922_meltup_flip_120',
 'ai_cs_daily_v3_peer_roe_resid_accel_120',
 'ai_cs_fund_gross_profit_margin_surprise_120',
 'ai_cs_fund_account_receivable_turnover_rate_surprise_240',
 'ai_cs_daily_v5_impact_beta_ret_impact_value_price_lag_20',
 'ai_cs_r22_turnover_without_price_stable_rank_20',
 'ai_cs_r175_gap_vwap_disagree_high_turn_abs_volume_confirm_w40',
 'ai_cs_fund_ep_change_120',
 'ai_cs_daily_v6_stable_qv_debt_cover_value_low_idio_10',
 'ai_cs_r110_volume_absorb_plain_mean_w40',
 'ai_cs_fund_net_profit_growth_surprise_120',
 'ai_cs_daily_v4_cash_defense_dividend_lowvol_price_lag_20',
 'ai_cs_r16_gap_abs_range_mean_rank_w20',
 'ai_cs_r83_vwap_turn_crowd_plain_body_range_confirm_w40',
 'ai_cs_combo_quality_growth_low_idio_vol_60',
 'ai_cs_idx852_residual_skew_reversal_40',
 'ai_cs_dv01_gap_follow_with_turnover_w40',
 'ai_cs_daily_v3_recognition_undervalued_growth_score_stability_10',
 'ai_cs_ind_residual_path_efficiency_3',
 'ai_cs_daily_v3_liq_turn_cluster_scaled_20',
 'ai_cs_daily_v3_recognition_undervalued_growth_vol_compress_10',
 'ai_cs_fund_return_on_equity_change_120',
 'ai_cs_daily_v3_path_vwap_stretch_fade_jump_damped_60',
 'ai_cs_daily_v3_regime_ind_up_vol_compress_40',
 'ai_cs_r160_volume_absorb_high_turn_abs_turn_confirm_w40',
 'ai_cs_daily_v3_liq_wet_exhaustion_crowd_flip_20',
 'ai_cs_daily_v3_path_gap_up_fade_level_60',
 'ai_cs_r169_winrate_memory_high_turn_body_confirm_w40',
 'ai_cs_ind_idio_vol_compression_3',
 'ai_cs_dv02_participation_volatility_penalty_40',
 'ai_cs_b_liq_turnover_compression_40',
 'ai_cs_fund_free_cash_flow_company_indz_price_confirm_240',
 'ai_cs_daily_v4_capacity_capacity_range_discount_stable_20',
 'ai_cs_fund_net_profit_margin_surprise_120',
 'ai_cs_fund_operating_cash_flow_indz_price_confirm_120',
 'ai_cs_daily_v3_peer_gross_margin_resid_accel_120',
 'ai_cs_daily_v3_regime_turn_hot_range_eff_120',
 'ai_cs_r09_trend_volume_confirmed_path_w40',
 'ai_cs_daily_v3_recognition_fcf_value_vwap_absorb_20',
 'ai_cs_daily_v3_recognition_efficiency_growth_score_delta_60',
 'ai_cs_r231_vwap_revert_high_range_range_adj_w40',
 'ai_cs_r83_vwap_turn_crowd_plain_lower_shadow_confirm_w40',
 'ai_cs_daily_v6_rotation_small_corr_drop_dividend_60',
 'ai_cs_r99_overextension_reversal_plain_raw_level_x_core_body_range_corr_w40',
 'ai_cs_fund_market_cap_indz_price_divergence_240',
 'ai_cs_r57_liquidity_replenish_raw_level_x_abs_vwap_confirm_w40',
 'ai_cs_ind_vwap_crowded_reversal_40',
 'ai_cs_r127_gap_follow_path_high_turn_abs_turn_confirm_w20',
 'ai_cs_r263_pv_diverge_volume_high_range_body_confirm_w40',
 'ai_cs_daily_v5_cash_beta_div_low_beta_low_idio_10',
 'ai_cs_fund_market_cap_indz_low_idio_vol_20',
 'ai_cs_daily_v4_beta_term_beta_stability_quality_change_60',
 'ai_cs_daily_v3_recognition_fcf_value_score_delta_10',
 'ai_cs_fund_operating_revenue_growth_indz_change_120',
 'ai_cs_daily_v3_path_gap_down_repair_shape_60',
 'ai_cs_r67_turnover_stability_dryup_confirm_10',
 'ai_cs_daily_v3_recognition_capital_efficiency_score_delta_60',
 'ai_cs_fund_pb_indz_change_20',
 'ai_cs_fund_account_receivable_turnover_rate_indz_change_120',
 'ai_cs_daily_v6_accounting_smooth_roe_growth_spread_stable_60',
 'ai_cs_r115_amihud_impact_plain_downside_gate_w40',
 'ai_cs_daily_v4_cap_alloc_roic_proxy_industry_disagree_20',
 'ai_cs_fund_gross_profit_growth_change_120',
 'ai_cs_daily_v3_path_volume_close_support_shape_40',
 'ai_cs_fund_ocf_to_debt_stable_level_60',
 'ai_cs_r171_clv_volume_confirm_high_turn_body_range_confirm_w40',
 'ai_cs_combo_fcf_value_low_idio_vol_60',
 'ai_cs_daily_v3_peer_dividend_industry_repair_10',
 'ai_cs_daily_v5_value_capacity_ep_size_capacity_corr_drop_40',
 'ai_cs_r30_vwap_premium_decay_raw_level_x_abs_turn_confirm_w20',
 'ai_cs_r17_turnover_stability_stable_rank_10',
 'ai_cs_daily_v3_liq_gap_per_turn_jump_damped_10',
 'ai_cs_ind_leader_follow_strength_10',
 'ai_cs_r83_vwap_turn_crowd_plain_max_gap_w20',
 'ai_cs_r64_crowded_profit_unwind_raw_level_x_abs_vwap_confirm_w20',
 'ai_cs_fund_ev_to_ebitda_change_60',
 'ai_cs_daily_v3_liq_upper_pressure_per_turn_scaled_60',
 'ai_cs_fund_free_cash_flow_equity_change_60',
 'ai_cs_fund_free_cash_flow_equity_surprise_120',
 'ai_cs_r22_turnover_volatility_discount_mean_raw_w10',
 'ai_cs_r129_open_panic_recover_high_turn_raw_level_x_abs_vwap_confirm_w20',
 'ai_cs_r160_volume_absorb_high_turn_range_adj_w40',
 'ai_cs_daily_v3_recognition_efficiency_growth_low_corr_10',
 'ai_cs_r129_open_panic_recover_high_turn_max_gap_w20',
 'ai_cs_fund_pe_change_60',
 'ai_cs_daily_v4_cash_defense_dividend_neglect_stable_20',
 'ai_cs_fund_sp_indz_change_60',
 'ai_cs_daily_v5_value_capacity_bp_float_capacity_price_lag_10',
 'ai_cs_r06_gap_low_reclaim_quality_w40',
 'ai_cs_idx905_ind_dual_residual_strength_10',
 'ai_cs_daily_v6_repair_low_corr_repair_div_10',
 'ai_cs_daily_v3_peer_dividend_scarce_quality_20',
 'ai_cs_fund_market_cap_indz_change_20',
 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_gap_body_spread_40',
 'ai_cs_daily_v3_peer_revenue_growth_resid_accel_120',
 'ai_cs_daily_v3_liq_turn_vs_volume_level_20',
 'ai_cs_daily_v4_capacity_ep_capacity_jump_guard_40',
 'ai_cs_daily_v3_recognition_fcf_value_score_stability_10',
 'ai_cs_daily_v4_beta_term_beta_stability_value_change_20',
 'ai_cs_r80_close_pressure_vwap_plain_lower_shadow_confirm_w20',
 'ai_cs_fund_free_cash_flow_company_change_60',
 'ai_cs_idx852_residual_illiq_penalty_10',
 'ai_cs_r175_gap_vwap_disagree_high_turn_abs_range_confirm_w40',
 'ai_cs_fund_net_profit_margin_change_60',
 'ai_cs_fund_ep_indz_change_60',
 'ai_cs_daily_v3_path_gap_up_fade_shape_120',
 'ai_cs_fund_ps_indz_change_20',
 'ai_cs_r160_volume_absorb_high_turn_body_confirm_w40',
 'ai_cs_fund_gross_profit_growth_surprise_240',
 'ai_cs_r252_new_low_reclaim_high_range_clv_gate_w40',
 'ai_cs_daily_v4_beta_term_def_beta_cash_change_40',
 'ai_cs_fund_inventory_turnover_indz_low_idio_vol_20',
 'ai_cs_r260_volume_absorb_high_range_lower_shadow_confirm_w40',
 'ai_cs_r175_gap_vwap_disagree_high_turn_max_gap_w20',
 'ai_cs_daily_v3_peer_dividend_resid_accel_10',
 'ai_cs_daily_v4_acc_cycle_inventory_margin_quality_unrecognized_20',
 'ai_cs_fund_total_asset_turnover_indz_change_120',
 'ai_cs_daily_v6_liq_sponsor_dry_repair_small_value_10',
 'ai_cs_r128_open_absorb_clv_high_turn_max_gap_w20',
 'ai_cs_fund_pe_indz_change_60',
 'ai_cs_daily_v4_acc_cycle_gross_net_gap_control_jump_guard_60',
 'ai_cs_daily_v5_impact_beta_ret_impact_value_ind_down_20',
 'ai_cs_daily_v3_path_overnight_intraday_conflict_shape_120',
 'ai_cs_daily_v3_regime_turn_hot_vol_compress_10',
 'ai_cs_daily_v3_path_volume_upper_pressure_jump_damped_60',
 'ai_cs_r179_open_panic_recover_low_turn_body_range_confirm_w20',
 'ai_cs_daily_v4_cash_defense_dividend_debt_cover_price_lag_20',
 'ai_cs_daily_v4_beta_term_def_beta_dividend_stable_20',
 'ai_cs_daily_v3_path_gap_up_fade_hot_flip_60',
 'ai_cs_daily_v3_path_weakday_vwap_recover_selfz_120',
 'ai_cs_r228_open_absorb_clv_high_range_abs_turn_confirm_w20',
 'ai_cs_daily_v3_liq_range_elasticity_crowd_flip_120',
 'ai_cs_daily_v5_impact_beta_gap_impact_dividend_corr_drop_60',
 'ai_cs_r17_turnover_stability_neg_mean_rank_20',
 'ai_cs_r229_open_panic_recover_high_range_abs_turn_confirm_w20',
 'ai_cs_daily_v4_cash_defense_ocf_debt_value_jump_guard_40',
 'ai_cs_r24_vwap_reward_risk_dev_rank_w40',
 'ai_cs_r22_turnover_without_price_dev_raw_40',
 'ai_cs_daily_v5_accounting_price_inventory_margin_price_lag_vwap_10',
 'ai_cs_daily_v5_cash_beta_fcf_roa_beta_low_idio_10',
 'ai_cs_daily_v6_accounting_smooth_rev_asset_light_stable_10',
 'ai_cs_fund_gross_profit_margin_change_60',
 'ai_cs_daily_v4_acc_cycle_receivable_inventory_stability_risk_persist_10',
 'ai_cs_fund_book_to_market_indz_change_20',
 'ai_cs_daily_v3_liq_price_elasticity_level_10',
 'ai_cs_daily_v3_liq_gap_liquidity_absorb_scaled_60',
 'ai_cs_r37_lower_shadow_absorb_mean_change_w40',
 'ai_cs_r58_turnover_spike_exhaust_raw_level_x_abs_turn_confirm_w20',
 'ai_cs_r141_range_expand_risk_high_turn_min_gap_w20',
 'ai_cs_daily_v6_stable_qv_turnover_quality_vwap_absorb_10',
 'ai_cs_ind_upper_shadow_relative_exhaustion_3',
 'ai_cs_daily_v4_capacity_capacity_beta_shadow_stable_10',
 'ai_cs_fund_ev_to_ebitda_indz_change_20',
 'ai_cs_fund_operating_cash_flow_surprise_240',
 'ai_cs_daily_v3_liq_gap_per_turn_scaled_10',
 'ai_cs_daily_v4_capacity_capacity_vwap_absorb_carry_20',
 'ai_cs_daily_v3_path_vwap_stretch_fade_shape_40',
 'ai_cs_daily_v5_value_capacity_ep_size_capacity_price_lag_10',
 'ai_cs_daily_v3_path_overnight_intraday_conflict_hot_flip_60',
 'ai_cs_fund_free_cash_flow_company_surprise_240',
 'ai_cs_fund_ev_to_ebitda_surprise_120',
 'ai_cs_daily_v6_stable_qv_debt_cover_value_price_lag_10',
 'ai_cs_r252_new_low_reclaim_high_range_abs_turn_confirm_w20',
 'ai_cs_fund_a_share_market_val_in_circulation_indz_level_240',
 'ai_cs_daily_v3_path_vwap_stretch_fade_hot_flip_40',
 'ai_cs_r137_upper_reject_high_turn_min_gap_w20',
 'ai_cs_r214_pv_diverge_turnover_low_turn_raw_level_x_core_ret_turn_confirm_w40',
 'ai_cs_daily_v6_accounting_smooth_sales_velocity_stable_40',
 'ai_cs_r110_volume_absorb_plain_body_confirm_w20',
 'ai_cs_r50_new_high_breakout_raw_level_x_abs_range_confirm_w20',
 'ai_cs_daily_v4_cash_defense_fcf_equity_quality_price_lag_20',
 'ai_cs_fund_inventory_turnover_indz_price_confirm_20',
 'ai_cs_fund_return_on_equity_indz_change_120',
 'ai_cs_combo_efficiency_growth_low_turnover_calm_60',
 'ai_cs_daily_v3_liq_turn_cluster_jump_damped_20',
 'ai_cs_r87_upper_reject_plain_min_gap_w40',
 'ai_cs_daily_v5_impact_beta_ret_impact_value_corr_drop_60',
 'ai_cs_daily_v5_impact_beta_body_impact_quality_corr_drop_60',
 'ai_cs_fund_return_on_asset_indz_price_divergence_240',
 'ai_cs_idx852_residual_vol_breakout_60',
 'ai_cs_daily_v4_acc_cycle_asset_growth_efficiency_neglect_20',
 'ai_cs_idx852_beta_instability_penalty_5',
 'ai_cs_r22_crowded_gap_fail_neg_mean_raw_w40',
 'ai_cs_daily_v3_liq_turn_accel_ind_stress_10',
 'ai_cs_daily_v4_cap_alloc_value_investment_gap_industry_disagree_20',
 'ai_cs_daily_v3_path_closepos_consistency_selfz_10',
 'ai_cs_daily_v5_accounting_price_inventory_margin_price_lag_lag_ind_10',
 'ai_cs_fund_pb_indz_price_divergence_240',
 'ai_cs_daily_v6_repair_low_corr_repair_book_20',
 'ai_cs_daily_v4_acc_cycle_inventory_margin_quality_risk_persist_10',
 'ai_cs_daily_v3_liq_upper_liquidity_mismatch_crowd_flip_120',
 'ai_cs_r266_turnover_stability_high_range_raw_level_20',
 'ai_cs_r06_gap_open_liquidity_penalty_w20',
 'ai_cs_daily_v4_capacity_capacity_rebalance_supply_jump_guard_20',
 'ai_cs_daily_v4_cash_defense_fcf_value_company_price_lag_20',
 'ai_cs_daily_v3_path_gap_down_repair_level_60',
 'ai_cs_r258_liquidity_replenish_high_range_gap_shock_gate_w20',
 'ai_cs_combo_undervalued_growth_low_turnover_calm_60',
 'ai_cs_r17_volume_stability_neg_mean_rank_w10',
 'ai_cs_combo_fcf_value_low_turnover_calm_60',
 'ai_cs_daily_v6_rotation_small_corr_drop_margin_value_40',
 'ai_cs_daily_v3_regime_idx_hot_intraday_follow_60',
 'ai_cs_idx906_residual_vol_compression_3',
 'ai_cs_r17_turnover_stability_dev_rank_20',
 'ai_cs_daily_v5_impact_beta_wet_exhaust_beta_beta_stable_60',
 'ai_cs_daily_v5_accounting_price_gross_net_control_price_lag_vwap_10',
 'ai_cs_daily_v4_beta_term_corr_drop_value_change_60',
 'ai_cs_daily_v3_path_volume_upper_pressure_selfz_20',
 'ai_cs_daily_v3_regime_idx_hot_vwap_recover_20',
 'ai_cs_daily_v3_liq_turn_cluster_crowd_flip_10',
 'ai_cs_daily_v3_path_body_range_tension_selfz_10',
 'ai_cs_r137_upper_reject_high_turn_abs_range_confirm_w20',
 'ai_cs_daily_v4_cap_alloc_funding_resilience_industry_disagree_20',
 'ai_cs_fund_pe_surprise_240',
 'ai_cs_daily_v3_path_overnight_intraday_conflict_selfz_10',
 'ai_cs_daily_v3_liq_gap_per_turn_level_10',
 'ai_cs_daily_v3_regime_range_quiet_vol_compress_60',
 'ai_cs_daily_v3_regime_idx_cold_gap_fade_60',
 'ai_cs_daily_v6_repair_idx_lag_growth_120',
 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_clv_mean_3',
 'ai_cs_daily_v3_regime_idx_hot_ind_resid_40',
 'ai_cs_daily_v4_capacity_quality_capacity_price_lag_10',
 'ai_cs_r135_body_failure_high_turn_body_range_confirm_w40',
 'ai_cs_r20_range_break_hold_stable_rank_w5',
 'ai_cs_daily_v5_regime_cross_industry_beats_small_small_value_10',
 'ai_cs_daily_v5_impact_beta_support_impact_value_ind_down_10',
 'ai_cs_daily_v3_path_vwap_stretch_fade_selfz_10',
 'ai_cs_r78_open_absorb_clv_plain_max_gap_w40',
 'ai_cs_idx300_down_market_resilience_20',
 'ai_cs_daily_v4_capacity_value_impact_buffer_jump_guard_20',
 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_body_range_turn_40',
 'ai_cs_daily_v3_path_volume_close_support_selfz_10',
 'ai_cs_daily_v3_regime_turn_hot_intraday_follow_40',
 'ai_cs_daily_v3_liq_turn_vs_volume_crowd_flip_60',
 'ai_cs_r160_volume_absorb_high_turn_body_range_confirm_w10',
 'ai_cs_r160_volume_absorb_high_turn_mean_w10',
 'ai_cs_r260_volume_absorb_high_range_body_range_confirm_w10',
 'ai_cs_r102_new_low_reclaim_plain_abs_range_confirm_w10',
 'ai_cs_r102_new_low_reclaim_plain_raw_level_x_abs_range_confirm_w10',
 'ai_cs_r174_body_vwap_confirm_high_turn_min_gap_w20']

    SPECS_SELECTED = [{'base_name': 'ai_cs_r160_volume_absorb_high_turn_mean',
  'expr_tpl': 'ts_mean(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + 1e-12)) * ((high - low) '
              '/ (close + 1e-12))), 0), {w})',
  'param_grid': {'w': [20, 10]},
  'sub_category': 'AI4000R160VolumeAbsorbHighTurn',
  'idea': '放量吸收反转，高换手状态。 滚动均值持久性。',
  'missing': []},
 {'base_name': 'ai_cs_r174_body_vwap_confirm_high_turn_clv_gate',
  'expr_tpl': 'ts_mean(ts_where(((close - low) / (high - low + 1e-12)) > 0.5, ts_where((turnover / '
              '(ts_mean(turnover, {w}) + 1e-12)) > 1, ((close / (open + 1e-12) - 1) * (close / (vwap + '
              '1e-12) - 1)), 0), -1 * (ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, ((close / '
              '(open + 1e-12) - 1) * (close / (vwap + 1e-12) - 1)), 0))), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R174BodyVwapConfirmHighTurn',
  'idea': '实体与 VWAP 偏离确认，高换手状态。 收盘位置门控。',
  'missing': []},
 {'base_name': 'ai_cs_r123_range_turnover_confirm_plain_body_confirm',
  'expr_tpl': 'ts_mean((((((high - low) / (close + 1e-12)) * (turnover / (ts_mean(turnover, {w}) + '
              '1e-12))))) * (close / (open + 1e-12) - 1), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R123RangeTurnoverConfirmPlain',
  'idea': '振幅由换手确认，原始状态。 strict corr 扩展候选：body_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r110_volume_absorb_plain_mean',
  'expr_tpl': 'ts_mean(((-1 * (close / (ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + '
              '1e-12)) * ((high - low) / (close + 1e-12)))), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R110VolumeAbsorbPlain',
  'idea': '放量吸收反转，原始状态。 滚动均值持久性。',
  'missing': []},
 {'base_name': 'ai_cs_r07_amihud_like_efficiency',
  'expr_tpl': '-1 * cs_rank(ts_mean(ts_abs(close / (ts_delay(close, 1) + 1e-12) - 1) / (turnover + 1e-12), '
              '{w}))',
  'param_grid': {'w': [3]},
  'sub_category': 'Round07LiquidityImpact',
  'idea': '单位成交额引起的价格波动越大，流动性冲击越强。',
  'missing': []},
 {'base_name': 'ai_cs_r160_volume_absorb_high_turn_range_adj',
  'expr_tpl': 'ts_mean(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + 1e-12)) * ((high - low) '
              '/ (close + 1e-12))), 0), {w}) / (ts_mean(((high - low) / (close + 1e-12)), {w}) + 1e-12)',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R160VolumeAbsorbHighTurn',
  'idea': '放量吸收反转，高换手状态。 振幅调整后强度。',
  'missing': []},
 {'base_name': 'ai_cs_r83_vwap_turn_crowd_plain_body_range_confirm',
  'expr_tpl': 'ts_mean((((-1 * (close / (vwap + 1e-12) - 1) * (turnover / (ts_mean(turnover, {w}) + '
              '1e-12))))) * (ts_abs(close - open) / (high - low + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R083VwapTurnCrowdPlain',
  'idea': 'VWAP 溢价叠加换手拥挤，原始状态。 strict corr 扩展候选：body_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r09_trend_volume_confirmed_path',
  'expr_tpl': 'cs_rank(ts_corr(close / (ts_delay(close, 1) + 1e-12) - 1, volume / (ts_mean(volume, {w}) + '
              '1e-12), {w}))',
  'param_grid': {'w': [40]},
  'sub_category': 'Round09TrendQuality',
  'idea': 'Round09TrendQuality',
  'missing': []},
 {'base_name': 'ai_cs_b_pv_turnover_return_corr_neg',
  'expr_tpl': '-1 * ts_corr(turnover / (ts_delay(turnover, 1) + 1e-12) - 1, close / (ts_delay(close, 1) + '
              '1e-12) - 1, {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'PriceVolumeDivergence',
  'idea': 'Negative signed turnover-return coupling may indicate distribution or liquidity-demand mismatch.',
  'missing': []},
 {'base_name': 'ai_cs_r58_turnover_spike_exhaust_raw_level_x_abs_turn_confirm',
  'expr_tpl': 'ts_mean(ts_abs((-1 * (turnover / (ts_mean(turnover, {w}) + 1e-12)) * ts_abs((close / '
              '(ts_delay(close, 1) + 1e-12) - 1)))) * (turnover / (ts_mean(turnover, {w}) + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'Mega1000R58TurnoverSpikeExhaust',
  'idea': '成交额尖峰后的拥挤消耗。 原始信号强度。 strict corr 扩展候选：abs_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r110_volume_absorb_plain_body_confirm',
  'expr_tpl': 'ts_mean((((-1 * (close / (ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) '
              '+ 1e-12)) * ((high - low) / (close + 1e-12))))) * (close / (open + 1e-12) - 1), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R110VolumeAbsorbPlain',
  'idea': '放量吸收反转，原始状态。 strict corr 扩展候选：body_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r06_gap_low_reclaim_quality',
  'expr_tpl': 'cs_rank(ts_mean(ts_where(open < ts_delay(close, 1), (close - low) / (high - low + 1e-12), 0), '
              '{w}))',
  'param_grid': {'w': [40]},
  'sub_category': 'Round06GapAuctionProxy',
  'idea': '低开后收回日内低点，反映开盘抛压被承接。',
  'missing': []},
 {'base_name': 'ai_cs_r79_open_panic_recover_plain_downside_gate',
  'expr_tpl': 'ts_mean(ts_where((close / (ts_delay(close, 1) + 1e-12) - 1) < 0, ((-1 * (open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * ((ts_where(open < close, open, close) - low) / (high - '
              'low + 1e-12)))), 0), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R079OpenPanicRecoverPlain',
  'idea': '开盘恐慌后的下影线修复，原始状态。 strict corr 扩展候选：downside_gate。',
  'missing': []},
 {'base_name': 'ai_cs_r234_body_push_quality_high_range_gap_shock_gate',
  'expr_tpl': 'ts_mean(ts_where(ts_abs((open / (ts_delay(close, 1) + 1e-12) - 1)) > ts_mean(ts_abs((open / '
              '(ts_delay(close, 1) + 1e-12) - 1)), {w}), ts_where(((high - low) / (close + 1e-12)) > '
              'ts_mean(((high - low) / (close + 1e-12)), {w}), ((close / (open + 1e-12) - 1) * (ts_abs(close '
              '- open) / (high - low + 1e-12))), 0), 0), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R234BodyPushQualityHighRange',
  'idea': '实体推进质量，高振幅状态。 strict corr 扩展候选：gap_shock_gate。',
  'missing': []},
 {'base_name': 'ai_cs_r160_volume_absorb_high_turn_body_range_confirm',
  'expr_tpl': 'ts_mean((ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + 1e-12)) * ((high - low) '
              '/ (close + 1e-12))), 0)) * (ts_abs(close - open) / (high - low + 1e-12)), {w})',
  'param_grid': {'w': [10]},
  'sub_category': 'AI4000R160VolumeAbsorbHighTurn',
  'idea': '放量吸收反转，高换手状态。 strict corr 扩展候选：body_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r252_new_low_reclaim_high_range_abs_turn_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where(((high - low) / (close + 1e-12)) > ts_mean(((high - low) / (close + '
              '1e-12)), {w}), ((close / (ts_min(low, {w}) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + '
              '1e-12))), 0)) * (turnover / (ts_mean(turnover, {w}) + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R252NewLowReclaimHighRange',
  'idea': '新低收复确认，高振幅状态。 strict corr 扩展候选：abs_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r30_vwap_premium_decay_raw_level_x_abs_turn_confirm',
  'expr_tpl': 'ts_mean(ts_abs((-1 * (close / (vwap + 1e-12) - 1) * (turnover / (ts_mean(turnover, {w}) + '
              '1e-12)))) * (turnover / (ts_mean(turnover, {w}) + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'Mega1000R30VwapPremiumDecay',
  'idea': '收盘高于成交均价后的溢价回落风险。 原始信号强度。 strict corr 扩展候选：abs_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r171_clv_volume_confirm_high_turn_body_range_confirm',
  'expr_tpl': 'ts_mean((ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, ((((close - low) / (high '
              '- low + 1e-12)) - 0.5) * (volume / (ts_mean(volume, {w}) + 1e-12))), 0)) * (ts_abs(close - '
              'open) / (high - low + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R171ClvVolumeConfirmHighTurn',
  'idea': '收盘位置被成交量确认，高换手状态。 strict corr 扩展候选：body_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r127_gap_follow_path_high_turn_abs_turn_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, ((open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (close / (open + 1e-12) - 1)), 0)) * (turnover / '
              '(ts_mean(turnover, {w}) + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R127GapFollowPathHighTurn',
  'idea': '跳空延续后的日内确认，高换手状态。 strict corr 扩展候选：abs_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r115_amihud_impact_plain_downside_gate',
  'expr_tpl': 'ts_mean(ts_where((close / (ts_delay(close, 1) + 1e-12) - 1) < 0, ((-1 * ts_mean(ts_abs((close '
              '/ (ts_delay(close, 1) + 1e-12) - 1)) / (turnover + 1e-12), {w}))), 0), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R115AmihudImpactPlain',
  'idea': '价格冲击成本，原始状态。 strict corr 扩展候选：downside_gate。',
  'missing': []},
 {'base_name': 'ai_cs_r260_volume_absorb_high_range_body_range_confirm',
  'expr_tpl': 'ts_mean((ts_where(((high - low) / (close + 1e-12)) > ts_mean(((high - low) / (close + '
              '1e-12)), {w}), (-1 * (close / (ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, '
              '{w}) + 1e-12)) * ((high - low) / (close + 1e-12))), 0)) * (ts_abs(close - open) / (high - low '
              '+ 1e-12)), {w})',
  'param_grid': {'w': [10]},
  'sub_category': 'AI4000R260VolumeAbsorbHighRange',
  'idea': '放量吸收反转，高振幅状态。 strict corr 扩展候选：body_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r40_range_expansion_risk_raw_level_x_ret_confirm',
  'expr_tpl': 'ts_mean(((-1 * (((high - low) / (close + 1e-12)) - ts_mean(((high - low) / (close + 1e-12)), '
              '{w})))) * (close / (ts_delay(close, 1) + 1e-12) - 1), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'Mega1000R40RangeExpansionRisk',
  'idea': '振幅扩张后路径噪声和风险暴露。 原始信号强度。 strict corr 扩展候选：ret_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r129_open_panic_recover_high_turn_raw_level_x_abs_vwap_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * ((ts_where(open < close, open, close) - low) / (high - '
              'low + 1e-12))), 0)) * ts_abs((close / (vwap + 1e-12) - 1)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R129OpenPanicRecoverHighTurn',
  'idea': '开盘恐慌后的下影线修复，高换手状态。 strict corr 扩展候选：abs_vwap_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r99_overextension_reversal_plain_raw_level_x_core_body_range_corr',
  'expr_tpl': 'ts_corr((close / (open + 1e-12) - 1), ((high - low) / (close + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R099OverextensionReversalPlain',
  'idea': '过度延伸后的回撤收益，原始状态。 strict corr 扩展候选：core_body_range_corr。',
  'missing': []},
 {'base_name': 'ai_cs_r169_winrate_memory_high_turn_body_confirm',
  'expr_tpl': 'ts_mean((ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (ts_sum(ts_where((close '
              '/ (ts_delay(close, 1) + 1e-12) - 1) > 0, 1, 0), {w}) / ({w} + 1e-12)), 0)) * (close / (open + '
              '1e-12) - 1), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R169WinrateMemoryHighTurn',
  'idea': '收益方向胜率记忆，高换手状态。 strict corr 扩展候选：body_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r20_range_break_hold_stable_rank',
  'expr_tpl': 'cs_rank(ts_mean(ts_where(close > ts_delay(ts_max(high, {w}), 1), (close - low) / (high - low '
              '+ 1e-12), 0), {w}) / (ts_std(ts_where(close > ts_delay(ts_max(high, {w}), 1), (close - low) / '
              '(high - low + 1e-12), 0), {w}) + 1e-12))',
  'param_grid': {'w': [5]},
  'sub_category': 'Round20ExtremePathReversal',
  'idea': '新高新低、过度延伸、回撤修复和极端路径反转。',
  'missing': []},
 {'base_name': 'ai_cs_r229_open_panic_recover_high_range_abs_turn_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where(((high - low) / (close + 1e-12)) > ts_mean(((high - low) / (close + '
              '1e-12)), {w}), (-1 * (open / (ts_delay(close, 1) + 1e-12) - 1) * ((ts_where(open < close, '
              'open, close) - low) / (high - low + 1e-12))), 0)) * (turnover / (ts_mean(turnover, {w}) + '
              '1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R229OpenPanicRecoverHighRange',
  'idea': '开盘恐慌后的下影线修复，高振幅状态。 strict corr 扩展候选：abs_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r175_gap_vwap_disagree_high_turn_max_gap',
  'expr_tpl': '(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / (ts_delay(close, '
              '1) + 1e-12) - 1) * (close / (vwap + 1e-12) - 1)), 0)) - ts_max(ts_where((turnover / '
              '(ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / (ts_delay(close, 1) + 1e-12) - 1) * '
              '(close / (vwap + 1e-12) - 1)), 0), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R175GapVwapDisagreeHighTurn',
  'idea': '跳空与 VWAP 偏离冲突，高换手状态。 strict corr 扩展候选：max_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r80_close_pressure_vwap_plain_lower_shadow_confirm',
  'expr_tpl': 'ts_mean(((((close / (vwap + 1e-12) - 1) * (((close - low) / (high - low + 1e-12)) - 0.5) * '
              '(turnover / (ts_mean(turnover, {w}) + 1e-12))))) * ((ts_where(open < close, open, close) - '
              'low) / (high - low + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R080ClosePressureVwapPlain',
  'idea': '收盘相对 VWAP 的执行压力，原始状态。 strict corr 扩展候选：lower_shadow_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r24_vwap_reward_risk_dev_rank',
  'expr_tpl': 'cs_rank((ts_mean(close / (vwap + 1e-12) - 1, {w}) / (ts_std(close / (vwap + 1e-12) - 1, {w}) '
              '+ 1e-12)) - ts_mean(ts_mean(close / (vwap + 1e-12) - 1, {w}) / (ts_std(close / (vwap + 1e-12) '
              '- 1, {w}) + 1e-12), {w}))',
  'param_grid': {'w': [40]},
  'sub_category': 'Round24RewardRiskEfficiency',
  'idea': '单位风险收益、单位振幅收益、成交效率和风险预算质量。',
  'missing': []},
 {'base_name': 'ai_cs_r22_turnover_volatility_discount_mean_raw',
  'expr_tpl': 'ts_mean(-1 * ts_std(turnover / (ts_mean(turnover, {w}) + 1e-12), {w}), {w})',
  'param_grid': {'w': [10]},
  'sub_category': 'Round22CrowdingUnwindRawCounterpart',
  'idea': '去掉上一版最外层 cs_rank 的 raw 对照，保留原始强度信息；评估层再决定是否 rank/zscore。',
  'missing': []},
 {'base_name': 'ai_cs_r102_new_low_reclaim_plain_abs_range_confirm',
  'expr_tpl': 'ts_mean(ts_abs((((close / (ts_min(low, {w}) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + '
              '1e-12))))) * ((high - low) / (close + 1e-12)), {w})',
  'param_grid': {'w': [10]},
  'sub_category': 'AI4000R102NewLowReclaimPlain',
  'idea': '新低收复确认，原始状态。 strict corr 扩展候选：abs_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r102_new_low_reclaim_plain_raw_level_x_abs_range_confirm',
  'expr_tpl': 'ts_mean(ts_abs((((close / (ts_min(low, {w}) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + '
              '1e-12))))) * ((high - low) / (close + 1e-12)), {w})',
  'param_grid': {'w': [10]},
  'sub_category': 'AI4000R102NewLowReclaimPlain',
  'idea': '新低收复确认，原始状态。 strict corr 扩展候选：abs_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r231_vwap_revert_high_range_range_adj',
  'expr_tpl': 'ts_mean(ts_where(((high - low) / (close + 1e-12)) > ts_mean(((high - low) / (close + 1e-12)), '
              '{w}), (-1 * (close / (vwap + 1e-12) - 1)), 0), {w}) / (ts_mean(((high - low) / (close + '
              '1e-12)), {w}) + 1e-12)',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R231VwapRevertHighRange',
  'idea': 'VWAP 偏离均值回归，高振幅状态。 振幅调整后强度。',
  'missing': []},
 {'base_name': 'ai_cs_r263_pv_diverge_volume_high_range_body_confirm',
  'expr_tpl': 'ts_mean((ts_where(((high - low) / (close + 1e-12)) > ts_mean(((high - low) / (close + '
              '1e-12)), {w}), ((close / (ts_delay(close, {w}) + 1e-12) - 1) - ts_mean((volume / '
              '(ts_mean(volume, {w}) + 1e-12)), {w})), 0)) * (close / (open + 1e-12) - 1), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R263PvDivergeVolumeHighRange',
  'idea': '价格与成交量背离，高振幅状态。 strict corr 扩展候选：body_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r141_range_expand_risk_high_turn_min_gap',
  'expr_tpl': '(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * ts_delta(((high - low) / '
              '(close + 1e-12)), 1) * (turnover / (ts_mean(turnover, {w}) + 1e-12))), 0)) - '
              'ts_min(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * ts_delta(((high - '
              'low) / (close + 1e-12)), 1) * (turnover / (ts_mean(turnover, {w}) + 1e-12))), 0), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R141RangeExpandRiskHighTurn',
  'idea': '振幅扩张后的风险释放，高换手状态。 strict corr 扩展候选：min_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r83_vwap_turn_crowd_plain_max_gap',
  'expr_tpl': '(((-1 * (close / (vwap + 1e-12) - 1) * (turnover / (ts_mean(turnover, {w}) + 1e-12))))) - '
              'ts_max(((-1 * (close / (vwap + 1e-12) - 1) * (turnover / (ts_mean(turnover, {w}) + 1e-12)))), '
              '{w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R083VwapTurnCrowdPlain',
  'idea': 'VWAP 溢价叠加换手拥挤，原始状态。 strict corr 扩展候选：max_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r64_crowded_profit_unwind_raw_level_x_abs_vwap_confirm',
  'expr_tpl': 'ts_mean(ts_abs((-1 * ts_where((close / (ts_delay(close, {w}) + 1e-12) - 1) > 0, (close / '
              '(ts_delay(close, {w}) + 1e-12) - 1) * (turnover / (ts_mean(turnover, {w}) + 1e-12)), 0))) * '
              'ts_abs((close / (vwap + 1e-12) - 1)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'Mega1000R64CrowdedProfitUnwind',
  'idea': '放量上涨后的获利拥挤退出风险。 原始信号强度。 strict corr 扩展候选：abs_vwap_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r228_open_absorb_clv_high_range_abs_turn_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where(((high - low) / (close + 1e-12)) > ts_mean(((high - low) / (close + '
              '1e-12)), {w}), (-1 * (open / (ts_delay(close, 1) + 1e-12) - 1) * (((close - low) / (high - '
              'low + 1e-12)) - 0.5)), 0)) * (turnover / (ts_mean(turnover, {w}) + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R228OpenAbsorbClvHighRange',
  'idea': '开盘跳空被收盘位置吸收，高振幅状态。 strict corr 扩展候选：abs_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_dv01_gap_follow_with_turnover',
  'expr_tpl': 'cs_rank(ts_mean((close / (open + 1e-12) - 1) * (turnover / (ts_mean(turnover, {w}) + 1e-12)), '
              '{w}))',
  'param_grid': {'w': [40]},
  'sub_category': 'DiverseStrictR01OvernightIntraday',
  'idea': 'strict 5轮结果：隔夜跳空与日内回补/失败之间的路径迁移。',
  'missing': []},
 {'base_name': 'ai_cs_r258_liquidity_replenish_high_range_gap_shock_gate',
  'expr_tpl': 'ts_mean(ts_where(ts_abs((open / (ts_delay(close, 1) + 1e-12) - 1)) > ts_mean(ts_abs((open / '
              '(ts_delay(close, 1) + 1e-12) - 1)), {w}), ts_where(((high - low) / (close + 1e-12)) > '
              'ts_mean(((high - low) / (close + 1e-12)), {w}), ((close / (ts_delay(close, {w}) + 1e-12) - 1) '
              '* (turnover / (ts_mean(turnover, {w}) + 1e-12))), 0), 0), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R258LiquidityReplenishHighRange',
  'idea': '流动性回补确认，高振幅状态。 strict corr 扩展候选：gap_shock_gate。',
  'missing': []},
 {'base_name': 'ai_cs_r73_vol_regime_switch_range_adj',
  'expr_tpl': 'ts_mean(ts_where(ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), {w}) > '
              'ts_mean(ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), {w}), {w}), -1 * (close / '
              '(ts_delay(close, {w}) + 1e-12) - 1), (close / (ts_delay(close, {w}) + 1e-12) - 1)), {w}) / '
              '(ts_mean(((high - low) / (close + 1e-12)), {w}) + 1e-12)',
  'param_grid': {'w': [10]},
  'sub_category': 'Mega1000R73VolRegimeSwitch',
  'idea': '高低波动状态切换后的条件收益。 振幅调整。',
  'missing': []},
 {'base_name': 'ai_cs_r129_open_panic_recover_high_turn_max_gap',
  'expr_tpl': '(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / (ts_delay(close, '
              '1) + 1e-12) - 1) * ((ts_where(open < close, open, close) - low) / (high - low + 1e-12))), 0)) '
              '- ts_max(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * ((ts_where(open < close, open, close) - low) / (high - '
              'low + 1e-12))), 0), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R129OpenPanicRecoverHighTurn',
  'idea': '开盘恐慌后的下影线修复，高换手状态。 strict corr 扩展候选：max_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r252_new_low_reclaim_high_range_clv_gate',
  'expr_tpl': 'ts_mean(ts_where(((close - low) / (high - low + 1e-12)) > 0.5, ts_where(((high - low) / '
              '(close + 1e-12)) > ts_mean(((high - low) / (close + 1e-12)), {w}), ((close / (ts_min(low, '
              '{w}) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + 1e-12))), 0), -1 * (ts_where(((high - '
              'low) / (close + 1e-12)) > ts_mean(((high - low) / (close + 1e-12)), {w}), ((close / '
              '(ts_min(low, {w}) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + 1e-12))), 0))), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R252NewLowReclaimHighRange',
  'idea': '新低收复确认，高振幅状态。 收盘位置门控。',
  'missing': []},
 {'base_name': 'ai_cs_r128_open_absorb_clv_high_turn_max_gap',
  'expr_tpl': '(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / (ts_delay(close, '
              '1) + 1e-12) - 1) * (((close - low) / (high - low + 1e-12)) - 0.5)), 0)) - '
              'ts_max(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (((close - low) / (high - low + 1e-12)) - 0.5)), 0), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R128OpenAbsorbClvHighTurn',
  'idea': '开盘跳空被收盘位置吸收，高换手状态。 strict corr 扩展候选：max_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r129_open_panic_recover_high_turn_gap_confirm',
  'expr_tpl': 'ts_mean((ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * ((ts_where(open < close, open, close) - low) / (high - '
              'low + 1e-12))), 0)) * (open / (ts_delay(close, 1) + 1e-12) - 1), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R129OpenPanicRecoverHighTurn',
  'idea': '开盘恐慌后的下影线修复，高换手状态。 跳空确认。',
  'missing': []},
 {'base_name': 'ai_cs_r50_new_high_breakout_raw_level_x_abs_range_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where(high > ts_delay(ts_max(high, {w}), 1), ((close - low) / (high - low + '
              '1e-12)), 0)) * ((high - low) / (close + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'Mega1000R50NewHighBreakout',
  'idea': '突破过去高点后的趋势确认。 原始信号强度。 strict corr 扩展候选：abs_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r214_pv_diverge_turnover_low_turn_raw_level_x_core_ret_turn_confirm',
  'expr_tpl': 'ts_mean((close / (ts_delay(close, 1) + 1e-12) - 1) * (turnover / (ts_mean(turnover, {w}) + '
              '1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R214PvDivergeTurnoverLowTurn',
  'idea': '价格与换手背离，低换手状态。 strict corr 扩展候选：core_ret_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r83_vwap_turn_crowd_plain_lower_shadow_confirm',
  'expr_tpl': 'ts_mean((((-1 * (close / (vwap + 1e-12) - 1) * (turnover / (ts_mean(turnover, {w}) + '
              '1e-12))))) * ((ts_where(open < close, open, close) - low) / (high - low + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R083VwapTurnCrowdPlain',
  'idea': 'VWAP 溢价叠加换手拥挤，原始状态。 strict corr 扩展候选：lower_shadow_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r137_upper_reject_high_turn_min_gap',
  'expr_tpl': '(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * ((high - ts_where(open > '
              'close, open, close)) / (high - low + 1e-12)) * ts_abs((close / (ts_delay(close, 1) + 1e-12) - '
              '1))), 0)) - ts_min(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * ((high - '
              'ts_where(open > close, open, close)) / (high - low + 1e-12)) * ts_abs((close / '
              '(ts_delay(close, 1) + 1e-12) - 1))), 0), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R137UpperRejectHighTurn',
  'idea': '上影线抛压反转，高换手状态。 strict corr 扩展候选：min_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r06_gap_open_liquidity_penalty',
  'expr_tpl': '-1 * cs_rank(ts_mean(ts_abs(open / (ts_delay(close, 1) + 1e-12) - 1) * volume / '
              '(ts_mean(volume, {w}) + 1e-12), {w}))',
  'param_grid': {'w': [20]},
  'sub_category': 'Round06GapAuctionProxy',
  'idea': '放量跳空代表开盘冲击拥挤，作为风险惩罚。',
  'missing': []},
 {'base_name': 'ai_cs_r135_body_failure_high_turn_body_range_confirm',
  'expr_tpl': 'ts_mean((ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (close / (open + '
              '1e-12) - 1) * ((high - ts_where(open > close, open, close)) / (high - low + 1e-12))), 0)) * '
              '(ts_abs(close - open) / (high - low + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R135BodyFailureHighTurn',
  'idea': '实体失败与反向修复，高换手状态。 strict corr 扩展候选：body_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r174_body_vwap_confirm_high_turn_min_gap',
  'expr_tpl': '(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, ((close / (open + 1e-12) - 1) * '
              '(close / (vwap + 1e-12) - 1)), 0)) - ts_min(ts_where((turnover / (ts_mean(turnover, {w}) + '
              '1e-12)) > 1, ((close / (open + 1e-12) - 1) * (close / (vwap + 1e-12) - 1)), 0), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R174BodyVwapConfirmHighTurn',
  'idea': '实体与 VWAP 偏离确认，高换手状态。 strict corr 扩展候选：min_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r137_upper_reject_high_turn_abs_range_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * ((high - '
              'ts_where(open > close, open, close)) / (high - low + 1e-12)) * ts_abs((close / '
              '(ts_delay(close, 1) + 1e-12) - 1))), 0)) * ((high - low) / (close + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R137UpperRejectHighTurn',
  'idea': '上影线抛压反转，高换手状态。 strict corr 扩展候选：abs_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r160_volume_absorb_high_turn_body_confirm',
  'expr_tpl': 'ts_mean((ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + 1e-12)) * ((high - low) '
              '/ (close + 1e-12))), 0)) * (close / (open + 1e-12) - 1), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R160VolumeAbsorbHighTurn',
  'idea': '放量吸收反转，高换手状态。 strict corr 扩展候选：body_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r260_volume_absorb_high_range_lower_shadow_confirm',
  'expr_tpl': 'ts_mean((ts_where(((high - low) / (close + 1e-12)) > ts_mean(((high - low) / (close + '
              '1e-12)), {w}), (-1 * (close / (ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, '
              '{w}) + 1e-12)) * ((high - low) / (close + 1e-12))), 0)) * ((ts_where(open < close, open, '
              'close) - low) / (high - low + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R260VolumeAbsorbHighRange',
  'idea': '放量吸收反转，高振幅状态。 strict corr 扩展候选：lower_shadow_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r78_open_absorb_clv_plain_max_gap',
  'expr_tpl': '(((-1 * (open / (ts_delay(close, 1) + 1e-12) - 1) * (((close - low) / (high - low + 1e-12)) - '
              '0.5)))) - ts_max(((-1 * (open / (ts_delay(close, 1) + 1e-12) - 1) * (((close - low) / (high - '
              'low + 1e-12)) - 0.5))), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R078OpenAbsorbClvPlain',
  'idea': '开盘跳空被收盘位置吸收，原始状态。 strict corr 扩展候选：max_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r179_open_panic_recover_low_turn_body_range_confirm',
  'expr_tpl': 'ts_mean((ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) < 1, (-1 * (open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * ((ts_where(open < close, open, close) - low) / (high - '
              'low + 1e-12))), 0)) * (ts_abs(close - open) / (high - low + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R179OpenPanicRecoverLowTurn',
  'idea': '开盘恐慌后的下影线修复，低换手状态。 strict corr 扩展候选：body_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r57_liquidity_replenish_raw_level_x_abs_vwap_confirm',
  'expr_tpl': 'ts_mean(ts_abs(((turnover / (ts_mean(turnover, {w}) + 1e-12)) * ((close - low) / (high - low '
              '+ 1e-12)))) * ts_abs((close / (vwap + 1e-12) - 1)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'Mega1000R57LiquidityReplenish',
  'idea': '缩量后成交恢复并伴随收盘改善。 原始信号强度。 strict corr 扩展候选：abs_vwap_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r175_gap_vwap_disagree_high_turn_abs_volume_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (close / (vwap + 1e-12) - 1)), 0)) * (volume / '
              '(ts_mean(volume, {w}) + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R175GapVwapDisagreeHighTurn',
  'idea': '跳空与 VWAP 偏离冲突，高换手状态。 strict corr 扩展候选：abs_volume_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r160_volume_absorb_high_turn_abs_turn_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (volume / (ts_mean(volume, {w}) + 1e-12)) * ((high - low) '
              '/ (close + 1e-12))), 0)) * (turnover / (ts_mean(turnover, {w}) + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R160VolumeAbsorbHighTurn',
  'idea': '放量吸收反转，高换手状态。 strict corr 扩展候选：abs_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r16_gap_abs_range_mean_rank',
  'expr_tpl': 'cs_rank(ts_mean(ts_abs(open / (ts_delay(close, 1) + 1e-12) - 1) / ((high - low) / (close + '
              '1e-12) + 1e-12), {w}))',
  'param_grid': {'w': [20]},
  'sub_category': 'Round16GapAuctionAbsorption',
  'idea': '开盘冲击、跳空吸收、日内回补和 VWAP 确认。',
  'missing': []},
 {'base_name': 'ai_cs_r175_gap_vwap_disagree_high_turn_abs_range_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) > 1, (-1 * (open / '
              '(ts_delay(close, 1) + 1e-12) - 1) * (close / (vwap + 1e-12) - 1)), 0)) * ((high - low) / '
              '(close + 1e-12)), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R175GapVwapDisagreeHighTurn',
  'idea': '跳空与 VWAP 偏离冲突，高换手状态。 strict corr 扩展候选：abs_range_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r196_return_efficiency_low_turn_abs_turn_confirm',
  'expr_tpl': 'ts_mean(ts_abs(ts_where((turnover / (ts_mean(turnover, {w}) + 1e-12)) < 1, ((close / '
              '(ts_delay(close, {w}) + 1e-12) - 1) / (ts_sum(ts_abs((close / (ts_delay(close, 1) + 1e-12) - '
              '1)), {w}) + 1e-12)), 0)) * (turnover / (ts_mean(turnover, {w}) + 1e-12)), {w})',
  'param_grid': {'w': [20]},
  'sub_category': 'AI4000R196ReturnEfficiencyLowTurn',
  'idea': '路径效率收益，低换手状态。 strict corr 扩展候选：abs_turn_confirm。',
  'missing': []},
 {'base_name': 'ai_cs_r17_volume_stability_neg_mean_rank',
  'expr_tpl': '-1 * cs_rank(ts_mean(1 / (ts_std(volume / (ts_mean(volume, {w}) + 1e-12), {w}) + 1e-12), '
              '{w}))',
  'param_grid': {'w': [10]},
  'sub_category': 'Round17LiquidityDryupReplenish',
  'idea': '缩量漂移、流动性枯竭、成交恢复和单位成交价格推进。',
  'missing': []},
 {'base_name': 'ai_cs_r22_crowded_gap_fail_neg_mean_raw',
  'expr_tpl': '-1 * (ts_mean(-1 * ts_abs(open / (ts_delay(close, 1) + 1e-12) - 1) * (turnover / '
              '(ts_mean(turnover, {w}) + 1e-12)) * ((high - ts_where(open > close, open, close)) / (high - '
              'low + 1e-12)), {w}))',
  'param_grid': {'w': [40]},
  'sub_category': 'Round22CrowdingUnwindRawCounterpart',
  'idea': '去掉上一版最外层 cs_rank 的 raw 对照，保留原始强度信息；评估层再决定是否 rank/zscore。',
  'missing': []},
 {'base_name': 'ai_cs_r87_upper_reject_plain_min_gap',
  'expr_tpl': '(((-1 * ((high - ts_where(open > close, open, close)) / (high - low + 1e-12)) * ts_abs((close '
              '/ (ts_delay(close, 1) + 1e-12) - 1))))) - ts_min(((-1 * ((high - ts_where(open > close, open, '
              'close)) / (high - low + 1e-12)) * ts_abs((close / (ts_delay(close, 1) + 1e-12) - 1)))), {w})',
  'param_grid': {'w': [40]},
  'sub_category': 'AI4000R087UpperRejectPlain',
  'idea': '上影线抛压反转，原始状态。 strict corr 扩展候选：min_gap。',
  'missing': []},
 {'base_name': 'ai_cs_r37_lower_shadow_absorb_mean_change',
  'expr_tpl': 'ts_mean(((ts_where(open < close, open, close) - low) / (high - low + 1e-12)), {w}) - '
              'ts_delay(ts_mean(((ts_where(open < close, open, close) - low) / (high - low + 1e-12)), {w}), '
              '{w})',
  'param_grid': {'w': [40]},
  'sub_category': 'Mega1000R37LowerShadowAbsorb',
  'idea': '下影线代表下方承接和恐慌吸收。 滚动均值变化。',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_922_growth_gap_skew_idx922_pullback_flip_120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + (-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + (-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + (-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120)))))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissGrowthGapSkewPullbackFlip',
  'idea': '成长/效率 surprise、000922.SSE gap 结构和行业残差偏度反转共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_regime_switch。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_300_growth_gap_skew_idx922_extreme_flip_120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + (-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + (-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + (-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))), '
              'cs_rank(cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + (-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 120))))))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissGrowthGapSkewRegimeFlip',
  'idea': '成长/效率 surprise、000300.SSE gap 结构和行业残差偏度反转共振；当中证红利/价值代理 20 日极端上涨或 60 日向上中的急跌回撤时翻转方向。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_regime_switch。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_pullback_flip_120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, 120)) / '
              '(ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 120))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) '
              '+ cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, (high - '
              'close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, 120)) / '
              '(ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 120)))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'idea': '质量 surprise、000922.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_regime_switch。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_pullback_flip_60',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, 60)) / '
              '(ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '60)) + ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 60))), cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, (high - '
              'close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, 60)) / '
              '(ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '60)) + ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 60)))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissQualityDefensePullbackFlip',
  'idea': '质量 surprise、000906.SSE gap 结构和指数下跌日抗跌共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_regime_switch。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_922_quality_gap_defensive_idx922_meltup_flip_120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 120)) / (ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) '
              '+ cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, (high - '
              'close) / (high - low + 1e-12), 0), 120)) + ts_mean(ts_where(idx922_r1 < 0, (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - idx922_r1, 0), 120))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, 120)) / '
              '(ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + ts_mean(ts_where(idx922_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx922_r1, 0), 120)))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'idea': '质量 surprise、000922.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_regime_switch。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_impact_range_turn_crowd_flip_10',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 10) + 1e-12)) > 1.5, -1 * cs_rank(ts_mean(((high - '
              'low) / (close + 1e-12)) / (turnover + 1e-12), 10)), cs_rank(ts_mean(((high - low) / (close + '
              '1e-12)) / (turnover + 1e-12), 10)))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityCrowdFlipV3',
  'idea': '振幅冲击成本 在成交拥挤时翻转。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_906_quality_gap_defensive_idx922_meltup_flip_60',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank(cs_rank((fund_return_on_equity_ttm - '
              'ts_mean(fund_return_on_equity_ttm, 60)) / (ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + '
              'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, (high - '
              'close) / (high - low + 1e-12), 0), 60)) + ts_mean(ts_where(idx906_r1 < 0, (close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 0), 60))), '
              'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, 60)) / '
              '(ts_std(fund_return_on_equity_ttm, 60) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '60)) + ts_mean(ts_where(idx906_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx906_r1, 0), 60)))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissQualityDefenseMeltupFlip',
  'idea': '质量 surprise、000906.SSE gap 结构和指数下跌日抗跌共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_regime_switch。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_905_quality_gap_defensive_120',
  'expr_tpl': 'cs_rank(cs_rank((fund_return_on_equity_ttm - ts_mean(fund_return_on_equity_ttm, 120)) / '
              '(ts_std(fund_return_on_equity_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120)) + ts_mean(ts_where(idx905_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'idx905_r1, 0), 120))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissQualityDefense',
  'idea': '训练期近似有效的质量 surprise、000905.SSE 高开耗尽原方向和指数下跌日抗跌残差的共振。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_composite。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_corr_drop_value_neglect_40',
  'expr_tpl': 'cs_rank(ts_mean((-1 * ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, '
              '40), 1) * fund_book_to_market_ratio_ttm), 40) * (1 / (ts_mean((turnover / (ts_mean(turnover, '
              '40) + 1e-12)), 40) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '小盘相关性下降时的价值独立性；beta 结构处在低成交关注下。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_regime_cross_stock_low_corr_dividend_20',
  'expr_tpl': 'ts_where(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 20) < 0.5, '
              'cs_rank(ts_mean(fund_dividend_yield_ttm, 20)), -1 * cs_rank(ts_mean(fund_dividend_yield_ttm, '
              '20)))',
  'param_grid': None,
  'sub_category': 'DailyGuidedRegimeCrossV5',
  'idea': '个股低小盘相关状态下使用股息，非该状态反向。 来源：daily_v3_v6_recovered:daily_guided_regime_cross_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_size_turn_capacity_jump_guard_10',
  'expr_tpl': 'cs_rank(ts_mean(((1 / (fund_market_cap + 1e-12)) * (1 / (ts_mean((turnover / '
              '(ts_mean(turnover, 10) + 1e-12)), 10) + 1e-12))), 10) / (ts_mean(ts_abs((((1 / '
              '(fund_market_cap + 1e-12)) * (1 / (ts_mean((turnover / (ts_mean(turnover, 10) + 1e-12)), 10) '
              '+ 1e-12)))) - ts_delay(((1 / (fund_market_cap + 1e-12)) * (1 / (ts_mean((turnover / '
              '(ts_mean(turnover, 10) + 1e-12)), 10) + 1e-12))), 1)), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '小市值且成交关注低的容量折价；容量信号低跳变。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx922_relative_gap_exhaustion_3',
  'expr_tpl': '-1 * cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx922_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 3))',
  'param_grid': None,
  'sub_category': 'IndexGapStructure',
  'idea': '相对 000922.SSE 高开后的冲高回落耗尽。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx905_relative_gap_exhaustion_3',
  'expr_tpl': '-1 * cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx905_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 3))',
  'param_grid': None,
  'sub_category': 'IndexGapStructure',
  'idea': '相对 000905.SSE 高开后的冲高回落耗尽。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_margin_growth_vol_compress_20',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_gross_profit_margin_ttm_ind_z) + '
              'cs_rank(fund_net_profit_margin_ttm_ind_z) + cs_rank(fund_gross_profit_growth_ratio_ttm_ind_z) '
              '+ cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z)), 20) * (1 / (ts_std((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 20) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionMarginGrowthConsensusV3',
  'idea': '利润率和毛利/收入成长共同确认。 的认知滞后/确认：基本面好且残差波动压缩。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx906_relative_gap_exhaustion_3',
  'expr_tpl': '-1 * cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1 - idx906_r1) > 0, '
              '(high - close) / (high - low + 1e-12), 0), 3))',
  'param_grid': None,
  'sub_category': 'IndexGapStructure',
  'idea': '相对 000906.SSE 高开后的冲高回落耗尽。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_regime_cross_stock_low_corr_small_value_20',
  'expr_tpl': 'ts_where(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 20) < 0.5, '
              'cs_rank(ts_mean((fund_book_to_market_ratio_ttm * (1 / (fund_market_cap + 1e-12))), 20)), -1 * '
              'cs_rank(ts_mean((fund_book_to_market_ratio_ttm * (1 / (fund_market_cap + 1e-12))), 20)))',
  'param_grid': None,
  'sub_category': 'DailyGuidedRegimeCrossV5',
  'idea': '个股低小盘相关状态下使用小市值价值，非该状态反向。 来源：daily_v3_v6_recovered:daily_guided_regime_cross_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_def_beta_cash_stable_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_ocf_to_debt_ttm * -1 * ts_beta((close / (ts_delay(close, 1) + 1e-12) - '
              '1), idx852_r1, 20)), 20) / (ts_std((fund_ocf_to_debt_ttm * -1 * ts_beta((close / '
              '(ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 20)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '低小盘 beta 的现金覆盖防御；beta 期限结构稳定性。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_efficiency_growth_vol_compress_20',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_total_asset_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z) + '
              'cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z)), 20) * (1 / (ts_std((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 20) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionEfficiencyGrowthConsensusV3',
  'idea': '经营效率改善和收入成长同时占优。 的认知滞后/确认：基本面好且残差波动压缩。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_corr_rise_risk_neglect_120',
  'expr_tpl': 'cs_rank(ts_mean((-1 * ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, '
              '120), 1) * ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), 120)), 120) * (1 / '
              '(ts_mean((turnover / (ts_mean(turnover, 120) + 1e-12)), 120) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '小盘相关性上升和个股波动风险；beta 结构处在低成交关注下。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_ind_residual_skew_reversal_20',
  'expr_tpl': '-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 20))',
  'param_grid': None,
  'sub_category': 'IndustryResidualShape',
  'idea': '行业残差收益正偏后的反转检验。 来源：daily_data_dimension_archive_20260518:industry。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_peer_market_cap_resid_accel_20',
  'expr_tpl': 'cs_rank((fund_market_cap_ind_resid - ts_delay(fund_market_cap_ind_resid, 20)) / '
              '(ts_std(fund_market_cap_ind_resid, 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPeerResidualAccelerationV3',
  'idea': '市值 行业内残差加速。 来源：daily_v3_v6_recovered:daily_peer_dispersion_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_regime_cross_stock_low_corr_capacity_cash_20',
  'expr_tpl': 'ts_where(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 20) < 0.5, '
              'cs_rank(ts_mean((fund_cfp_ratio_ttm * (1 / (ts_mean((turnover / (ts_mean(turnover, 20) + '
              '1e-12)), 20) + 1e-12))), 20)), -1 * cs_rank(ts_mean((fund_cfp_ratio_ttm * (1 / '
              '(ts_mean((turnover / (ts_mean(turnover, 20) + 1e-12)), 20) + 1e-12))), 20)))',
  'param_grid': None,
  'sub_category': 'DailyGuidedRegimeCrossV5',
  'idea': '个股低小盘相关状态下使用低关注现金流收益，非该状态反向。 来源：daily_v3_v6_recovered:daily_guided_regime_cross_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_net_profit_growth_change_120',
  'expr_tpl': 'cs_rank(fund_net_profit_growth_ratio_ttm - ts_delay(fund_net_profit_growth_ratio_ttm, 120))',
  'param_grid': None,
  'sub_category': 'FundamentalGrowthChange',
  'idea': 'net_profit_growth_ratio_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_operating_revenue_growth_surprise_120',
  'expr_tpl': 'cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalGrowthSurprise',
  'idea': 'operating_revenue_growth_ratio_ttm 相对自身历史均值的偏离。 '
          '来源：daily_data_dimension_archive_20260518:fundamental。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_capacity_range_discount_jump_guard_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_ep_ratio_ttm * (1 / (ts_mean(((high - low) / (close + 1e-12)), 20) + '
              '1e-12))), 20) / (ts_mean(ts_abs(((fund_ep_ratio_ttm * (1 / (ts_mean(((high - low) / (close + '
              '1e-12)), 20) + 1e-12)))) - ts_delay((fund_ep_ratio_ttm * (1 / (ts_mean(((high - low) / (close '
              '+ 1e-12)), 20) + 1e-12))), 1)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '盈利收益率要求低振幅消耗；容量信号低跳变。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_cash_beta_ocf_low_beta_low_idio_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_ocf_to_debt_ttm * -1 * ts_beta((close / (ts_delay(close, 1) + 1e-12) - '
              '1), idx852_r1, 10)) * ((1 / (ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1), '
              '10) + 1e-12))), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedCashBetaV5',
  'idea': '经营现金覆盖与低小盘 beta；低行业残差风险。 来源：daily_v3_v6_recovered:daily_guided_cash_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_300_growth_lowvol_gap_idx922_pullback_flip_120',
  'expr_tpl': 'ts_where(idx922_r20 < -0.08, ts_where(idx922_r60 > 0, -1 * '
              '(cs_rank((cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120))) / (ts_std((close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 120) + 1e-12))), '
              'cs_rank((cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120))) / (ts_std((close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 120) + 1e-12))), '
              'cs_rank((cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120))) / (ts_std((close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 120) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissLowVolPullbackFlip',
  'idea': '成长/效率 surprise、000300.SSE gap 结构和低指数残差波动共振；只在中期向上后的短期急跌回撤中翻转，处理 2021-10 类阶段性失效。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_regime_switch。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_total_asset_turnover_surprise_240',
  'expr_tpl': 'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 240)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 240) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalEfficiencySurprise',
  'idea': 'total_asset_turnover_ttm 相对自身历史均值的偏离。 来源：daily_data_dimension_archive_20260518:fundamental。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cash_defense_dividend_lowvol_jump_guard_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_dividend_yield_ttm * (1 / (ts_std(((close / (ts_delay(close, 1) + '
              '1e-12) - 1) - ind_r1), 20) + 1e-12))), 20) / (ts_mean(ts_abs(((fund_dividend_yield_ttm * (1 / '
              '(ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1), 20) + 1e-12)))) - '
              'ts_delay((fund_dividend_yield_ttm * (1 / (ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) '
              '- ind_r1), 20) + 1e-12))), 1)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyCashDividendDefenseV4',
  'idea': '股息率和低残差风险；现金股息低跳变。 来源：daily_v3_v6_recovered:daily_cash_dividend_defense_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_operating_revenue_growth_change_120',
  'expr_tpl': 'cs_rank(fund_operating_revenue_growth_ratio_ttm - '
              'ts_delay(fund_operating_revenue_growth_ratio_ttm, 120))',
  'param_grid': None,
  'sub_category': 'FundamentalGrowthChange',
  'idea': 'operating_revenue_growth_ratio_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_capacity_rebalance_demand_jump_guard_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_book_to_market_ratio_ttm * ts_mean(((close - low) / (high - low + '
              '1e-12)) / ((turnover / (ts_mean(turnover, 20) + 1e-12)) + 1e-12), 20)), 20) / '
              '(ts_mean(ts_abs(((fund_book_to_market_ratio_ttm * ts_mean(((close - low) / (high - low + '
              '1e-12)) / ((turnover / (ts_mean(turnover, 20) + 1e-12)) + 1e-12), 20))) - '
              'ts_delay((fund_book_to_market_ratio_ttm * ts_mean(((close - low) / (high - low + 1e-12)) / '
              '((turnover / (ts_mean(turnover, 20) + 1e-12)) + 1e-12), 20)), 1)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '价值信号低成交收盘需求；容量信号低跳变。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_corr_drop_value_stable_60',
  'expr_tpl': 'cs_rank(ts_mean((-1 * ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, '
              '60), 1) * fund_book_to_market_ratio_ttm), 60) / (ts_std((-1 * ts_delta(ts_corr((close / '
              '(ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 60), 1) * fund_book_to_market_ratio_ttm), 60) + '
              '1e-12))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '小盘相关性下降时的价值独立性；beta 期限结构稳定性。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_return_on_asset_surprise_120',
  'expr_tpl': 'cs_rank((fund_return_on_asset_ttm - ts_mean(fund_return_on_asset_ttm, 120)) / '
              '(ts_std(fund_return_on_asset_ttm, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalQualitySurprise',
  'idea': 'return_on_asset_ttm 相对自身历史均值的偏离。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_overnight_intraday_conflict_level_60',
  'expr_tpl': 'cs_rank(ts_mean((open / (ts_delay(close, 1) + 1e-12) - 1) - (close / (open + 1e-12) - 1), '
              '60))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryV3',
  'idea': '隔夜与日内冲突 的路径均值。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_ind_range_adjusted_relative_strength_3',
  'expr_tpl': 'cs_rank(((close / (ts_delay(close, 3) + 1e-12) - 1) - ind_r3) / (ts_mean((high - low) / '
              '(close + 1e-12), 3) - ind_range3 + 1e-12))',
  'param_grid': None,
  'sub_category': 'IndustryVolAdjusted',
  'idea': '相对行业收益除以相对行业振幅消耗。 来源：daily_data_dimension_archive_20260518:industry。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_value_capacity_ep_size_capacity_low_idio_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_ep_ratio_ttm * (1 / (fund_market_cap + 1e-12))) * ((1 / (ts_std(((close '
              '/ (ts_delay(close, 1) + 1e-12) - 1) - ind_r1), 10) + 1e-12))), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedValueCapacityV5',
  'idea': '盈利收益率与小市值容量；叠加低行业残差风险。 来源：daily_v3_v6_recovered:daily_guided_value_capacity_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_impact_beta_ret_impact_value_low_idio_10',
  'expr_tpl': 'cs_rank((ts_mean(ts_abs((close / (ts_delay(close, 1) + 1e-12) - 1)) / (turnover + 1e-12), 10) '
              '* fund_book_to_market_ratio_ttm) * ((1 / (ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) '
              '- ind_r1), 10) + 1e-12))))',
  'param_grid': None,
  'sub_category': 'DailyGuidedImpactBetaV5',
  'idea': '收益冲击成本与价值折价交叉；叠加低行业残差风险。 来源：daily_v3_v6_recovered:daily_guided_impact_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cash_defense_dividend_lowvol_stable_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_dividend_yield_ttm * (1 / (ts_std(((close / (ts_delay(close, 1) + '
              '1e-12) - 1) - ind_r1), 20) + 1e-12))), 20) / (ts_std((fund_dividend_yield_ttm * (1 / '
              '(ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1), 20) + 1e-12))), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyCashDividendDefenseV4',
  'idea': '股息率和低残差风险；现金股息稳定性。 来源：daily_v3_v6_recovered:daily_cash_dividend_defense_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_fcf_value_low_corr_10',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 10) * (1 - ts_abs(ts_corr((close / '
              '(ts_delay(close, 1) + 1e-12) - 1), ind_r1, 10))))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionFcfValueConsensusV3',
  'idea': '自由现金流和估值安全边际共同确认。 的认知滞后/确认：基本面好且低行业相关。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_size_turn_capacity_stable_10',
  'expr_tpl': 'cs_rank(ts_mean(((1 / (fund_market_cap + 1e-12)) * (1 / (ts_mean((turnover / '
              '(ts_mean(turnover, 10) + 1e-12)), 10) + 1e-12))), 10) / (ts_std(((1 / (fund_market_cap + '
              '1e-12)) * (1 / (ts_mean((turnover / (ts_mean(turnover, 10) + 1e-12)), 10) + 1e-12))), 10) + '
              '1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '小市值且成交关注低的容量折价；容量信号稳定性。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_value_impact_buffer_carry_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_book_to_market_ratio_ttm / (ts_mean(ts_abs((close / (ts_delay(close, 1) '
              '+ 1e-12) - 1)) / (turnover + 1e-12), 10) + 1e-12)), 10))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '价值信号除以冲击成本；容量约束 carry。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_nearmiss_300_growth_lowvol_gap_idx922_meltup_flip_120',
  'expr_tpl': 'ts_where(idx922_r20 > 0.10, -1 * (cs_rank((cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120))) / (ts_std((close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 120) + 1e-12))), '
              'cs_rank((cs_rank((fund_operating_revenue_growth_ratio_ttm - '
              'ts_mean(fund_operating_revenue_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm, 120) + 1e-12)) + '
              'cs_rank((fund_total_asset_turnover_ttm - ts_mean(fund_total_asset_turnover_ttm, 120)) / '
              '(ts_std(fund_total_asset_turnover_ttm, 120) + 1e-12)) + cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1 - idx300_r1) > 0, (high - close) / (high - low + 1e-12), 0), '
              '120))) / (ts_std((close / (ts_delay(close, 1) + 1e-12) - 1) - idx300_r1, 120) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'CompositeNearMissLowVolMeltupFlip',
  'idea': '成长/效率 surprise、000300.SSE gap 结构和低指数残差波动共振；只在指数 20 日极端上涨时翻转，处理 melt-up 中质量/防御因子的阶段性反向。 '
          '来源：daily_data_dimension_archive_20260518:cross_source_regime_switch。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_peer_roe_resid_accel_120',
  'expr_tpl': 'cs_rank((fund_return_on_equity_ttm_ind_resid - ts_delay(fund_return_on_equity_ttm_ind_resid, '
              '120)) / (ts_std(fund_return_on_equity_ttm_ind_resid, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPeerResidualAccelerationV3',
  'idea': 'ROE 行业内残差加速。 来源：daily_v3_v6_recovered:daily_peer_dispersion_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_gross_profit_margin_surprise_120',
  'expr_tpl': 'cs_rank((fund_gross_profit_margin_ttm - ts_mean(fund_gross_profit_margin_ttm, 120)) / '
              '(ts_std(fund_gross_profit_margin_ttm, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalQualitySurprise',
  'idea': 'gross_profit_margin_ttm 相对自身历史均值的偏离。 来源：daily_data_dimension_archive_20260518:fundamental。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_account_receivable_turnover_rate_surprise_240',
  'expr_tpl': 'cs_rank((fund_account_receivable_turnover_rate_ttm - '
              'ts_mean(fund_account_receivable_turnover_rate_ttm, 240)) / '
              '(ts_std(fund_account_receivable_turnover_rate_ttm, 240) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalEfficiencySurprise',
  'idea': 'account_receivable_turnover_rate_ttm 相对自身历史均值的偏离。 '
          '来源：daily_data_dimension_archive_20260518:fundamental。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_impact_beta_ret_impact_value_price_lag_20',
  'expr_tpl': 'cs_rank((ts_mean(ts_abs((close / (ts_delay(close, 1) + 1e-12) - 1)) / (turnover + 1e-12), 20) '
              '* fund_book_to_market_ratio_ttm) * ((-1 * ((close / (ts_delay(close, 20) + 1e-12) - 1) - '
              'ind_r20))))',
  'param_grid': None,
  'sub_category': 'DailyGuidedImpactBetaV5',
  'idea': '收益冲击成本与价值折价交叉；叠加行业相对价格尚未反映。 来源：daily_v3_v6_recovered:daily_guided_impact_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_r22_turnover_without_price_stable_rank_20',
  'expr_tpl': 'cs_rank(ts_mean((turnover / (ts_mean(turnover, 20) + 1e-12)) - ts_abs(close / '
              '(ts_delay(close, 1) + 1e-12) - 1), 20) / (ts_std((turnover / (ts_mean(turnover, 20) + 1e-12)) '
              '- ts_abs(close / (ts_delay(close, 1) + 1e-12) - 1), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'Round22CrowdingUnwind',
  'idea': '成交拥挤、放量无效、资金冲击和拥挤释放。 来源：strict2636_theme_recovered_from_ai_cs。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_ep_change_120',
  'expr_tpl': 'cs_rank(fund_ep_ratio_ttm - ts_delay(fund_ep_ratio_ttm, 120))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationChange',
  'idea': 'ep_ratio_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_stable_qv_debt_cover_value_low_idio_10',
  'expr_tpl': 'cs_rank(ts_mean(((fund_ocf_to_debt_ttm * fund_book_to_market_ratio_ttm)) * ((1 / '
              '(ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1), 10) + 1e-12))), 10))',
  'param_grid': None,
  'sub_category': 'DailyV6StableQualityValue',
  'idea': '债务现金覆盖与价值，叠加低行业残差风险，偏稳定慢收益来源。 来源：daily_v3_v6_recovered:daily_v6_stable_quality_value。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_net_profit_growth_surprise_120',
  'expr_tpl': 'cs_rank((fund_net_profit_growth_ratio_ttm - ts_mean(fund_net_profit_growth_ratio_ttm, 120)) / '
              '(ts_std(fund_net_profit_growth_ratio_ttm, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalGrowthSurprise',
  'idea': 'net_profit_growth_ratio_ttm 相对自身历史均值的偏离。 来源：daily_data_dimension_archive_20260518:fundamental。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cash_defense_dividend_lowvol_price_lag_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_dividend_yield_ttm * (1 / (ts_std(((close / (ts_delay(close, 1) + '
              '1e-12) - 1) - ind_r1), 20) + 1e-12))), 20) * -1 * ((close / (ts_delay(close, 20) + 1e-12) - '
              '1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'DailyCashDividendDefenseV4',
  'idea': '股息率和低残差风险；现金股息尚未被行业相对价格反映。 来源：daily_v3_v6_recovered:daily_cash_dividend_defense_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_quality_growth_low_idio_vol_60',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_return_on_equity_ttm_ind_z) + '
              'cs_rank(fund_return_on_asset_ttm_ind_z) + cs_rank(fund_gross_profit_margin_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z) + '
              'cs_rank(fund_net_profit_growth_ratio_ttm_ind_z)), 60) / (ts_std((close / (ts_delay(close, 1) '
              '+ 1e-12) - 1) - ind_r1, 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'CompositeQualityGrowthConsensusLowVol',
  'idea': '盈利质量和收入/利润成长同时占优。 并要求行业残差波动较低。 来源：daily_data_dimension_archive_20260518:fundamental_composite。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx852_residual_skew_reversal_40',
  'expr_tpl': '-1 * cs_rank(ts_skew((close / (ts_delay(close, 1) + 1e-12) - 1) - idx852_r1, 40))',
  'param_grid': None,
  'sub_category': 'IndexResidualShape',
  'idea': '指数残差收益正偏后的反转检验。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_undervalued_growth_score_stability_10',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_ep_ratio_ttm_ind_z) + cs_rank(fund_sp_ratio_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z) + '
              'cs_rank(fund_net_profit_growth_ratio_ttm_ind_z)), 10) * '
              '((ts_mean((cs_rank(fund_ep_ratio_ttm_ind_z) + cs_rank(fund_sp_ratio_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z) + '
              'cs_rank(fund_net_profit_growth_ratio_ttm_ind_z)), 10) / '
              '(ts_std((cs_rank(fund_ep_ratio_ttm_ind_z) + cs_rank(fund_sp_ratio_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z) + '
              'cs_rank(fund_net_profit_growth_ratio_ttm_ind_z)), 10) + 1e-12))))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionUndervaluedGrowthConsensusV3',
  'idea': '低估值收入/利润成长组合。 的认知滞后/确认：基本面组合稳定。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_ind_residual_path_efficiency_3',
  'expr_tpl': 'cs_rank(ts_mean(((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1) / ((high - low) / '
              '(close + 1e-12) - ind_range1 + 1e-12), 3))',
  'param_grid': None,
  'sub_category': 'IndustryPathEfficiency',
  'idea': '单位相对行业振幅带来的残差收益。 来源：daily_data_dimension_archive_20260518:industry。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_turn_cluster_scaled_20',
  'expr_tpl': 'cs_rank(ts_mean((turnover / (ts_mean(turnover, 20) + 1e-12)) / (ts_mean((turnover / '
              '(ts_mean(turnover, 20) + 1e-12)), 5) + 1e-12), 20) / (ts_std((turnover / (ts_mean(turnover, '
              '20) + 1e-12)) / (ts_mean((turnover / (ts_mean(turnover, 20) + 1e-12)), 5) + 1e-12), 20) + '
              '1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityScaledV3',
  'idea': '成交活跃聚集 除以自身波动。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_undervalued_growth_vol_compress_10',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_ep_ratio_ttm_ind_z) + cs_rank(fund_sp_ratio_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z) + '
              'cs_rank(fund_net_profit_growth_ratio_ttm_ind_z)), 10) * (1 / (ts_std((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 10) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionUndervaluedGrowthConsensusV3',
  'idea': '低估值收入/利润成长组合。 的认知滞后/确认：基本面好且残差波动压缩。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_return_on_equity_change_120',
  'expr_tpl': 'cs_rank(fund_return_on_equity_ttm - ts_delay(fund_return_on_equity_ttm, 120))',
  'param_grid': None,
  'sub_category': 'FundamentalQualityChange',
  'idea': 'return_on_equity_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_vwap_stretch_fade_jump_damped_60',
  'expr_tpl': 'cs_rank(ts_mean(-1 * (close / (vwap + 1e-12) - 1) * ((high - low) / (close + 1e-12)), 60) / '
              '(ts_mean(ts_abs((-1 * (close / (vwap + 1e-12) - 1) * ((high - low) / (close + 1e-12))) - '
              'ts_delay(-1 * (close / (vwap + 1e-12) - 1) * ((high - low) / (close + 1e-12)), 1)), 60) + '
              '1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryJumpDampedV3',
  'idea': 'VWAP 拉伸回落 的低跳变稳定版本。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_ind_up_vol_compress_40',
  'expr_tpl': 'ts_where(ind_r20 > 0, cs_rank(ts_mean(-1 * ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), '
              '40), 40)), -1 * cs_rank(ts_mean(-1 * ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), 40), '
              '40)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '行业短期上行状态下使用波动压缩，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_wet_exhaustion_crowd_flip_20',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 20) + 1e-12)) > 1.5, -1 * cs_rank(ts_mean(-1 * (close '
              '/ (ts_delay(close, 1) + 1e-12) - 1) * (turnover / (ts_mean(turnover, 20) + 1e-12)), 20)), '
              'cs_rank(ts_mean(-1 * (close / (ts_delay(close, 1) + 1e-12) - 1) * (turnover / '
              '(ts_mean(turnover, 20) + 1e-12)), 20)))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityCrowdFlipV3',
  'idea': '放量涨跌耗尽 在成交拥挤时翻转。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_gap_up_fade_level_60',
  'expr_tpl': 'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1) > 0, (open - close) / '
              '(open + 1e-12), 0), 60))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryV3',
  'idea': '高开回落 的路径均值。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_ind_idio_vol_compression_3',
  'expr_tpl': '-1 * cs_rank(ts_std((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 3))',
  'param_grid': None,
  'sub_category': 'IndustryResidualVol',
  'idea': '相对行业残差波动压缩。 来源：daily_data_dimension_archive_20260518:industry。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_dv02_participation_volatility_penalty_40',
  'expr_tpl': '-1 * cs_rank(ts_std(turnover / (ts_mean(turnover, 40) + 1e-12), 40))',
  'param_grid': None,
  'sub_category': 'DiverseStrictR02ParticipationCrowding',
  'idea': 'strict 5轮结果：成交参与度、拥挤交易和放量无效/缩量有效。 来源：strict2636_theme_recovered_from_ai_cs。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_b_liq_turnover_compression_40',
  'expr_tpl': '-1 * cs_zscore(ts_std(turnover / (ts_mean(turnover, 40) + 1e-12), 40))',
  'param_grid': None,
  'sub_category': 'Liquidity',
  'idea': 'Compressed turnover variability after normalization may identify stable participation regimes. '
          '来源：strict2636_theme_recovered_from_ai_cs。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_free_cash_flow_company_indz_price_confirm_240',
  'expr_tpl': 'cs_rank(ts_mean(fund_free_cash_flow_company_per_share_ttm_ind_z, 240) * ((close / '
              '(ts_delay(close, 240) + 1e-12) - 1) - ind_r240))',
  'param_grid': None,
  'sub_category': 'FundamentalCashflowIndustryNeutralPriceConfirm',
  'idea': 'free_cash_flow_company_per_share_ttm 行业内优势被行业残差价格趋势确认。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_capacity_range_discount_stable_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_ep_ratio_ttm * (1 / (ts_mean(((high - low) / (close + 1e-12)), 20) + '
              '1e-12))), 20) / (ts_std((fund_ep_ratio_ttm * (1 / (ts_mean(((high - low) / (close + 1e-12)), '
              '20) + 1e-12))), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '盈利收益率要求低振幅消耗；容量信号稳定性。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_net_profit_margin_surprise_120',
  'expr_tpl': 'cs_rank((fund_net_profit_margin_ttm - ts_mean(fund_net_profit_margin_ttm, 120)) / '
              '(ts_std(fund_net_profit_margin_ttm, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalQualitySurprise',
  'idea': 'net_profit_margin_ttm 相对自身历史均值的偏离。 来源：daily_data_dimension_archive_20260518:fundamental。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_operating_cash_flow_indz_price_confirm_120',
  'expr_tpl': 'cs_rank(ts_mean(fund_operating_cash_flow_per_share_ttm_ind_z, 120) * ((close / '
              '(ts_delay(close, 120) + 1e-12) - 1) - ind_r120))',
  'param_grid': None,
  'sub_category': 'FundamentalCashflowIndustryNeutralPriceConfirm',
  'idea': 'operating_cash_flow_per_share_ttm 行业内优势被行业残差价格趋势确认。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_peer_gross_margin_resid_accel_120',
  'expr_tpl': 'cs_rank((fund_gross_profit_margin_ttm_ind_resid - '
              'ts_delay(fund_gross_profit_margin_ttm_ind_resid, 120)) / '
              '(ts_std(fund_gross_profit_margin_ttm_ind_resid, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPeerResidualAccelerationV3',
  'idea': '毛利率 行业内残差加速。 来源：daily_v3_v6_recovered:daily_peer_dispersion_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_turn_hot_range_eff_120',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 120) + 1e-12)) > 1.5, cs_rank(ts_mean((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) / (((high - low) / (close + 1e-12)) + 1e-12), 120)), -1 * '
              'cs_rank(ts_mean((close / (ts_delay(close, 1) + 1e-12) - 1) / (((high - low) / (close + '
              '1e-12)) + 1e-12), 120)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '个股成交拥挤状态下使用路径效率，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_fcf_value_vwap_absorb_20',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 20) * (ts_mean((close / (vwap + 1e-12) - 1), '
              '20)))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionFcfValueConsensusV3',
  'idea': '自由现金流和估值安全边际共同确认。 的认知滞后/确认：基本面好且 VWAP 吸收。 '
          '来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_efficiency_growth_score_delta_60',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_total_asset_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z) + '
              'cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z)), 60) * '
              '((ts_mean((cs_rank(fund_total_asset_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z) + '
              'cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z)), 60) - '
              'ts_delay(ts_mean((cs_rank(fund_total_asset_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z) + '
              'cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z)), 60), 60))))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionEfficiencyGrowthConsensusV3',
  'idea': '经营效率改善和收入成长同时占优。 的认知滞后/确认：基本面组合自身改善。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_rotation_small_corr_drop_dividend_60',
  'expr_tpl': 'cs_rank(ts_mean(((-1 * ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), '
              'idx852_r1, 60), 1))) * (fund_dividend_yield_ttm), 60))',
  'param_grid': None,
  'sub_category': 'DailyV6IndexRotationCarry',
  'idea': '小盘相关性下降 与 股息 交叉，捕捉指数轮动中的慢因子暴露。 来源：daily_v3_v6_recovered:daily_v6_index_rotation_carry。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_market_cap_indz_price_divergence_240',
  'expr_tpl': 'cs_rank(ts_mean(fund_market_cap_ind_z, 240) * (-1 * ((close / (ts_delay(close, 240) + 1e-12) '
              '- 1) - ind_r240)))',
  'param_grid': None,
  'sub_category': 'FundamentalSizeIndustryNeutralContrarian',
  'idea': 'market_cap 行业内优势但价格相对行业滞后，测试修复收益。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_ind_vwap_crowded_reversal_40',
  'expr_tpl': '-1 * cs_rank(ts_mean(close / (vwap + 1e-12) - 1, 40) - ind_vwap_dev40)',
  'param_grid': None,
  'sub_category': 'IndustryVwap',
  'idea': '个股相对行业持续收在 VWAP 上方后的拥挤反转。 来源：daily_data_dimension_archive_20260518:industry。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_cash_beta_div_low_beta_low_idio_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_dividend_yield_ttm * -1 * ts_beta((close / (ts_delay(close, 1) + 1e-12) '
              '- 1), idx852_r1, 10)) * ((1 / (ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1), '
              '10) + 1e-12))), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedCashBetaV5',
  'idea': '股息与低小盘 beta；低行业残差风险。 来源：daily_v3_v6_recovered:daily_guided_cash_beta_v5。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_market_cap_indz_low_idio_vol_20',
  'expr_tpl': 'cs_rank(ts_mean(fund_market_cap_ind_z, 20) / (ts_std((close / (ts_delay(close, 1) + 1e-12) - '
              '1) - ind_r1, 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalSizeIndustryNeutralLowVol',
  'idea': 'market_cap 行业内优势叠加低行业残差波动。 来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_beta_stability_quality_change_60',
  'expr_tpl': 'cs_rank(ts_mean(((fund_return_on_equity_ttm / (ts_std(ts_beta((close / (ts_delay(close, 1) + '
              '1e-12) - 1), idx852_r1, 60), 60) + 1e-12))) - ts_delay((fund_return_on_equity_ttm / '
              '(ts_std(ts_beta((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 60), 60) + 1e-12)), '
              '1), 60))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '小盘 beta 稳定质量；beta 期限结构的慢速变化。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_fcf_value_score_delta_10',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 10) * '
              '((ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 10) - '
              'ts_delay(ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 10), 10))))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionFcfValueConsensusV3',
  'idea': '自由现金流和估值安全边际共同确认。 的认知滞后/确认：基本面组合自身改善。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_operating_revenue_growth_indz_change_120',
  'expr_tpl': 'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z - '
              'ts_delay(fund_operating_revenue_growth_ratio_ttm_ind_z, 120))',
  'param_grid': None,
  'sub_category': 'FundamentalGrowthIndustryNeutralChange',
  'idea': 'operating_revenue_growth_ratio_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_gap_down_repair_shape_60',
  'expr_tpl': 'cs_rank(ts_skew(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1) < 0, (close - open) / '
              '(open + 1e-12), 0), 60))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryShapeV3',
  'idea': '低开修复 的分布偏态。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_r67_turnover_stability_dryup_confirm_10',
  'expr_tpl': 'ts_mean(((-1 * ts_std((turnover / (ts_mean(turnover, 10) + 1e-12)), 10))) * '
              '(ts_mean(turnover, 10) / (turnover + 1e-12)), 10)',
  'param_grid': None,
  'sub_category': 'Mega1000R67TurnoverStability',
  'idea': '成交参与稳定降低冲击成本。 缩量低拥挤确认。 来源：strict2636_theme_recovered_from_ai_cs。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_capital_efficiency_score_delta_60',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_return_on_asset_ttm_ind_z) + '
              'cs_rank(fund_total_asset_turnover_ttm_ind_z) + cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z)), 60) * '
              '((ts_mean((cs_rank(fund_return_on_asset_ttm_ind_z) + '
              'cs_rank(fund_total_asset_turnover_ttm_ind_z) + cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z)), 60) - '
              'ts_delay(ts_mean((cs_rank(fund_return_on_asset_ttm_ind_z) + '
              'cs_rank(fund_total_asset_turnover_ttm_ind_z) + cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z)), 60), 60))))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionCapitalEfficiencyConsensusV3',
  'idea': '资产回报和营运周转效率共同占优。 的认知滞后/确认：基本面组合自身改善。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_pb_indz_change_20',
  'expr_tpl': 'cs_rank(fund_pb_ratio_ttm_ind_z - ts_delay(fund_pb_ratio_ttm_ind_z, 20))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationIndustryNeutralChange',
  'idea': 'pb_ratio_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_account_receivable_turnover_rate_indz_change_120',
  'expr_tpl': 'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z - '
              'ts_delay(fund_account_receivable_turnover_rate_ttm_ind_z, 120))',
  'param_grid': None,
  'sub_category': 'FundamentalEfficiencyIndustryNeutralChange',
  'idea': 'account_receivable_turnover_rate_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_accounting_smooth_roe_growth_spread_stable_60',
  'expr_tpl': 'cs_rank(ts_mean((fund_return_on_equity_ttm - fund_net_asset_growth_ratio_ttm), 60) / '
              '(ts_std((fund_return_on_equity_ttm - fund_net_asset_growth_ratio_ttm), 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyV6AccountingSmooth',
  'idea': 'ROE 高于净资产扩张；低波动稳定版本。 来源：daily_v3_v6_recovered:daily_v6_accounting_smooth。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cap_alloc_roic_proxy_industry_disagree_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_return_on_asset_ttm * fund_total_asset_turnover_ttm), 20) * -1 * '
              '((close / (ts_delay(close, 20) + 1e-12) - 1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'DailyCapitalAllocationV4',
  'idea': 'ROA 和资产周转构造的资本回报代理；资本配置优势和行业相对价格背离。 来源：daily_v3_v6_recovered:daily_capital_allocation_v4。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_gross_profit_growth_change_120',
  'expr_tpl': 'cs_rank(fund_gross_profit_growth_ratio_ttm - ts_delay(fund_gross_profit_growth_ratio_ttm, '
              '120))',
  'param_grid': None,
  'sub_category': 'FundamentalGrowthChange',
  'idea': 'gross_profit_growth_ratio_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_volume_close_support_shape_40',
  'expr_tpl': 'cs_rank(ts_skew(((close - low) / (high - low + 1e-12)) * (turnover / (ts_mean(turnover, 40) + '
              '1e-12)), 40))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryShapeV3',
  'idea': '放量收盘承接 的分布偏态。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_ocf_to_debt_stable_level_60',
  'expr_tpl': 'cs_rank(ts_mean(fund_ocf_to_debt_ttm, 60) / (ts_std(fund_ocf_to_debt_ttm, 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalCashflowStability',
  'idea': 'ocf_to_debt_ttm 水平高且自身波动低。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_fcf_value_low_idio_vol_60',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 60) / (ts_std((close / (ts_delay(close, 1) + '
              '1e-12) - 1) - ind_r1, 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'CompositeFcfValueConsensusLowVol',
  'idea': '自由现金流和估值安全边际共同确认。 并要求行业残差波动较低。 来源：daily_data_dimension_archive_20260518:fundamental_composite。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_peer_dividend_industry_repair_10',
  'expr_tpl': 'cs_rank(ts_mean(fund_dividend_yield_ttm_ind_mean, 10) * (-1 * ind_r10))',
  'param_grid': None,
  'sub_category': 'DailyPeerIndustryRepairV3',
  'idea': '股息率 所在行业基本面强但价格滞后。 来源：daily_v3_v6_recovered:daily_peer_dispersion_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_value_capacity_ep_size_capacity_corr_drop_40',
  'expr_tpl': 'cs_rank(ts_mean((fund_ep_ratio_ttm * (1 / (fund_market_cap + 1e-12))) * ((-1 * '
              'ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 40), 1))), 40))',
  'param_grid': None,
  'sub_category': 'DailyGuidedValueCapacityV5',
  'idea': '盈利收益率与小市值容量；叠加中证1000相关性下降。 来源：daily_v3_v6_recovered:daily_guided_value_capacity_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_r17_turnover_stability_stable_rank_10',
  'expr_tpl': 'cs_rank(ts_mean(1 / (ts_std(turnover / (ts_mean(turnover, 10) + 1e-12), 10) + 1e-12), 10) / '
              '(ts_std(1 / (ts_std(turnover / (ts_mean(turnover, 10) + 1e-12), 10) + 1e-12), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'Round17LiquidityDryupReplenish',
  'idea': '缩量漂移、流动性枯竭、成交恢复和单位成交价格推进。 来源：strict2636_theme_recovered_from_ai_cs。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_gap_per_turn_jump_damped_10',
  'expr_tpl': 'cs_rank(ts_mean(ts_abs((open / (ts_delay(close, 1) + 1e-12) - 1)) / ((turnover / '
              '(ts_mean(turnover, 10) + 1e-12)) + 1e-12), 10) / (ts_mean(ts_abs((ts_abs((open / '
              '(ts_delay(close, 1) + 1e-12) - 1)) / ((turnover / (ts_mean(turnover, 10) + 1e-12)) + 1e-12)) '
              '- ts_delay(ts_abs((open / (ts_delay(close, 1) + 1e-12) - 1)) / ((turnover / '
              '(ts_mean(turnover, 10) + 1e-12)) + 1e-12), 1)), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityJumpDampedV3',
  'idea': '单位成交跳空 的低跳变形态。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_ind_leader_follow_strength_10',
  'expr_tpl': 'cs_rank(ind_r10 * cs_rank(close / (ts_delay(close, 10) + 1e-12) - 1 - ind_r10))',
  'param_grid': None,
  'sub_category': 'IndustryLeaderFollower',
  'idea': '强行业中仍领先行业的个股。 来源：daily_data_dimension_archive_20260518:industry。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_ev_to_ebitda_change_60',
  'expr_tpl': 'cs_rank(fund_ev_to_ebitda_ttm - ts_delay(fund_ev_to_ebitda_ttm, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationChange',
  'idea': 'ev_to_ebitda_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_upper_pressure_per_turn_scaled_60',
  'expr_tpl': 'cs_rank(ts_mean(((high - close) / (high - low + 1e-12)) / ((turnover / (ts_mean(turnover, 60) '
              '+ 1e-12)) + 1e-12), 60) / (ts_std(((high - close) / (high - low + 1e-12)) / ((turnover / '
              '(ts_mean(turnover, 60) + 1e-12)) + 1e-12), 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityScaledV3',
  'idea': '低成交上影压力 除以自身波动。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_free_cash_flow_equity_change_60',
  'expr_tpl': 'cs_rank(fund_free_cash_flow_equity_per_share_ttm - '
              'ts_delay(fund_free_cash_flow_equity_per_share_ttm, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalCashflowChange',
  'idea': 'free_cash_flow_equity_per_share_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_free_cash_flow_equity_surprise_120',
  'expr_tpl': 'cs_rank((fund_free_cash_flow_equity_per_share_ttm - '
              'ts_mean(fund_free_cash_flow_equity_per_share_ttm, 120)) / '
              '(ts_std(fund_free_cash_flow_equity_per_share_ttm, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalCashflowSurprise',
  'idea': 'free_cash_flow_equity_per_share_ttm 相对自身历史均值的偏离。 '
          '来源：daily_data_dimension_archive_20260518:fundamental。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_efficiency_growth_low_corr_10',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_total_asset_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z) + '
              'cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z)), 10) * (1 - ts_abs(ts_corr((close / '
              '(ts_delay(close, 1) + 1e-12) - 1), ind_r1, 10))))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionEfficiencyGrowthConsensusV3',
  'idea': '经营效率改善和收入成长同时占优。 的认知滞后/确认：基本面好且低行业相关。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_pe_change_60',
  'expr_tpl': 'cs_rank(fund_pe_ratio_ttm - ts_delay(fund_pe_ratio_ttm, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationChange',
  'idea': 'pe_ratio_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cash_defense_dividend_neglect_stable_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_dividend_yield_ttm * (1 / (ts_mean((turnover / (ts_mean(turnover, 20) + '
              '1e-12)), 20) + 1e-12))), 20) / (ts_std((fund_dividend_yield_ttm * (1 / (ts_mean((turnover / '
              '(ts_mean(turnover, 20) + 1e-12)), 20) + 1e-12))), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyCashDividendDefenseV4',
  'idea': '股息收益未被成交关注；现金股息稳定性。 来源：daily_v3_v6_recovered:daily_cash_dividend_defense_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_sp_indz_change_60',
  'expr_tpl': 'cs_rank(fund_sp_ratio_ttm_ind_z - ts_delay(fund_sp_ratio_ttm_ind_z, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationIndustryNeutralChange',
  'idea': 'sp_ratio_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_value_capacity_bp_float_capacity_price_lag_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_book_to_market_ratio_ttm * (1 / (fund_a_share_market_val_in_circulation '
              '+ 1e-12))) * ((-1 * ((close / (ts_delay(close, 10) + 1e-12) - 1) - ind_r10))), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedValueCapacityV5',
  'idea': '账面价值与小流通市值容量；叠加行业相对价格滞后。 来源：daily_v3_v6_recovered:daily_guided_value_capacity_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx905_ind_dual_residual_strength_10',
  'expr_tpl': 'cs_rank(((close / (ts_delay(close, 10) + 1e-12) - 1) - idx905_r10) * ((close / '
              '(ts_delay(close, 10) + 1e-12) - 1) - ind_r10))',
  'param_grid': None,
  'sub_category': 'IndexIndustryInteraction',
  'idea': '同时跑赢 000905.SSE 与行业的双重残差强势。 来源：daily_data_dimension_archive_20260518:index_industry。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_repair_low_corr_repair_div_10',
  'expr_tpl': 'cs_rank(ts_mean(((1 - ts_abs(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, '
              '10))) * (-1 * ((close / (ts_delay(close, 10) + 1e-12) - 1) - ind_r10))) * '
              '(fund_dividend_yield_ttm), 10))',
  'param_grid': None,
  'sub_category': 'DailyV6FundamentalRepair',
  'idea': '低相关滞后修复 与 股息率 交叉，寻找未充分定价的修复。 来源：daily_v3_v6_recovered:daily_v6_fundamental_repair。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_peer_dividend_scarce_quality_20',
  'expr_tpl': 'cs_rank(ts_mean(fund_dividend_yield_ttm_ind_z, 20) / '
              '(ts_mean(fund_dividend_yield_ttm_ind_std, 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPeerScarcityV3',
  'idea': '股息率 行业内稀缺优势。 来源：daily_v3_v6_recovered:daily_peer_dispersion_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_market_cap_indz_change_20',
  'expr_tpl': 'cs_rank(fund_market_cap_ind_z - ts_delay(fund_market_cap_ind_z, 20))',
  'param_grid': None,
  'sub_category': 'FundamentalSizeIndustryNeutralChange',
  'idea': 'market_cap 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_gap_body_spread_40',
  'expr_tpl': 'ts_mean((open / (ts_delay(close, 1) + 1e-12) - 1) - (close / (open + 1e-12) - 1), 40)',
  'param_grid': None,
  'sub_category': 'AI4000R116TurnoverStabilityPlain',
  'idea': '换手稳定性，原始状态。 strict corr 扩展候选：core_gap_body_spread。 来源：strict2636_theme_recovered_from_ai_cs。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_peer_revenue_growth_resid_accel_120',
  'expr_tpl': 'cs_rank((fund_operating_revenue_growth_ratio_ttm_ind_resid - '
              'ts_delay(fund_operating_revenue_growth_ratio_ttm_ind_resid, 120)) / '
              '(ts_std(fund_operating_revenue_growth_ratio_ttm_ind_resid, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPeerResidualAccelerationV3',
  'idea': '收入成长 行业内残差加速。 来源：daily_v3_v6_recovered:daily_peer_dispersion_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_turn_vs_volume_level_20',
  'expr_tpl': 'cs_rank(ts_mean((turnover / (ts_delay(turnover, 1) + 1e-12) - 1) - (volume / '
              '(ts_delay(volume, 1) + 1e-12) - 1), 20))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityV3',
  'idea': '成交额和成交量加速差 的平滑水平。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_ep_capacity_jump_guard_40',
  'expr_tpl': 'cs_rank(ts_mean((fund_ep_ratio_ttm * (1 / (ts_mean((turnover / (ts_mean(turnover, 40) + '
              '1e-12)), 40) + 1e-12))), 40) / (ts_mean(ts_abs(((fund_ep_ratio_ttm * (1 / (ts_mean((turnover '
              '/ (ts_mean(turnover, 40) + 1e-12)), 40) + 1e-12)))) - ts_delay((fund_ep_ratio_ttm * (1 / '
              '(ts_mean((turnover / (ts_mean(turnover, 40) + 1e-12)), 40) + 1e-12))), 1)), 40) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '盈利收益率处在低成交容量压力下；容量信号低跳变。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_recognition_fcf_value_score_stability_10',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 10) * '
              '((ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 10) / '
              '(ts_std((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 10) + 1e-12))))',
  'param_grid': None,
  'sub_category': 'DailyRecognitionFcfValueConsensusV3',
  'idea': '自由现金流和估值安全边际共同确认。 的认知滞后/确认：基本面组合稳定。 来源：daily_v3_v6_recovered:daily_fundamental_recognition_v3。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_beta_stability_value_change_20',
  'expr_tpl': 'cs_rank(ts_mean(((fund_book_to_market_ratio_ttm / (ts_std(ts_beta((close / (ts_delay(close, '
              '1) + 1e-12) - 1), idx852_r1, 20), 20) + 1e-12))) - ts_delay((fund_book_to_market_ratio_ttm / '
              '(ts_std(ts_beta((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 20), 20) + 1e-12)), '
              '1), 20))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '小盘 beta 稳定价值；beta 期限结构的慢速变化。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_free_cash_flow_company_change_60',
  'expr_tpl': 'cs_rank(fund_free_cash_flow_company_per_share_ttm - '
              'ts_delay(fund_free_cash_flow_company_per_share_ttm, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalCashflowChange',
  'idea': 'free_cash_flow_company_per_share_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx852_residual_illiq_penalty_10',
  'expr_tpl': '-1 * cs_rank(ts_mean(ts_abs((close / (ts_delay(close, 1) + 1e-12) - 1) - idx852_r1) / '
              '(turnover / (ts_mean(turnover, 10) + 1e-12) + 1e-12), 10))',
  'param_grid': None,
  'sub_category': 'IndexLiquidityImpact',
  'idea': '单位成交活跃度承受的指数残差波动过高。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_net_profit_margin_change_60',
  'expr_tpl': 'cs_rank(fund_net_profit_margin_ttm - ts_delay(fund_net_profit_margin_ttm, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalQualityChange',
  'idea': 'net_profit_margin_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_ep_indz_change_60',
  'expr_tpl': 'cs_rank(fund_ep_ratio_ttm_ind_z - ts_delay(fund_ep_ratio_ttm_ind_z, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationIndustryNeutralChange',
  'idea': 'ep_ratio_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_gap_up_fade_shape_120',
  'expr_tpl': 'cs_rank(ts_skew(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1) > 0, (open - close) / '
              '(open + 1e-12), 0), 120))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryShapeV3',
  'idea': '高开回落 的分布偏态。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_ps_indz_change_20',
  'expr_tpl': 'cs_rank(fund_ps_ratio_ttm_ind_z - ts_delay(fund_ps_ratio_ttm_ind_z, 20))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationIndustryNeutralChange',
  'idea': 'ps_ratio_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_gross_profit_growth_surprise_240',
  'expr_tpl': 'cs_rank((fund_gross_profit_growth_ratio_ttm - ts_mean(fund_gross_profit_growth_ratio_ttm, '
              '240)) / (ts_std(fund_gross_profit_growth_ratio_ttm, 240) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalGrowthSurprise',
  'idea': 'gross_profit_growth_ratio_ttm 相对自身历史均值的偏离。 来源：daily_data_dimension_archive_20260518:fundamental。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_def_beta_cash_change_40',
  'expr_tpl': 'cs_rank(ts_mean(((fund_ocf_to_debt_ttm * -1 * ts_beta((close / (ts_delay(close, 1) + 1e-12) - '
              '1), idx852_r1, 40))) - ts_delay((fund_ocf_to_debt_ttm * -1 * ts_beta((close / '
              '(ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 40)), 1), 40))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '低小盘 beta 的现金覆盖防御；beta 期限结构的慢速变化。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_inventory_turnover_indz_low_idio_vol_20',
  'expr_tpl': 'cs_rank(ts_mean(fund_inventory_turnover_ttm_ind_z, 20) / (ts_std((close / (ts_delay(close, 1) '
              '+ 1e-12) - 1) - ind_r1, 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalEfficiencyIndustryNeutralLowVol',
  'idea': 'inventory_turnover_ttm 行业内优势叠加低行业残差波动。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_peer_dividend_resid_accel_10',
  'expr_tpl': 'cs_rank((fund_dividend_yield_ttm_ind_resid - ts_delay(fund_dividend_yield_ttm_ind_resid, 10)) '
              '/ (ts_std(fund_dividend_yield_ttm_ind_resid, 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPeerResidualAccelerationV3',
  'idea': '股息率 行业内残差加速。 来源：daily_v3_v6_recovered:daily_peer_dispersion_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_acc_cycle_inventory_margin_quality_unrecognized_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_inventory_turnover_ttm * fund_gross_profit_margin_ttm), 20) * -1 * '
              '((close / (ts_delay(close, 20) + 1e-12) - 1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'DailyAccountingCycleV4',
  'idea': '存货周转和毛利率同时成立的经营质量；经营质量尚未被行业相对价格反映。 来源：daily_v3_v6_recovered:daily_accounting_cycle_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_total_asset_turnover_indz_change_120',
  'expr_tpl': 'cs_rank(fund_total_asset_turnover_ttm_ind_z - ts_delay(fund_total_asset_turnover_ttm_ind_z, '
              '120))',
  'param_grid': None,
  'sub_category': 'FundamentalEfficiencyIndustryNeutralChange',
  'idea': 'total_asset_turnover_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_liq_sponsor_dry_repair_small_value_10',
  'expr_tpl': 'cs_rank(ts_mean((ts_mean((-1 * ((close / (ts_delay(close, 10) + 1e-12) - 1) - ind_r10)) / '
              '(ts_mean((turnover / (ts_mean(turnover, 10) + 1e-12)), 10) + 1e-12), 10)) * '
              '((fund_book_to_market_ratio_ttm * (1 / (fund_market_cap + 1e-12)))), 10))',
  'param_grid': None,
  'sub_category': 'DailyV6LiquiditySponsorship',
  'idea': '低成交相对滞后修复 与 小市值价值 交叉，识别低关注或低冲击的可交易慢信号。 来源：daily_v3_v6_recovered:daily_v6_liquidity_sponsorship。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_pe_indz_change_60',
  'expr_tpl': 'cs_rank(fund_pe_ratio_ttm_ind_z - ts_delay(fund_pe_ratio_ttm_ind_z, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationIndustryNeutralChange',
  'idea': 'pe_ratio_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_acc_cycle_gross_net_gap_control_jump_guard_60',
  'expr_tpl': 'cs_rank(ts_mean(-1 * (fund_gross_profit_margin_ttm - fund_net_profit_margin_ttm), 60) / '
              '(ts_mean(ts_abs((-1 * (fund_gross_profit_margin_ttm - fund_net_profit_margin_ttm)) - '
              'ts_delay(-1 * (fund_gross_profit_margin_ttm - fund_net_profit_margin_ttm), 1)), 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyAccountingCycleV4',
  'idea': '毛利到净利损耗较低的费用控制；要求会计链条低跳变。 来源：daily_v3_v6_recovered:daily_accounting_cycle_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_impact_beta_ret_impact_value_ind_down_20',
  'expr_tpl': 'cs_rank((ts_mean(ts_abs((close / (ts_delay(close, 1) + 1e-12) - 1)) / (turnover + 1e-12), 20) '
              '* fund_book_to_market_ratio_ttm) * (ts_mean(ts_where(ind_r1 < 0, (close / (ts_delay(close, 1) '
              '+ 1e-12) - 1) - ind_r1, 0), 20)))',
  'param_grid': None,
  'sub_category': 'DailyGuidedImpactBetaV5',
  'idea': '收益冲击成本与价值折价交叉；叠加行业压力期抗跌。 来源：daily_v3_v6_recovered:daily_guided_impact_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_overnight_intraday_conflict_shape_120',
  'expr_tpl': 'cs_rank(ts_skew((open / (ts_delay(close, 1) + 1e-12) - 1) - (close / (open + 1e-12) - 1), '
              '120))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryShapeV3',
  'idea': '隔夜与日内冲突 的分布偏态。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_turn_hot_vol_compress_10',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 10) + 1e-12)) > 1.5, cs_rank(ts_mean(-1 * '
              'ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), 10), 10)), -1 * cs_rank(ts_mean(-1 * '
              'ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), 10), 10)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '个股成交拥挤状态下使用波动压缩，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_volume_upper_pressure_jump_damped_60',
  'expr_tpl': 'cs_rank(ts_mean(((high - close) / (high - low + 1e-12)) * (turnover / (ts_mean(turnover, 60) '
              '+ 1e-12)), 60) / (ts_mean(ts_abs((((high - close) / (high - low + 1e-12)) * (turnover / '
              '(ts_mean(turnover, 60) + 1e-12))) - ts_delay(((high - close) / (high - low + 1e-12)) * '
              '(turnover / (ts_mean(turnover, 60) + 1e-12)), 1)), 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryJumpDampedV3',
  'idea': '放量上影压力 的低跳变稳定版本。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cash_defense_dividend_debt_cover_price_lag_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_dividend_yield_ttm * fund_ocf_to_debt_ttm), 20) * -1 * ((close / '
              '(ts_delay(close, 20) + 1e-12) - 1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'DailyCashDividendDefenseV4',
  'idea': '股息率被债务现金覆盖；现金股息尚未被行业相对价格反映。 来源：daily_v3_v6_recovered:daily_cash_dividend_defense_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_def_beta_dividend_stable_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_dividend_yield_ttm * -1 * ts_beta((close / (ts_delay(close, 1) + 1e-12) '
              '- 1), idx852_r1, 20)), 20) / (ts_std((fund_dividend_yield_ttm * -1 * ts_beta((close / '
              '(ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 20)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '低小盘 beta 的股息防御；beta 期限结构稳定性。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_gap_up_fade_hot_flip_60',
  'expr_tpl': 'ts_where(idx852_r20 > 0.08, -1 * cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + '
              '1e-12) - 1) > 0, (open - close) / (open + 1e-12), 0), 60)), cs_rank(ts_mean(ts_where((open / '
              '(ts_delay(close, 1) + 1e-12) - 1) > 0, (open - close) / (open + 1e-12), 0), 60)))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryHotFlipV3',
  'idea': '高开回落 在中证1000短期过热时翻转。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_weakday_vwap_recover_selfz_120',
  'expr_tpl': 'cs_rank(((ts_where(close < open, (close / (vwap + 1e-12) - 1), 0)) - ts_mean(ts_where(close < '
              'open, (close / (vwap + 1e-12) - 1), 0), 120)) / (ts_std(ts_where(close < open, (close / (vwap '
              '+ 1e-12) - 1), 0), 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetrySelfZV3',
  'idea': '弱实体日 VWAP 修复 的自身状态突变。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_range_elasticity_crowd_flip_120',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 120) + 1e-12)) > 1.5, -1 * cs_rank(ts_mean(((high - '
              'low) / (close + 1e-12)) / ((turnover / (ts_delay(turnover, 1) + 1e-12) - 1) + 1e-12), 120)), '
              'cs_rank(ts_mean(((high - low) / (close + 1e-12)) / ((turnover / (ts_delay(turnover, 1) + '
              '1e-12) - 1) + 1e-12), 120)))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityCrowdFlipV3',
  'idea': '振幅对成交加速弹性 在成交拥挤时翻转。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_impact_beta_gap_impact_dividend_corr_drop_60',
  'expr_tpl': 'cs_rank((ts_mean(ts_abs((open / (ts_delay(close, 1) + 1e-12) - 1)) / ((turnover / '
              '(ts_mean(turnover, 60) + 1e-12)) + 1e-12), 60) * fund_dividend_yield_ttm) * ((-1 * '
              'ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 60), 1))))',
  'param_grid': None,
  'sub_category': 'DailyGuidedImpactBetaV5',
  'idea': '跳空冲击成本与股息防御交叉；叠加中证1000相关性下降。 来源：daily_v3_v6_recovered:daily_guided_impact_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_r17_turnover_stability_neg_mean_rank_20',
  'expr_tpl': '-1 * cs_rank(ts_mean(1 / (ts_std(turnover / (ts_mean(turnover, 20) + 1e-12), 20) + 1e-12), '
              '20))',
  'param_grid': None,
  'sub_category': 'Round17LiquidityDryupReplenish',
  'idea': '缩量漂移、流动性枯竭、成交恢复和单位成交价格推进。 来源：strict2636_theme_recovered_from_ai_cs。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cash_defense_ocf_debt_value_jump_guard_40',
  'expr_tpl': 'cs_rank(ts_mean((fund_ocf_to_debt_ttm * fund_book_to_market_ratio_ttm), 40) / '
              '(ts_mean(ts_abs(((fund_ocf_to_debt_ttm * fund_book_to_market_ratio_ttm)) - '
              'ts_delay((fund_ocf_to_debt_ttm * fund_book_to_market_ratio_ttm), 1)), 40) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyCashDividendDefenseV4',
  'idea': '经营现金流覆盖债务的价值防御；现金股息低跳变。 来源：daily_v3_v6_recovered:daily_cash_dividend_defense_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_r22_turnover_without_price_dev_raw_40',
  'expr_tpl': '((turnover / (ts_mean(turnover, 40) + 1e-12)) - ts_abs(close / (ts_delay(close, 1) + 1e-12) - '
              '1)) - ts_mean((turnover / (ts_mean(turnover, 40) + 1e-12)) - ts_abs(close / (ts_delay(close, '
              '1) + 1e-12) - 1), 40)',
  'param_grid': None,
  'sub_category': 'Round22CrowdingUnwindRawCounterpart',
  'idea': '去掉上一版最外层 cs_rank 的 raw 对照，保留原始强度信息；评估层再决定是否 rank/zscore。 '
          '来源：strict2636_theme_recovered_from_ai_cs。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_accounting_price_inventory_margin_price_lag_vwap_10',
  'expr_tpl': 'cs_rank(ts_mean(((fund_inventory_turnover_ttm * fund_gross_profit_margin_ttm)) * '
              '(ts_mean((close / (vwap + 1e-12) - 1), 10)), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedAccountingPriceV5',
  'idea': '存货周转毛利质量；VWAP 吸收。 来源：daily_v3_v6_recovered:daily_guided_accounting_price_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_cash_beta_fcf_roa_beta_low_idio_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_free_cash_flow_company_per_share_ttm * fund_return_on_asset_ttm * -1 * '
              'ts_beta((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 10)) * ((1 / (ts_std(((close / '
              '(ts_delay(close, 1) + 1e-12) - 1) - ind_r1), 10) + 1e-12))), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedCashBetaV5',
  'idea': '公司自由现金流质量与低 beta；低行业残差风险。 来源：daily_v3_v6_recovered:daily_guided_cash_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_accounting_smooth_rev_asset_light_stable_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_operating_revenue_growth_ratio_ttm - '
              'fund_total_asset_growth_ratio_ttm), 10) / (ts_std((fund_operating_revenue_growth_ratio_ttm - '
              'fund_total_asset_growth_ratio_ttm), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyV6AccountingSmooth',
  'idea': '收入增长高于资产扩张；低波动稳定版本。 来源：daily_v3_v6_recovered:daily_v6_accounting_smooth。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_gross_profit_margin_change_60',
  'expr_tpl': 'cs_rank(fund_gross_profit_margin_ttm - ts_delay(fund_gross_profit_margin_ttm, 60))',
  'param_grid': None,
  'sub_category': 'FundamentalQualityChange',
  'idea': 'gross_profit_margin_ttm 的中期变化。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_acc_cycle_receivable_inventory_stability_risk_persist_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_account_receivable_turnover_rate_ttm / (fund_inventory_turnover_ttm + '
              '1e-12)), 10) * (1 / (ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1), 10) + '
              '1e-12)))',
  'param_grid': None,
  'sub_category': 'DailyAccountingCycleV4',
  'idea': '应收与存货周转结构稳定性；只奖励低残差风险下的持续经营质量。 来源：daily_v3_v6_recovered:daily_accounting_cycle_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_book_to_market_indz_change_20',
  'expr_tpl': 'cs_rank(fund_book_to_market_ratio_ttm_ind_z - ts_delay(fund_book_to_market_ratio_ttm_ind_z, '
              '20))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationIndustryNeutralChange',
  'idea': 'book_to_market_ratio_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_price_elasticity_level_10',
  'expr_tpl': 'cs_rank(ts_mean((close / (ts_delay(close, 1) + 1e-12) - 1) / ((turnover / (ts_delay(turnover, '
              '1) + 1e-12) - 1) + 1e-12), 10))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityV3',
  'idea': '价格对成交加速弹性 的平滑水平。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_gap_liquidity_absorb_scaled_60',
  'expr_tpl': 'cs_rank(ts_mean(-1 * (open / (ts_delay(close, 1) + 1e-12) - 1) * (turnover / '
              '(ts_mean(turnover, 60) + 1e-12)), 60) / (ts_std(-1 * (open / (ts_delay(close, 1) + 1e-12) - '
              '1) * (turnover / (ts_mean(turnover, 60) + 1e-12)), 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityScaledV3',
  'idea': '放量吸收跳空 除以自身波动。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_stable_qv_turnover_quality_vwap_absorb_10',
  'expr_tpl': 'cs_rank(ts_mean(((fund_total_asset_turnover_ttm * fund_return_on_asset_ttm)) * '
              '(ts_mean((close / (vwap + 1e-12) - 1), 10)), 10))',
  'param_grid': None,
  'sub_category': 'DailyV6StableQualityValue',
  'idea': '资产周转质量，叠加VWAP 吸收，偏稳定慢收益来源。 来源：daily_v3_v6_recovered:daily_v6_stable_quality_value。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_ind_upper_shadow_relative_exhaustion_3',
  'expr_tpl': '-1 * cs_rank(ts_mean((high - close) / (high - low + 1e-12), 3) - ind_upper_shadow3)',
  'param_grid': None,
  'sub_category': 'IndustryKlineStructure',
  'idea': '上影回落强于行业平均的耗尽结构。 来源：daily_data_dimension_archive_20260518:industry。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_capacity_beta_shadow_stable_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_book_to_market_ratio_ttm * (1 - ts_abs(ts_corr((close / '
              '(ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 10)))), 10) / '
              '(ts_std((fund_book_to_market_ratio_ttm * (1 - ts_abs(ts_corr((close / (ts_delay(close, 1) + '
              '1e-12) - 1), idx852_r1, 10)))), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '价值信号低指数相关；容量信号稳定性。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_ev_to_ebitda_indz_change_20',
  'expr_tpl': 'cs_rank(fund_ev_to_ebitda_ttm_ind_z - ts_delay(fund_ev_to_ebitda_ttm_ind_z, 20))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationIndustryNeutralChange',
  'idea': 'ev_to_ebitda_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_operating_cash_flow_surprise_240',
  'expr_tpl': 'cs_rank((fund_operating_cash_flow_per_share_ttm - '
              'ts_mean(fund_operating_cash_flow_per_share_ttm, 240)) / '
              '(ts_std(fund_operating_cash_flow_per_share_ttm, 240) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalCashflowSurprise',
  'idea': 'operating_cash_flow_per_share_ttm 相对自身历史均值的偏离。 '
          '来源：daily_data_dimension_archive_20260518:fundamental。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_gap_per_turn_scaled_10',
  'expr_tpl': 'cs_rank(ts_mean(ts_abs((open / (ts_delay(close, 1) + 1e-12) - 1)) / ((turnover / '
              '(ts_mean(turnover, 10) + 1e-12)) + 1e-12), 10) / (ts_std(ts_abs((open / (ts_delay(close, 1) + '
              '1e-12) - 1)) / ((turnover / (ts_mean(turnover, 10) + 1e-12)) + 1e-12), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityScaledV3',
  'idea': '单位成交跳空 除以自身波动。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_capacity_vwap_absorb_carry_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_book_to_market_ratio_ttm * ts_mean((close / (vwap + 1e-12) - 1), 20)), '
              '20))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '价值信号被 VWAP 吸收确认；容量约束 carry。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_vwap_stretch_fade_shape_40',
  'expr_tpl': 'cs_rank(ts_skew(-1 * (close / (vwap + 1e-12) - 1) * ((high - low) / (close + 1e-12)), 40))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryShapeV3',
  'idea': 'VWAP 拉伸回落 的分布偏态。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_value_capacity_ep_size_capacity_price_lag_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_ep_ratio_ttm * (1 / (fund_market_cap + 1e-12))) * ((-1 * ((close / '
              '(ts_delay(close, 10) + 1e-12) - 1) - ind_r10))), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedValueCapacityV5',
  'idea': '盈利收益率与小市值容量；叠加行业相对价格滞后。 来源：daily_v3_v6_recovered:daily_guided_value_capacity_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_overnight_intraday_conflict_hot_flip_60',
  'expr_tpl': 'ts_where(idx852_r20 > 0.08, -1 * cs_rank(ts_mean((open / (ts_delay(close, 1) + 1e-12) - 1) - '
              '(close / (open + 1e-12) - 1), 60)), cs_rank(ts_mean((open / (ts_delay(close, 1) + 1e-12) - 1) '
              '- (close / (open + 1e-12) - 1), 60)))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryHotFlipV3',
  'idea': '隔夜与日内冲突 在中证1000短期过热时翻转。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_free_cash_flow_company_surprise_240',
  'expr_tpl': 'cs_rank((fund_free_cash_flow_company_per_share_ttm - '
              'ts_mean(fund_free_cash_flow_company_per_share_ttm, 240)) / '
              '(ts_std(fund_free_cash_flow_company_per_share_ttm, 240) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalCashflowSurprise',
  'idea': 'free_cash_flow_company_per_share_ttm 相对自身历史均值的偏离。 '
          '来源：daily_data_dimension_archive_20260518:fundamental。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_ev_to_ebitda_surprise_120',
  'expr_tpl': 'cs_rank((fund_ev_to_ebitda_ttm - ts_mean(fund_ev_to_ebitda_ttm, 120)) / '
              '(ts_std(fund_ev_to_ebitda_ttm, 120) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationSurprise',
  'idea': 'ev_to_ebitda_ttm 相对自身历史均值的偏离。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_stable_qv_debt_cover_value_price_lag_10',
  'expr_tpl': 'cs_rank(ts_mean(((fund_ocf_to_debt_ttm * fund_book_to_market_ratio_ttm)) * ((-1 * ((close / '
              '(ts_delay(close, 10) + 1e-12) - 1) - ind_r10))), 10))',
  'param_grid': None,
  'sub_category': 'DailyV6StableQualityValue',
  'idea': '债务现金覆盖与价值，叠加行业相对价格滞后，偏稳定慢收益来源。 来源：daily_v3_v6_recovered:daily_v6_stable_quality_value。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_a_share_market_val_in_circulation_indz_level_240',
  'expr_tpl': 'cs_rank(ts_mean(fund_a_share_market_val_in_circulation_ind_z, 240))',
  'param_grid': None,
  'sub_category': 'FundamentalSizeIndustryNeutral',
  'idea': 'a_share_market_val_in_circulation 的行业内 zscore 慢变量水平。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_vwap_stretch_fade_hot_flip_40',
  'expr_tpl': 'ts_where(idx852_r20 > 0.08, -1 * cs_rank(ts_mean(-1 * (close / (vwap + 1e-12) - 1) * ((high - '
              'low) / (close + 1e-12)), 40)), cs_rank(ts_mean(-1 * (close / (vwap + 1e-12) - 1) * ((high - '
              'low) / (close + 1e-12)), 40)))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryHotFlipV3',
  'idea': 'VWAP 拉伸回落 在中证1000短期过热时翻转。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_accounting_smooth_sales_velocity_stable_40',
  'expr_tpl': 'cs_rank(ts_mean((fund_operating_revenue_growth_ratio_ttm * fund_total_asset_turnover_ttm), '
              '40) / (ts_std((fund_operating_revenue_growth_ratio_ttm * fund_total_asset_turnover_ttm), 40) '
              '+ 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyV6AccountingSmooth',
  'idea': '销售增长周转速度；低波动稳定版本。 来源：daily_v3_v6_recovered:daily_v6_accounting_smooth。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cash_defense_fcf_equity_quality_price_lag_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_free_cash_flow_equity_per_share_ttm * fund_return_on_equity_ttm), 20) * '
              '-1 * ((close / (ts_delay(close, 20) + 1e-12) - 1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'DailyCashDividendDefenseV4',
  'idea': '股权自由现金流和 ROE；现金股息尚未被行业相对价格反映。 来源：daily_v3_v6_recovered:daily_cash_dividend_defense_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_inventory_turnover_indz_price_confirm_20',
  'expr_tpl': 'cs_rank(ts_mean(fund_inventory_turnover_ttm_ind_z, 20) * ((close / (ts_delay(close, 20) + '
              '1e-12) - 1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'FundamentalEfficiencyIndustryNeutralPriceConfirm',
  'idea': 'inventory_turnover_ttm 行业内优势被行业残差价格趋势确认。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_return_on_equity_indz_change_120',
  'expr_tpl': 'cs_rank(fund_return_on_equity_ttm_ind_z - ts_delay(fund_return_on_equity_ttm_ind_z, 120))',
  'param_grid': None,
  'sub_category': 'FundamentalQualityIndustryNeutralChange',
  'idea': 'return_on_equity_ttm 行业内 zscore 的中期变化。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_efficiency_growth_low_turnover_calm_60',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_total_asset_turnover_ttm_ind_z) + '
              'cs_rank(fund_account_receivable_turnover_rate_ttm_ind_z) + '
              'cs_rank(fund_inventory_turnover_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z)), 60) / (ts_mean(turnover / '
              '(ts_mean(turnover, 60) + 1e-12), 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'CompositeEfficiencyGrowthConsensusLowCrowding',
  'idea': '经营效率改善和收入成长同时占优。 且不依赖放量拥挤。 来源：daily_data_dimension_archive_20260518:fundamental_composite。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_turn_cluster_jump_damped_20',
  'expr_tpl': 'cs_rank(ts_mean((turnover / (ts_mean(turnover, 20) + 1e-12)) / (ts_mean((turnover / '
              '(ts_mean(turnover, 20) + 1e-12)), 5) + 1e-12), 20) / (ts_mean(ts_abs(((turnover / '
              '(ts_mean(turnover, 20) + 1e-12)) / (ts_mean((turnover / (ts_mean(turnover, 20) + 1e-12)), 5) '
              '+ 1e-12)) - ts_delay((turnover / (ts_mean(turnover, 20) + 1e-12)) / (ts_mean((turnover / '
              '(ts_mean(turnover, 20) + 1e-12)), 5) + 1e-12), 1)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityJumpDampedV3',
  'idea': '成交活跃聚集 的低跳变形态。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_impact_beta_ret_impact_value_corr_drop_60',
  'expr_tpl': 'cs_rank((ts_mean(ts_abs((close / (ts_delay(close, 1) + 1e-12) - 1)) / (turnover + 1e-12), 60) '
              '* fund_book_to_market_ratio_ttm) * ((-1 * ts_delta(ts_corr((close / (ts_delay(close, 1) + '
              '1e-12) - 1), idx852_r1, 60), 1))))',
  'param_grid': None,
  'sub_category': 'DailyGuidedImpactBetaV5',
  'idea': '收益冲击成本与价值折价交叉；叠加中证1000相关性下降。 来源：daily_v3_v6_recovered:daily_guided_impact_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_impact_beta_body_impact_quality_corr_drop_60',
  'expr_tpl': 'cs_rank((ts_mean(ts_abs(((close - open) / (high - low + 1e-12))) / ((turnover / '
              '(ts_mean(turnover, 60) + 1e-12)) + 1e-12), 60) * fund_return_on_asset_ttm) * ((-1 * '
              'ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 60), 1))))',
  'param_grid': None,
  'sub_category': 'DailyGuidedImpactBetaV5',
  'idea': '实体冲击成本与 ROA 质量交叉；叠加中证1000相关性下降。 来源：daily_v3_v6_recovered:daily_guided_impact_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_return_on_asset_indz_price_divergence_240',
  'expr_tpl': 'cs_rank(ts_mean(fund_return_on_asset_ttm_ind_z, 240) * (-1 * ((close / (ts_delay(close, 240) '
              '+ 1e-12) - 1) - ind_r240)))',
  'param_grid': None,
  'sub_category': 'FundamentalQualityIndustryNeutralContrarian',
  'idea': 'return_on_asset_ttm 行业内优势但价格相对行业滞后，测试修复收益。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx852_residual_vol_breakout_60',
  'expr_tpl': 'cs_rank(ts_delta(ts_std((close / (ts_delay(close, 1) + 1e-12) - 1) - idx852_r1, 60), 1))',
  'param_grid': None,
  'sub_category': 'IndexResidualVol',
  'idea': '指数残差波动扩张。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_acc_cycle_asset_growth_efficiency_neglect_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_operating_revenue_growth_ratio_ttm - '
              'fund_total_asset_growth_ratio_ttm), 20) * (1 / (ts_mean((turnover / (ts_mean(turnover, 20) + '
              '1e-12)), 20) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'DailyAccountingCycleV4',
  'idea': '收入增长高于资产扩张的轻资产效率；经营质量处于低成交关注。 来源：daily_v3_v6_recovered:daily_accounting_cycle_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx852_beta_instability_penalty_5',
  'expr_tpl': '-1 * cs_rank(ts_std(ts_beta(close / (ts_delay(close, 1) + 1e-12) - 1, idx852_r1, 5), 5))',
  'param_grid': None,
  'sub_category': 'IndexBetaStability',
  'idea': '指数 beta 不稳定带来的风险惩罚。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_turn_accel_ind_stress_10',
  'expr_tpl': 'cs_rank(ts_mean((turnover / (ts_delay(turnover, 1) + 1e-12) - 1), 10) * '
              'ts_mean(ts_where(ind_r1 < 0, (close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 0), 10))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityStressV3',
  'idea': '成交额一阶加速 在行业压力期的抗跌确认。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cap_alloc_value_investment_gap_industry_disagree_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_book_to_market_ratio_ttm - fund_total_asset_growth_ratio_ttm), 20) * -1 '
              '* ((close / (ts_delay(close, 20) + 1e-12) - 1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'DailyCapitalAllocationV4',
  'idea': '账面价值相对资产扩张的投资约束；资本配置优势和行业相对价格背离。 来源：daily_v3_v6_recovered:daily_capital_allocation_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_closepos_consistency_selfz_10',
  'expr_tpl': 'cs_rank(((((close - low) / (high - low + 1e-12)) / (ts_std(((close - low) / (high - low + '
              '1e-12)), 10) + 1e-12)) - ts_mean(((close - low) / (high - low + 1e-12)) / (ts_std(((close - '
              'low) / (high - low + 1e-12)), 10) + 1e-12), 10)) / (ts_std(((close - low) / (high - low + '
              '1e-12)) / (ts_std(((close - low) / (high - low + 1e-12)), 10) + 1e-12), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetrySelfZV3',
  'idea': '收盘位置稳定性 的自身状态突变。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_accounting_price_inventory_margin_price_lag_lag_ind_10',
  'expr_tpl': 'cs_rank(ts_mean(((fund_inventory_turnover_ttm * fund_gross_profit_margin_ttm)) * ((-1 * '
              '((close / (ts_delay(close, 10) + 1e-12) - 1) - ind_r10))), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedAccountingPriceV5',
  'idea': '存货周转毛利质量；行业相对价格滞后。 来源：daily_v3_v6_recovered:daily_guided_accounting_price_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_pb_indz_price_divergence_240',
  'expr_tpl': 'cs_rank(ts_mean(fund_pb_ratio_ttm_ind_z, 240) * (-1 * ((close / (ts_delay(close, 240) + '
              '1e-12) - 1) - ind_r240)))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationIndustryNeutralContrarian',
  'idea': 'pb_ratio_ttm 行业内优势但价格相对行业滞后，测试修复收益。 '
          '来源：daily_data_dimension_archive_20260518:fundamental_industry_neutral。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_repair_low_corr_repair_book_20',
  'expr_tpl': 'cs_rank(ts_mean(((1 - ts_abs(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, '
              '20))) * (-1 * ((close / (ts_delay(close, 20) + 1e-12) - 1) - ind_r20))) * '
              '(fund_book_to_market_ratio_ttm), 20))',
  'param_grid': None,
  'sub_category': 'DailyV6FundamentalRepair',
  'idea': '低相关滞后修复 与 账面价值 交叉，寻找未充分定价的修复。 来源：daily_v3_v6_recovered:daily_v6_fundamental_repair。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_acc_cycle_inventory_margin_quality_risk_persist_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_inventory_turnover_ttm * fund_gross_profit_margin_ttm), 10) * (1 / '
              '(ts_std(((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1), 10) + 1e-12)))',
  'param_grid': None,
  'sub_category': 'DailyAccountingCycleV4',
  'idea': '存货周转和毛利率同时成立的经营质量；只奖励低残差风险下的持续经营质量。 来源：daily_v3_v6_recovered:daily_accounting_cycle_v4。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_upper_liquidity_mismatch_crowd_flip_120',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 120) + 1e-12)) > 1.5, -1 * cs_rank(ts_mean(((high - '
              'close) / (high - low + 1e-12)) - (turnover / (ts_mean(turnover, 120) + 1e-12)), 120)), '
              'cs_rank(ts_mean(((high - close) / (high - low + 1e-12)) - (turnover / (ts_mean(turnover, 120) '
              '+ 1e-12)), 120)))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityCrowdFlipV3',
  'idea': '上影压力与成交不匹配 在成交拥挤时翻转。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_r266_turnover_stability_high_range_raw_level_20',
  'expr_tpl': '(ts_where(((high - low) / (close + 1e-12)) > ts_mean(((high - low) / (close + 1e-12)), 20), '
              '(-1 * ts_std((turnover / (ts_mean(turnover, 20) + 1e-12)), 20)), 0))',
  'param_grid': None,
  'sub_category': 'AI4000R266TurnoverStabilityHighRange',
  'idea': '换手稳定性，高振幅状态。 原始信号强度。 来源：strict2636_theme_recovered_from_ai_cs。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_capacity_rebalance_supply_jump_guard_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_dividend_yield_ttm * -1 * ts_mean(((high - close) / (high - low + '
              '1e-12)) * (turnover / (ts_mean(turnover, 20) + 1e-12)), 20)), 20) / '
              '(ts_mean(ts_abs(((fund_dividend_yield_ttm * -1 * ts_mean(((high - close) / (high - low + '
              '1e-12)) * (turnover / (ts_mean(turnover, 20) + 1e-12)), 20))) - '
              'ts_delay((fund_dividend_yield_ttm * -1 * ts_mean(((high - close) / (high - low + 1e-12)) * '
              '(turnover / (ts_mean(turnover, 20) + 1e-12)), 20)), 1)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '股息信号避开放量上影供给；容量信号低跳变。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cash_defense_fcf_value_company_price_lag_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_free_cash_flow_company_per_share_ttm * fund_book_to_market_ratio_ttm), '
              '20) * -1 * ((close / (ts_delay(close, 20) + 1e-12) - 1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'DailyCashDividendDefenseV4',
  'idea': '公司自由现金流价值；现金股息尚未被行业相对价格反映。 来源：daily_v3_v6_recovered:daily_cash_dividend_defense_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_gap_down_repair_level_60',
  'expr_tpl': 'cs_rank(ts_mean(ts_where((open / (ts_delay(close, 1) + 1e-12) - 1) < 0, (close - open) / '
              '(open + 1e-12), 0), 60))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetryV3',
  'idea': '低开修复 的路径均值。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_undervalued_growth_low_turnover_calm_60',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_ep_ratio_ttm_ind_z) + cs_rank(fund_sp_ratio_ttm_ind_z) + '
              'cs_rank(fund_operating_revenue_growth_ratio_ttm_ind_z) + '
              'cs_rank(fund_net_profit_growth_ratio_ttm_ind_z)), 60) / (ts_mean(turnover / '
              '(ts_mean(turnover, 60) + 1e-12), 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'CompositeUndervaluedGrowthConsensusLowCrowding',
  'idea': '低估值收入/利润成长组合。 且不依赖放量拥挤。 来源：daily_data_dimension_archive_20260518:fundamental_composite。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_combo_fcf_value_low_turnover_calm_60',
  'expr_tpl': 'cs_rank(ts_mean((cs_rank(fund_free_cash_flow_company_per_share_ttm_ind_z) + '
              'cs_rank(fund_free_cash_flow_equity_per_share_ttm_ind_z) + cs_rank(fund_cfp_ratio_ttm_ind_z) + '
              'cs_rank(fund_book_to_market_ratio_ttm_ind_z)), 60) / (ts_mean(turnover / (ts_mean(turnover, '
              '60) + 1e-12), 60) + 1e-12))',
  'param_grid': None,
  'sub_category': 'CompositeFcfValueConsensusLowCrowding',
  'idea': '自由现金流和估值安全边际共同确认。 且不依赖放量拥挤。 来源：daily_data_dimension_archive_20260518:fundamental_composite。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_rotation_small_corr_drop_margin_value_40',
  'expr_tpl': 'cs_rank(ts_mean(((-1 * ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), '
              'idx852_r1, 40), 1))) * ((fund_net_profit_margin_ttm * fund_ep_ratio_ttm)), 40))',
  'param_grid': None,
  'sub_category': 'DailyV6IndexRotationCarry',
  'idea': '小盘相关性下降 与 利润率价值 交叉，捕捉指数轮动中的慢因子暴露。 来源：daily_v3_v6_recovered:daily_v6_index_rotation_carry。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_idx_hot_intraday_follow_60',
  'expr_tpl': 'ts_where(idx852_r20 > 0.08, cs_rank(ts_mean((close / (open + 1e-12) - 1), 60)), -1 * '
              'cs_rank(ts_mean((close / (open + 1e-12) - 1), 60)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '中证1000短期过热状态下使用日内跟随，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx906_residual_vol_compression_3',
  'expr_tpl': '-1 * cs_rank(ts_std((close / (ts_delay(close, 1) + 1e-12) - 1) - idx906_r1, 3))',
  'param_grid': None,
  'sub_category': 'IndexResidualVol',
  'idea': '指数残差波动压缩。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_r17_turnover_stability_dev_rank_20',
  'expr_tpl': 'cs_rank((1 / (ts_std(turnover / (ts_mean(turnover, 20) + 1e-12), 20) + 1e-12)) - ts_mean(1 / '
              '(ts_std(turnover / (ts_mean(turnover, 20) + 1e-12), 20) + 1e-12), 20))',
  'param_grid': None,
  'sub_category': 'Round17LiquidityDryupReplenish',
  'idea': '缩量漂移、流动性枯竭、成交恢复和单位成交价格推进。 来源：strict2636_theme_recovered_from_ai_cs。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_impact_beta_wet_exhaust_beta_beta_stable_60',
  'expr_tpl': 'cs_rank((-1 * ts_mean((close / (ts_delay(close, 1) + 1e-12) - 1) * (turnover / '
              '(ts_mean(turnover, 60) + 1e-12)), 60) * (ts_beta((close / (ts_delay(close, 1) + 1e-12) - 1), '
              'idx852_r1, 60) - ts_beta((close / (ts_delay(close, 1) + 1e-12) - 1), idx300_r1, 60))) * ((1 / '
              '(ts_std(ts_beta((close / (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 60), 60) + 1e-12))))',
  'param_grid': None,
  'sub_category': 'DailyGuidedImpactBetaV5',
  'idea': '放量涨跌耗尽与小盘 beta 差交叉；叠加 beta 稳定性。 来源：daily_v3_v6_recovered:daily_guided_impact_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_accounting_price_gross_net_control_price_lag_vwap_10',
  'expr_tpl': 'cs_rank(ts_mean((-1 * (fund_gross_profit_margin_ttm - fund_net_profit_margin_ttm)) * '
              '(ts_mean((close / (vwap + 1e-12) - 1), 10)), 10))',
  'param_grid': None,
  'sub_category': 'DailyGuidedAccountingPriceV5',
  'idea': '费用损耗控制；VWAP 吸收。 来源：daily_v3_v6_recovered:daily_guided_accounting_price_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_beta_term_corr_drop_value_change_60',
  'expr_tpl': 'cs_rank(ts_mean(((-1 * ts_delta(ts_corr((close / (ts_delay(close, 1) + 1e-12) - 1), '
              'idx852_r1, 60), 1) * fund_book_to_market_ratio_ttm)) - ts_delay((-1 * ts_delta(ts_corr((close '
              '/ (ts_delay(close, 1) + 1e-12) - 1), idx852_r1, 60), 1) * fund_book_to_market_ratio_ttm), 1), '
              '60))',
  'param_grid': None,
  'sub_category': 'DailyBetaTermStructureV4',
  'idea': '小盘相关性下降时的价值独立性；beta 期限结构的慢速变化。 来源：daily_v3_v6_recovered:daily_beta_term_structure_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_volume_upper_pressure_selfz_20',
  'expr_tpl': 'cs_rank(((((high - close) / (high - low + 1e-12)) * (turnover / (ts_mean(turnover, 20) + '
              '1e-12))) - ts_mean(((high - close) / (high - low + 1e-12)) * (turnover / (ts_mean(turnover, '
              '20) + 1e-12)), 20)) / (ts_std(((high - close) / (high - low + 1e-12)) * (turnover / '
              '(ts_mean(turnover, 20) + 1e-12)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetrySelfZV3',
  'idea': '放量上影压力 的自身状态突变。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_idx_hot_vwap_recover_20',
  'expr_tpl': 'ts_where(idx852_r20 > 0.08, cs_rank(ts_mean((close / (vwap + 1e-12) - 1), 20)), -1 * '
              'cs_rank(ts_mean((close / (vwap + 1e-12) - 1), 20)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '中证1000短期过热状态下使用VWAP 修复，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_turn_cluster_crowd_flip_10',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 10) + 1e-12)) > 1.5, -1 * cs_rank(ts_mean((turnover / '
              '(ts_mean(turnover, 10) + 1e-12)) / (ts_mean((turnover / (ts_mean(turnover, 10) + 1e-12)), 5) '
              '+ 1e-12), 10)), cs_rank(ts_mean((turnover / (ts_mean(turnover, 10) + 1e-12)) / '
              '(ts_mean((turnover / (ts_mean(turnover, 10) + 1e-12)), 5) + 1e-12), 10)))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityCrowdFlipV3',
  'idea': '成交活跃聚集 在成交拥挤时翻转。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_body_range_tension_selfz_10',
  'expr_tpl': 'cs_rank(((((close - open) / (high - low + 1e-12)) / (((high - low) / (close + 1e-12)) + '
              '1e-12)) - ts_mean(((close - open) / (high - low + 1e-12)) / (((high - low) / (close + 1e-12)) '
              '+ 1e-12), 10)) / (ts_std(((close - open) / (high - low + 1e-12)) / (((high - low) / (close + '
              '1e-12)) + 1e-12), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetrySelfZV3',
  'idea': '实体和振幅张力 的自身状态突变。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_cap_alloc_funding_resilience_industry_disagree_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_ocf_to_debt_ttm + fund_dividend_yield_ttm - fund_ev_to_ebitda_ttm), 20) '
              '* -1 * ((close / (ts_delay(close, 20) + 1e-12) - 1) - ind_r20))',
  'param_grid': None,
  'sub_category': 'DailyCapitalAllocationV4',
  'idea': '现金覆盖、股息和估值压力的融资韧性；资本配置优势和行业相对价格背离。 来源：daily_v3_v6_recovered:daily_capital_allocation_v4。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_fund_pe_surprise_240',
  'expr_tpl': 'cs_rank((fund_pe_ratio_ttm - ts_mean(fund_pe_ratio_ttm, 240)) / (ts_std(fund_pe_ratio_ttm, '
              '240) + 1e-12))',
  'param_grid': None,
  'sub_category': 'FundamentalValuationSurprise',
  'idea': 'pe_ratio_ttm 相对自身历史均值的偏离。 来源：daily_data_dimension_archive_20260518:fundamental。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_overnight_intraday_conflict_selfz_10',
  'expr_tpl': 'cs_rank((((open / (ts_delay(close, 1) + 1e-12) - 1) - (close / (open + 1e-12) - 1)) - '
              'ts_mean((open / (ts_delay(close, 1) + 1e-12) - 1) - (close / (open + 1e-12) - 1), 10)) / '
              '(ts_std((open / (ts_delay(close, 1) + 1e-12) - 1) - (close / (open + 1e-12) - 1), 10) + '
              '1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetrySelfZV3',
  'idea': '隔夜与日内冲突 的自身状态突变。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_gap_per_turn_level_10',
  'expr_tpl': 'cs_rank(ts_mean(ts_abs((open / (ts_delay(close, 1) + 1e-12) - 1)) / ((turnover / '
              '(ts_mean(turnover, 10) + 1e-12)) + 1e-12), 10))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityV3',
  'idea': '单位成交跳空 的平滑水平。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_range_quiet_vol_compress_60',
  'expr_tpl': 'ts_where(((high - low) / (close + 1e-12)) < ts_mean(((high - low) / (close + 1e-12)), 20), '
              'cs_rank(ts_mean(-1 * ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), 60), 60)), -1 * '
              'cs_rank(ts_mean(-1 * ts_std((close / (ts_delay(close, 1) + 1e-12) - 1), 60), 60)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '个股振幅低于中期状态下使用波动压缩，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_idx_cold_gap_fade_60',
  'expr_tpl': 'ts_where(idx852_r20 < -0.08, cs_rank(ts_mean(-1 * (open / (ts_delay(close, 1) + 1e-12) - 1), '
              '60)), -1 * cs_rank(ts_mean(-1 * (open / (ts_delay(close, 1) + 1e-12) - 1), 60)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '中证1000短期过冷状态下使用跳空回落，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v6_repair_idx_lag_growth_120',
  'expr_tpl': 'cs_rank(ts_mean(((-1 * ((close / (ts_delay(close, 120) + 1e-12) - 1) - ts_mean(idx852_r1, '
              '120)))) * (fund_net_profit_growth_ratio_ttm), 120))',
  'param_grid': None,
  'sub_category': 'DailyV6FundamentalRepair',
  'idea': '指数相对价格滞后修复 与 利润成长 交叉，寻找未充分定价的修复。 来源：daily_v3_v6_recovered:daily_v6_fundamental_repair。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_clv_mean_3',
  'expr_tpl': 'ts_mean(((close - low) / (high - low + 1e-12)), 3)',
  'param_grid': None,
  'sub_category': 'AI4000R116TurnoverStabilityPlain',
  'idea': '换手稳定性，原始状态。 strict corr 扩展候选：core_clv_mean。 来源：strict2636_theme_recovered_from_ai_cs。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_idx_hot_ind_resid_40',
  'expr_tpl': 'ts_where(idx852_r20 > 0.08, cs_rank(ts_mean((close / (ts_delay(close, 1) + 1e-12) - 1) - '
              'ind_r1, 40)), -1 * cs_rank(ts_mean((close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 40)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '中证1000短期过热状态下使用行业残差，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily '
          'base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_quality_capacity_price_lag_10',
  'expr_tpl': 'cs_rank(ts_mean((fund_return_on_equity_ttm * (1 / (ts_mean((turnover / (ts_mean(turnover, 10) '
              '+ 1e-12)), 10) + 1e-12))), 10) * -1 * ((close / (ts_delay(close, 10) + 1e-12) - 1) - '
              'ind_r10))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': 'ROE 优势处在低成交关注下；容量信号尚未被行业相对价格反映。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_regime_cross_industry_beats_small_small_value_10',
  'expr_tpl': 'ts_where(ind_r20 > idx852_r20, cs_rank(ts_mean((fund_book_to_market_ratio_ttm * (1 / '
              '(fund_market_cap + 1e-12))), 10)), -1 * cs_rank(ts_mean((fund_book_to_market_ratio_ttm * (1 / '
              '(fund_market_cap + 1e-12))), 10)))',
  'param_grid': None,
  'sub_category': 'DailyGuidedRegimeCrossV5',
  'idea': '行业强于小盘指数状态下使用小市值价值，非该状态反向。 来源：daily_v3_v6_recovered:daily_guided_regime_cross_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v5_impact_beta_support_impact_value_ind_down_10',
  'expr_tpl': 'cs_rank((ts_mean(((close - low) / (high - low + 1e-12)) / ((turnover / (ts_mean(turnover, 10) '
              '+ 1e-12)) + 1e-12), 10) * fund_book_to_market_ratio_ttm) * (ts_mean(ts_where(ind_r1 < 0, '
              '(close / (ts_delay(close, 1) + 1e-12) - 1) - ind_r1, 0), 10)))',
  'param_grid': None,
  'sub_category': 'DailyGuidedImpactBetaV5',
  'idea': '低成交收盘承接与价值交叉；叠加行业压力期抗跌。 来源：daily_v3_v6_recovered:daily_guided_impact_beta_v5。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_vwap_stretch_fade_selfz_10',
  'expr_tpl': 'cs_rank(((-1 * (close / (vwap + 1e-12) - 1) * ((high - low) / (close + 1e-12))) - ts_mean(-1 '
              '* (close / (vwap + 1e-12) - 1) * ((high - low) / (close + 1e-12)), 10)) / (ts_std(-1 * (close '
              '/ (vwap + 1e-12) - 1) * ((high - low) / (close + 1e-12)), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetrySelfZV3',
  'idea': 'VWAP 拉伸回落 的自身状态突变。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_idx300_down_market_resilience_20',
  'expr_tpl': 'cs_rank(ts_mean(ts_where(idx300_r1 < 0, close / (ts_delay(close, 1) + 1e-12) - 1 - idx300_r1, '
              '0), 20))',
  'param_grid': None,
  'sub_category': 'IndexRegimeCapture',
  'idea': '指数下跌日个股相对抗跌能力。 来源：daily_data_dimension_archive_20260518:index。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v4_capacity_value_impact_buffer_jump_guard_20',
  'expr_tpl': 'cs_rank(ts_mean((fund_book_to_market_ratio_ttm / (ts_mean(ts_abs((close / (ts_delay(close, 1) '
              '+ 1e-12) - 1)) / (turnover + 1e-12), 20) + 1e-12)), 20) / '
              '(ts_mean(ts_abs(((fund_book_to_market_ratio_ttm / (ts_mean(ts_abs((close / (ts_delay(close, '
              '1) + 1e-12) - 1)) / (turnover + 1e-12), 20) + 1e-12))) - '
              'ts_delay((fund_book_to_market_ratio_ttm / (ts_mean(ts_abs((close / (ts_delay(close, 1) + '
              '1e-12) - 1)) / (turnover + 1e-12), 20) + 1e-12)), 1)), 20) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityCapacityV4',
  'idea': '价值信号除以冲击成本；容量信号低跳变。 来源：daily_v3_v6_recovered:daily_liquidity_capacity_v4。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_r116_turnover_stability_plain_raw_level_x_core_body_range_turn_40',
  'expr_tpl': 'ts_mean((ts_abs(close - open) / (high - low + 1e-12)) * (turnover / (ts_mean(turnover, 40) + '
              '1e-12)), 40)',
  'param_grid': None,
  'sub_category': 'AI4000R116TurnoverStabilityPlain',
  'idea': '换手稳定性，原始状态。 strict corr 扩展候选：core_body_range_turn。 来源：strict2636_theme_recovered_from_ai_cs。 '
          'daily base quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_path_volume_close_support_selfz_10',
  'expr_tpl': 'cs_rank(((((close - low) / (high - low + 1e-12)) * (turnover / (ts_mean(turnover, 10) + '
              '1e-12))) - ts_mean(((close - low) / (high - low + 1e-12)) * (turnover / (ts_mean(turnover, '
              '10) + 1e-12)), 10)) / (ts_std(((close - low) / (high - low + 1e-12)) * (turnover / '
              '(ts_mean(turnover, 10) + 1e-12)), 10) + 1e-12))',
  'param_grid': None,
  'sub_category': 'DailyPathAsymmetrySelfZV3',
  'idea': '放量收盘承接 的自身状态突变。 来源：daily_v3_v6_recovered:daily_path_asymmetry_v3。 daily base quality-pass '
          'representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_regime_turn_hot_intraday_follow_40',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 40) + 1e-12)) > 1.5, cs_rank(ts_mean((close / (open + '
              '1e-12) - 1), 40)), -1 * cs_rank(ts_mean((close / (open + 1e-12) - 1), 40)))',
  'param_grid': None,
  'sub_category': 'DailyStateTransitionV3',
  'idea': '个股成交拥挤状态下使用日内跟随，非该状态反向，测试状态切换收益。 来源：daily_v3_v6_recovered:daily_state_transition_v3。 daily base '
          'quality-pass representative.',
  'missing': []},
 {'base_name': 'ai_cs_daily_v3_liq_turn_vs_volume_crowd_flip_60',
  'expr_tpl': 'ts_where((turnover / (ts_mean(turnover, 60) + 1e-12)) > 1.5, -1 * cs_rank(ts_mean((turnover / '
              '(ts_delay(turnover, 1) + 1e-12) - 1) - (volume / (ts_delay(volume, 1) + 1e-12) - 1), 60)), '
              'cs_rank(ts_mean((turnover / (ts_delay(turnover, 1) + 1e-12) - 1) - (volume / '
              '(ts_delay(volume, 1) + 1e-12) - 1), 60)))',
  'param_grid': None,
  'sub_category': 'DailyLiquidityElasticityCrowdFlipV3',
  'idea': '成交额和成交量加速差 在成交拥挤时翻转。 来源：daily_v3_v6_recovered:daily_liquidity_elasticity_v3。 daily base '
          'quality-pass representative.',
  'missing': []}]

    FACTOR_NAMES = FACTOR_NAMES_SELECTED
    SPECS = SPECS_SELECTED

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
        self._add_specs(self.SPECS_SELECTED)

    def _add_specs(self, specs: list[dict[str, object]]) -> None:
        for spec in specs:
            self.add_parametric_feature(
                base_name=str(spec["base_name"]),
                expr_tpl=str(spec["expr_tpl"]),
                param_grid=spec["param_grid"],
                category=self.CATEGORY,
                sub_category=str(spec.get("sub_category", "")),
            )


assert len(AiSelectedFactors.FACTOR_NAMES_CORR_SELECTED_286) == 286
assert len(set(AiSelectedFactors.FACTOR_NAMES_CORR_SELECTED_286)) == 286
assert len(AiSelectedFactors.FACTOR_NAMES_STRICT_SELECTED_6) == 6
assert len(set(AiSelectedFactors.FACTOR_NAMES_STRICT_SELECTED_6)) == 6
assert len(AiSelectedFactors.FACTOR_NAMES_SELECTED) == 292
assert len(set(AiSelectedFactors.FACTOR_NAMES_SELECTED)) == 292
assert len(AiSelectedFactors.SPECS_SELECTED) == 291
assert all(name.startswith("ai_cs") for name in AiSelectedFactors.FACTOR_NAMES_SELECTED)
assert all(str(spec["base_name"]).startswith("ai_cs") for spec in AiSelectedFactors.SPECS_SELECTED)
