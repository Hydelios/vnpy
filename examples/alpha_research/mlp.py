#!/usr/bin/env python
"""
MLP (多层感知器) 模型的 Alpha 研究示例
使用 Alpha158 因子库训练神经网络模型预测股票收益
"""

import sys
sys.path.append('F:\\git\\vnpy_hub\\vnpy')

# 导入部分保持在顶层
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import polars as pl
import numpy as np
import importlib
from datetime import datetime
from functools import partial

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, Segment, AlphaDataset, AlphaModel
from vnpy.alpha.dataset import (
    process_drop_na,
    process_robust_zscore_norm,
    process_fill_na,
    process_cs_rank_norm,
    to_datetime
)
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.dataset.datasets.alpha_158_polars import Alpha158Polars
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha.strategy import BacktestingEngine
import vnpy.alpha.strategy.strategies.equity_demo_strategy as equity_demo_strategy


def prepare_data(lab: AlphaLab, name: str, index_symbol: str, 
                 start: str, end: str, use_multiprocess: bool = False, use_polars: bool = True):
    """
    准备数据并计算因子
    
    Args:
        lab: AlphaLab实例
        name: 任务名称
        index_symbol: 指数代码
        start: 开始日期
        end: 结束日期
        use_multiprocess: 是否使用多进程
        use_polars: 是否使用Polars优化版本
    
    Returns:
        dataset: 准备好的数据集
        component_symbols: 成分股列表
    """
    print("=" * 60)
    print("1. 准备数据")
    print("=" * 60)
    
    # 设置任务参数
    interval = Interval.DAILY
    extended_days = 100
    
    # 加载所有成分股代码
    print(f"加载 {index_symbol} 成分股...")
    component_symbols = lab.load_component_symbols(index_symbol, start, end)
    print(f"找到 {len(component_symbols)} 只成分股")
    
    # 加载成分股数据
    print("加载K线数据...")
    df = lab.load_bar_df(component_symbols, interval, start, end, extended_days)
    print(f"数据shape: {df.shape}")
    
    # 设置数据时间段
    train_period = ("2008-01-01", "2020-12-31")
    valid_period = ("2021-01-01", "2023-12-31")
    test_period = ("2024-01-01", "2025-7-31")
    
    print(f"训练期: {train_period}")
    print(f"验证期: {valid_period}")
    print(f"测试期: {test_period}")
    
    # 创建数据集对象
    if use_polars:
        print("\n创建 Alpha158Polars 数据集 (优化版)...")
        dataset = Alpha158Polars(
            df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
        )
    else:
        print("\n创建 Alpha158 数据集...")
        dataset = Alpha158(
            df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
        )
    
    # 添加数据预处理器
    print("添加数据预处理器...")
    fit_start_time = to_datetime(train_period[0])
    fit_end_time = to_datetime(train_period[1])
    
    dataset.add_processor("infer", partial(process_robust_zscore_norm, 
                                          fit_start_time=fit_start_time, 
                                          fit_end_time=fit_end_time))
    dataset.add_processor("infer", partial(process_fill_na, fill_value=0, fill_label=False))
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_rank_norm, names=["label"]))
    
    # 收集指数成分过滤器
    filters = lab.load_component_filters(index_symbol, start, end)
    
    # 准备特征和标签数据
    print("\n计算因子特征...")
    if use_multiprocess:
        print("使用多进程计算 (max_workers=3)")
        dataset.prepare_data(filters, max_workers=3)
    else:
        print("使用单进程计算 (更稳定)")
        dataset.prepare_data(filters, max_workers=1)
    
    # 保存到文件缓存
    print(f"\n保存数据集到缓存: {name}")
    lab.save_dataset(name, dataset)
    
    return dataset, component_symbols


def train_model(lab: AlphaLab, name: str, dataset: AlphaDataset):
    """
    训练 MLP 模型
    
    Args:
        lab: AlphaLab实例
        name: 任务名称
        dataset: 数据集
    
    Returns:
        model: 训练好的模型
    """
    print("\n" + "=" * 60)
    print("2. 模型训练")
    print("=" * 60)
    
    # 创建模型对象
    print("创建 MLP 模型...")
    kwargs = {
        "input_size": 158,
        "hidden_sizes": (256,),
        "lr": 0.002,
        "optimizer": "adam",
        "n_epochs": 8000,
        "batch_size": 8192,
        "weight_decay": 0.0002,
        "seed": 42
    }
    
    print(f"模型参数: {kwargs}")
    model = MlpModel(**kwargs)
    
    # 使用数据集训练模型
    print("\n开始训练模型...")
    model.fit(dataset)
    
    # 查看模型细节
    model.detail()
    
    # 保存模型
    print(f"保存模型: {name}")
    lab.save_model(name, model)
    
    return model


def predict_signal(lab: AlphaLab, name: str, dataset: AlphaDataset):
    """
    生成预测信号
    
    Args:
        lab: AlphaLab实例
        name: 任务名称
        dataset: 数据集
    
    Returns:
        signal: 预测信号
    """
    print("\n" + "=" * 60)
    print("3. 预测信号")
    print("=" * 60)
    
    # 加载模型
    model = lab.load_model(name)
    
    # 用模型在测试集上预测
    print("在测试集上预测...")
    pre = model.predict(dataset, Segment.TEST)
    
    # 加载测试集数据
    df_t = dataset.fetch_infer(Segment.TEST)
    
    # 合并预测信号列
    df_t = df_t.with_columns(pl.Series(pre).alias("signal"))
    
    # 提取信号数据
    signal = df_t["datetime", "vt_symbol", "signal"]
    
    # 保存信号数据
    print(f"保存信号: {name}")
    lab.save_signal(name, signal)
    
    return signal


def run_backtest(lab: AlphaLab, name: str, index_symbol: str, 
                 component_symbols: list, signal: pl.DataFrame):
    """
    运行策略回测
    
    Args:
        lab: AlphaLab实例
        name: 任务名称
        index_symbol: 指数代码
        component_symbols: 成分股列表
        signal: 预测信号
    """
    print("\n" + "=" * 60)
    print("4. 策略回测")
    print("=" * 60)
    
    # 重载策略类
    importlib.reload(equity_demo_strategy)
    EquityDemoStrategy = equity_demo_strategy.EquityDemoStrategy
    
    # 创建回测引擎对象
    print("创建回测引擎...")
    engine = BacktestingEngine(lab)
    
    # 设置回测参数
    engine.set_parameters(
        vt_symbols=component_symbols,
        interval=Interval.DAILY,
        start=datetime(2024, 1, 1),
        end=datetime(2025, 7, 31),
        capital=100000000
    )
    
    # 添加策略实例
    setting = {"top_k": 30, "n_drop": 3, "hold_thresh": 3}
    engine.add_strategy(EquityDemoStrategy, setting, signal)
    
    # 执行回测任务
    print("执行回测...")
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()
    
    # 显示结果
    engine.show_chart()
    engine.show_performance(benchmark_symbol=index_symbol)


def main():
    """主函数"""
    # 切换到脚本所在目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 创建数据中心
    lab = AlphaLab("./lab/csi300")
    
    # 设置任务参数
    name = "300_mlp"
    index_symbol = "000300.SSE"
    start = "2008-01-01"
    end = "2025-08-01"
    
    # 询问是否使用多进程
    use_multiprocess = True  # 使用多进程加速计算
    use_polars = True  # 使用Polars优化版本，大幅提升计算速度
    
    try:
        # 1. 准备数据
        dataset, component_symbols = prepare_data(
            lab, name, index_symbol, start, end, use_multiprocess, use_polars
        )
        
        # 特征表现分析（可选）
        print("\n分析 rsv_5 因子表现...")
        dataset.show_feature_performance("rsv_5")
        
        # 2. 训练模型
        model = train_model(lab, name, dataset)
        
        # 3. 生成预测信号
        signal = predict_signal(lab, name, dataset)
        
        # 检查信号绩效
        print("\n检查信号绩效...")
        dataset.show_signal_performance(signal)
        
        # 4. 运行回测
        run_backtest(lab, name, index_symbol, component_symbols, signal)
        
        print("\n" + "=" * 60)
        print("完成！所有结果已保存")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Windows 多进程支持
    from multiprocessing import freeze_support
    freeze_support()
    
    # 运行主函数
    main()