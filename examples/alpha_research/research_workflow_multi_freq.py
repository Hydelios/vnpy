"""
多频率因子计算示例脚本

展示如何使用新的多频率支持功能进行因子计算和存储：
1. 创建不同频率的数据集（日线、10分钟线）
2. 分别计算和保存因子
3. 训练模型并保存
4. 生成和保存交易信号
"""

import sys
sys.path.append('F:\\git\\vnpy_hub\\vnpy')

from datetime import datetime
import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, AlphaModel
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.lgb_model import LgbModel


def process_daily_data():
    """处理日线数据"""
    print("\n" + "="*60)
    print("处理日线数据 (1d)")
    print("="*60)
    
    # 创建数据中心 - 使用已下载数据的目录
    lab = AlphaLab("./lab/test_multi_freq")
    
    # 参数设置
    symbols = ["000001.SZSE", "000002.SZSE", "000004.SZSE", "000005.SZSE", "000006.SZSE"]
    start = "2024-01-01"
    end = "2024-01-31"
    interval_str = "1d"
    
    # 加载日线数据
    print(f"加载 {len(symbols)} 只股票的日线数据...")
    df = lab.load_bar_df(symbols, Interval.DAILY, start, end, extended_days=100)
    
    if df is None or len(df) == 0:
        print("无法加载日线数据")
        return
    
    print(f"成功加载 {len(df)} 条日线数据")
    
    # 创建日线数据集，指定interval
    dataset = Alpha158(
        df=df,
        train_period=("2024-01-01", "2024-01-10"),
        valid_period=("2024-01-11", "2024-01-20"),
        test_period=("2024-01-21", "2024-01-31"),
        interval=interval_str,  # 指定频率标签
        enable_cache=True,
        cache_dir=str(lab.lab_path / "factor_cache")
    )
    
    print(f"创建数据集，interval: {dataset.interval}")
    
    # 准备数据（计算因子）
    print("计算因子特征...")
    dataset.prepare_data(max_workers=1)
    
    # 保存数据集（自动带上频率标签）
    lab.save_dataset("alpha158", dataset)
    print(f"保存数据集: alpha158_{interval_str}.pkl")
    
    # 训练模型
    print("训练LightGBM模型...")
    model = LgbModel(seed=42)
    model.fit(dataset)
    
    # 保存模型（自动带上频率标签）
    lab.save_model("lgb_model", model, interval=interval_str)
    print(f"保存模型: lgb_model_{interval_str}.pkl")
    
    # 生成信号
    print("生成交易信号...")
    from vnpy.alpha.dataset.utility import Segment
    predictions = model.predict(dataset, Segment.TEST)
    df_test = dataset.fetch_infer(Segment.TEST)
    df_test = df_test.with_columns(pl.Series(predictions).alias("signal"))
    signal = df_test["datetime", "vt_symbol", "signal"]
    
    # 保存信号（自动带上频率标签）
    lab.save_signal("trading_signal", signal, interval=interval_str)
    print(f"保存信号: trading_signal_{interval_str}.parquet")
    
    return True


def process_10min_data():
    """处理10分钟线数据"""
    print("\n" + "="*60)
    print("处理10分钟线数据 (10m)")
    print("="*60)
    
    # 创建数据中心 - 使用已下载数据的目录
    lab = AlphaLab("./lab/test_multi_freq")
    
    # 参数设置
    symbols = ["000001.SZSE", "000002.SZSE", "000004.SZSE", "000005.SZSE", "000006.SZSE"]
    start = "2024-01-01"
    end = "2024-01-31"
    interval_str = "10m"
    
    # 加载10分钟线数据 - 使用改进后的 load_bar_df
    print(f"加载 {len(symbols)} 只股票的10分钟线数据...")
    # 现在 load_bar_df 支持 interval_value 参数了
    df = lab.load_bar_df(symbols, Interval.MINUTE, start, end, extended_days=5, interval_value="10m")
    
    if df is None or len(df) == 0:
        print("无法加载10分钟线数据")
        return
    
    print(f"成功加载 {len(df)} 条10分钟线数据")
    
    # 创建10分钟线数据集，指定interval
    dataset = Alpha158(
        df=df,
        train_period=("2024-01-01", "2024-01-15"),
        valid_period=("2024-01-16", "2024-01-23"),
        test_period=("2024-01-24", "2024-01-31"),
        interval=interval_str,  # 指定频率标签
        enable_cache=True,
        cache_dir=str(lab.lab_path / "factor_cache")
    )
    
    print(f"创建数据集，interval: {dataset.interval}")
    
    # 准备数据（计算因子）
    print("计算因子特征...")
    dataset.prepare_data(max_workers=1)
    
    # 保存数据集（自动带上频率标签）
    lab.save_dataset("alpha158", dataset)
    print(f"保存数据集: alpha158_{interval_str}.pkl")
    
    # 训练模型
    print("训练LightGBM模型...")
    model = LgbModel(seed=42)
    model.fit(dataset)
    
    # 保存模型（自动带上频率标签）
    lab.save_model("lgb_model", model, interval=interval_str)
    print(f"保存模型: lgb_model_{interval_str}.pkl")
    
    # 生成信号
    print("生成交易信号...")
    from vnpy.alpha.dataset.utility import Segment
    predictions = model.predict(dataset, Segment.TEST)
    df_test = dataset.fetch_infer(Segment.TEST)
    df_test = df_test.with_columns(pl.Series(predictions).alias("signal"))
    signal = df_test["datetime", "vt_symbol", "signal"]
    
    # 保存信号（自动带上频率标签）
    lab.save_signal("trading_signal", signal, interval=interval_str)
    print(f"保存信号: trading_signal_{interval_str}.parquet")
    
    return True


def load_and_compare():
    """加载并比较不同频率的数据"""
    print("\n" + "="*60)
    print("加载并比较不同频率的数据")
    print("="*60)
    
    lab = AlphaLab("./lab/test_multi_freq")
    
    # 加载日线数据集
    dataset_1d = lab.load_dataset("alpha158", interval="1d")
    if dataset_1d:
        print(f"成功加载日线数据集，interval: {dataset_1d.interval}")
    else:
        print("无法加载日线数据集")
    
    # 加载10分钟线数据集
    dataset_10m = lab.load_dataset("alpha158", interval="10m")
    if dataset_10m:
        print(f"成功加载10分钟线数据集，interval: {dataset_10m.interval}")
    else:
        print("无法加载10分钟线数据集")
    
    # 加载日线模型
    model_1d = lab.load_model("lgb_model", interval="1d")
    if model_1d:
        print("成功加载日线模型")
    
    # 加载10分钟线模型
    model_10m = lab.load_model("lgb_model", interval="10m")
    if model_10m:
        print("成功加载10分钟线模型")
    
    # 加载日线信号
    signal_1d = lab.load_signal("trading_signal", interval="1d")
    if signal_1d is not None:
        print(f"成功加载日线信号，数据量: {len(signal_1d)}")
    
    # 加载10分钟线信号
    signal_10m = lab.load_signal("trading_signal", interval="10m")
    if signal_10m is not None:
        print(f"成功加载10分钟线信号，数据量: {len(signal_10m)}")
    
    # 显示缓存目录结构
    print("\n缓存目录结构:")
    cache_dir = lab.lab_path / "factor_cache"
    if cache_dir.exists():
        for subdir in cache_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*.parquet"))
                print(f"  - {subdir.name}/: {len(files)} 个因子文件")


def main():
    """主函数"""
    print("="*60)
    print("多频率因子计算示例")
    print("="*60)
    
    # 处理日线数据
    success_1d = process_daily_data()
    
    # 处理10分钟线数据
    success_10m = process_10min_data()
    
    # 加载并比较
    if success_1d or success_10m:
        load_and_compare()
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)


if __name__ == "__main__":
    main()