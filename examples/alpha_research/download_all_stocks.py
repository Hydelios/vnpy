"""
全市场股票数据下载脚本

功能：
1. 下载全市场所有股票的K线数据
2. 支持日线、分钟线等多种周期
3. 自动保存到AlphaLab数据中心
4. 设置回测参数配置

使用方法：
    python download_all_stocks.py [开始日期] [结束日期]
    
    示例：
    python download_all_stocks.py  # 使用默认日期
    python download_all_stocks.py 2010-01-01  # 指定开始日期
    python download_all_stocks.py 2010-01-01 2024-12-31  # 指定开始和结束日期
"""

import sys
sys.path.append('F:\\git\\vnpy_hub\\vnpy')
sys.path.append('F:\\git\\vnpy_hub\\vnpy_rqdata')

import argparse
from datetime import datetime, timedelta
from typing import List, Optional

from tqdm import tqdm
import rqdatac as rq
import pandas as pd

from vnpy.trader.database import DB_TZ
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest, BarData
from vnpy.trader.utility import round_to, ZoneInfo
from vnpy.alpha import AlphaLab, logger


def get_all_stocks() -> List[str]:
    """
    获取全市场股票列表
    
    Returns:
        所有股票的vnpy格式代码列表
    """
    print("正在获取全市场股票列表...")
    
    try:
        # 获取所有股票
        df = rq.all_instruments(type='CS')  # CS表示普通股票
        
        # 转换为vnpy格式的股票代码列表
        all_symbols = []
        for _, row in df.iterrows():
            rq_symbol = row['order_book_id']
            # 转换格式：XSHG->SSE, XSHE->SZSE
            vt_symbol = rq_symbol.replace("XSHG", "SSE").replace("XSHE", "SZSE")
            all_symbols.append(vt_symbol)
        
        print(f"成功获取 {len(all_symbols)} 只股票")
        return all_symbols
        
    except Exception as e:
        logger.error(f"获取股票列表失败：{e}")
        return []


def download_bar_data_extended(lab: AlphaLab, symbols: List[str], start: datetime, end: datetime, 
                              interval_str: str) -> List[str]:
    """
    下载扩展周期的K线数据（10分钟、30分钟等）
    直接调用米筐API支持非标准周期
    
    Args:
        lab: AlphaLab实例
        symbols: 股票代码列表
        start: 开始时间
        end: 结束时间
        interval_str: 周期字符串（如"10m", "30m"）
    
    Returns:
        下载失败的股票列表
    """
    # 这些周期vnpy不直接支持，但米筐支持
    # 保存时统一用MINUTE标识
    rq_frequency = interval_str
    save_interval = Interval.MINUTE
    
    # 使用自定义下载函数
    from pandas import DataFrame
    from rqdatac.services.get_price import get_price
    from rqdatac.services.calendar import get_next_trading_date
    
    CHINA_TZ = ZoneInfo("Asia/Shanghai")
    failed_symbols = []
    
    print(f"\n开始下载 {len(symbols)} 只股票的{rq_frequency}数据...")
    print(f"时间范围：{start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')}")
    
    # 初始化米筐
    datafeed = get_datafeed()
    if not datafeed.init():
        print("米筐数据服务初始化失败")
        return symbols
    
    # 根据频率计算时间调整（米筐返回K线结束时间，vnpy使用开始时间）
    adjustment_map = {
        "1m": timedelta(minutes=1),
        "10m": timedelta(minutes=10),
        "30m": timedelta(minutes=30),
        "60m": timedelta(hours=1),
    }
    adjustment = adjustment_map.get(rq_frequency, timedelta())
    
    for vt_symbol in tqdm(symbols, desc="下载进度"):
        try:
            symbol, exchange_str = vt_symbol.split(".")
            exchange = Exchange(exchange_str)
            
            # 转换为米筐代码
            if exchange == Exchange.SSE:
                rq_symbol = f"{symbol}.XSHG"
            elif exchange == Exchange.SZSE:
                rq_symbol = f"{symbol}.XSHE"
            else:
                rq_symbol = vt_symbol
            
            # 字段列表
            fields = ["open", "high", "low", "close", "volume", "total_turnover"]
            
            # 对于股票查询后复权K线数据
            if rq_symbol.endswith(".XSHG") or rq_symbol.endswith(".XSHE"):
                adjust_type = "post_volume"
            else:
                adjust_type = "none"
            
            # 调用米筐API
            df: DataFrame = get_price(
                rq_symbol,
                frequency=rq_frequency,
                fields=fields,
                start_date=start,
                end_date=get_next_trading_date(end),
                adjust_type=adjust_type
            )
            
            if df is not None and not df.empty:
                # 填充NaN为0
                df.fillna(0, inplace=True)
                
                bars = []
                for row in df.itertuples():
                    dt = row.Index[1].to_pydatetime() - adjustment
                    dt = dt.replace(tzinfo=CHINA_TZ)
                    
                    if dt >= end:
                        break
                    
                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange,
                        interval=save_interval,  # 使用MINUTE作为保存的interval
                        datetime=dt,
                        open_price=round_to(row.open, 0.000001),
                        high_price=round_to(row.high, 0.000001),
                        low_price=round_to(row.low, 0.000001),
                        close_price=round_to(row.close, 0.000001),
                        volume=row.volume,
                        turnover=row.total_turnover,
                        open_interest=0,
                        gateway_name="RQ"
                    )
                    bars.append(bar)
                
                if bars:
                    lab.save_bar_data(bars, interval_value=rq_frequency)
                else:
                    logger.error(f"下载{vt_symbol}数据失败：无有效数据")
                    failed_symbols.append(vt_symbol)
            else:
                logger.error(f"下载{vt_symbol}数据失败：无数据返回")
                failed_symbols.append(vt_symbol)
                
        except Exception as e:
            logger.error(f"下载{vt_symbol}数据失败：{e}")
            failed_symbols.append(vt_symbol)
    
    return failed_symbols


def download_bar_data(lab: AlphaLab, symbols: List[str], start: datetime, end: datetime, 
                     interval: Interval = Interval.DAILY, interval_value: str = None) -> List[str]:
    """
    下载K线数据（标准周期）
    
    Args:
        lab: AlphaLab实例
        symbols: 股票代码列表
        start: 开始时间
        end: 结束时间
        interval: K线周期
        interval_value: 周期标识（如"1m", "60m"）
    
    Returns:
        下载失败的股票列表
    """
    failed_symbols = []
    
    print(f"\n开始下载 {len(symbols)} 只股票的{interval.value}数据...")
    print(f"时间范围：{start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')}")
    
    # 初始化数据服务
    datafeed = get_datafeed()
    
    for vt_symbol in tqdm(symbols, desc="下载进度"):
        try:
            symbol, exchange_str = vt_symbol.split(".")
            
            req = HistoryRequest(
                symbol=symbol, 
                exchange=Exchange(exchange_str), 
                start=start, 
                end=end, 
                interval=interval
            )
            
            bars = datafeed.query_bar_history(req)
            
            if bars:
                lab.save_bar_data(bars, interval_value=interval_value)
            else:
                logger.error(f"下载{vt_symbol}数据失败：无数据返回")
                failed_symbols.append(vt_symbol)
                
        except Exception as e:
            logger.error(f"下载{vt_symbol}数据失败：{e}")
            failed_symbols.append(vt_symbol)
    
    return failed_symbols


def set_contract_settings(lab: AlphaLab, symbols: List[str], settings: dict):
    """
    为股票列表设置回测参数
    
    Args:
        lab: AlphaLab实例
        symbols: 股票代码列表
        settings: 回测参数配置
    """
    print(f"\n为 {len(symbols)} 只股票设置回测参数...")
    
    for vt_symbol in tqdm(symbols, desc="设置回测参数"):
        try:
            lab.add_contract_setting(
                vt_symbol,
                long_rate=settings["long_rate"],
                short_rate=settings["short_rate"],
                size=settings["size"],
                pricetick=settings["pricetick"],
            )
        except Exception as e:
            logger.error(f"设置{vt_symbol}回测参数失败：{e}")


def save_download_report(lab: AlphaLab, total: int, failed_symbols: List[str]):
    """
    保存下载报告
    
    Args:
        lab: AlphaLab实例
        total: 总股票数量
        failed_symbols: 失败的股票列表
    """
    report_path = lab.lab_path.joinpath("download_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 全市场股票数据下载报告\n")
        f.write(f"# 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 统计信息\n")
        f.write(f"- 总股票数量：{total}\n")
        f.write(f"- 成功下载：{total - len(failed_symbols)}\n")
        f.write(f"- 下载失败：{len(failed_symbols)}\n\n")
        
        if failed_symbols:
            f.write(f"## 失败股票列表\n")
            for symbol in failed_symbols:
                f.write(f"- {symbol}\n")
    
    print(f"\n下载报告已保存到：{report_path}")


def verify_data(lab: AlphaLab, sample_symbols: List[str], start: datetime, end: datetime):
    """
    验证数据是否正确保存
    
    Args:
        lab: AlphaLab实例
        sample_symbols: 抽样股票列表
        start: 开始时间
        end: 结束时间
    """
    print("\n数据验证...")
    
    for symbol in sample_symbols[:3]:  # 验证前3只股票
        bars = lab.load_bar_data(symbol, start, end, Interval.DAILY)
        if bars:
            print(f"  - {symbol}: 成功加载 {len(bars)} 条数据")
        else:
            print(f"  - {symbol}: 未能加载数据")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='下载全市场股票K线数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python download_all_stocks.py                    # 使用默认日期（2007-01-01 至 昨天）
  python download_all_stocks.py 2010-01-01         # 指定开始日期（2010-01-01 至 昨天）
  python download_all_stocks.py 2010-01-01 2024-12-31  # 指定日期范围
  python download_all_stocks.py --interval 10m     # 下载10分钟线
  python download_all_stocks.py --interval 30m     # 下载30分钟线
  python download_all_stocks.py --interval 60m     # 下载60分钟线
        """
    )
    
    # 获取昨天的日期作为默认结束日期
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    parser.add_argument(
        'start_date', 
        nargs='?', 
        default='2007-01-01',
        help='开始日期，格式：YYYY-MM-DD（默认：2007-01-01）'
    )
    
    parser.add_argument(
        'end_date', 
        nargs='?', 
        default=yesterday,
        help=f'结束日期，格式：YYYY-MM-DD（默认：昨天 {yesterday}）'
    )
    
    parser.add_argument(
        '--task-name', 
        default='all_stocks',
        help='任务名称，用于创建数据文件夹（默认：all_stocks）'
    )
    
    parser.add_argument(
        '--interval', 
        default='1d',
        choices=['1d', '1m', '10m', '30m', '60m'],
        help='K线周期：1d(日线), 1m(1分钟), 10m(10分钟), 30m(30分钟), 60m(60分钟)（默认：1d）'
    )
    
    parser.add_argument(
        '--no-contract', 
        action='store_true',
        help='不设置回测参数'
    )
    
    parser.add_argument(
        '--no-verify', 
        action='store_true',
        help='不验证下载的数据'
    )
    
    args = parser.parse_args()
    
    # 验证日期格式
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        parser.error(f"日期格式错误：{e}")
    
    return args


def main():
    """主函数"""
    
    # ========== 解析命令行参数 ==========
    args = parse_arguments()
    
    # ========== 配置参数 ==========
    
    # 任务名称（用于创建数据文件夹）
    task_name = args.task_name
    
    # 时间范围设置
    start_date = args.start_date
    end_date = args.end_date
    
    # K线周期映射
    interval_map = {
        '1d': Interval.DAILY,
        '1m': Interval.MINUTE,
        '60m': Interval.HOUR,
        # 10m和30m会使用扩展函数处理
    }
    interval = interval_map.get(args.interval, Interval.MINUTE)
    
    # 回测参数配置
    contract_settings = {
        "long_rate": 5/10000,     # 做多手续费率（万分之五）
        "short_rate": 10/10000,   # 做空手续费率（千分之一）
        "size": 1,                 # 合约乘数
        "pricetick": 0.0001,      # 最小价格变动
    }
    
    # 是否设置回测参数
    set_contract_params = not args.no_contract
    
    # 是否验证数据
    verify_download = not args.no_verify
    
    # ========== 初始化环境 ==========
    
    print("="*60)
    print("全市场股票数据下载工具")
    print("="*60)
    print(f"\n下载参数：")
    print(f"  - 开始日期：{start_date}")
    print(f"  - 结束日期：{end_date}")
    print(f"  - K线周期：{args.interval}")
    print(f"  - 任务名称：{task_name}")
    print(f"  - 设置回测参数：{'是' if set_contract_params else '否'}")
    print(f"  - 验证数据：{'是' if verify_download else '否'}")
    
    # 创建投研实验室
    lab = AlphaLab(f"./lab/{task_name}")
    print(f"\n数据保存路径：./lab/{task_name}/")
    
    # 初始化数据服务
    print("\n初始化米筐数据服务...")
    datafeed = get_datafeed()
    if not datafeed.init():
        print("米筐数据服务初始化失败，请检查配置")
        return
    print("米筐数据服务初始化成功")
    
    # 转换时间格式
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=DB_TZ)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=DB_TZ)
    
    # ========== 获取股票列表 ==========
    
    all_symbols = get_all_stocks()
    
    if not all_symbols:
        print("未能获取股票列表，程序退出")
        return
    
    # ========== 下载数据 ==========
    
    # 根据不同的周期类型选择下载函数
    if args.interval in ['10m', '30m']:
        # 使用扩展下载函数处理10分钟、30分钟
        failed_symbols = download_bar_data_extended(lab, all_symbols, start, end, args.interval)
    else:
        # 使用标准下载函数处理日线、1分钟、60分钟
        # 对于分钟级数据，传递interval_value以便保存到正确的子文件夹
        interval_value = args.interval if args.interval in ['1m', '60m'] else None
        failed_symbols = download_bar_data(lab, all_symbols, start, end, interval, interval_value)
    
    # 输出统计信息
    print(f"\n下载完成！")
    print(f"  - 成功：{len(all_symbols) - len(failed_symbols)} 只")
    print(f"  - 失败：{len(failed_symbols)} 只")
    
    if failed_symbols:
        print("\n失败的股票列表（前20个）：")
        for symbol in failed_symbols[:20]:
            print(f"  - {symbol}")
        if len(failed_symbols) > 20:
            print(f"  ... 还有 {len(failed_symbols) - 20} 只")
    
    # ========== 设置回测参数 ==========
    
    if set_contract_params:
        successful_symbols = [s for s in all_symbols if s not in failed_symbols]
        set_contract_settings(lab, successful_symbols, contract_settings)
    
    # ========== 保存下载报告 ==========
    
    save_download_report(lab, len(all_symbols), failed_symbols)
    
    # ========== 数据验证 ==========
    
    if verify_download and all_symbols:
        successful_symbols = [s for s in all_symbols if s not in failed_symbols]
        if successful_symbols:
            verify_data(lab, successful_symbols, start, end)
    
    # ========== 完成 ==========
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print(f"数据保存在：./lab/{task_name}/")
    print("="*60)
    
    return len(all_symbols) - len(failed_symbols)  # 返回成功下载的数量


if __name__ == "__main__":
    # 执行主函数
    success_count = main()
    print(f"\n程序执行完毕，成功下载 {success_count} 只股票数据")