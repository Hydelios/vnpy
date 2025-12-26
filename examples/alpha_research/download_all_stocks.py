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
sys.path.append('/home/hyd/research/vnpy_hub/vnpy')
sys.path.append('/home/hyd/research/vnpy_hub/vnpy_rqdata')

import argparse
import json
import configparser
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

from tqdm import tqdm
import rqdatac as rq
import pandas as pd
import polars as pl
from pathlib import Path

from vnpy.trader.database import DB_TZ
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest, BarData
from vnpy.trader.utility import round_to, ZoneInfo
from vnpy.alpha import AlphaLab, logger


DEFAULT_INDEX_LIST: list[dict[str, str]] = [
    {"vnpy": "000016.SSE", "rq": "000016.XSHG", "name": "上证50"},
    {"vnpy": "000300.SSE", "rq": "000300.XSHG", "name": "沪深300"},
    {"vnpy": "000905.SSE", "rq": "000905.XSHG", "name": "中证500"},
    {"vnpy": "000906.SSE", "rq": "000906.XSHG", "name": "中证800"},
    {"vnpy": "000852.SSE", "rq": "000852.XSHG", "name": "中证1000"},
    {"vnpy": "932000.SSE", "rq": "932000.XSHG", "name": "中证2000"},
    {"vnpy": "000688.SSE", "rq": "000688.XSHG", "name": "科创50"},
    {"vnpy": "000922.SSE", "rq": "000922.XSHG", "name": "中证红利"},
]


def _is_truthy(value: object) -> bool:
    """将配置值解析为布尔值（兼容 bool/str/int）。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    uniq: list[str] = []
    for item in items:
        if item in seen:
            continue
        uniq.append(item)
        seen.add(item)
    return uniq


def _rq_to_vt_symbol(order_book_id: str) -> str:
    """将米筐 order_book_id 转换为 vnpy vt_symbol。"""
    return order_book_id.replace("XSHG", "SSE").replace("XSHE", "SZSE")


def _parse_symbol_list(value: object) -> list[str]:
    """解析逗号分隔字符串或列表为 vt_symbol 列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [str(value).strip()] if str(value).strip() else []


def get_stock_symbols() -> list[str]:
    """获取全市场股票（CS）列表，返回 vt_symbol 列表。"""
    print("正在获取全市场股票列表...")
    try:
        df = rq.all_instruments(type="CS")  # CS表示普通股票
        symbols = [_rq_to_vt_symbol(s) for s in df["order_book_id"].tolist()]
        symbols = _dedup_keep_order(symbols)
        print(f"成功获取 {len(symbols)} 只股票")
        return symbols
    except Exception as e:
        logger.error(f"获取股票列表失败：{e}")
        return []


def get_all_index_symbols() -> list[str]:
    """获取全市场指数（INDX）列表，返回 vt_symbol 列表。"""
    print("正在获取全市场指数列表...")
    try:
        df = rq.all_instruments(type="INDX")
        symbols = [_rq_to_vt_symbol(s) for s in df["order_book_id"].tolist()]
        symbols = _dedup_keep_order(symbols)
        print(f"成功获取 {len(symbols)} 只指数")
        return symbols
    except Exception as e:
        logger.error(f"获取指数列表失败：{e}")
        return []


def get_default_index_symbols(cfg: dict) -> list[str]:
    """获取默认指数列表：优先读取配置 index_list，否则使用内置 DEFAULT_INDEX_LIST。"""
    cfg_list = cfg.get("index_list")
    if isinstance(cfg_list, list) and cfg_list:
        symbols: list[str] = []
        for item in cfg_list:
            if not isinstance(item, dict):
                continue
            vt = str(item.get("vnpy", "")).strip()
            if vt:
                symbols.append(vt)
        if symbols:
            return _dedup_keep_order(symbols)
    return [d["vnpy"] for d in DEFAULT_INDEX_LIST]


def _load_config_file(path: str | Path) -> dict:
    """读取配置文件（优先JSON，回退INI）。文件不存在则返回空dict。"""
    p = Path(path)
    if not p.exists():
        return {}
    # try JSON first
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    # fallback to INI
    cfg = configparser.ConfigParser()
    try:
        cfg.read(p, encoding="utf-8")
        # 优先使用名为 download 的 section，否则合并所有节
        data: dict = {}
        if "download" in cfg:
            data = dict(cfg["download"])
        else:
            for sec in cfg.sections():
                data.update(cfg[sec])
        # 规范化布尔/字符串
        def to_bool(v: str) -> bool:
            return str(v).strip().lower() in {"1", "true", "yes", "on"}
        for k in list(data.keys()):
            v = data[k]
            if k in {
                "no_contract",
                "no_verify",
                "full",
                "incremental",
                "verify",
                "set_contract",
                "include_index",
                "all_index",
                "merge_index",
            }:
                data[k] = to_bool(v)
        return data
    except Exception:
        return {}


def get_incremental_start(
    lab: AlphaLab,
    vt_symbol: str,
    interval: Interval,
    interval_value: Optional[str],
    default_start: datetime,
) -> datetime:
    """
    计算单个标的的增量起点：重复“最后一天”。
    - 若本地无文件，返回 default_start。
    - 若存在文件：读取最大 datetime，将起点设为该日00:00（带DB_TZ）。
    """
    # 定位 parquet 路径
    if interval == Interval.DAILY:
        file_path: Path = lab.daily_path.joinpath(f"{vt_symbol}.parquet")
    elif interval == Interval.HOUR:
        file_path = lab.minute_60m_path.joinpath(f"{vt_symbol}.parquet")
    elif interval == Interval.MINUTE:
        folder = lab.get_minute_folder_path(interval, interval_value)
        file_path = folder.joinpath(f"{vt_symbol}.parquet")
    else:
        return default_start

    if not file_path.exists():
        return default_start

    try:
        df = pl.read_parquet(file_path, columns=["datetime"])  # 仅读时间列
        if df.is_empty():
            return default_start
        last_dt = df.select(pl.max("datetime")).to_series()[0]
        # 归零到当日 00:00，并加回DB_TZ
        inc_start = last_dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=DB_TZ)
        # 防止配置的 default_start 晚于本地数据
        return max(inc_start, default_start)
    except Exception:
        return default_start


def download_bar_data_extended(lab: AlphaLab, symbols: List[str], start: datetime, end: datetime, 
                              interval_str: str, full_mode: bool = False,
                              index_symbols: Set[str] | None = None) -> List[str]:
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
    index_symbols = index_symbols or set()
    
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
            if vt_symbol in index_symbols:
                adjust_type = "none"
            elif rq_symbol.endswith(".XSHG") or rq_symbol.endswith(".XSHE"):
                adjust_type = "post_volume"
            else:
                adjust_type = "none"
            
            # 调用米筐API
            # 计算增量起点（重复最后一天）
            inc_start = start if full_mode else get_incremental_start(
                lab, vt_symbol, save_interval, interval_str, start
            )

            if inc_start >= end:
                # 已最新，跳过
                if not full_mode:
                    tqdm.write(f"[增量] {vt_symbol} 已最新，跳过")
                continue

            if not full_mode:
                fmt = "%Y-%m-%d %H:%M"
                tqdm.write(f"[增量] {vt_symbol}：{inc_start.strftime(fmt)} -> {end.strftime(fmt)} ({rq_frequency})")

            df: DataFrame = get_price(
                rq_symbol,
                frequency=rq_frequency,
                fields=fields,
                start_date=inc_start,
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
                    if not full_mode:
                        tqdm.write(f"[增量] {vt_symbol} 新增 {len(bars)} 条")
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
                     interval: Interval = Interval.DAILY, interval_value: str = None,
                     full_mode: bool = False,
                     index_symbols: Set[str] | None = None) -> List[str]:
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
    index_symbols = index_symbols or set()
    
    print(f"\n开始下载 {len(symbols)} 个标的的{interval.value}数据...")
    print(f"时间范围：{start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')}")
    
    # 初始化数据服务
    datafeed = get_datafeed()
    
    for vt_symbol in tqdm(symbols, desc="下载进度"):
        try:
            symbol, exchange_str = vt_symbol.split(".")
            # 增量起点（重复最后一天），全量模式则使用传入的start
            req_start = start if full_mode else get_incremental_start(
                lab, vt_symbol, interval, interval_value, start
            )

            if req_start >= end:
                # 已最新，跳过
                if not full_mode:
                    tqdm.write(f"[增量] {vt_symbol} 已最新，跳过")
                continue

            if not full_mode:
                fmt = "%Y-%m-%d" if interval == Interval.DAILY else "%Y-%m-%d %H:%M"
                tqdm.write(f"[增量] {vt_symbol}：{req_start.strftime(fmt)} -> {end.strftime(fmt)}")

            if vt_symbol in index_symbols:
                # 指数：直接用米筐 get_price 下载，避免 datafeed 侧对 .XSHG/.XSHE 自动前复权
                from rqdatac.services.get_price import get_price
                from rqdatac.services.calendar import get_next_trading_date

                CHINA_TZ = ZoneInfo("Asia/Shanghai")
                exchange = Exchange(exchange_str)
                if exchange == Exchange.SSE:
                    rq_symbol = f"{symbol}.XSHG"
                elif exchange == Exchange.SZSE:
                    rq_symbol = f"{symbol}.XSHE"
                else:
                    rq_symbol = vt_symbol

                if interval == Interval.DAILY:
                    rq_frequency = "1d"
                    adjustment = timedelta()
                elif interval == Interval.HOUR:
                    rq_frequency = "60m"
                    adjustment = timedelta(hours=1)
                elif interval == Interval.MINUTE:
                    adjustment = timedelta(minutes=1)
                else:
                    raise ValueError(f"不支持的周期：{interval.value}")

                df = get_price(
                    rq_symbol,
                    frequency=rq_frequency,
                    fields=["open", "high", "low", "close", "volume", "total_turnover"],
                    start_date=req_start,
                    end_date=get_next_trading_date(end),
                    adjust_type="none",
                )

                bars = []
                if df is not None and not df.empty:
                    df.fillna(0, inplace=True)
                    for row in df.itertuples():
                        row_index = row.Index
                        ts = row_index[1] if isinstance(row_index, tuple) else row_index
                        dt = ts.to_pydatetime() - adjustment
                        dt = dt.replace(tzinfo=CHINA_TZ)
                        if dt >= end:
                            break

                        bar = BarData(
                            symbol=symbol,
                            exchange=exchange,
                            interval=interval,
                            datetime=dt,
                            open_price=round_to(row.open, 0.000001),
                            high_price=round_to(row.high, 0.000001),
                            low_price=round_to(row.low, 0.000001),
                            close_price=round_to(row.close, 0.000001),
                            volume=row.volume,
                            turnover=row.total_turnover,
                            open_interest=0,
                            gateway_name="RQ",
                        )
                        bars.append(bar)
            else:
                req = HistoryRequest(
                    symbol=symbol,
                    exchange=Exchange(exchange_str),
                    start=req_start,
                    end=end,
                    interval=interval
                )
                bars = datafeed.query_bar_history(req)
            
            if bars:
                lab.save_bar_data(bars, interval_value=interval_value)
                if not full_mode:
                    tqdm.write(f"[增量] {vt_symbol} 新增 {len(bars)} 条")
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
        f.write(f"# 数据下载报告\n")
        f.write(f"# 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 统计信息\n")
        f.write(f"- 总标的数量：{total}\n")
        f.write(f"- 成功下载：{total - len(failed_symbols)}\n")
        f.write(f"- 下载失败：{len(failed_symbols)}\n\n")
        
        if failed_symbols:
            f.write(f"## 失败标的列表\n")
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
        bars = lab.load_bar_data(symbol, Interval.DAILY, start, end)
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
  python download_all_stocks.py --targets index --include-index   # 下载默认指数列表（见 data_download.json 的 index_list）
  python download_all_stocks.py --targets index --all-index       # 下载全量指数列表（INDX）
  python download_all_stocks.py --targets index --index-symbols 000300.SSE,000852.SSE  # 指定下载的指数
  python download_all_stocks.py --targets both --include-index    # 股票+默认指数（可用 --merge-index 合并保存）
        """
    )
    
    parser.add_argument(
        'start_date', 
        nargs='?', 
        default=None,
        help='开始日期，格式：YYYY-MM-DD（若未提供，将从配置或默认2007-01-01读取）'
    )
    
    parser.add_argument(
        'end_date', 
        nargs='?', 
        default=None,
        help='结束日期，格式：YYYY-MM-DD（若未提供，将从配置或默认昨天读取）'
    )
    
    parser.add_argument(
        '--task-name', 
        default=None,
        help='任务名称，用于创建数据文件夹（可由配置覆盖，默认：all_stocks）'
    )
    
    parser.add_argument(
        '--interval', 
        default=None,
        choices=['1d', '1m', '10m', '30m', '60m'],
        help='K线周期：1d(日线), 1m(1分钟), 10m(10分钟), 30m(30分钟), 60m(60分钟)（可由配置覆盖，默认：1d）'
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

    parser.add_argument(
        '--full',
        action='store_true',
        help='强制全量下载（默认进行增量下载，重复最后一天）'
    )

    parser.add_argument(
        '--config',
        default='data_download.json',
        help='下载参数配置文件路径（JSON或INI），默认：data_download.json'
    )

    parser.add_argument(
        '--targets',
        choices=['stocks', 'index', 'both'],
        default=None,
        help='下载范围：stocks(仅股票)、index(仅指数)、both(股票+指数)；也可在配置中设置 targets'
    )

    parser.add_argument(
        '--include-index',
        action='store_true',
        help='下载默认指数列表（见配置 index_list）；与 --targets 搭配使用，或作为快速开关'
    )

    parser.add_argument(
        '--all-index',
        action='store_true',
        help='下载全量指数列表（INDX），优先级高于 index_list/index_symbols'
    )

    parser.add_argument(
        '--index-symbols',
        default=None,
        help='指定下载的指数列表（vnpy代码，逗号分隔），例如 000300.SSE,000852.SSE'
    )

    parser.add_argument(
        '--merge-index',
        action='store_true',
        help='将指数行情与股票保存到同一目录（默认保存到 ./lab/{task_name}/index/）'
    )

    parser.add_argument(
        '--index-dir',
        default=None,
        help='指数保存子目录名（默认 index；仅在未启用 --merge-index 时生效）'
    )
    
    args = parser.parse_args()

    return args


def main():
    """主函数"""
    
    # ========== 解析命令行参数 ==========
    args = parse_arguments()

    # ========== 读取配置文件并合并参数 ==========
    cfg = _load_config_file(args.config)

    # 调试输出：配置文件读取情况与关键参数
    try:
        cfg_exists = Path(args.config).exists()
        print(f"[配置] 文件: {args.config} ({'已找到' if cfg_exists else '未找到'})")
        if cfg:
            print(
                f"[配置] interval={cfg.get('interval')}, start_date={cfg.get('start_date')}, end_date={cfg.get('end_date')}, "
                f"task_name={cfg.get('task_name')}, mode={cfg.get('mode')}, full={cfg.get('full')}, "
                f"targets={cfg.get('targets')}, include_index={cfg.get('include_index')}, all_index={cfg.get('all_index')}, "
                f"merge_index={cfg.get('merge_index')}, index_dir={cfg.get('index_dir')}"
            )
        else:
            print("[配置] 配置为空或解析失败，使用默认/命令行参数")
    except Exception:
        pass

    # 计算默认日期
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # 合并参数：CLI > 配置 > 默认
    start_date = args.start_date or cfg.get('start_date') or '2007-01-01'
    end_date = args.end_date or cfg.get('end_date') or yesterday
    task_name = args.task_name or cfg.get('task_name') or 'all_stocks'
    interval_arg = args.interval or cfg.get('interval') or '1d'

    # 增量/全量：默认增量。支持配置字段 full 或 mode='full'
    cfg_full = False
    mode = str(cfg.get('mode', '')).lower()
    if cfg.get('full') is True or mode == 'full':
        cfg_full = True
    full_mode = args.full or cfg_full

    # 合同/校验布尔参数：CLI 的 no_* 优先，其次配置 set_contract/verify/no_*
    if args.no_contract:
        set_contract_params = False
    else:
        if 'no_contract' in cfg:
            set_contract_params = not bool(cfg['no_contract'])
        elif 'set_contract' in cfg:
            set_contract_params = bool(cfg['set_contract'])
        else:
            set_contract_params = True

    if args.no_verify:
        verify_download = False
    else:
        if 'no_verify' in cfg:
            verify_download = not bool(cfg['no_verify'])
        elif 'verify' in cfg:
            verify_download = bool(cfg['verify'])
        else:
            verify_download = True

    # 下载范围：CLI > 配置 > 兼容 include_index > 默认 stocks
    targets = args.targets
    if targets is None:
        if args.include_index:
            targets = "both"
        else:
            targets = cfg.get("targets")
    if targets is None:
        include_index = _is_truthy(cfg.get("include_index")) or _is_truthy(cfg.get("download_index"))
        targets = "both" if include_index else "stocks"
    targets = str(targets).strip().lower()
    if targets not in {"stocks", "index", "both"}:
        print(f"[配置] targets={targets} 非法，已回退为 stocks")
        targets = "stocks"

    # 指数下载参数
    all_index = args.all_index or _is_truthy(cfg.get("all_index")) or _is_truthy(cfg.get("index_all"))
    merge_index = args.merge_index or _is_truthy(cfg.get("merge_index"))
    index_dir = str(args.index_dir or cfg.get("index_dir") or "index").strip() or "index"

    # 自定义指数列表：CLI > 配置 index_symbols > 配置 index_list/内置列表
    index_symbols_override: list[str] = _parse_symbol_list(args.index_symbols)
    if not index_symbols_override:
        index_symbols_override = _parse_symbol_list(cfg.get("index_symbols"))

    # 生效参数调试打印
    try:
        print(
            f"[生效] interval={interval_arg}, start_date={start_date}, end_date={end_date}, "
            f"task_name={task_name}, mode={'full' if full_mode else 'incremental'}, "
            f"targets={targets}, all_index={all_index}, merge_index={merge_index}, index_dir={index_dir}"
        )
    except Exception:
        pass

    # 验证日期格式
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"日期格式错误：{e}")
        return
    
    # ========== 配置参数 ==========
    
    # K线周期映射
    interval_map = {
        '1d': Interval.DAILY,
        '1m': Interval.MINUTE,
        '60m': Interval.HOUR,
        # 10m和30m会使用扩展函数处理
    }
    interval = interval_map.get(interval_arg, Interval.MINUTE)
    
    # 回测参数配置
    contract_settings = {
        "long_rate": 5/10000,     # 做多手续费率（万分之五）
        "short_rate": 10/10000,   # 做空手续费率（千分之一）
        "size": 1,                 # 合约乘数
        "pricetick": 0.0001,      # 最小价格变动
    }
    
    # full_mode, set_contract_params, verify_download 已在上方合并
    
    # ========== 初始化环境 ==========
    
    print("="*60)
    print("全市场股票数据下载工具")
    print("="*60)
    print(f"\n下载参数：")
    print(f"  - 开始日期：{start_date}")
    print(f"  - 结束日期：{end_date}")
    print(f"  - K线周期：{interval_arg}")
    print(f"  - 下载模式：{'全量' if full_mode else '增量(重复最后一天)'}")
    print(f"  - 任务名称：{task_name}")
    targets_text = {"stocks": "仅股票", "index": "仅指数", "both": "股票+指数"}[targets]
    print(f"  - 下载范围：{targets_text}")
    if targets in {"index", "both"}:
        if all_index:
            index_src = "全量指数(INDX)"
        elif index_symbols_override:
            index_src = f"自定义列表({len(index_symbols_override)}个)"
        else:
            index_src = "默认指数列表(index_list)"
        print(f"  - 指数列表：{index_src}")
        print(f"  - 指数保存：{'合并到主目录' if merge_index else f'子目录 {index_dir}/'}")
    print(f"  - 设置回测参数：{'是' if set_contract_params else '否'}")
    print(f"  - 验证数据：{'是' if verify_download else '否'}")
    
    # 创建投研实验室
    base_lab = AlphaLab(f"./lab/{task_name}")
    print(f"\n股票数据保存路径：./lab/{task_name}/")
    index_lab = base_lab
    if targets in {"index", "both"} and not merge_index:
        index_lab = AlphaLab(str(base_lab.lab_path.joinpath(index_dir)))
        print(f"指数数据保存路径：./lab/{task_name}/{index_dir}/")
    
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
    
    # ========== 获取标的列表 ==========
    stock_symbols: list[str] = []
    index_symbols: list[str] = []
    index_symbols_set: set[str] = set()

    if targets in {"stocks", "both"}:
        stock_symbols = get_stock_symbols()
        if not stock_symbols and targets == "stocks":
            print("未能获取股票列表，程序退出")
            return 0

    if targets in {"index", "both"}:
        if all_index:
            index_symbols = get_all_index_symbols()
        elif index_symbols_override:
            index_symbols = _dedup_keep_order(index_symbols_override)
            print(f"使用自定义指数列表，共 {len(index_symbols)} 个")
        else:
            index_symbols = get_default_index_symbols(cfg)
            print(f"使用默认指数列表，共 {len(index_symbols)} 个")

        index_symbols_set = set(index_symbols)
        if not index_symbols and targets == "index":
            print("未能获取指数列表，程序退出")
            return 0
    
    # ========== 下载数据 ==========
    failed_stock_symbols: list[str] = []
    failed_index_symbols: list[str] = []
    failed_all_symbols: list[str] = []
    all_symbols: list[str] = []

    def run_download(lab: AlphaLab, symbols: list[str], index_set: set[str]) -> list[str]:
        """按 interval_arg 调用相应下载函数，返回失败列表。"""
        if not symbols:
            return []
        if interval_arg in ["10m", "30m"]:
            return download_bar_data_extended(
                lab,
                symbols,
                start,
                end,
                interval_arg,
                full_mode=full_mode,
                index_symbols=index_set,
            )
        # 使用标准下载函数处理日线、1分钟、60分钟
        interval_value = interval_arg if interval_arg in ["1m", "60m"] else None
        return download_bar_data(
            lab,
            symbols,
            start,
            end,
            interval,
            interval_value,
            full_mode=full_mode,
            index_symbols=index_set,
        )

    if targets == "stocks":
        failed_stock_symbols = run_download(base_lab, stock_symbols, set())
    elif targets == "index":
        failed_index_symbols = run_download(index_lab, index_symbols, index_symbols_set)
    else:  # both
        if merge_index:
            all_symbols = _dedup_keep_order([*stock_symbols, *index_symbols])
            failed_all_symbols = run_download(base_lab, all_symbols, index_symbols_set)
            failed_stock_symbols = [s for s in failed_all_symbols if s not in index_symbols_set]
            failed_index_symbols = [s for s in failed_all_symbols if s in index_symbols_set]
        else:
            failed_stock_symbols = run_download(base_lab, stock_symbols, set())
            failed_index_symbols = run_download(index_lab, index_symbols, index_symbols_set)

    # 输出统计信息
    print("\n下载完成！")
    success_count = 0
    if targets in {"stocks", "both"}:
        success_stock = len(stock_symbols) - len(failed_stock_symbols)
        print(f"  - 股票：成功 {success_stock} / {len(stock_symbols)}")
        success_count += success_stock
        if failed_stock_symbols:
            print("\n失败的股票列表（前20个）：")
            for symbol in failed_stock_symbols[:20]:
                print(f"  - {symbol}")
            if len(failed_stock_symbols) > 20:
                print(f"  ... 还有 {len(failed_stock_symbols) - 20} 只")

    if targets in {"index", "both"}:
        success_index = len(index_symbols) - len(failed_index_symbols)
        print(f"  - 指数：成功 {success_index} / {len(index_symbols)}")
        success_count += success_index
        if failed_index_symbols:
            print("\n失败的指数列表（前20个）：")
            for symbol in failed_index_symbols[:20]:
                print(f"  - {symbol}")
            if len(failed_index_symbols) > 20:
                print(f"  ... 还有 {len(failed_index_symbols) - 20} 个")
    
    # ========== 设置回测参数 ==========
    
    if set_contract_params and targets in {"stocks", "both"}:
        successful_stock_symbols = [s for s in stock_symbols if s not in failed_stock_symbols]
        set_contract_settings(base_lab, successful_stock_symbols, contract_settings)
    
    # ========== 保存下载报告 ==========
    if targets == "stocks":
        save_download_report(base_lab, len(stock_symbols), failed_stock_symbols)
    elif targets == "index":
        save_download_report(index_lab, len(index_symbols), failed_index_symbols)
    else:  # both
        if merge_index:
            save_download_report(base_lab, len(all_symbols), failed_all_symbols)
        else:
            save_download_report(base_lab, len(stock_symbols), failed_stock_symbols)
            save_download_report(index_lab, len(index_symbols), failed_index_symbols)
    
    # ========== 数据验证 ==========
    
    if verify_download:
        if targets in {"stocks", "both"}:
            successful_stock_symbols = [s for s in stock_symbols if s not in failed_stock_symbols]
            if successful_stock_symbols:
                verify_data(base_lab, successful_stock_symbols, start, end)
        if targets in {"index", "both"}:
            successful_index_symbols = [s for s in index_symbols if s not in failed_index_symbols]
            if successful_index_symbols:
                verify_data(index_lab, successful_index_symbols, start, end)
    
    # ========== 完成 ==========
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print(f"股票数据保存在：./lab/{task_name}/")
    if targets in {"index", "both"} and not merge_index:
        print(f"指数数据保存在：./lab/{task_name}/{index_dir}/")
    print("="*60)
    
    return success_count  # 返回成功下载的数量


if __name__ == "__main__":
    # 执行主函数
    success_count = main()
    print(f"\n程序执行完毕，成功下载 {success_count} 个标的数据")
