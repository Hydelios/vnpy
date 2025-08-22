# %% [markdown]
# # 数据下载脚本
# 
# 支持多种下载模式：
# - 全市场股票下载
# - 指数成分股下载
# - 自定义股票列表下载

# %% [markdown]
# ## 1. 导入模块

# %%
import sys
sys.path.append('F:\\git\\vnpy_hub\\vnpy')
sys.path.append('F:\\git\\vnpy_hub\\vnpy_rqdata')

# %%
# 加载基础模块
from datetime import datetime
from typing import List, Optional

from tqdm import tqdm
import rqdatac as rq
import pandas as pd

from vnpy.trader.database import DB_TZ
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest
from vnpy.alpha import AlphaLab, logger

# %% [markdown]
# ## 2. 下载参数配置

# %%
# 下载模式选择：'all_stocks' | 'index_components' | 'custom_list' | 'multiple_indexes'
DOWNLOAD_MODE = "multiple_indexes"  # 可以改为 'all_stocks', 'index_components' 或 'custom_list'

# 任务名称（用于创建数据文件夹）
task_name = "all_stocks"

# 时间范围设置
start_date = "2007-01-01"
end_date = "2025-08-20"

# 单个指数参数（当 DOWNLOAD_MODE = 'index_components' 时使用）
index_symbol = "000300.SSE"  # vnpy格式
rq_index_symbol = "000300.XSHG"  # 米筐格式

# 多个指数列表（当 DOWNLOAD_MODE = 'multiple_indexes' 时使用）
index_list = [
    {"vnpy": "000016.SSE", "rq": "000016.XSHG", "name": "上证50"},
    {"vnpy": "000300.SSE", "rq": "000300.XSHG", "name": "沪深300"},
    {"vnpy": "000905.SSE", "rq": "000905.XSHG", "name": "中证500"},
    {"vnpy": "000906.SSE", "rq": "000906.XSHG", "name": "中证800"},
    {"vnpy": "000852.SSE", "rq": "000852.XSHG", "name": "中证1000"},
    {"vnpy": "932000.SSE", "rq": "932000.XSHG", "name": "中证2000"},
    {"vnpy": "000688.SSE", "rq": "000688.XSHG", "name": "科创50"},
    {"vnpy": "000922.SSE", "rq": "000922.XSHG", "name": "中证红利"},
]

# 自定义股票列表（当 DOWNLOAD_MODE = 'custom_list' 时使用）
custom_symbols = [
    "000001.SZSE",
    "000002.SZSE", 
    "600000.SSE",
    # 添加更多股票...
]

# 回测参数配置
contract_settings = {
    "long_rate": 5/10000,     # 做多手续费率
    "short_rate": 10/10000,   # 做空手续费率
    "size": 1,                 # 合约乘数
    "pricetick": 0.0001,      # 最小价格变动
}

# %% [markdown]
# ## 3. 初始化环境

# %%
# 创建投研实验室
lab = AlphaLab(f"./lab/{task_name}")

# 初始化数据服务
datafeed = get_datafeed()
datafeed.init()

# 转换时间格式
start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=DB_TZ)
end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=DB_TZ)

# %% [markdown]
# ## 4. 获取股票列表

# %%
def get_all_stocks() -> List[str]:
    """获取全市场股票列表"""
    df = rq.all_instruments(type='CS')  # CS表示普通股票
    
    # 转换为vnpy格式的股票代码列表
    all_symbols = []
    for _, row in df.iterrows():
        rq_symbol = row['order_book_id']
        # 转换格式：XSHG->SSE, XSHE->SZSE
        vt_symbol = rq_symbol.replace("XSHG", "SSE").replace("XSHE", "SZSE")
        all_symbols.append(vt_symbol)
    
    print(f"共获取到 {len(all_symbols)} 只股票")
    return all_symbols


def get_index_components(rq_index_symbol: str, index_symbol: str, 
                        start_date: str, end_date: str) -> List[str]:
    """获取指数成分股列表"""
    # 下载指数成分股历史数据
    data = rq.index_components(rq_index_symbol, start_date=start_date, end_date=end_date)
    
    # 转换合约代码
    index_components = {}
    all_component_symbols = set()  # 使用集合去重
    
    for dt, rq_symbols in data.items():
        vt_symbols: list = []
        
        for rq_symbol in rq_symbols:
            vt_symbol = rq_symbol.replace("XSHG", "SSE").replace("XSHE", "SZSE")
            vt_symbols.append(vt_symbol)
            all_component_symbols.add(vt_symbol)
        
        index_components[dt.strftime("%Y-%m-%d")] = vt_symbols
    
    # 保存成分股数据到数据中心
    lab.save_component_data(index_symbol, index_components)
    
    # 返回所有出现过的成分股（去重后）
    component_list = list(all_component_symbols)
    print(f"共获取到 {len(component_list)} 只成分股")
    
    # 加上指数本身
    component_list.append(index_symbol)
    
    return component_list


def get_multiple_indexes_components(index_list: List[dict], 
                                   start_date: str, end_date: str) -> List[str]:
    """获取多个指数的成分股列表（合并去重）"""
    all_symbols = set()  # 使用集合自动去重
    
    for index_info in index_list:
        vnpy_symbol = index_info["vnpy"]
        rq_symbol = index_info["rq"]
        name = index_info["name"]
        
        print(f"\n正在获取 {name}({vnpy_symbol}) 成分股...")
        
        try:
            # 下载指数成分股历史数据
            data = rq.index_components(rq_symbol, start_date=start_date, end_date=end_date)
            
            # 转换合约代码并保存
            index_components = {}
            index_symbols = set()
            
            for dt, rq_symbols in data.items():
                vt_symbols: list = []
                
                for symbol in rq_symbols:
                    vt_symbol = symbol.replace("XSHG", "SSE").replace("XSHE", "SZSE")
                    vt_symbols.append(vt_symbol)
                    index_symbols.add(vt_symbol)
                    all_symbols.add(vt_symbol)  # 添加到总集合
                
                index_components[dt.strftime("%Y-%m-%d")] = vt_symbols
            
            # 保存每个指数的成分股数据
            lab.save_component_data(vnpy_symbol, index_components)
            print(f"  - {name} 历史成分股数量：{len(index_symbols)}")
            
            # 添加指数本身
            all_symbols.add(vnpy_symbol)
            
        except Exception as e:
            print(f"  - 获取 {name} 成分股失败：{e}")
    
    # 转换为列表并排序
    symbol_list = sorted(list(all_symbols))
    print(f"\n总计获取到 {len(symbol_list)} 只不重复的股票和指数")
    
    return symbol_list


# %%
# 根据模式获取股票列表
if DOWNLOAD_MODE == "all_stocks":
    print("模式：下载全市场股票")
    task_symbols = get_all_stocks()
    
elif DOWNLOAD_MODE == "index_components":
    print(f"模式：下载指数成分股 - {index_symbol}")
    task_symbols = get_index_components(rq_index_symbol, index_symbol, start_date, end_date)
    task_symbols=[]
    
elif DOWNLOAD_MODE == "multiple_indexes":
    print("模式：下载多个指数成分股")
    task_symbols = get_multiple_indexes_components(index_list, start_date, end_date)
    task_symbols=[]
    
elif DOWNLOAD_MODE == "custom_list":
    print("模式：下载自定义股票列表")
    task_symbols = custom_symbols
    print(f"共 {len(task_symbols)} 只股票")
    
else:
    raise ValueError(f"不支持的下载模式: {DOWNLOAD_MODE}")

# %% [markdown]
# ## 5. 下载数据

# %%
def download_bar_data(symbols: List[str], start: datetime, end: datetime, 
                     interval: Interval = Interval.DAILY) -> dict:
    """下载K线数据
    
    返回下载失败的股票列表
    """
    failed_symbols = []
    
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
                lab.save_bar_data(bars)
            else:
                logger.error(f"下载{vt_symbol}数据失败：无数据返回")
                failed_symbols.append(vt_symbol)
                
        except Exception as e:
            logger.error(f"下载{vt_symbol}数据失败：{e}")
            failed_symbols.append(vt_symbol)
    
    return failed_symbols


# %%
# 执行下载
print(f"\n开始下载 {len(task_symbols)} 只股票的日线数据...")
print(f"时间范围：{start_date} 至 {end_date}")

failed_symbols = download_bar_data(task_symbols, start, end, Interval.DAILY)

# 输出统计信息
print(f"\n下载完成！")
print(f"成功：{len(task_symbols) - len(failed_symbols)} 只")
print(f"失败：{len(failed_symbols)} 只")

if failed_symbols:
    print("\n失败的股票列表：")
    for symbol in failed_symbols[:10]:  # 只显示前10个
        print(f"  - {symbol}")
    if len(failed_symbols) > 10:
        print(f"  ... 还有 {len(failed_symbols) - 10} 只")

# %% [markdown]
# ## 6. 设置回测参数

# %%
def set_contract_settings(symbols: List[str], settings: dict):
    """为股票列表设置回测参数"""
    for vt_symbol in tqdm(symbols, desc="设置回测参数"):
        lab.add_contract_setting(
            vt_symbol,
            long_rate=settings["long_rate"],
            short_rate=settings["short_rate"],
            size=settings["size"],
            pricetick=settings["pricetick"],
        )


# %%
# 为成功下载的股票设置回测参数
successful_symbols = [s for s in task_symbols if s not in failed_symbols]
print(f"\n为 {len(successful_symbols)} 只股票设置回测参数...")
set_contract_settings(successful_symbols, contract_settings)

print("\n所有任务完成！")
print(f"数据保存路径：./lab/{task_name}/")

