"""
指数成分股列表下载脚本

功能：
1. 下载多个指数的历史成分股列表
2. 保存成分股变化历史到AlphaLab
3. 支持单个或多个指数下载
4. 自动去重合并成分股列表

使用方法：
    python download_index_components.py [开始日期] [结束日期]
    
    示例：
    python download_index_components.py  # 使用默认日期
    python download_index_components.py 2010-01-01  # 指定开始日期
    python download_index_components.py 2010-01-01 2024-12-31  # 指定开始和结束日期
"""

import sys
sys.path.append('/home/hyd/research/vnpy_hub/vnpy')
sys.path.append('/home/hyd/research/vnpy_hub/vnpy_rqdata')

import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Set

import rqdatac as rq
from vnpy.trader.datafeed import get_datafeed
from vnpy.alpha import AlphaLab, logger


def get_index_components(lab: AlphaLab, rq_index_symbol: str, index_symbol: str, 
                        start_date: str, end_date: str) -> Set[str]:
    """
    获取单个指数的成分股列表
    
    Args:
        lab: AlphaLab实例
        rq_index_symbol: 米筐格式的指数代码
        index_symbol: vnpy格式的指数代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        所有历史成分股的集合
    """
    try:
        # 下载指数成分股历史数据
        data = rq.index_components(rq_index_symbol, start_date=start_date, end_date=end_date)
        
        # 转换合约代码
        index_components = {}
        all_component_symbols = set()
        
        for dt, rq_symbols in data.items():
            vt_symbols: list = []
            
            for rq_symbol in rq_symbols:
                vt_symbol = rq_symbol.replace("XSHG", "SSE").replace("XSHE", "SZSE")
                vt_symbols.append(vt_symbol)
                all_component_symbols.add(vt_symbol)
            
            index_components[dt.strftime("%Y-%m-%d")] = vt_symbols
        
        # 保存成分股数据到数据中心
        lab.save_component_data(index_symbol, index_components)
        
        print(f"  - 成功获取 {len(all_component_symbols)} 只历史成分股")
        
        # 加上指数本身
        all_component_symbols.add(index_symbol)
        
        return all_component_symbols
        
    except Exception as e:
        logger.error(f"获取指数 {index_symbol} 成分股失败：{e}")
        return set()


def get_multiple_indexes_components(lab: AlphaLab, index_list: List[Dict], 
                                   start_date: str, end_date: str) -> List[str]:
    """
    获取多个指数的成分股列表（合并去重）
    
    Args:
        lab: AlphaLab实例
        index_list: 指数列表，每个元素包含vnpy、rq、name字段
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        所有指数成分股的列表（去重后）
    """
    all_symbols = set()  # 使用集合自动去重
    
    print("\n开始获取指数成分股列表...")
    print(f"时间范围：{start_date} 至 {end_date}\n")
    
    for index_info in index_list:
        vnpy_symbol = index_info["vnpy"]
        rq_symbol = index_info["rq"]
        name = index_info["name"]
        
        print(f"正在获取 {name}({vnpy_symbol}) 成分股...")
        
        # 获取单个指数的成分股
        index_symbols = get_index_components(lab, rq_symbol, vnpy_symbol, start_date, end_date)
        
        if index_symbols:
            all_symbols.update(index_symbols)
            print(f"  - 累计股票数量：{len(all_symbols)}")
        else:
            print(f"  - 获取失败，跳过")
    
    # 转换为列表并排序
    symbol_list = sorted(list(all_symbols))
    
    print(f"\n汇总统计：")
    print(f"  - 获取指数数量：{len(index_list)}")
    print(f"  - 不重复股票和指数总数：{len(symbol_list)}")
    
    return symbol_list


def save_component_summary(lab: AlphaLab, symbols: List[str], filename: str = "all_components.txt"):
    """
    保存成分股汇总列表到文件
    
    Args:
        lab: AlphaLab实例
        symbols: 股票代码列表
        filename: 保存的文件名
    """
    import os
    
    # 确保目录存在
    component_dir = lab.lab_path.joinpath("component")
    component_dir.mkdir(exist_ok=True)
    
    # 保存到文件
    file_path = component_dir.joinpath(filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"# 指数成分股汇总列表\n")
        f.write(f"# 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 股票总数：{len(symbols)}\n\n")
        
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    print(f"\n成分股列表已保存到：{file_path}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='下载指数成分股列表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python download_index_components.py                    # 使用默认日期（2007-01-01 至 昨天）
  python download_index_components.py 2010-01-01         # 指定开始日期（2010-01-01 至 昨天）
  python download_index_components.py 2010-01-01 2024-12-31  # 指定日期范围
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
    
    # 指数列表配置
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
    
    # 是否下载单个指数（设为True则只下载一个指数）
    single_index_mode = False
    single_index = {"vnpy": "000300.SSE", "rq": "000300.XSHG", "name": "沪深300"}
    
    # ========== 初始化环境 ==========
    
    print("="*60)
    print("指数成分股列表下载工具")
    print("="*60)
    print(f"\n下载参数：")
    print(f"  - 开始日期：{start_date}")
    print(f"  - 结束日期：{end_date}")
    print(f"  - 任务名称：{task_name}")
    
    # 创建投研实验室
    lab = AlphaLab(f"./lab/{task_name}")
    print(f"\n数据保存路径：./lab/{task_name}/")
    
    # 初始化数据服务（米筐数据）
    print("\n初始化米筐数据服务...")
    datafeed = get_datafeed()
    if not datafeed.init():
        print("米筐数据服务初始化失败，请检查配置")
        return
    print("米筐数据服务初始化成功")
    
    # ========== 获取成分股列表 ==========
    
    if single_index_mode:
        # 单个指数模式
        print(f"\n模式：下载单个指数成分股 - {single_index['name']}")
        components = get_index_components(
            lab, 
            single_index["rq"], 
            single_index["vnpy"], 
            start_date, 
            end_date
        )
        all_symbols = sorted(list(components))
        
    else:
        # 多个指数模式
        print(f"\n模式：下载多个指数成分股")
        all_symbols = get_multiple_indexes_components(lab, index_list, start_date, end_date)
    
    # ========== 保存汇总列表 ==========
    
    if all_symbols:
        save_component_summary(lab, all_symbols)
        
        # 显示部分结果
        print("\n成分股列表示例（前10个）：")
        for symbol in all_symbols[:10]:
            print(f"  - {symbol}")
        if len(all_symbols) > 10:
            print(f"  ... 还有 {len(all_symbols) - 10} 个")
    else:
        print("\n未获取到任何成分股数据")
    
    # ========== 完成 ==========
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print("="*60)
    
    # 返回成分股列表，便于其他脚本调用
    return all_symbols


if __name__ == "__main__":
    # 执行主函数
    symbols = main()