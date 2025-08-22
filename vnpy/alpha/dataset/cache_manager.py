"""
因子缓存管理工具类

提供因子依赖分析、缓存管理和分层并行计算功能
"""

import re
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any
from multiprocessing import get_context
from multiprocessing.context import BaseContext

import polars as pl
from tqdm import tqdm

from ..logger import logger
from .utility import calculate_by_expression, calculate_by_polars


class FactorCacheManager:
    """因子缓存管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None, interval: str = "1d") -> None:
        """
        初始化缓存管理器
        
        Parameters:
        -----------
        cache_dir: Optional[str]
            缓存目录路径，如果为None则创建临时目录
        interval: str
            数据频率标识，如 1d, 1m, 10m, 30m, 60m
        """
        self.interval = interval
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            # 按频率创建子目录
            self.cache_dir = self.cache_dir / interval
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="factor_cache_"))
            self.cache_dir = temp_dir / interval
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cached_factors: Dict[str, Path] = {}  # 记录已缓存的因子路径
        logger.info(f"因子缓存目录: {self.cache_dir} (频率: {interval})")
    
    def analyze_dependencies(
        self, 
        expressions: Dict[str, Union[str, pl.expr.expr.Expr]]
    ) -> Dict[str, List[str]]:
        """
        分析因子之间的依赖关系
        
        Parameters:
        -----------
        expressions: Dict[str, Union[str, pl.expr.expr.Expr]]
            因子表达式字典
            
        Returns:
        --------
        Dict[str, List[str]]
            每个因子依赖的其他因子列表
        """
        dependencies = {}
        
        for name, expr in expressions.items():
            deps = []
            if isinstance(expr, str):
                # 使用正则表达式查找引用的其他因子
                for other_name in expressions:
                    if other_name != name:
                        # 匹配因子名作为独立的词（前后是非字母数字字符或边界）
                        pattern = r'\b' + re.escape(other_name) + r'\b'
                        if re.search(pattern, expr):
                            deps.append(other_name)
            dependencies[name] = deps
        
        return dependencies
    
    def topological_sort(
        self, 
        expressions: Dict[str, Union[str, pl.expr.expr.Expr]]
    ) -> List[List[str]]:
        """
        对因子进行拓扑排序，返回分层列表
        
        Parameters:
        -----------
        expressions: Dict[str, Union[str, pl.expr.expr.Expr]]
            因子表达式字典
            
        Returns:
        --------
        List[List[str]]
            分层的因子名称列表，每层可以并行计算
        """
        dependencies = self.analyze_dependencies(expressions)
        
        layers = []
        remaining = dict(dependencies)
        processed = set()
        
        while remaining:
            # 找出当前层：没有依赖或所有依赖都已处理
            current_layer = []
            for name, deps in list(remaining.items()):
                if not deps or all(dep in processed for dep in deps):
                    current_layer.append(name)
            
            if not current_layer:
                # 存在循环依赖
                logger.warning(f"检测到循环依赖，剩余因子: {list(remaining.keys())}")
                # 将剩余的都放入一层
                current_layer = list(remaining.keys())
            
            layers.append(current_layer)
            
            # 更新已处理集合和剩余字典
            for name in current_layer:
                processed.add(name)
                remaining.pop(name, None)
        
        return layers
    
    def load_factor(self, name: str) -> Optional[pl.DataFrame]:
        """
        从缓存加载单个因子
        
        Parameters:
        -----------
        name: str
            因子名称
            
        Returns:
        --------
        Optional[pl.DataFrame]
            因子数据，如果不存在则返回None
        """
        # 使用带频率标识的缓存键
        cache_key = f"{name}_{self.interval}"
        if cache_key in self.cached_factors:
            cache_path = self.cached_factors[cache_key]
            if cache_path.exists():
                return pl.read_parquet(cache_path)
        
        # 尝试从文件系统查找
        cache_path = self.cache_dir / f"{name}.parquet"
        if cache_path.exists():
            self.cached_factors[cache_key] = cache_path
            return pl.read_parquet(cache_path)
        
        return None
    
    def save_factor(self, name: str, data: pl.DataFrame) -> None:
        """
        保存单个因子到缓存
        
        Parameters:
        -----------
        name: str
            因子名称
        data: pl.DataFrame
            因子数据
        """
        # 使用不带频率的文件名，因为已经在子目录中
        cache_path = self.cache_dir / f"{name}.parquet"
        data.write_parquet(cache_path)
        
        # 使用带频率标识的缓存键
        cache_key = f"{name}_{self.interval}"
        self.cached_factors[cache_key] = cache_path
    
    def calculate_factors_with_cache(
        self,
        df: pl.DataFrame,
        expressions: Dict[str, Union[str, pl.expr.expr.Expr]],
        n_jobs: int = 4,
        context: Optional[BaseContext] = None
    ) -> pl.DataFrame:
        """
        带缓存的因子计算主方法
        
        Parameters:
        -----------
        df: pl.DataFrame
            原始数据
        expressions: Dict[str, Union[str, pl.expr.expr.Expr]]
            因子表达式字典
        n_jobs: int
            并行进程数
        context: Optional[BaseContext]
            多进程上下文
            
        Returns:
        --------
        pl.DataFrame
            包含所有计算结果的DataFrame
        """
        if not expressions:
            return df
        
        # 获取分层结构
        layers = self.topological_sort(expressions)
        logger.info(f"因子依赖分析完成，共{len(layers)}层")
        for i, layer in enumerate(layers):
            logger.info(f"第{i+1}层: {layer}")
        
        # 累积的DataFrame，包含原始数据和所有已计算的因子
        accumulated_df = df.clone()
        
        # 逐层计算
        for layer_idx, layer_factors in enumerate(layers):
            logger.info(f"开始计算第{layer_idx + 1}层，包含{len(layer_factors)}个因子")
            
            # 检查缓存
            factors_to_calculate = []
            for name in layer_factors:
                cached_data = self.load_factor(name)
                if cached_data is not None:
                    logger.info(f"从缓存加载因子: {name}")
                    # 只添加data列到accumulated_df
                    accumulated_df = accumulated_df.with_columns(
                        cached_data["data"].alias(name)
                    )
                else:
                    factors_to_calculate.append(name)
            
            # 并行计算需要计算的因子
            if factors_to_calculate:
                if n_jobs == 1:
                    # 单进程计算
                    for name in tqdm(factors_to_calculate, desc=f"Layer {layer_idx + 1}"):
                        result = self._calculate_single_factor(
                            accumulated_df, name, expressions[name]
                        )
                        accumulated_df = accumulated_df.with_columns(
                            result.alias(name)
                        )
                        # 保存到缓存
                        # 检测使用的字段名
                        symbol_col = "symbol" if "symbol" in accumulated_df.columns else "vt_symbol"
                        factor_df = pl.DataFrame({
                            "datetime": accumulated_df["datetime"],
                            symbol_col: accumulated_df[symbol_col],
                            "data": result
                        })
                        self.save_factor(name, factor_df)
                else:
                    # 多进程计算
                    args_list = [
                        (accumulated_df, name, expressions[name])
                        for name in factors_to_calculate
                    ]
                    
                    ctx = context or get_context("spawn")
                    with ctx.Pool(n_jobs) as pool:
                        results = list(tqdm(
                            pool.imap(_calculate_factor_wrapper, args_list),
                            total=len(args_list),
                            desc=f"Layer {layer_idx + 1}"
                        ))
                    
                    # 添加计算结果到accumulated_df并缓存
                    for name, result in results:
                        accumulated_df = accumulated_df.with_columns(
                            result.alias(name)
                        )
                        # 保存到缓存
                        # 检测使用的字段名
                        symbol_col = "symbol" if "symbol" in accumulated_df.columns else "vt_symbol"
                        factor_df = pl.DataFrame({
                            "datetime": accumulated_df["datetime"],
                            symbol_col: accumulated_df[symbol_col],
                            "data": result
                        })
                        self.save_factor(name, factor_df)
        
        return accumulated_df
    
    def _calculate_single_factor(
        self,
        df: pl.DataFrame,
        name: str,
        expression: Union[str, pl.expr.expr.Expr]
    ) -> pl.Series:
        """
        计算单个因子
        
        Parameters:
        -----------
        df: pl.DataFrame
            包含所有必要数据的DataFrame
        name: str
            因子名称
        expression: Union[str, pl.expr.expr.Expr]
            因子表达式
            
        Returns:
        --------
        pl.Series
            计算结果
        """
        if isinstance(expression, pl.expr.expr.Expr):
            result = calculate_by_polars(df, expression)["data"]
        else:
            result = calculate_by_expression(df, expression)["data"]
        return result
    
    def clear_cache(self) -> None:
        """清理所有缓存文件"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            logger.info(f"已清理缓存目录: {self.cache_dir}")
        self.cached_factors.clear()
    
    def __del__(self):
        """析构函数，可选择性清理临时缓存"""
        # 如果是临时目录，可以选择清理
        if self.cache_dir and "temp" in str(self.cache_dir).lower():
            try:
                self.clear_cache()
                if self.cache_dir.exists():
                    self.cache_dir.rmdir()
            except Exception as e:
                logger.warning(f"清理临时缓存目录失败: {e}")


def _calculate_factor_wrapper(args: Tuple[pl.DataFrame, str, Union[str, pl.expr.expr.Expr]]) -> Tuple[str, pl.Series]:
    """
    多进程计算的包装函数
    
    Parameters:
    -----------
    args: Tuple
        (DataFrame, 因子名称, 因子表达式)
        
    Returns:
    --------
    Tuple[str, pl.Series]
        (因子名称, 计算结果)
    """
    df, name, expression = args
    
    if isinstance(expression, pl.expr.expr.Expr):
        result = calculate_by_polars(df, expression)["data"]
    else:
        result = calculate_by_expression(df, expression)["data"]
    
    return name, result