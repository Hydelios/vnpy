import json
import shutil
import shelve
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from functools import lru_cache

import polars as pl

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval
from vnpy.trader.utility import extract_vt_symbol

from .logger import logger
from .dataset import AlphaDataset, to_datetime, Segment
from .model import AlphaModel


class AlphaLab:
    """Alpha Research Laboratory"""

    def __init__(self, lab_path: str) -> None:
        """Constructor"""
        # Set data paths
        self.lab_path: Path = Path(lab_path)

        self.daily_path: Path = self.lab_path.joinpath("daily")
        self.minute_path: Path = self.lab_path.joinpath("minute")
        self.component_path: Path = self.lab_path.joinpath("component")

        # 为不同分钟级别创建子文件夹
        self.minute_1m_path: Path = self.minute_path.joinpath("1m")
        self.minute_10m_path: Path = self.minute_path.joinpath("10m")
        self.minute_30m_path: Path = self.minute_path.joinpath("30m")
        self.minute_60m_path: Path = self.minute_path.joinpath("60m")

        self.dataset_path: Path = self.lab_path.joinpath("dataset")
        self.model_path: Path = self.lab_path.joinpath("model")
        self.signal_path: Path = self.lab_path.joinpath("signal")

        self.contract_path: Path = self.lab_path.joinpath("contract.json")

        # Create folders
        for path in [
            self.lab_path,
            self.daily_path,
            self.minute_path,
            self.minute_1m_path,
            self.minute_10m_path,
            self.minute_30m_path,
            self.minute_60m_path,
            self.component_path,
            self.dataset_path,
            self.model_path,
            self.signal_path
        ]:
            if not path.exists():
                path.mkdir(parents=True)

    def get_minute_folder_path(self, interval: Interval, interval_value: str = None) -> Path:
        """获取分钟级别数据的文件夹路径"""
        if interval == Interval.MINUTE:
            # 1分钟线，如果有特定的interval_value则使用对应文件夹
            if interval_value == "10m":
                return self.minute_10m_path
            elif interval_value == "30m":
                return self.minute_30m_path
            else:
                return self.minute_1m_path
        elif interval == Interval.HOUR:
            # 60分钟线
            return self.minute_60m_path
        else:
            return self.minute_path

    def save_bar_data(self, bars: list[BarData], interval_value: str = None) -> None:
        """Save bar data
        
        Args:
            bars: K线数据列表
            interval_value: 可选的周期标识（如"10m", "30m"），用于区分不同的分钟级别
        """
        if not bars:
            return

        # Get file path
        bar: BarData = bars[0]

        if bar.interval == Interval.DAILY:
            file_path: Path = self.daily_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.MINUTE:
            # 根据interval_value选择正确的文件夹
            folder_path = self.get_minute_folder_path(bar.interval, interval_value)
            file_path = folder_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval == Interval.HOUR:
            # 60分钟线保存到60m文件夹
            file_path = self.minute_60m_path.joinpath(f"{bar.vt_symbol}.parquet")
        elif bar.interval:
            logger.error(f"Unsupported interval {bar.interval.value}")
            return

        data: list = []
        for bar in bars:
            bar_data: dict = {
                "datetime": bar.datetime.replace(tzinfo=None),
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
                "turnover": bar.turnover,
                "open_interest": bar.open_interest
            }
            data.append(bar_data)

        new_df: pl.DataFrame = pl.DataFrame(data)

        # If file exists, read and merge
        if file_path.exists():
            old_df: pl.DataFrame = pl.read_parquet(file_path)

            new_df = pl.concat([old_df, new_df])

            new_df = new_df.unique(subset=["datetime"])

            new_df = new_df.sort("datetime")

        # Save to file
        new_df.write_parquet(file_path)

    def load_bar_data(
        self,
        vt_symbol: str,
        interval: Interval | str,
        start: datetime | str,
        end: datetime | str,
        interval_value: str = None
    ) -> list[BarData]:
        """Load bar data
        
        Args:
            vt_symbol: 合约代码
            interval: K线周期
            start: 开始时间
            end: 结束时间
            interval_value: 可选的周期标识（如"10m", "30m"）
        """
        # Convert types
        if isinstance(interval, str):
            interval = Interval(interval)

        start = to_datetime(start)
        end = to_datetime(end)

        # Get folder path
        if interval == Interval.DAILY:
            folder_path: Path = self.daily_path
        elif interval == Interval.MINUTE:
            folder_path = self.get_minute_folder_path(interval, interval_value)
        elif interval == Interval.HOUR:
            # 60分钟线从60m文件夹读取
            folder_path = self.minute_60m_path
        else:
            logger.error(f"Unsupported interval {interval.value}")
            return []

        # Check if file exists
        file_path: Path = folder_path.joinpath(f"{vt_symbol}.parquet")
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            return []

        # Open file
        df: pl.DataFrame = pl.read_parquet(file_path)

        # Filter by date range
        df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

        # Convert to BarData objects
        bars: list[BarData] = []

        symbol, exchange = extract_vt_symbol(vt_symbol)

        for row in df.iter_rows(named=True):
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"],
                interval=interval,
                open_price=row["open"],
                high_price=row["high"],
                low_price=row["low"],
                close_price=row["close"],
                volume=row["volume"],
                turnover=row["turnover"],
                open_interest=row["open_interest"],
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def load_bar_df(
        self,
        vt_symbols: list[str],
        interval: Interval | str,
        start: datetime | str,
        end: datetime | str,
        extended_days: int,
        interval_value: str = None
    ) -> pl.DataFrame | None:
        """Load bar data as DataFrame
        
        Args:
            vt_symbols: 股票代码列表
            interval: K线周期
            start: 开始时间
            end: 结束时间
            extended_days: 扩展天数
            interval_value: 可选的周期标识（如"10m", "30m"），用于区分不同的分钟级别
        """
        if not vt_symbols:
            return None

        # Convert types
        if isinstance(interval, str):
            interval = Interval(interval)

        start = to_datetime(start) - timedelta(days=extended_days)
        end = to_datetime(end) + timedelta(days=extended_days // 10)

        # Get folder path - 支持多种分钟周期
        if interval == Interval.DAILY:
            folder_path: Path = self.daily_path
        elif interval == Interval.MINUTE:
            # 根据 interval_value 选择正确的子文件夹
            if interval_value == "10m":
                folder_path = self.minute_10m_path
            elif interval_value == "30m":
                folder_path = self.minute_30m_path
            elif interval_value == "1m":
                folder_path = self.minute_1m_path
            else:
                # 默认使用 1m 文件夹
                folder_path = self.minute_1m_path
        elif interval == Interval.HOUR:
            # 60分钟线从 60m 文件夹读取
            folder_path = self.minute_60m_path
        else:
            logger.error(f"Unsupported interval {interval.value}")
            return None

        # Read data for each symbol
        dfs: list = []

        for vt_symbol in vt_symbols:
            # Check if file exists
            file_path: Path = folder_path.joinpath(f"{vt_symbol}.parquet")
            if not file_path.exists():
                logger.error(f"File {file_path} does not exist")
                continue

            # Open file
            df: pl.DataFrame = pl.read_parquet(file_path)

            # Filter by date range
            df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))

            # Specify data types
            df = df.with_columns(
                pl.col("open").cast(pl.Float32),
                pl.col("high").cast(pl.Float32),
                pl.col("low").cast(pl.Float32),
                pl.col("close").cast(pl.Float32),
                pl.col("volume").cast(pl.Float32),
                pl.col("turnover").cast(pl.Float32),
                pl.col("open_interest").cast(pl.Float32),
                (pl.col("turnover") / pl.col("volume")).cast(pl.Float32).alias("vwap")
            )

            # Check for empty data
            if df.is_empty():
                continue

            # Normalize prices
            close_0: float = df.select(pl.col("close")).item(0, 0)

            df = df.with_columns(
                (pl.col("open") / close_0).alias("open"),
                (pl.col("high") / close_0).alias("high"),
                (pl.col("low") / close_0).alias("low"),
                (pl.col("close") / close_0).alias("close"),
            )

            # Convert zeros to NaN for suspended trading days
            numeric_columns: list = df.columns[1:]                              # Extract numeric columns

            mask: pl.Series = df[numeric_columns].sum_horizontal() == 0         # Sum by row, if 0 then suspended

            df = df.with_columns(                                               # Convert suspended day values to NaN
                [pl.when(mask).then(float("nan")).otherwise(pl.col(col)).alias(col) for col in numeric_columns]
            )

            # Add symbol column
            df = df.with_columns(pl.lit(vt_symbol).alias("vt_symbol"))

            # Cache in list
            dfs.append(df)

        # Concatenate results
        result_df: pl.DataFrame = pl.concat(dfs)
        return result_df

    def save_component_data(
        self,
        index_symbol: str,
        index_components: dict[str, list[str]]
    ) -> None:
        """Save index component data"""
        file_path: Path = self.component_path.joinpath(f"{index_symbol}")

        with shelve.open(str(file_path)) as db:
            db.update(index_components)

    @lru_cache      # noqa
    def load_component_data(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> dict[datetime, list[str]]:
        """Load index component data as DataFrame"""
        file_path: Path = self.component_path.joinpath(f"{index_symbol}")

        start = to_datetime(start)
        end = to_datetime(end)

        with shelve.open(str(file_path)) as db:
            keys: list[str] = list(db.keys())
            keys.sort()

            index_components: dict[datetime, list[str]] = {}
            for key in keys:
                dt: datetime = datetime.strptime(key, "%Y-%m-%d")
                if start <= dt <= end:
                    index_components[dt] = db[key]

            return index_components

    def load_component_symbols(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> list[str]:
        """Collect index component symbols"""
        index_components: dict[datetime, list[str]] = self.load_component_data(
            index_symbol,
            start,
            end
        )

        component_symbols: set[str] = set()

        for vt_symbols in index_components.values():
            component_symbols.update(vt_symbols)

        return list(component_symbols)

    def load_component_filters(
        self,
        index_symbol: str,
        start: datetime | str,
        end: datetime | str
    ) -> dict[str, list[tuple[datetime, datetime]]]:
        """Collect index component duration filters"""
        index_components: dict[datetime, list[str]] = self.load_component_data(
            index_symbol,
            start,
            end
        )

        # Get all trading dates and sort
        trading_dates: list[datetime] = sorted(index_components.keys())

        # Initialize component duration dictionary
        component_filters: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)

        # Get all component symbols
        all_symbols: set[str] = set()
        for vt_symbols in index_components.values():
            all_symbols.update(vt_symbols)

        # Iterate through each component to identify its duration in the index
        for vt_symbol in all_symbols:
            period_start: datetime | None = None
            period_end: datetime | None = None

            # Iterate through each trading day to identify continuous holding periods
            for trading_date in trading_dates:
                if vt_symbol in index_components[trading_date]:
                    if period_start is None:
                        period_start = trading_date

                    period_end = trading_date
                else:
                    if period_start and period_end:
                        component_filters[vt_symbol].append((period_start, period_end))
                        period_start = None
                        period_end = None

            # Handle the last holding period
            if period_start and period_end:
                component_filters[vt_symbol].append((period_start, period_end))

        return component_filters

    def add_contract_setting(
        self,
        vt_symbol: str,
        long_rate: float,
        short_rate: float,
        size: float,
        pricetick: float
    ) -> None:
        """Add contract information"""
        contracts: dict = {}

        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        contracts[vt_symbol] = {
            "long_rate": long_rate,
            "short_rate": short_rate,
            "size": size,
            "pricetick": pricetick
        }

        with open(self.contract_path, mode="w+", encoding="UTF-8") as f:
            json.dump(
                contracts,
                f,
                indent=4,
                ensure_ascii=False
            )

    def load_contract_setttings(self) -> dict:
        """Load contract settings"""
        contracts: dict = {}

        if self.contract_path.exists():
            with open(self.contract_path, encoding="UTF-8") as f:
                contracts = json.load(f)

        return contracts

    def save_dataset(
        self,
        name: str,
        dataset: AlphaDataset,
        interval: str | None = None,
        format: str = "parquet",
        params: dict | None = None,
    ) -> None:
        """Save dataset

        - 默认使用 Parquet+JSON（目录 `{name}[_{interval}]`），含可选 `params.json`。
        - 兼容 pkl：传入 `format='pkl'` 时写入 `{name}[_{interval}].pkl`。
        """
        # 如果传入了运行参数，则挂载到对象，便于后续 pickling
        if params is not None:
            try:
                setattr(dataset, "params", params)
            except Exception:
                pass

        if format == "pkl":
            # 兼容旧格式：单文件 pkl
            if interval:
                file_path: Path = self.dataset_path.joinpath(f"{name}_{interval}.pkl")
            else:
                file_path = self.dataset_path.joinpath(f"{name}.pkl")

            with open(file_path, mode="wb") as f:
                pickle.dump(dataset, f)
            return

        # 新格式：Parquet 目录 + params.json
        dir_name = f"{name}_{interval}" if interval else name
        base_dir: Path = self.dataset_path.joinpath(dir_name)
        base_dir.mkdir(parents=True, exist_ok=True)

        # 写入各视图（存在即写）
        frames = {
            "raw": getattr(dataset, "raw_df", None),
            "infer": getattr(dataset, "infer_df", None),
            "learn": getattr(dataset, "learn_df", None),
            "result": getattr(dataset, "result_df", None),
        }
        for frame_name, df in frames.items():
            if df is None:
                continue
            if isinstance(df, pl.DataFrame):
                # 统一 schema：datetime 无时区，字符串列标准化，排序稳定
                cols = df.columns
                if "datetime" in cols:
                    df = df.with_columns(pl.col("datetime").cast(pl.Datetime))
                if "vt_symbol" in cols:
                    df = df.with_columns(pl.col("vt_symbol").cast(pl.Utf8))
                try:
                    df = df.sort(["datetime", "vt_symbol"])  # type: ignore[arg-type]
                except Exception:
                    pass
                file_path = base_dir.joinpath(f"{frame_name}.parquet")
                # 压缩写入（zstd），不存在则创建，存在则覆盖
                df.write_parquet(file_path, compression="zstd")

        # 写入 params.json（始终写入：优先用户传入，其次对象自带；否则从数据集推导）
        derived: dict = {
            "interval": getattr(dataset, "interval", "1d"),
        }
        try:
            dp = getattr(dataset, "data_periods", {})
            if dp:
                derived.update({
                    "train_period": list(dp.get(Segment.TRAIN, ("", ""))),
                    "valid_period": list(dp.get(Segment.VALID, ("", ""))),
                    "test_period": list(dp.get(Segment.TEST, ("", ""))),
                })
        except Exception:
            pass

        params_obj = params if params is not None else getattr(dataset, "params", None)
        final_params = dict(derived)
        if isinstance(params_obj, dict):
            # 用户/对象参数覆盖推导值
            final_params.update(params_obj)

        # 回写到对象，便于后续 pickling
        try:
            setattr(dataset, "params", final_params)
        except Exception:
            pass

        params_path = base_dir.joinpath("params.json")
        with open(params_path, mode="w", encoding="utf-8") as f:
            json.dump(final_params, f, ensure_ascii=False, indent=2)

    def load_dataset(
        self,
        name: str,
        interval: str | None = None,
        format: str | None = None,
    ) -> AlphaDataset | None:
        """Load dataset with optional interval tag

        - 默认优先 Parquet 目录（`{name}[_{interval}]`），不存在再回退 pkl。
        - 允许显式指定 `format`：`parquet` 或 `pkl`。
        """
        # 1) 尝试 Parquet 目录
        if format in (None, "parquet"):
            candidates: list[Path] = []
            if interval:
                candidates.append(self.dataset_path.joinpath(f"{name}_{interval}"))
            candidates.append(self.dataset_path.joinpath(name))

            for base_dir in candidates:
                if base_dir.exists() and base_dir.is_dir():
                    # 读取各视图
                    def _read(path: Path) -> pl.DataFrame | None:
                        return pl.read_parquet(path) if path.exists() else None

                    def _sort(df: pl.DataFrame | None) -> pl.DataFrame | None:
                        if df is None:
                            return None
                        # 标准化 schema 并排序，保证与内存流程一致
                        try:
                            out = df
                            if "datetime" in out.columns:
                                out = out.with_columns(pl.col("datetime").cast(pl.Datetime))
                            if "vt_symbol" in out.columns:
                                out = out.with_columns(pl.col("vt_symbol").cast(pl.Utf8))
                            return out.sort(["datetime", "vt_symbol"])  # type: ignore[arg-type]
                        except Exception:
                            return df

                    raw_df = _sort(_read(base_dir.joinpath("raw.parquet")))
                    infer_df = _sort(_read(base_dir.joinpath("infer.parquet")))
                    learn_df = _sort(_read(base_dir.joinpath("learn.parquet")))
                    result_df = _sort(_read(base_dir.joinpath("result.parquet")))

                    # 读取运行参数（如存在）
                    params_path = base_dir.joinpath("params.json")
                    params_obj: dict | None = None
                    if params_path.exists():
                        try:
                            with open(params_path, encoding="utf-8") as f:
                                params_obj = json.load(f)
                        except Exception:
                            params_obj = None

                    # 选择一个可用的基础 DataFrame（用于构造 AlphaDataset）
                    base_df = None
                    for _df in (result_df, raw_df, infer_df, learn_df):
                        if _df is not None:
                            base_df = _df
                            break
                    if base_df is None:
                        logger.error(f"Dataset directory {base_dir} is empty")
                        return None

                    # 从 params 中恢复 period/interval，缺失则用数据范围兜底
                    def _to_periods(df: pl.DataFrame) -> tuple[str, str]:
                        dt_min = df["datetime"].min()
                        dt_max = df["datetime"].max()
                        try:
                            start = dt_min.strftime("%Y-%m-%d")  # type: ignore[attr-defined]
                            end = dt_max.strftime("%Y-%m-%d")    # type: ignore[attr-defined]
                        except Exception:
                            # 若列为字符串，直接返回
                            start = str(dt_min)
                            end = str(dt_max)
                        return start, end

                    if params_obj and all(k in params_obj for k in ("train_period", "valid_period", "test_period")):
                        train_period = tuple(params_obj["train_period"])  # type: ignore[assignment]
                        valid_period = tuple(params_obj["valid_period"])  # type: ignore[assignment]
                        test_period = tuple(params_obj["test_period"])    # type: ignore[assignment]
                        interval_str = params_obj.get("interval", "1d")
                    else:
                        if params_path.exists():
                            logger.warning(f"Invalid or empty params.json at {params_path}, falling back to full-range periods")
                        p = _to_periods(base_df)
                        train_period = valid_period = test_period = p
                        interval_str = "1d"

                    dataset = AlphaDataset(
                        df=base_df,
                        train_period=train_period, valid_period=valid_period, test_period=test_period,
                        interval=interval_str,
                    )

                    # 回填各视图（存在才设置）
                    if raw_df is not None:
                        dataset.raw_df = raw_df
                    if infer_df is not None:
                        dataset.infer_df = infer_df
                    if learn_df is not None:
                        dataset.learn_df = learn_df
                    if result_df is not None:
                        dataset.result_df = result_df

                    # 回填运行参数
                    if params_obj is not None:
                        try:
                            setattr(dataset, "params", params_obj)
                        except Exception:
                            pass

                    return dataset

            if format == "parquet":
                # 显式要求 parquet 但未找到
                logger.error(f"Dataset directory {name} not found (interval={interval})")
                return None

        # 2) 回退到 pkl
        if format in (None, "pkl"):
            if interval:
                file_path: Path = self.dataset_path.joinpath(f"{name}_{interval}.pkl")
            else:
                file_path = self.dataset_path.joinpath(f"{name}.pkl")

            if not file_path.exists() and interval:
                file_path = self.dataset_path.joinpath(f"{name}.pkl")

            if not file_path.exists():
                logger.error(f"Dataset file {name} does not exist")
                return None

            with open(file_path, mode="rb") as f:
                dataset: AlphaDataset = pickle.load(f)
                return dataset

        # 3) 都未找到
        logger.error(f"Dataset {name} not found")
        return None

    def remove_dataset(self, name: str) -> bool:
        """Remove dataset (supports Parquet dir or pkl file)

        - 若存在同名目录：递归删除目录。
        - 否则尝试删除同名 pkl 文件。
        """
        dir_path: Path = self.dataset_path.joinpath(name)
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")

        removed = False
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
            removed = True
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            removed = True

        if not removed:
            logger.error(f"Dataset {name} does not exist")
        return removed

    def list_all_datasets(self) -> list[str]:
        """List all datasets (parquet dirs and pkl files)"""
        names: set[str] = set()
        # pkl files
        names.update(file.stem for file in self.dataset_path.glob("*.pkl"))
        # parquet directories
        for p in self.dataset_path.iterdir():
            if p.is_dir():
                names.add(p.name)
        return sorted(names)

    def save_model(self, name: str, model: AlphaModel, interval: str | None = None) -> None:
        """Save model with optional interval tag"""
        # 构建文件名
        if interval:
            file_path: Path = self.model_path.joinpath(f"{name}_{interval}.pkl")
        else:
            file_path: Path = self.model_path.joinpath(f"{name}.pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(model, f)

    def load_model(self, name: str, interval: str | None = None) -> AlphaModel | None:
        """Load model with optional interval tag"""
        # 构建文件名
        if interval:
            file_path: Path = self.model_path.joinpath(f"{name}_{interval}.pkl")
        else:
            file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        
        # 如果带 interval 的文件不存在，尝试不带 interval 的文件（向后兼容）
        if not file_path.exists() and interval:
            file_path = self.model_path.joinpath(f"{name}.pkl")
        
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            model: AlphaModel = pickle.load(f)
            return model

    def remove_model(self, name: str) -> bool:
        """Remove model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            logger.error(f"Model file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_models(self) -> list[str]:
        """List all models"""
        return [file.stem for file in self.model_path.glob("*.pkl")]

    def save_signal(self, name: str, signal: pl.DataFrame, interval: str | None = None) -> None:
        """Save signal with optional interval tag"""
        # 构建文件名
        if interval:
            file_path: Path = self.signal_path.joinpath(f"{name}_{interval}.parquet")
        else:
            file_path: Path = self.signal_path.joinpath(f"{name}.parquet")

        signal.write_parquet(file_path)

    def load_signal(self, name: str, interval: str | None = None) -> pl.DataFrame | None:
        """Load signal with optional interval tag"""
        # 构建文件名
        if interval:
            file_path: Path = self.signal_path.joinpath(f"{name}_{interval}.parquet")
        else:
            file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        
        # 如果带 interval 的文件不存在，尝试不带 interval 的文件（向后兼容）
        if not file_path.exists() and interval:
            file_path = self.signal_path.joinpath(f"{name}.parquet")
        
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return None

        return pl.read_parquet(file_path)

    def remove_signal(self, name: str) -> bool:
        """Remove signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        if not file_path.exists():
            logger.error(f"Signal file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_signals(self) -> list[str]:
        """List all signals"""
        return [file.stem for file in self.model_path.glob("*.parquet")]
