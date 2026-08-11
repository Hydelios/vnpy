from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import rqdatac

uri = "tcp://license:RBw9XfgLg0IQjE2DpZY-JT44Y875QkRhIrs4TEE271Rm6Nh5K4S2Z-dzrOlf4gny0T4-DFtWmO9f3HM5Sja_4_TgfQ2ECfhlG3soYBFSPRYvUlq71Fki4kf8j51UDkqszSKHGSCdsetY0tw_ofy-TAxPIUN31P-yjGFGKmbgwJs=ObE9AI6CI-Iikt8fzg6uk_0yfHp5l5dnpOVXtzFXFZ75dpaYG_W8tKvzMM76-5E9sqNAsVfqyyQNqvWvlSAgQZhWk6wtYNRF5txbvF88uQ16-PTvW3QwJ_977HauCREGU2POhYhCJUtY9jxnyE5JoSs82B7LvCmR_xWlw0vQ9Eg=@rqdatad-pro.ricequant.com:16011"
rqdatac.init(uri)

RUN_POSITION_ANALYSES = False
RUN_BARRA_EXPOSURE = False

DEFAULT_COMPONENT_DIR = Path(
    "C:/Users/GTZQ/Documents/PythonProjs/框架修改文件/数据下载/lab/all_stocks/component"
)
DEFAULT_BARRA_DIR = Path(
    "C:/Users/GTZQ/Documents/PythonProjs/框架修改文件/数据下载/barra数据下载/lab/all_stocks/barra_exposure_v2/by_factor"
)
DEFAULT_INDUSTRY_PATH = Path(
    "C:/Users/GTZQ/Documents/PythonProjs/框架修改文件/数据下载/lab/all_stocks/common/原始数据包/industry.csv"
)

INDEX_FILES = {
    "沪深300": "indexweight_000300.SSE.parquet",
    "中证500": "indexweight_000905.SSE.parquet",
    "中证1000": "indexweight_000852.SSE.parquet",
}

MARKET_CAP_BUCKETS = [
    (50, "50亿以下"),
    (100, "50-100亿"),
    (300, "100-300亿"),
    (500, "300-500亿"),
    (1000, "500-1000亿"),
]

MARKET_CAP_BUCKET_ORDER = [
    "50亿以下",
    "50-100亿",
    "100-300亿",
    "300-500亿",
    "500-1000亿",
    "1000亿以上",
]

INDEX_CATEGORY_ORDER = ["沪深300", "中证500", "中证1000", "其他"]

DEFAULT_BARRA_FACTORS = [
    "size",
    "liquidity",
    "leverage",
    "earnings_variability",
    "earnings_quality",
    "profitability",
    "investment_quality",
    "book_to_price",
    "earnings_yield",
    "longterm_reversal",
    "growth",
    "momentum",
    "mid_cap",
    "beta",
    "residual_volatility",
    "dividend_yield",
]

DEFAULT_ANALYSES = [
    {
        "name": "industry_daily",
        "category_col": "所属行业",
        "output_name": "行业持仓分析.xlsx",
    },
    {
        "name": "index_distribution",
        "category_col": "指数分类",
        "category_order": INDEX_CATEGORY_ORDER,
        "output_name": "指数分类持仓分布.xlsx",
    },
    {
        "name": "market_cap_distribution",
        "category_col": "流通市值区间",
        "category_order": MARKET_CAP_BUCKET_ORDER,
        "drop_null_cols": ["流通市值（亿元）"],
        "output_name": "流通市值持仓分布.xlsx",
    },
]


class PositionAnalysisError(ValueError):
    pass


def init_rqdatac(uri: str | None = None, **kwargs: Any) -> None:
    if uri:
        rqdatac.init(uri, **kwargs)
    else:
        rqdatac.init(**kwargs)


def wind_to_rq_code_expr(column: str = "万得代码") -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.replace("\\.SH$", ".XSHG")
        .str.replace("\\.SZ$", ".XSHE")
    )


def component_code_to_wind_expr(column: str = "vt_symbol") -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.replace("\\.SSE$", ".SH")
        .str.replace("\\.SZSE$", ".SZ")
    )


def normalize_trade_date(df: pl.DataFrame, date_col: str = "持仓日期") -> pl.DataFrame:
    if "交易日" in df.columns:
        source_col = "交易日"
    elif date_col in df.columns:
        source_col = date_col
    else:
        raise PositionAnalysisError(f"持仓数据缺少日期列：交易日 或 {date_col}")

    date_text = pl.col(source_col).cast(pl.Utf8)
    parsed_date = (
        pl.when(pl.col("_交易日文本").str.contains(r"^\d{8}$"))
        .then(pl.col("_交易日文本").str.strptime(pl.Date, "%Y%m%d", strict=False))
        .otherwise(
            pl.col("_交易日文本")
            .str.replace(r" 00:00:00$", "")
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
    )

    return (
        df.with_columns(date_text.alias("_交易日文本"))
        .with_columns(parsed_date.alias("交易日"))
        .drop("_交易日文本")
    )


def load_position(position_path: str | Path) -> pl.DataFrame:
    path = Path(position_path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pl.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise PositionAnalysisError(f"不支持的持仓文件类型：{path.suffix}")


def load_index_components(component_dir: str | Path = DEFAULT_COMPONENT_DIR) -> pl.DataFrame:
    component_dir = Path(component_dir)
    frames = []

    for priority, (index_name, file_name) in enumerate(INDEX_FILES.items()):
        path = component_dir / file_name
        frames.append(
            pl.scan_parquet(path)
            .select(["date", "vt_symbol"])
            .with_columns(
                pl.lit(index_name).alias("指数分类"),
                pl.lit(priority).alias("_priority"),
            )
        )

    return (
        pl.concat(frames)
        .with_columns(
            pl.col("date").str.to_date().alias("交易日"),
            component_code_to_wind_expr("vt_symbol").alias("万得代码"),
        )
        .select(["交易日", "万得代码", "指数分类", "_priority"])
        .collect()
        .sort(["交易日", "万得代码", "_priority"])
        .unique(subset=["交易日", "万得代码"], keep="first", maintain_order=True)
        .drop("_priority")
    )


def filter_trading_days(position: pl.DataFrame) -> pl.DataFrame:
    if "交易日" not in position.columns:
        raise PositionAnalysisError("过滤交易日前需要先生成 交易日")

    start_date = position["交易日"].min()
    end_date = position["交易日"].max()
    trading_dates = rqdatac.get_trading_dates(
        start_date=start_date,
        end_date=end_date,
        market="cn",
    )
    trading_dates = [dt.date() if hasattr(dt, "date") else dt for dt in trading_dates]

    return position.filter(pl.col("交易日").is_in(trading_dates))


def fetch_share_data(position: pl.DataFrame) -> pl.DataFrame:
    if "交易日" not in position.columns or "米筐代码" not in position.columns:
        raise PositionAnalysisError("获取股本数据前需要先生成 交易日 和 米筐代码")

    start_date = position["交易日"].min()
    end_date = position["交易日"].max()
    order_book_ids = position["米筐代码"].drop_nulls().unique().to_list()

    shares = rqdatac.get_shares(
        order_book_ids,
        start_date=start_date,
        end_date=end_date,
        fields=["circulation_a", "total"],
        expect_df=True,
    )

    if shares is None or shares.empty:
        return pl.DataFrame(
            schema={
                "交易日": pl.Date,
                "米筐代码": pl.Utf8,
                "流通股本": pl.Float64,
                "总股本": pl.Float64,
            }
        )

    return (
        pl.from_pandas(shares.reset_index())
        .rename({"date": "交易日", "order_book_id": "米筐代码", "circulation_a": "流通股本", "total": "总股本"})
        .with_columns(pl.col("交易日").cast(pl.Date))
        .select(["交易日", "米筐代码", "流通股本", "总股本"])
    )


def add_market_cap_columns(position: pl.DataFrame, price_col: str = "当日价格") -> pl.DataFrame:
    missing = [col for col in ["流通股本", "总股本", price_col] if col not in position.columns]
    if missing:
        raise PositionAnalysisError(f"计算市值缺少列：{missing}")

    return position.with_columns(
        (pl.col("流通股本").cast(pl.Float64, strict=False) * pl.col(price_col).cast(pl.Float64, strict=False)).alias("流通市值"),
        (pl.col("总股本").cast(pl.Float64, strict=False) * pl.col(price_col).cast(pl.Float64, strict=False)).alias("总市值"),
    ).with_columns(
        (pl.col("流通市值") / 100000000).alias("流通市值（亿元）"),
        (pl.col("总市值") / 100000000).alias("总市值（亿元）"),
    )


def choose_market_cap_column(position: pl.DataFrame, preferred_col: str | None = None) -> str:
    candidates = [preferred_col, "流通市值（亿元）", "流通市值"]
    for col in candidates:
        if col and col in position.columns:
            return col
    raise PositionAnalysisError("持仓数据缺少流通市值列，请先计算市值或传入 market_cap_col")


def add_market_cap_bucket(
    position: pl.DataFrame,
    market_cap_col: str | None = None,
    bucket_col: str = "流通市值区间",
) -> pl.DataFrame:
    market_cap_col = choose_market_cap_column(position, market_cap_col)
    position = position.with_columns(pl.col(market_cap_col).cast(pl.Float64, strict=False))

    expr = pl.when(pl.col(market_cap_col).is_null()).then(pl.lit(None))
    for upper_bound, label in MARKET_CAP_BUCKETS:
        expr = expr.when(pl.col(market_cap_col) < upper_bound).then(pl.lit(label))

    return position.with_columns(expr.otherwise(pl.lit("1000亿以上")).alias(bucket_col))


def prepare_position(
    position_path: str | Path,
    component_dir: str | Path = DEFAULT_COMPONENT_DIR,
    market_cap_col: str | None = None,
    fetch_shares: bool = True,
    price_col: str = "当日价格",
) -> pl.DataFrame:
    position = load_position(position_path)

    if "万得代码" not in position.columns:
        raise PositionAnalysisError("持仓数据缺少 万得代码 列")

    position = normalize_trade_date(position).with_columns(
        wind_to_rq_code_expr("万得代码").alias("米筐代码")
    )
    position = filter_trading_days(position)

    if fetch_shares:
        share_data = fetch_share_data(position)
        position = position.join(share_data, on=["交易日", "米筐代码"], how="left")

    index_components = load_index_components(component_dir)
    position = (
        position.join(index_components, on=["交易日", "万得代码"], how="left")
        .with_columns(pl.col("指数分类").fill_null("其他"))
    )

    position = add_market_cap_columns(position, price_col=price_col)

    return add_market_cap_bucket(
        position,
        market_cap_col=market_cap_col,
    )


def filter_date_range(
    df: pl.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    result = df.with_columns(pl.col("交易日").cast(pl.Date))
    if start_date:
        result = result.filter(pl.col("交易日") >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d"))
    if end_date:
        result = result.filter(pl.col("交易日") <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d"))
    return result


def build_distribution_table(
    position: pl.DataFrame,
    category_col: str,
    value_col: str = "持仓市值（人民币）",
    start_date: str | None = None,
    end_date: str | None = None,
    category_order: list[str] | None = None,
    drop_null_cols: list[str] | None = None,
) -> pl.DataFrame:
    missing = [col for col in ["交易日", category_col, value_col] if col not in position.columns]
    if missing:
        raise PositionAnalysisError(f"生成分布表缺少列：{missing}")

    source = filter_date_range(position, start_date=start_date, end_date=end_date)
    if drop_null_cols:
        source = source.drop_nulls([col for col in drop_null_cols if col in source.columns])

    result = (
        source
        .group_by(["交易日", category_col])
        .agg(pl.col(value_col).sum().alias("分类持仓市值"))
        .with_columns(
            (pl.col("分类持仓市值") / pl.col("分类持仓市值").sum().over("交易日")).alias("持仓市值占比")
        )
        .select(["交易日", category_col, "持仓市值占比"])
        .pivot(values="持仓市值占比", index="交易日", on=category_col)
        .fill_null(0)
        .sort("交易日")
    )

    if category_order:
        ordered_cols = [col for col in category_order if col in result.columns]
        other_cols = [col for col in result.columns if col not in {"交易日", *ordered_cols}]
        result = result.select(["交易日", *ordered_cols, *other_cols])

    return result


def load_barra_factor(
    factor: str,
    barra_dir: str | Path = DEFAULT_BARRA_DIR,
) -> pl.DataFrame:
    path = Path(barra_dir) / f"{factor}.parquet"
    if not path.exists():
        raise PositionAnalysisError(f"Barra因子文件不存在：{path}")

    return (
        pl.scan_parquet(path)
        .select(["trade_date", "vt_symbol", "exposure"])
        .with_columns(
            pl.col("trade_date").str.to_date().alias("交易日"),
            component_code_to_wind_expr("vt_symbol").alias("万得代码"),
            pl.col("exposure").cast(pl.Float64, strict=False).alias(factor),
        )
        .select(["交易日", "万得代码", factor])
        .collect()
    )


def build_barra_exposure_table(
    position: pl.DataFrame,
    barra_dir: str | Path = DEFAULT_BARRA_DIR,
    factors: list[str] | None = None,
    value_col: str = "持仓市值（人民币）",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    missing = [col for col in ["交易日", "万得代码", value_col] if col not in position.columns]
    if missing:
        raise PositionAnalysisError(f"计算Barra暴露缺少列：{missing}")

    factors = factors or DEFAULT_BARRA_FACTORS
    source = filter_date_range(position, start_date=start_date, end_date=end_date)
    results = []

    for factor in factors:
        factor_df = load_barra_factor(factor, barra_dir=barra_dir)
        factor_position = (
            source.join(factor_df, on=["交易日", "万得代码"], how="left")
            .drop_nulls([factor, value_col])
        )
        if factor_position.is_empty():
            continue

        result = (
            factor_position
            .group_by("交易日")
            .agg(
                (
                    (pl.col(value_col).cast(pl.Float64, strict=False) * pl.col(factor)).sum()
                    / pl.col(value_col).cast(pl.Float64, strict=False).sum()
                ).alias(factor)
            )
            .select(["交易日", factor])
        )
        results.append(result)

    if not results:
        return pl.DataFrame({"交易日": source.select("交易日").unique().sort("交易日")["交易日"]})

    barra_exposure = results[0]
    for result in results[1:]:
        barra_exposure = barra_exposure.join(result, on="交易日", how="outer", coalesce=True)

    return barra_exposure.sort("交易日")


def build_index_barra_exposure_table(
    component_dir: str | Path = DEFAULT_COMPONENT_DIR,
    barra_dir: str | Path = DEFAULT_BARRA_DIR,
    factors: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    index_files: dict[str, str] | None = None,
) -> pl.DataFrame:
    factors = factors or DEFAULT_BARRA_FACTORS
    index_files = index_files or INDEX_FILES
    component_dir = Path(component_dir)
    results = []

    for index_name, file_name in index_files.items():
        index_weight = (
            pl.scan_parquet(component_dir / file_name)
            .select(["date", "vt_symbol", "weight"])
            .with_columns(
                pl.col("date").str.to_date().alias("交易日"),
                component_code_to_wind_expr("vt_symbol").alias("万得代码"),
                pl.col("weight").cast(pl.Float64, strict=False).alias("指数权重"),
            )
            .select(["交易日", "万得代码", "指数权重"])
            .collect()
        )
        index_weight = filter_date_range(index_weight, start_date=start_date, end_date=end_date)

        index_result = index_weight.select("交易日").unique().sort("交易日")
        for factor in factors:
            factor_df = load_barra_factor(factor, barra_dir=barra_dir)
            factor_position = (
                index_weight.join(factor_df, on=["交易日", "万得代码"], how="left")
                .drop_nulls(["指数权重", factor])
            )
            if factor_position.is_empty():
                index_result = index_result.with_columns(pl.lit(None).cast(pl.Float64).alias(factor))
                continue

            factor_result = (
                factor_position
                .group_by("交易日")
                .agg(
                    (
                        (pl.col("指数权重") * pl.col(factor)).sum()
                        / pl.col("指数权重").sum()
                    ).alias(factor)
                )
            )
            index_result = index_result.join(factor_result, on="交易日", how="left")

        for factor in factors:
            if factor not in index_result.columns:
                index_result = index_result.with_columns(pl.lit(None).cast(pl.Float64).alias(factor))

        index_result = index_result.with_columns(pl.lit(index_name).alias("指数名称"))
        results.append(index_result)

    if not results:
        return pl.DataFrame(schema={"交易日": pl.Date, "指数名称": pl.Utf8})

    return pl.concat(results, how="diagonal").select(["交易日", "指数名称", *factors]).sort(["交易日", "指数名称"])


def load_industry_data(industry_path: str | Path = DEFAULT_INDUSTRY_PATH) -> pl.DataFrame:
    path = Path(industry_path)
    if not path.exists():
        raise PositionAnalysisError(f"行业数据文件不存在：{path}")

    return (
        pl.scan_csv(path)
        .select(["trade_date", "vt_symbol", "first_industry_name"])
        .with_columns(
            pl.col("trade_date").str.to_date().alias("交易日"),
            component_code_to_wind_expr("vt_symbol").alias("万得代码"),
            pl.col("first_industry_name").alias("所属行业"),
        )
        .select(["交易日", "万得代码", "所属行业"])
        .collect()
    )


def build_index_industry_distribution_table(
    component_dir: str | Path = DEFAULT_COMPONENT_DIR,
    industry_path: str | Path = DEFAULT_INDUSTRY_PATH,
    start_date: str | None = None,
    end_date: str | None = None,
    index_files: dict[str, str] | None = None,
) -> pl.DataFrame:
    index_files = index_files or INDEX_FILES
    component_dir = Path(component_dir)
    industry = load_industry_data(industry_path)
    results = []

    for index_name, file_name in index_files.items():
        index_weight = (
            pl.scan_parquet(component_dir / file_name)
            .select(["date", "vt_symbol", "weight"])
            .with_columns(
                pl.col("date").str.to_date().alias("交易日"),
                component_code_to_wind_expr("vt_symbol").alias("万得代码"),
                pl.col("weight").cast(pl.Float64, strict=False).alias("指数权重"),
            )
            .select(["交易日", "万得代码", "指数权重"])
            .collect()
        )
        index_weight = filter_date_range(index_weight, start_date=start_date, end_date=end_date)

        index_industry = (
            index_weight.join(industry, on=["交易日", "万得代码"], how="left")
            .drop_nulls(["所属行业", "指数权重"])
            .group_by(["交易日", "所属行业"])
            .agg(pl.col("指数权重").sum().alias("行业权重"))
            .with_columns(
                (pl.col("行业权重") / pl.col("行业权重").sum().over("交易日")).alias("行业占比")
            )
            .select(["交易日", "所属行业", "行业占比"])
            .pivot(values="行业占比", index="交易日", on="所属行业")
            .fill_null(0)
            .sort("交易日")
            .with_columns(pl.lit(index_name).alias("指数名称"))
        )
        results.append(index_industry)

    if not results:
        return pl.DataFrame(schema={"交易日": pl.Date, "指数名称": pl.Utf8})

    cols = pl.concat(results, how="diagonal").columns
    ordered_cols = ["交易日", "指数名称", *[col for col in cols if col not in {"交易日", "指数名称"}]]
    return pl.concat(results, how="diagonal").select(ordered_cols).sort(["交易日", "指数名称"])


def build_analysis_table(
    position: pl.DataFrame,
    analysis: dict[str, Any],
    value_col: str = "持仓市值（人民币）",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    return build_distribution_table(
        position,
        category_col=analysis["category_col"],
        value_col=analysis.get("value_col", value_col),
        start_date=start_date,
        end_date=end_date,
        category_order=analysis.get("category_order"),
        drop_null_cols=analysis.get("drop_null_cols"),
    )


def build_analysis_tables(
    position: pl.DataFrame,
    value_col: str = "持仓市值（人民币）",
    start_date: str | None = None,
    end_date: str | None = None,
    analyses: list[dict[str, Any]] | None = None,
) -> dict[str, pl.DataFrame]:
    analyses = analyses or DEFAULT_ANALYSES
    return {
        analysis["name"]: build_analysis_table(
            position,
            analysis,
            value_col=value_col,
            start_date=start_date,
            end_date=end_date,
        )
        for analysis in analyses
    }


def write_analysis_outputs(
    tables: dict[str, pl.DataFrame],
    output_dir: str | Path,
    analyses: list[dict[str, Any]] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses = analyses or DEFAULT_ANALYSES
    output_names = {
        analysis["name"]: analysis["output_name"]
        for analysis in analyses
        if "output_name" in analysis
    }
    output_names["barra_exposure"] = "Barra因子暴露.xlsx"
    output_names["index_barra_exposure"] = "指数Barra因子暴露.xlsx"
    output_names["index_industry_distribution"] = "指数行业分布.xlsx"

    for key, file_name in output_names.items():
        if key in tables:
            tables[key].write_excel(output_dir / file_name)


def analyze_position_file(
    position_path: str | Path,
    output_dir: str | Path | None = None,
    component_dir: str | Path = DEFAULT_COMPONENT_DIR,
    barra_dir: str | Path = DEFAULT_BARRA_DIR,
    industry_path: str | Path = DEFAULT_INDUSTRY_PATH,
    market_cap_col: str | None = None,
    value_col: str = "持仓市值（人民币）",
    start_date: str | None = None,
    end_date: str | None = None,
    fetch_shares: bool = True,
    write_outputs: bool = True,
    price_col: str = "当日价格",
    analyses: list[dict[str, Any]] | None = None,
    barra_factors: list[str] | None = None,
    run_position_analyses: bool = RUN_POSITION_ANALYSES,
    run_barra_exposure: bool = RUN_BARRA_EXPOSURE,
    run_index_barra_exposure: bool = RUN_BARRA_EXPOSURE,
    run_index_industry_distribution: bool = RUN_POSITION_ANALYSES,
) -> dict[str, pl.DataFrame]:
    position_path = Path(position_path)
    output_dir = Path(output_dir) if output_dir else position_path.parent

    position = prepare_position(
        position_path=position_path,
        component_dir=component_dir,
        market_cap_col=market_cap_col,
        fetch_shares=fetch_shares,
        price_col=price_col,
    )

    tables = {"position": position}

    if run_position_analyses:
        tables.update(
            build_analysis_tables(
                position,
                value_col=value_col,
                start_date=start_date,
                end_date=end_date,
                analyses=analyses,
            )
        )

    if run_barra_exposure:
        tables["barra_exposure"] = build_barra_exposure_table(
            position,
            barra_dir=barra_dir,
            factors=barra_factors,
            value_col=value_col,
            start_date=start_date,
            end_date=end_date,
        )

    date_range = filter_date_range(position, start_date=start_date, end_date=end_date)
    index_start_date = date_range["交易日"].min()
    index_end_date = date_range["交易日"].max()
    index_start_text = index_start_date.isoformat() if index_start_date else None
    index_end_text = index_end_date.isoformat() if index_end_date else None

    if run_index_barra_exposure:
        tables["index_barra_exposure"] = build_index_barra_exposure_table(
            component_dir=component_dir,
            barra_dir=barra_dir,
            factors=barra_factors,
            start_date=index_start_text,
            end_date=index_end_text,
        )

    if run_index_industry_distribution:
        tables["index_industry_distribution"] = build_index_industry_distribution_table(
            component_dir=component_dir,
            industry_path=industry_path,
            start_date=index_start_text,
            end_date=index_end_text,
        )

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        position.write_excel(output_dir / "持仓分析明细.xlsx")
        write_analysis_outputs(tables, output_dir, analyses=analyses)

    return tables


if __name__ == "__main__":
    result = analyze_position_file(
        position_path="C:/Users/GTZQ/Documents/PythonProjs/持仓信息.xlsx",
        output_dir="C:/Users/GTZQ/Documents/PythonProjs/持仓分析test",
        start_date="2026-06-01",
        end_date="2026-06-30",
        run_position_analyses=False,
        run_barra_exposure=False,
        run_index_barra_exposure=False,
        run_index_industry_distribution=True

    )
    print(result["position"])
    print(result["industry_daily"])
    print(result["index_distribution"])
    print(result["market_cap_distribution"])

