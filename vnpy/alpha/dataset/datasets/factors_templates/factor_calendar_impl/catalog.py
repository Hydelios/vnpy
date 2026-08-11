from __future__ import annotations

import csv
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .implementations import find_spec
from .metadata import CalendarFactorDef, ImplementationStatus
from .parser import ParsedFactor, formula_from_body, parse_calendar


YEARS = (2022, 2023, 2024, 2025, 2026)
EXPECTED_COUNTS = {2022: 365, 2023: 357, 2024: 366, 2025: 365, 2026: 362}

CATEGORY_MAP = {
    "技术": "量价技术类",
    "高频": "高频因子",
    "财务": "财务质量因子",
    "估值": "估值因子",
    "流动性": "流动性因子",
    "动量": "动量因子",
    "波动": "波动率因子",
    "投资": "投资因子",
    "规模": "规模因子",
    "成长": "成长因子",
    "分析师": "分析师预期类",
    "股东": "股东因子",
    "另类": "另类因子",
    "易类": "另类因子",
    "无形资产": "无形资产因子",
    "杠杆": "杠杆因子",
    "倍值": "估值因子",
    "武侯": "财务质量因子",
    "流动出": "高频因子",
}

DATA_BLOCKERS = {
    "sentiment_external": ("Sentiment Beta", "情绪贝塔"),
    "analyst": ("分析师", "一致预期", "目标价", "预期EPS", "预期PE"),
    "news": ("新闻", "舆情", "点击量", "自选股", "搜索"),
    "attention_events": ("龙虎榜",),
    "patent": ("专利", "权利要求", "审查期", "科技动量"),
    "supply_chain": ("客户动量", "供应链", "供应商", "上下游"),
    "shareholder": ("股东户数", "机构持股", "持仓机构", "股权集中度", "十大股东"),
    "northbound": ("北向", "北上"),
    "orderbook": (
        "委买", "委卖", "委托", "盘口", "买一", "卖一", "中间价", "买盘深度", "卖盘深度",
        "订单失衡", "订单斜率", "有效价差", "实现价差", "买卖价差", "报价价差",
    ),
    "tick_trade": (
        "单笔", "笔均", "逐笔", "主买", "主卖", "大单", "小单", "撤单", "成交笔数",
        "主动买卖", "主动净流入", "买卖方向", "成交方向", "交易时长", "个人投资者交易", "机构交易热度",
        "买单非流动性", "卖单非流动性",
    ),
    "call_auction": ("集合竞价", "开盘前竞价", "收盘竞价"),
    "fundamental_raw": (
        "单季度", "研发费用", "员工", "劳动力", "应计盈余", "资本支出", "股权发行",
        "债务发行", "财务费用率", "总营业费用率", "存货变动", "净经营资产", "金融负债",
        "盈余持续性", "盈余可预测性", "标准化营业", "标准化净利润", "异常毛利润", "产能利用率",
        "异常其他应收款", "异常销售管理费用", "股票发行", "负债增长", "存货增长", "股东盈余",
        "总资产市值比",
        "资产增长率-5年", "账面价值增长率-5年", "线性纯化净利润", "营业能力改善",
        "标准化调整的营业利润", "经营性现金流量净额占比", "非流动性经营资产变动",
        "流动性经营资产变动", "基本面隐含收益", "规模调整后的ROE",
        "异常资本投资", "特异市值", "财务动量",
    ),
    "macro_external": ("消费指数", "宏观", "CPI", "GDP"),
    "risk_free_rate": ("无风险利率", "r_f", "r_{f"),
    "factor_returns": ("RMRF", "SMB", "HML", "UMD", "Fama-French", "FF3"),
    "governance": ("公司治理",),
    "earnings_event": ("盈余公告", "业绩公告"),
    "listing_metadata": ("企业年龄", "IPO日"),
    "business_segments": ("复杂公司", "业务复杂度", "分行业营业收入"),
    "trade_direction": ("羊群行为", "LSV 模型", "散户的买卖非平衡性"),
    "option": ("隐含波动率", "期权"),
    "network": ("图谱", "节点度", "中心性", "Katz", "网络"),
}

# The source block names a factor family but omits a constant, horizon, or
# unique scalar output.  These are definition blockers, not data blockers.
FORMULA_BLOCKER_TITLES = {
    "质量因子", "质量增长", "趋势因子", "趋势动量",
    "钱德动量摆动平均指数",
    "换手率调整的日成交量为零的数量", "价格时滞",
    "一致交易", "一致买入交易", "一致买入交易因子", "一致卖出交易",
    "异常尾部概率E", "异常尾部概率S", "尾部风险", "极端下行风险",
    "分钟理想振幅", "日内振幅切割因子", "基于振幅切割的动量",
    "加权盈利频率", "加权上引线频率", "加权下引线频率", "K 线形态",
    "行业动量-纵向切割", "行业动量-横向切割", "Dimson贝塔", "Dimson 贝塔",
    "基于月度收益的市场Beta",
    "上方\\下方活动筹码占比", "上方\\下方锁定筹码占比",
    "收益季节性", "收益季节性反转", "基于排序的动量", "估值趋势偏离度",
    "跳跃强度", "累计跳跃绝对收益", "平均换手日内正跳跃收益",
    "平均日内正跳跃收益", "平均日内正跳跃标准差", "跳跃显著程度加权的上下行跳跃波动不对称性",
    "日内信噪比", "多层加权日内信噪比", "信噪比增强反转",
    "高频弹性因子", "“情绪溢出”因子", "“自信溢出”因子",
    "跌幅时间重心偏离", "时间重心偏离", "日内异常交易量", "持续异常交易量",
    "基于 PCA 的特质波动率", "特质日内收益波动率", "特质隔夜收益波动率", "特质换手波动率",
    "同时点峰岭数相关性",
    "跳跃关联相对动量", "CSSD 模型", "CSAD 模型",
    "相似反转", "相似低波", "筹码收益", "筹码收益调整", "筹码收益增强",
}


def _slug(title: str) -> str:
    ascii_words = re.findall(r"[A-Za-z0-9]+", title.lower())
    if ascii_words:
        return "_".join(ascii_words)[:48]
    # Stable readable-enough code without an optional transliteration package.
    return "u" + "_".join(f"{ord(ch):x}" for ch in title if "\u4e00" <= ch <= "\u9fff")[:44]


def _normalized_category(parsed: ParsedFactor) -> str:
    raw = f"{parsed.category} {parsed.raw_category} {parsed.title}"
    if parsed.category and parsed.category != "未分类":
        for needle, normalized in CATEGORY_MAP.items():
            if needle in parsed.category:
                return normalized
    for needle, normalized in CATEGORY_MAP.items():
        if needle in raw:
            return normalized
    return parsed.category or "未分类"


def _blocker(title: str, body: str) -> str:
    haystack = f"{title}\n{body}"
    for label, needles in DATA_BLOCKERS.items():
        if any(needle in haystack for needle in needles):
            return label
    return "unmapped_inputs"


def _duplicate_refs(title: str) -> tuple[str, ...]:
    refs = []
    technical = ("动量", "反转", "波动", "换手", "量价", "相对强弱", "随机指标", "异同均线")
    if any(token in title for token in technical):
        refs.append("alpha101/158/191/360:concept_overlap")
    report = ("非流动性", "Amihud", "筹码", "资本利得", "价量", "低波", "Beta")
    if any(token.lower() in title.lower() for token in report):
        refs.append("research_report_factors:concept_overlap")
    return tuple(refs)


class CalendarFactorCatalog:
    """Load, validate and export all supported factor-calendar definitions."""

    def __init__(self, source_dir: str | Path | None = None) -> None:
        if source_dir is None:
            source_dir = Path(__file__).resolve().parent.parent / "factor_calendar"
        self.source_dir = Path(source_dir)
        self._definitions: dict[int, list[CalendarFactorDef]] = {}

    def build_year(self, year: int) -> list[CalendarFactorDef]:
        if year not in YEARS:
            raise ValueError(f"unsupported calendar year: {year}; supported={YEARS}")
        if year in self._definitions:
            return list(self._definitions[year])
        path = self.source_dir / f"factors{year}.md"
        parsed = parse_calendar(path, year)
        definitions = [self._to_definition(year, path, item) for item in parsed]
        expected = EXPECTED_COUNTS[year]
        if len(definitions) != expected:
            raise ValueError(f"calendar {year}: expected {expected}, parsed {len(definitions)}")
        names = [item.name for item in definitions]
        if len(names) != len(set(names)):
            raise ValueError(f"calendar {year}: duplicate generated names")
        self._definitions[year] = definitions
        return list(definitions)

    def build_all(self) -> list[CalendarFactorDef]:
        return [item for year in YEARS for item in self.build_year(year)]

    def _to_definition(self, year: int, path: Path, parsed: ParsedFactor) -> CalendarFactorDef:
        identifier = parsed.calendar_id.replace(".", "_")
        name = f"calendar{year}_{identifier}_{_slug(parsed.title)}"
        formula = formula_from_body(parsed.body)
        spec = find_spec(parsed.title, year=year, calendar_id=parsed.calendar_id)
        if spec is not None:
            status = spec.status
            notes = spec.notes
            implemented_formula = spec.formula
            required = spec.required_fields
            frequency = spec.input_frequency
            key = spec.key
            warmup = spec.warmup
            params = spec.params
        elif not parsed.body or parsed.title.startswith("未恢复标题_"):
            status = ImplementationStatus.BLOCKED_SOURCE
            notes = "源文档标题或定义块未可靠恢复"
            implemented_formula = ""
            required = ()
            frequency = "unknown"
            key = ""
            warmup = 0
            params = {}
        elif not formula:
            status = ImplementationStatus.BLOCKED_FORMULA
            notes = "源文档未提供可执行的明确公式"
            implemented_formula = ""
            required = ()
            frequency = "unknown"
            key = ""
            warmup = 0
            params = {}
        elif parsed.title in FORMULA_BLOCKER_TITLES:
            status = ImplementationStatus.BLOCKED_FORMULA
            notes = "源定义缺少唯一执行所需的窗口、阈值或标量输出，未擅自补默认值"
            implemented_formula = ""
            required = ()
            frequency = "unknown"
            key = ""
            warmup = 0
            params = {}
        else:
            reason = _blocker(parsed.title, parsed.body)
            if reason == "unmapped_inputs":
                status = ImplementationStatus.PENDING_IMPLEMENTATION
                notes = "源公式已保留，现有OHLCV/分钟线/基本面大概率可覆盖；尚待专用算子映射与逐式校验"
                required = ()
            else:
                status = ImplementationStatus.BLOCKED_DATA
                notes = f"精确实现依赖当前数据审计未发现的数据域: {reason}"
                required = (reason,)
            implemented_formula = ""
            frequency = "1m" if ("分钟" in parsed.body or "高频" in parsed.category) else "1d"
            key = ""
            warmup = 0
            params = {}
        duplicates = _duplicate_refs(parsed.title)
        return CalendarFactorDef(
            name=name,
            calendar_year=year,
            calendar_id=parsed.calendar_id,
            title=parsed.title,
            category=parsed.category,
            sub_category=parsed.sub_category,
            raw_category=parsed.raw_category,
            normalized_category=_normalized_category(parsed),
            source_file=path.name,
            source_start_line=parsed.start_line,
            source_end_line=parsed.end_line,
            input_frequency=frequency,
            required_fields=required,
            original_formula=formula,
            implemented_formula=implemented_formula,
            implementation_status=status,
            implementation_notes=notes,
            implementation_key=key,
            duplicate_refs=duplicates,
            duplicate_type="concept_overlap" if duplicates else "",
            default_params=dict(params),
            warmup_requirement=warmup,
        )

    def export_manifests(self, output_dir: str | Path) -> list[Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        written = []
        for year in YEARS:
            rows = [item.to_row() for item in self.build_year(year)]
            csv_path = output / f"calendar_{year}.csv"
            json_path = output / f"calendar_{year}.json"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            written.extend((csv_path, json_path))
        return written

    def with_proxy(
        self,
        original: CalendarFactorDef,
        *,
        key: str,
        formula: str,
        required_fields: Iterable[str],
        notes: str,
    ) -> CalendarFactorDef:
        """Create an explicitly named proxy without replacing the blocked source row."""
        return replace(
            original,
            name=f"{original.name}_proxy",
            implemented_formula=formula,
            implementation_status=ImplementationStatus.PROXY,
            implementation_notes=notes,
            implementation_key=key,
            proxy_of=original.name,
            required_fields=tuple(required_fields),
        )
