# 因子日历独立实现

本目录只新增独立代码，不修改现有因子、算子注册表或 AlphaLab。支持的日历为
2022、2023、2024、2025、2026。

## 当前覆盖

| 日历 | 定义数 | 精确实现 | 显式代理 | 有数据待实现 | 确认缺数据 | 源定义不足 |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 365 | 176 | 0 | 0 | 140 | 49 |
| 2023 | 357 | 174 | 0 | 0 | 156 | 27 |
| 2024 | 366 | 141 | 0 | 0 | 176 | 49 |
| 2025 | 365 | 69 | 7 | 100 | 164 | 25 |
| 2026 | 362 | 119 | 7 | 0 | 173 | 63 |
| 合计 | 1815 | 679 | 14 | 100 | 809 | 213 |

每条定义都包含 `calendar_year`、`calendar_id`、原始/标准化分类、源文件和行号、
原公式、实现公式、输入频率、字段依赖、预热期、实现状态以及与已有因子的概念重复备注。
完整逐条结果位于 `manifests/calendar_<year>.csv` 和 JSON 文件。

状态含义：

- `exact`：已经有可执行实现，且公式与日历定义直接对应。
- `pending_implementation`：源公式明确，审计判断现有 OHLCV、分钟线或基本面字段可以覆盖，
  但尚未逐式实现和验证专用算子。
- `blocked_data`：需要分析师预期、新闻/搜索、专利、供应链、股东明细、北向、逐笔成交、
  订单簿、期权或关系网络等当前数据域未发现的数据。
- `blocked_formula`：日历只给出概念、步骤不完整或参数口径不足，不能声称为精确实现。
- `proxy`：实现和清单中必须显式标注替代口径。2025、2026 各有 7 项因源定义要求中信一级行业，
  而现有 `industry.csv` 只有申万行业；量价主体公式不变，但不能冒充精确实现。

## 已接入的数据

`CalendarLabDataAdapter` 直接读取 `playground/alpha_research/lab/all_stocks`：

- 日线 5,493 个文件；1 分钟和 10 分钟各 5,489 个文件；60 分钟 5,496 个文件。
- 30 分钟目录当前没有 parquet，不能把 30 分钟原公式直接标成精确实现；可在调用侧从
  1 分钟重采样，但必须保留该口径说明。
- K 线字段包含 datetime、OHLC、volume、turnover、open_interest，VWAP 由
  `turnover / volume` 派生。
- common 下已审计估值、经营、财务、成长、现金流和行业表；行业适配器按 `source`
  显式过滤，默认使用当前落盘的 `sws`，避免未来多个行业源产生重复键。
- component、price_limit、universe_daily 等文件已存在，但本独立执行器尚未将它们隐式加入
  因子计算，避免不透明的股票池或可交易性过滤。

分钟算子严格按 `trade_date + vt_symbol` 分组，隔夜跳空不会混入日内收益。新增算子涵盖
日内聚合、收益高阶矩、上下行半方差及占比、已实现波动率、尾盘占比/收益、最大回撤、
量价滞后相关、成交额自相关、成交量分布、有状态筹码参考成本、量谷相对价格、
小型板块 CSAD、注意力溢出、趋势资金、净支撑量与极端跟随行为。2022、2023、2024、
2026 的可用数据待映射条目已经清零；2025 首批新增了 17 个专用映射，其余待实现条目
继续保留在 manifest 中，供后续逐式校验。

## 使用

在仓库根目录执行：

```bash
conda run --no-capture-output -n python311 python -m \
  vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.cli manifests \
  --output-dir vnpy/vnpy/alpha/dataset/datasets/factors_templates/factor_calendar_impl/manifests

conda run --no-capture-output -n python311 python -m \
  vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.cli audit \
  --lab-path playground/alpha_research/lab/all_stocks

conda run --no-capture-output -n python311 python -m \
  vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.cli sample \
  --lab-path playground/alpha_research/lab/all_stocks --year 2026 \
  --start 2024-01-02 --end 2024-06-28 --max-factors 30
```

真实数据验收结果记录在 `manifests/sample_validation.json`。全历史、全市场计算未自动启动。

## Titan Calendar family

Calendar 的 C++ 定义、编译、缓存与调用均由
`playground/alpha_research/titan_jinja` 统一管理。当前目录只保留元数据、Python
参考实现、数据适配器和兼容导入，不定义或编译 C++。Calendar 保持为独立 family，
不修改 Alpha101/158/191 的表达式与算子语义：

- `factor_calendar_daily`：286 条定义可以直接编译为现有 C++ TS/CS 表达式。
- `titan_jinja/cpp/kernels/calendar/special_calendar.cpp`：专门实现量谷价格、3 个 CSAD、2 个注意力溢出以及 5 个
  趋势资金/净支撑类因子，共 11 条。
- `titan_jinja/cpp/kernels/calendar/special_calendar_2025.cpp`：实现 14 个分钟技术指标、平均异常日收益率、时序 CSAD
  和基于 PB 的前景价值，共 17 条；分钟指标统一执行严格 20 个有效分钟交易日线性衰减。
- 合计 314 条 C++ 因子（286 条表达式 + 28 条专用内核）。其余 379 条已实现定义当前
  不能直接编译，其中包含专用算子和现有编译器尚不支持的
  表达式；逐条原因位于
  `manifests/calendar_cpp_coverage.json`。

基本面字段在进入 C++ 前由 `CalendarLabDataAdapter` 按 `datetime + vt_symbol` 合并。
common 日频因子是按历史交易日调用供应商接口得到的 PIT 面板；C++ 只消费日频数值列，
不会将季度报告期直接当作公告可得日。行业字段在传入专用 C++ 模块前编码成整数 ID。

同步 Calendar 表达式 family 到 Titan 中央因子库：

```bash
conda run --no-capture-output -n python311 python \
  playground/alpha_research/titan_jinja/python/factors/sync_alpha_family.py \
  --families calendar
```

计算示例：

```bash
PYTHONPATH=vnpy conda run --no-capture-output -n python311 python -m \
  vnpy.alpha.dataset.datasets.factors_templates.factor_calendar_impl.cpp_backend.cli compute \
  --lab-path playground/alpha_research/lab/all_stocks \
  --start 2025-08-29 --end 2026-04-30 \
  --symbols 000001.SZSE,000002.SZSE,600000.SSE \
  --mode special --industry-source sws --num-threads 8 \
  --output /tmp/factor_calendar_cpp.parquet
```

专用模块由 Titan 首次编译到 `titan_jinja/generated/.cache/custom`，后续按源码、编译参数
和 Titan 头文件哈希复用。所有公开列名使用 `calendar_{年份}_` 前缀。至少需要
121 个日频交易日作为 CSAD 因子的增量预热窗口；分钟类因子最多需要 40 个有效分钟
交易日，但统一使用 121 日可简化生产增量配置。
