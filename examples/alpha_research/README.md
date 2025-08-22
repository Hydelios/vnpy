# 股票数据下载工具使用说明

本目录包含两个主要的数据下载脚本，用于从米筐数据源下载股票相关数据。

## 目录结构

```
alpha_research/
├── download_index_components.py  # 指数成分股列表下载
├── download_all_stocks.py        # 全市场股票K线数据下载
└── README.md                      # 本说明文档
```

## 前置条件

1. **米筐数据账号**：需要有效的米筐(RQData)账号，在 `vnpy/trader/setting.py` 中配置：
   ```python
   "datafeed.name": "rqdata",
   "datafeed.username": "your_license",
   "datafeed.password": "your_password"
   ```

2. **Python环境**：需要安装以下依赖：
   ```bash
   pip install vnpy
   pip install vnpy-rqdata
   pip install tqdm
   pip install pandas
   pip install polars
   ```

## 一、指数成分股下载工具

### 文件：`download_index_components.py`

### 功能说明
- 下载主要指数的历史成分股列表
- 保存每个指数的成分股变化历史
- 支持单个或多个指数批量下载
- 自动去重合并所有成分股

### 支持的指数
- 上证50 (000016.SSE)
- 沪深300 (000300.SSE)
- 中证500 (000905.SSE)
- 中证800 (000906.SSE)
- 中证1000 (000852.SSE)
- 中证2000 (932000.SSE)
- 科创50 (000688.SSE)
- 中证红利 (000922.SSE)

### 使用方法

```bash
# 1. 使用默认参数（2007-01-01 至 昨天）
python download_index_components.py

# 2. 指定开始日期（结束日期默认为昨天）
python download_index_components.py 2010-01-01

# 3. 指定完整日期范围
python download_index_components.py 2010-01-01 2024-12-31

# 4. 指定任务名称（数据保存目录）
python download_index_components.py --task-name my_indexes
```

### 输出文件
数据保存在 `./lab/{task_name}/` 目录下：
- `component/` - 各指数的历史成分股数据
  - `{index_symbol}.dat` - 指数成分股历史记录
  - `all_components.txt` - 所有成分股汇总列表
- `contract.json` - 股票回测参数配置

### 示例输出
```
============================================================
指数成分股列表下载工具
============================================================

下载参数：
  - 开始日期：2007-01-01
  - 结束日期：2024-12-31
  - 任务名称：major_indexes

正在获取 沪深300(000300.SSE) 成分股...
  - 成功获取 523 只历史成分股
  - 累计股票数量：523

正在获取 中证500(000905.SSE) 成分股...
  - 成功获取 681 只历史成分股
  - 累计股票数量：1089

汇总统计：
  - 获取指数数量：8
  - 不重复股票和指数总数：2156
```

## 二、全市场股票K线下载工具

### 文件：`download_all_stocks.py`

### 功能说明
- 下载全市场所有股票的K线数据
- 支持多种时间周期（日线、分钟线）
- 自动设置回测参数
- 数据验证和下载报告

### 支持的K线周期
- `1d` - 日线
- `1m` - 1分钟线
- `10m` - 10分钟线
- `30m` - 30分钟线
- `60m` - 60分钟线（小时线）

### 使用方法

#### 基础用法
```bash
# 1. 下载日线数据（默认）
python download_all_stocks.py

# 2. 指定时间范围
python download_all_stocks.py 2020-01-01 2024-12-31

# 3. 下载不同周期的数据
python download_all_stocks.py --interval 1m    # 1分钟线
python download_all_stocks.py --interval 10m   # 10分钟线
python download_all_stocks.py --interval 30m   # 30分钟线
python download_all_stocks.py --interval 60m   # 60分钟线
```

#### 高级选项
```bash
# 指定任务名称（数据保存目录）
python download_all_stocks.py --task-name my_data

# 不设置回测参数
python download_all_stocks.py --no-contract

# 不验证下载的数据
python download_all_stocks.py --no-verify

# 组合使用
python download_all_stocks.py 2024-01-01 2024-12-31 --interval 30m --task-name min30_data
```

### 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `start_date` | 开始日期 (YYYY-MM-DD) | 2007-01-01 |
| `end_date` | 结束日期 (YYYY-MM-DD) | 昨天 |
| `--interval` | K线周期 | 1d |
| `--task-name` | 任务名称/数据目录 | all_stocks |
| `--no-contract` | 不设置回测参数 | False |
| `--no-verify` | 不验证数据 | False |

### 数据存储结构
```
./lab/{task_name}/
├── daily/              # 日线数据
│   └── *.parquet      # 每个股票一个文件
├── minute/             # 分钟线数据
│   ├── 1m/            # 1分钟线
│   ├── 10m/           # 10分钟线
│   ├── 30m/           # 30分钟线
│   └── 60m/           # 60分钟线
├── contract.json       # 回测参数配置
└── download_report.txt # 下载报告
```

### 回测参数设置
脚本会自动为每只股票设置以下回测参数：
- 做多手续费率：万分之五 (0.05%)
- 做空手续费率：千分之一 (0.1%)
- 合约乘数：1
- 最小价格变动：0.0001

### 示例输出
```
============================================================
全市场股票数据下载工具
============================================================

下载参数：
  - 开始日期：2024-01-01
  - 结束日期：2024-12-31
  - K线周期：30m
  - 任务名称：all_stocks
  - 设置回测参数：是
  - 验证数据：是

正在获取全市场股票列表...
成功获取 5483 只股票

开始下载 5483 只股票的30m数据...
时间范围：2024-01-01 至 2024-12-31
下载进度: 100%|████████████| 5483/5483 [45:23<00:00, 2.01it/s]

下载完成！
  - 成功：5201 只
  - 失败：282 只

为 5201 只股票设置回测参数...
设置回测参数: 100%|████████████| 5201/5201 [02:15<00:00, 38.52it/s]

下载报告已保存到：./lab/all_stocks/download_report.txt
```

## 三、典型使用场景

### 场景1：获取主要指数成分股进行研究
```bash
# 1. 先下载指数成分股列表
python download_index_components.py 2020-01-01

# 2. 下载成分股的日线数据（全市场数据包含成分股）
python download_all_stocks.py 2020-01-01 --task-name index_stocks
```

### 场景2：准备回测数据
```bash
# 下载近3年的日线数据用于回测
python download_all_stocks.py 2022-01-01 --interval 1d
```

### 场景3：高频策略数据准备
```bash
# 下载最近1年的10分钟线数据
python download_all_stocks.py 2024-01-01 --interval 10m --task-name hf_data
```

### 场景4：指数成分股历史分析
```bash
# 下载所有指数的完整历史成分股数据
python download_index_components.py 2007-01-01
```

## 四、注意事项

1. **数据量提醒**
   - 全市场日线数据：约5000+股票 × 交易日数
   - 分钟线数据量更大，建议分批下载或缩短时间范围
   - 10分钟线：每天约24个数据点
   - 30分钟线：每天约8个数据点

2. **下载时间估算**
   - 日线数据：全市场约需30-60分钟
   - 1分钟线：建议下载近期数据，全量数据可能需要数小时
   - 10/30分钟线：约需1-2小时

3. **存储空间**
   - 日线数据：每只股票约1-5MB
   - 分钟线数据：根据周期不同，每只股票10-100MB
   - 建议预留足够的磁盘空间

4. **错误处理**
   - 脚本会记录下载失败的股票
   - 失败列表保存在 `download_report.txt`
   - 可根据失败列表重新下载特定股票

5. **米筐数据限制**
   - 部分退市股票可能无法获取数据
   - 新股上市初期可能数据不全
   - 建议定期更新数据

## 五、常见问题

### Q1: 下载失败怎么办？
A: 检查以下几点：
- 米筐账号是否有效
- 网络连接是否正常
- 查看 `download_report.txt` 中的失败列表
- 可以缩短时间范围重试

### Q2: 如何只下载特定股票？
A: 目前脚本下载全市场数据。如需特定股票，可以：
1. 修改 `get_all_stocks()` 函数
2. 或者下载后从数据文件中筛选

### Q3: 数据更新频率？
A: 建议：
- 日线数据：每天收盘后更新
- 分钟线数据：根据策略需要更新
- 成分股列表：每季度更新一次

### Q4: 如何验证数据完整性？
A: 脚本提供了数据验证功能：
- 自动抽样验证（默认开启）
- 查看 `download_report.txt` 统计信息
- 使用 AlphaLab 加载数据进行检查

## 六、技术支持

如遇到问题，请检查：
1. Python环境和依赖是否正确安装
2. 米筐账号配置是否正确
3. 查看日志中的具体错误信息

---

更新日期：2024-12-22
版本：v1.0