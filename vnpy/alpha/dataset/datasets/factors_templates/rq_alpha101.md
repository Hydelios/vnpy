CLOSE = Factor('close')
RETURNS = (CLOSE - REF(CLOSE, 1)) / REF(CLOSE, 1)
CAP = Factor('market_cap')
VWAP = Factor('total_turnover') / Factor('volume)

alpha001 = (RANK(TS_ARGMAX(SIGNEDPOWER(IF((RETURNS < 0), STDDEV(RETURNS, 20), CLOSE), 2.), 5)) - 0.5)

alpha002 = (-1 * CORRELATION(RANK(DELTA(LOG(VOLUME), 2)), RANK(((CLOSE - OPEN) / OPEN)), 6))

alpha003 = (-1 * CORRELATION(RANK(OPEN), RANK(VOLUME), 10))

alpha004 = (-1 * TS_RANK(RANK(LOW), 9))

alpha005 = (RANK((OPEN - (SUM(VWAP, 10) / 10))) * (-1 * ABS(RANK((CLOSE - VWAP)))))

alpha006 = (-1 * CORRELATION(OPEN, VOLUME, 10))

alpha007 = IF((ADV(20) < VOLUME), ((-1 * TS_RANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7))), (-1 * 1))

alpha008 = (-1 * RANK(((SUM(OPEN, 5) * SUM(RETURNS, 5)) - DELAY((SUM(OPEN, 5) * SUM(RETURNS, 5)), 10))))

alpha009 = IF((0 < TS_MIN(DELTA(CLOSE, 1), 5)), DELTA(CLOSE, 1), (IF((TS_MAX(DELTA(CLOSE, 1), 5) < 0), DELTA(CLOSE, 1), (-1 * DELTA(CLOSE, 1)))))

alpha010 = RANK(IF((0 < TS_MIN(DELTA(CLOSE, 1), 4)), DELTA(CLOSE, 1), IF((TS_MAX(DELTA(CLOSE, 1), 4) < 0), DELTA(CLOSE, 1), (-1 * DELTA(CLOSE, 1)))))

alpha011 = ((RANK(TS_MAX((VWAP - CLOSE), 3)) + RANK(TS_MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))

alpha012 = (SIGN(DELTA(VOLUME, 1)) * (-1 * DELTA(CLOSE, 1)))

alpha013 = (-1 * RANK(COVARIANCE(RANK(CLOSE), RANK(VOLUME), 5)))

alpha014 = ((-1 * RANK(DELTA(RETURNS, 3))) * CORRELATION(OPEN, VOLUME, 10))

alpha015 = (-1 * SUM(RANK(CORRELATION(RANK(HIGH), RANK(VOLUME), 3)), 3))

alpha016 = (-1 * RANK(COVARIANCE(RANK(HIGH), RANK(VOLUME), 5)))

alpha017 = (((-1 * RANK(TS_RANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TS_RANK((VOLUME / ADV(20)), 5)))

alpha018 = (-1 * RANK(((STDDEV(ABS((CLOSE - OPEN)), 5) + (CLOSE - OPEN)) + CORRELATION(CLOSE, OPEN, 10))))

alpha019 = ((-1 * SIGN(((CLOSE - DELAY(CLOSE, 7)) + DELTA(CLOSE, 7)))) * (1 + RANK((1 + SUM(RETURNS, 250)))))

alpha020 = (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))

alpha021 = IF((((SUM(CLOSE, 8) / 8) + STDDEV(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)), (-1 * 1), IF(((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STDDEV(CLOSE, 8))), 1, IF(((1 < (VOLUME / ADV(20))) | ((VOLUME / ADV(20)) == 1)), 1, (-1 * 1))))

alpha022 = (-1 * (DELTA(CORRELATION(HIGH, VOLUME, 5), 5) * RANK(STDDEV(CLOSE, 20))))

alpha023 = IF(((SUM(HIGH, 20) / 20) < HIGH), (-1 * DELTA(HIGH, 2)), 0)

alpha024 = IF((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) | ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)), (-1 * (CLOSE - TS_MIN(CLOSE, 100))), (-1 * DELTA(CLOSE, 3)))

alpha025 = RANK(((((-1 * RETURNS) * ADV(20)) * VWAP) * (HIGH - CLOSE)))

alpha026 = (-1 * TS_MAX(CORRELATION(TS_RANK(VOLUME, 5), TS_RANK(HIGH, 5), 5), 3))

alpha027 = IF((0.5 < RANK((SUM(CORRELATION(RANK(VOLUME), RANK(VWAP), 6), 2) / 2.0))), (-1 * 1), 1)

alpha028 = SCALE(((CORRELATION(ADV(20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE))

alpha029 = (MIN(PRODUCT(RANK(RANK(SCALE(LOG(SUM(TS_MIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1))))), 1), 5) + TS_RANK(DELAY((-1 * RETURNS), 6), 5))

alpha030 = (((1.0 - RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))

alpha031 = ((RANK(RANK(RANK(DECAY_LINEAR((-1 * RANK(RANK(DELTA(CLOSE, 10)))), 10)))) + RANK((-1 * DELTA(CLOSE, 3)))) + SIGN(SCALE(CORRELATION(ADV(20), LOW, 12))))

alpha032 = (SCALE(((SUM(CLOSE, 7) / 7) - CLOSE)) + (20 * SCALE(CORRELATION(VWAP, DELAY(CLOSE, 5), 230))))

alpha033 = RANK((-1 * ((1 - (OPEN / CLOSE))**1)))

alpha034 = RANK(((1 - RANK((STDDEV(RETURNS, 2) / STDDEV(RETURNS, 5)))) + (1 - RANK(DELTA(CLOSE, 1)))))

alpha035 = ((TS_RANK(VOLUME, 32) * (1 - TS_RANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TS_RANK(RETURNS, 32)))

alpha036 = (((((2.21 * RANK(CORRELATION((CLOSE - OPEN), DELAY(VOLUME, 1), 15))) + (0.7 * RANK((OPEN - CLOSE)))) + (0.73 * RANK(TS_RANK(DELAY((-1 * RETURNS), 6), 5)))) + RANK(ABS(CORRELATION(VWAP, ADV(20), 6)))) + (0.6 * RANK((((SUM(CLOSE, 200) / 200) - OPEN) * (CLOSE - OPEN)))))

alpha037 = (RANK(CORRELATION(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))

alpha038 = ((-1 * RANK(TS_RANK(CLOSE, 10))) * RANK((CLOSE / OPEN)))

alpha039 = ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAY_LINEAR((VOLUME / ADV(20)), 9)))))) * (1 + RANK(SUM(RETURNS, 250))))

alpha040 = ((-1 * RANK(STDDEV(HIGH, 10))) * CORRELATION(HIGH, VOLUME, 10))

alpha041 = (((HIGH * LOW)**0.5) - VWAP)

alpha042 = (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))

alpha043 = (TS_RANK((VOLUME / ADV(20)), 20) * TS_RANK((-1 * DELTA(CLOSE, 7)), 8))

alpha044 = (-1 * CORRELATION(HIGH, RANK(VOLUME), 5))

alpha045 = (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORRELATION(CLOSE, VOLUME, 2)) * RANK(CORRELATION(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))

alpha046 = IF((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))), (-1 * 1), IF(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0), 1, ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))

alpha047 = ((((RANK((1 / CLOSE)) * VOLUME) / ADV(20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))

alpha048 = (INDUSTRY_NEUTRALIZE(((CORRELATION(DELTA(CLOSE, 1), DELTA(DELAY(CLOSE, 1), 1), 250) * DELTA(CLOSE, 1)) / CLOSE)) / SUM(((DELTA(CLOSE, 1) / DELAY(CLOSE, 1))**2), 250))

alpha049 = IF(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < (-1 * 0.1)), 1, ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1))))

alpha050 = (-1 * TS_MAX(RANK(CORRELATION(RANK(VOLUME), RANK(VWAP), 5)), 5))

alpha051 = IF(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < (-1 * 0.05)), 1, ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1))))

alpha052 = ((((-1 * TS_MIN(LOW, 5)) + DELAY(TS_MIN(LOW, 5), 5)) * RANK(((SUM(RETURNS, 240) - SUM(RETURNS, 20)) / 220))) * TS_RANK(VOLUME, 5))

alpha053 = (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (CLOSE - LOW)), 9))

alpha054 = ((-1 * ((LOW - CLOSE) * (OPEN**5))) / ((LOW - HIGH) * (CLOSE**5)))

alpha055 = (-1 * CORRELATION(RANK(((CLOSE - TS_MIN(LOW, 12)) / (TS_MAX(HIGH, 12) - TS_MIN(LOW, 12)))), RANK(VOLUME), 6))

alpha056 = (0 - (1 * (RANK((SUM(RETURNS, 10) / SUM(SUM(RETURNS, 2), 3))) * RANK((RETURNS * CAP)))))

alpha057 = (0 - (1 * ((CLOSE - VWAP) / DECAY_LINEAR(RANK(TS_ARGMAX(CLOSE, 30)), 2))))

alpha058 = (-1 * TS_RANK(DECAY_LINEAR(CORRELATION(INDUSTRY_NEUTRALIZE(VWAP), VOLUME, 4), 8), 6))

alpha059 = (-1 * TS_RANK(DECAY_LINEAR(CORRELATION(INDUSTRY_NEUTRALIZE(((VWAP * 0.728317) + (VWAP * (1 - 0.728317)))), VOLUME, 4), 16), 8))

alpha060 = (0 - (1 * ((2 * SCALE(RANK(((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)) * VOLUME)))) - SCALE(RANK(TS_ARGMAX(CLOSE, 10))))))

alpha061 = (RANK((VWAP - TS_MIN(VWAP, 16))) < RANK(CORRELATION(VWAP, ADV(180), 18)))

alpha062 = ((RANK(CORRELATION(VWAP, SUM(ADV(20), 22), 10)) < RANK(((RANK(OPEN) + RANK(OPEN)) < (RANK(((HIGH + LOW) / 2)) + RANK(HIGH))))) * -1)

alpha063 = ((RANK(DECAY_LINEAR(DELTA(INDUSTRY_NEUTRALIZE(CLOSE), 2), 8)) - RANK(DECAY_LINEAR(CORRELATION(((VWAP * 0.318108) + (OPEN * (1 - 0.318108))), SUM(ADV(180), 37), 14), 12))) * -1)

alpha064 = ((RANK(CORRELATION(SUM(((OPEN * 0.178404) + (LOW * (1 - 0.178404))), 13), SUM(ADV(120), 13), 17)) < RANK(DELTA(((((HIGH + LOW) / 2) * 0.178404) + (VWAP * (1 - 0.178404))), 4))) * -1)

alpha065 = ((RANK(CORRELATION(((OPEN * 0.00817205) + (VWAP * (1 - 0.00817205))), SUM(ADV(60), 9), 6)) < RANK((OPEN - TS_MIN(OPEN, 14)))) * -1)

alpha066 = ((RANK(DECAY_LINEAR(DELTA(VWAP, 4), 7)) + TS_RANK(DECAY_LINEAR(((((LOW * 0.96633) + (LOW * (1 - 0.96633))) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)

alpha067 = ((RANK((HIGH - TS_MIN(HIGH, 2)))**RANK(CORRELATION(INDUSTRY_NEUTRALIZE(VWAP), INDUSTRY_NEUTRALIZE(ADV(20)), 6))) * -1)

alpha068 = ((TS_RANK(CORRELATION(RANK(HIGH), RANK(ADV(15)), 9), 14) < RANK(DELTA(((CLOSE * 0.518371) + (LOW * (1 - 0.518371))), 1))) * -1)

alpha069 = ((RANK(TS_MAX(DELTA(INDUSTRY_NEUTRALIZE(VWAP), 3), 5))**TS_RANK(CORRELATION(((CLOSE * 0.490655) + (VWAP * (1 - 0.490655))), ADV(20), 5), 9)) * -1)

alpha070 = ((RANK(DELTA(VWAP, 1))**TS_RANK(CORRELATION(INDUSTRY_NEUTRALIZE(CLOSE), ADV(50), 18), 18)) * -1)

alpha071 = MAX(TS_RANK(DECAY_LINEAR(CORRELATION(TS_RANK(CLOSE, 3), TS_RANK(ADV(180), 12), 18), 4), 16), TS_RANK(DECAY_LINEAR((RANK(((LOW + OPEN) - (VWAP + VWAP)))**2), 16), 4))

alpha072 = (RANK(DECAY_LINEAR(CORRELATION(((HIGH + LOW) / 2), ADV(40), 9), 10)) / RANK(DECAY_LINEAR(CORRELATION(TS_RANK(VWAP, 4), TS_RANK(VOLUME, 19), 7), 3)))

# Modify DELTA(VWAP, 4.72775) to DELTA(VWAP, 5)
alpha073 = (MAX(RANK(DECAY_LINEAR(DELTA(VWAP, 5), 3)), TS_RANK(DECAY_LINEAR(((DELTA(((OPEN * 0.147155) + (LOW * (1 - 0.147155))), 2) / ((OPEN * 0.147155) + (LOW * (1 - 0.147155)))) * -1), 3), 17)) * -1)

alpha074 = ((RANK(CORRELATION(CLOSE, SUM(ADV(30), 37), 15)) < RANK(CORRELATION(RANK(((HIGH * 0.0261661) + (VWAP * (1 - 0.0261661)))), RANK(VOLUME), 11))) * -1)

alpha075 = (RANK(CORRELATION(VWAP, VOLUME, 4)) < RANK(CORRELATION(RANK(LOW), RANK(ADV(50)), 12)))

# Modify TS_RANK(CORRELATION(INDUSTRY_NEUTRALIZE(LOW), ADV(81), 8), 19.569) to TS_RANK(CORRELATION(INDUSTRY_NEUTRALIZE(LOW), ADV(81), 8), 20)
alpha076 = (MAX(RANK(DECAY_LINEAR(DELTA(VWAP, 1), 12)), TS_RANK(DECAY_LINEAR(TS_RANK(CORRELATION(INDUSTRY_NEUTRALIZE(LOW), ADV(81), 8), 20), 17), 19)) * -1)

alpha077 = MIN(RANK(DECAY_LINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)), RANK(DECAY_LINEAR(CORRELATION(((HIGH + LOW) / 2), ADV(40), 3), 6)))

alpha078 = (RANK(CORRELATION(SUM(((LOW * 0.352233) + (VWAP * (1 - 0.352233))), 20), SUM(ADV(40), 20), 7))**RANK(CORRELATION(RANK(VWAP), RANK(VOLUME), 6)))

alpha079 = (RANK(DELTA(INDUSTRY_NEUTRALIZE(((CLOSE * 0.60733) + (OPEN * (1 - 0.60733)))), 1)) < RANK(CORRELATION(TS_RANK(VWAP, 4), TS_RANK(ADV(150), 9), 15)))

alpha080 = ((RANK(SIGN(DELTA(INDUSTRY_NEUTRALIZE(((OPEN * 0.868128) + (HIGH * (1 - 0.868128)))), 4)))**TS_RANK(CORRELATION(HIGH, ADV(10), 5), 6)) * -1)

alpha081 = ((RANK(LOG(PRODUCT(RANK((RANK(CORRELATION(VWAP, SUM(ADV(10), 50), 8))**4)), 15))) < RANK(CORRELATION(RANK(VWAP), RANK(VOLUME), 5))) * -1)

alpha082 = (MIN(RANK(DECAY_LINEAR(DELTA(OPEN, 1), 15)), TS_RANK(DECAY_LINEAR(CORRELATION(INDUSTRY_NEUTRALIZE(VOLUME), ((OPEN * 0.634196) + (OPEN * (1 - 0.634196))), 17), 7), 13)) * -1)

alpha083 = ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))

alpha084 = SIGNEDPOWER(TS_RANK((VWAP - TS_MAX(VWAP, 15)), 21), DELTA(CLOSE, 5))

alpha085 = (RANK(CORRELATION(((HIGH * 0.876703) + (CLOSE * (1 - 0.876703))), ADV(30), 10))**RANK(CORRELATION(TS_RANK(((HIGH + LOW) / 2), 4), TS_RANK(VOLUME, 10), 7)))

alpha086 = ((TS_RANK(CORRELATION(CLOSE, SUM(ADV(20), 15), 6), 20) < RANK(((OPEN + CLOSE) - (VWAP + OPEN)))) * -1)

alpha087 = (MAX(RANK(DECAY_LINEAR(DELTA(((CLOSE * 0.369701) + (VWAP * (1 - 0.369701))), 2), 3)), TS_RANK(DECAY_LINEAR(ABS(CORRELATION(INDUSTRY_NEUTRALIZE(ADV(81)), CLOSE, 13)), 5), 14)) * -1)

# Modify TS_RANK(ADV(60), 20.6966) to TS_RANK(ADV(60), 21),
alpha088 = MIN(RANK(DECAY_LINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TS_RANK(DECAY_LINEAR(CORRELATION(TS_RANK(CLOSE, 8), TS_RANK(ADV(60), 21), 8), 7), 3))

alpha089 = (TS_RANK(DECAY_LINEAR(CORRELATION(((LOW * 0.967285) + (LOW * (1 - 0.967285))), ADV(10), 7), 6), 4) - TS_RANK(DECAY_LINEAR(DELTA(INDUSTRY_NEUTRALIZE(VWAP), 3), 10), 15))

alpha090 = ((RANK((CLOSE - TS_MAX(CLOSE, 5)))**TS_RANK(CORRELATION(INDUSTRY_NEUTRALIZE(ADV(40)), LOW, 5), 3)) * -1)

alpha091 = ((TS_RANK(DECAY_LINEAR(DECAY_LINEAR(CORRELATION(INDUSTRY_NEUTRALIZE(CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAY_LINEAR(CORRELATION(VWAP, ADV(30), 4), 3))) * -1)

alpha092 = MIN(TS_RANK(DECAY_LINEAR(AS_FLOAT((((HIGH + LOW) / 2) + CLOSE) < (LOW + OPEN)), 15), 19), TS_RANK(DECAY_LINEAR(CORRELATION(RANK(LOW), RANK(ADV(30)), 8), 7), 7))

alpha093 = (TS_RANK(DECAY_LINEAR(CORRELATION(INDUSTRY_NEUTRALIZE(VWAP), ADV(81), 17), 20), 8) / RANK(DECAY_LINEAR(DELTA(((CLOSE * 0.524434) + (VWAP * (1 - 0.524434))), 3), 16)))

alpha094 = ((RANK((VWAP - TS_MIN(VWAP, 12)))**TS_RANK(CORRELATION(TS_RANK(VWAP, 20), TS_RANK(ADV(60), 4), 18), 3)) * -1)

alpha095 = (RANK((OPEN - TS_MIN(OPEN, 12))) < TS_RANK((RANK(CORRELATION(SUM(((HIGH + LOW) / 2), 19), SUM(ADV(40), 19), 13))**5), 12))

# Modify TS_RANK(ADV(60), 4.13242) to TS_RANK(ADV(60), 4)
alpha096 = MAX(TS_RANK(DECAY_LINEAR(CORRELATION(RANK(VWAP), RANK(VOLUME), 4), 4), 8), TS_RANK(DECAY_LINEAR(TS_ARGMAX(CORRELATION(TS_RANK(CLOSE, 7), TS_RANK(ADV(60), 4), 4), 13), 14), 13)) * -1

# Modify TS_RANK(LOW, 7.87871) to TS_RANK(LOW, 8)
alpha097 = ((RANK(DECAY_LINEAR(DELTA(INDUSTRY_NEUTRALIZE(((LOW * 0.721001) + (VWAP * (1 - 0.721001)))), 3), 20)) - TS_RANK(DECAY_LINEAR(TS_RANK(CORRELATION(TS_RANK(LOW, 8), TS_RANK(ADV(60), 17), 5), 19), 16), 7)) * -1)

alpha098 = (RANK(DECAY_LINEAR(CORRELATION(VWAP, SUM(ADV(5), 26), 6), 7)) - RANK(DECAY_LINEAR(TS_RANK(TS_ARGMIN(CORRELATION(RANK(OPEN), RANK(ADV(15)), 21), 9), 7), 8)))

alpha099 = ((RANK(CORRELATION(SUM(((HIGH + LOW) / 2), 20), SUM(ADV(60), 20), 9)) < RANK(CORRELATION(LOW, VOLUME, 6))) * -1)

alpha100 = (0 - (1 * (((1.5 * SCALE(INDUSTRY_NEUTRALIZE(INDUSTRY_NEUTRALIZE(RANK(((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)) * VOLUME)))))) - SCALE(INDUSTRY_NEUTRALIZE((CORRELATION(CLOSE, RANK(ADV(20)), 5) - RANK(TS_ARGMIN(CLOSE, 30)))))) * (VOLUME / ADV(20)))))

alpha101 = ((CLOSE - OPEN) / ((HIGH - LOW) + .001))


简单算子
处理单因子或双因子的当期 / 基础时间序列逻辑，涵盖数值转换、条件判断、滞后计算等场景：

算子	含义
ABS(X)	取因子的绝对值
LOG(X)	取因子的对数值
EXP(X)	e 的 x 次幂，x 指因子值。参考 np.exp
LOG(X)	取因子的自然对数
EQUAL(A, B)	比较两个因子或一个因子与一个常数
SIGN(X)	取因子值的符号，参考 np.sign
SIGNEDPOWER(X, c)	计算因子值的 c 次方，保持符号不变；c 为常数
MIN(A, B)	对A、B 两个因子，取其中较小的值
FMIN(A, B)	与 MIN 类似，但当其中一个值为 NaN 时，FMIN返回另一个值，MIN 则相反
MAX(A, B)	对A、B 两个因子，取其中较大的值
FMAX(A, B)	与 MAX 类似，但当其中一个值为 NaN 时，FMAX 返回另一个值，MAX 则相反
IF(C, A, B)	当因子 C 值为 True 时，返回因子 A 的值；反之则返回因子 B 的值
AS_FLOAT(X)	将因子 X 的值转换为浮点类型
REF(X, n)	返回 n 个交易日前 X 的值
DELAY(X, n)	REF 的别名
DELTA(X, n)	X - REF(X, n)
PCT_CHANGE(X, n)	X / REF(X, n) - 1.0
均线算子
专注于时间序列的平滑计算，覆盖动态移动平均、简单移动平均、指数移动平均等常用均线类型：

DMA
DMA(X, c) 基于常数权重 c 计算动态平均，公式为：

    y[0] = x[0]
    y[i] = y[i-1] * (1 - c) + x[i] * c
DMA(A, B) 对于序列a、b，DMA 变换的结果为：

    y[0] = a[0]
    y[i] = y[i-1] * (1 - b[i]) + a[i] * b[i]
MA 及 SMA
SMA_CN(X, n, m) 国内常用行情软件中的 SMA 算子，相当于 DMA(X, m / n)
SMA(X, n) MA 的别名
MA(X, n) 简单移动平均，与 talib 中 MA 相同
EMA
EMA_CN(X, n) 指数平滑移动平均，相当于 DMA(X, 2 / (N + 1))
EMA(X, n) 指数平滑移动平均，返回收盘价 n 个交易日的指数平滑移动平均
WMA
WMA(X, n) 与 talib 中 WMA 相同
DECAY_LINEAR(X, n) WMA 的别名
横截面算子
RANK
RANK(X， method='average', ascending=True) 对股票池中股票的因子值排序，返回归一化排名（名次 / 总数量）。对于 NaN 值，返回 NaN。

method: 排名方式，可选 average, min, max, first, dense，其意义与 pandas.DataFrame.rank 中 method 字段相同；
ascending: 排序方向（True 升序，False 降序），默认为True
例如：RANK(Factor('close'), 'first', True) 返回当日收盘价在过去股票池内的升序排序占比，若多项值相同，则按它们出现在数组中的顺序进行排序；
SCALE
SCALE(X, to=1) 对股票池中的因子值进行缩放，使其绝对值的和等于 to

例如：SCALE(Factor('close'), 10) 给定股票池，将股票池内收盘价量级调整进行调整，使绝对值之和等于 10
DEMEAN
DEMEAN(X) 对股票池中的因子值做中心化处理，从而使得和为 0

例如：DEMEAN(Factor('close')) 给定股票池，返回每只股票的收盘价减去收盘价的均值
CS_ZSCORE
CS_ZSCORE(X) 求股票池中因子值的 zscore

例如：CS_ZSCORE(Factor('close')) 给定股票池，返回收盘价在选定股票池的标准化值
QUANTILE
QUANTILE(X, bins=5, ascending=True) 将因子值按大小划分为 bins 组，返回对应的组序号。参考 pandas.qcut

例如：QUANTILE(Factor('close'), bins = 5, ascending = True) 表示对收盘价按升序分为 5 组
TOP
TOP(X, threshold=50, pct=False) 筛选股票池内 X 排名靠前的标的

当 pct 为 False 时，因子值处于 top threshold 的股票返回 1，其他返回 0；
当 pct 为 True 时，因子值处于前 threshold 百分位的股票返回 1，其他返回 0。
例如：TOP(Factor('close'), threshold=50, pct=False) 在股票池中排名前 50 的返回 1，其余为 0
BOTTOM
BOTTOM(X, threshold=50, pct=False) 与 TOP 类似，取排名较后的

例如：BOTTOM(Factor('close'), threshold=50, pct=False) 在股票池中排名最后 50 的返回 1，其余为 0
INDUSTRY_NEUTRALIZE
INDUSTRY_NEUTRALIZE(X) 对因子 X 进行行业中性化，默认按申万一级行业分类

例如：INDUSTRY_NEUTRALIZE(Factor('close')) 选定股票池，根据申万行业分类对收盘价进行行业中性化处理
CS_REGRESSION_RESIDUAL
CS_REGRESSION_RESIDUAL(Y, *X, add_const=True) 使用一个或多个因子 *X 作为自变量，Y 作为因变量进行回归，返回回归残差

例如：CS_REGRESSION_RESIDUAL (Factor('close'), Factor('open'), Factor('high'))
CS_FILLNA
CS_FILLNA(X) 用X所属行业均值填充缺失值

FIX
FIX(X, order_book_id) 固定返回因子 X 在标的 order_book_id 的值，如 FIX(Factor('close'), '000300.XSHG') 将总是返回 000300.XSHG 的收盘价。 常用于计算股票与指数关系，如 CORR(PCT_CHANGE(Facotr('close')), FIX(PCT_CHANGE(Factor('close')), '000300.XSHG')) 可计算股票收益率与沪深 300 收益率的相关性。

其他算子
AVEDEV
AVEDEV(X, n) X 最近 n 期因子值与均值的绝对偏差的平均值

STD
STD(X, n) X 最近 n 期因子值的标准差，与 talib 中 STDDEV 相同

例如：STD(Factor('close'), 10) 表示收盘价 10 个交易日的标准差
STDDEV
STDDEV(X, n) STD 的别名

VAR
VAR(X, n) X 最近 n 期因子值的方差，与 talib 中 VAR 相同

例如：VAR(Factor('close'), 10) 表示收盘价在过去 10 个交易日内的方差
CROSS
CROSS(A, B) （A[i-1] <= B[i-1]） and A[i] > B[i]，即 A 从下方穿越 B

例如：CROSS( MA(Factor('close'), 5), MA(Factor('close'), 10)) 返回 5 日均线与 10 日均线交叉
TS_SKEW
TS_SKEW(X, n) 最近 n 期因子值的偏度

例如：TS_SKEW(Factor('close'), 60) 返回收盘价过去 60 个交易日的偏度
TS_KURT
TS_KURT(X, n) 最近 n 期因子值的峰度

例如：TS_KURT(Factor('close'), 60) 返回收盘价过去 60 个交易日的峰度
SLOPE
SLOPE(X, n) 最近 n 期因子值的线性回归斜率，与 talib 中 LINEARREG_SLOPE 相同

SUM
SUM(X, n) 最近 n 期因子值的和

例如：SUM(Factor('volume'), 10) 统计 10 个交易日成交量的总和
PRODUCT
PRODUCT(X, n) 最近 n 期因子值的乘积

例如：PRODUCT(Factor('close'), 10) 表示过去 10 个交易日收盘价的移动平均求积
TS_MIN
TS_MIN(X, n) 最近 n 期因子值的最小值，与 talib 中 MIN 相同

例如：TS_MIN(Factor('close'), 5) 返回过去 5 个交易日中收盘价的最小值
TS_MAX
TS_MAX(X, n) 最近 n 期因子值的最大值，与 talib 中 MAX 相同

例如：TS_MAX(Factor('close'), 5) 返回过去 5 个交易日中收盘价的最大值
LLV
LLV(X, n) TS_MIN 的别名

例如：LLV(Factor('close'), 10) 表示 10 个交易日内收盘价的最小值
HHV
HHV(X, n) TS_MAX 的别名

例如：HHV(Factor('close'), 10) 表示 10 个交易日内收盘价的最大值
TS_ARGMIN
TS_ARGMIN(X, n) 最近 n 期因子值中最小值的位置索引，与 talib 中 MININDEX 相同

例如：TS_ARGMIN(Factor('close'), 5) 返回过去 5 个交易日中收盘价的最小值的位置索引
TS_ARGMAX
TS_ARGMAX(X, n) 最近 n 期因子值中最大值的位置索引，与 talib 中 MAXINDEX 相同

例如：TS_ARGMAX(Factor('close'), 5) 返回过去 5 个交易日中收盘价的最大值的索引
CORR
CORR(A, B, n) 因子 A 及 B 最近 n 期因子值的相关性，与 talib 中 CORREL 相同

例如：CORR(Factor('close'), Factor('open'), 66) 表示过去 66 个交易日收盘价与开盘价的相关系数
CORRELATION
CORRELATION(A, B, n) CORR 的别名

COVARIANCE
COVARIANCE(A, B, n) 因子 A 及 B 最近 n 期因子值的协方差

COV
COV(A, B, n) COVARIANCE 的别名

例如：COV(Factor('open'), Factor('close'), 66) 表示过去 66 个交易日收盘价与开盘价的协方差
COUNT
COUNT(X, n) 最近 n 期因子值中为 True 的数量

EVERY
EVERY(X, n) EQUAL(COUNT(X, n), n)

例如：EVERY(Factor('close') > Factor('open'), 10) 表示若 10 个交易日内一直是阳线，则返回 True，否则返回 False
TS_RANK
TS_RANK(X, n) 最新值在最近 n 期中的排名百分位

例如：如 N=5，今天的 X 排序为 3，则返回 0.6
TS_ZSCORE
TS_ZSCORE(X, n) 最新值在最近 n 期中的 zscore

例如：TS_ZSCORE (Factor('close'), 10) 表示收盘价在过去 10 个交易日的取值作为样本，返回当前交易日收盘价对应的标准分数
TS_REGRESSION
TS_REGRESSION(Y, X, n) 对最新 n 期因子值，使用 Y作为因变量，X作为自变量进行回归，回归系数作为结果

例如：TS_REGRESSION(Factor('close'),Factor('open'),60) 返回收盘价为因变量，开盘价为自变量在过去 60 个交易日的回归参数
TS_FILLNA
TS_FILLNA(X, nv, method='value') 缺失值填充 根据 method 的值，使用不同的方式对缺失值进行填充：

value: 将 NaN 替换为 nv
MA: 使用前 nv 期非NaN的因子均值填充
forward: 向前搜索直到找到非 NaN 的值，但不超过 nv 项，用找到的值填充
