---
TQ_show_start_date: 2026-01-20
---

# 1 公司治理类
### 1.1.1 公司治理因子

参考南开指数的构造及2002年1月7日证监会颁布的《上市公司治理准则》中的相关规定，从公司内部治理出发选取公司治理细分因子，并将其加权汇总成公司治理综合因子。因子共涉及股权结构与股东权利、董事会构成、管理层激励、信息披露与合规、激励约束机制5个方面：

公司治理因子 = α₁第一大股东持股比例 + α₂第二至十大股东持股比例  
+ α₃流通股比例 + α₄上市公司股东数量  
+ α₅独立董事占比 + α₆董事会委员数量  
+ α₇管理层薪酬 + α₈管理层持股数  
+ α₉受证监会、交易所等处罚情况 + α₁₀是否实施股权激励量

#### 说明
良好的公司治理能使公司在未来具有较高财务安全性，有利于提高公司的盈利能力，是公司产生潜在价值的源泉。

#### 参考文献
刘均伟, 2017, 别开生面:公司治理因子详解, 光大证券

---

### 1.1.2 公司治理指数

作者利用IRRC的公司治理条款构建了一个代表股东权利强度的“公司治理指数”，判断每一家公司的所有治理条款，如果治理条款减少了股东权利，就给指数加1分。

#### 说明
指数取值越高，代表公司的管理者权利越高，股东权利越低；指数取值越低，代表公司的管理者权利越低，股东权利越高，而拥有更强股东权利的公司具有更高的公司价值，更高的利润、更高的销售增长、更低的资本支出和更少的企业收购。

#### 参考文献
Gompers, P. A., Ishii, J. L., Metrick, A. Corporate Governance and Equity Price. Quarterly Journal of Economic, 2003, 118(1), 107-155;
# 2 量价技术类
## 2.1 技术因子-超买超卖
### 2.1.1 随机指标  

RSV = 100*(CLOSE-MIN(LOW,N))/(MAX(HIGH,N)-MIN(LOW,N))  
K = SMA(RSV,M1,1)  
D = SMA(K,M2,1)  
J = 3\*K-2\*D  

SMA(X,N,M)为加权移动平均: Y = (X\*M+Y\*(N-M))/N  
默认: N = 9、M1 = 3、M2 = 3

#### 说明
KD主要是研究最高价、最低价和收盘价之间的关系，同时综合了动量观念、强弱指标及移动平均线的优点，所以能够比较直观地研判行情，被广泛用于股市的中短期趋势分析中；K与D值永远介于0到100之间，D大于70时，行情呈现超买现象，D小于30时，行情呈现超卖现象。

### 2.1.2 异同离差乖离率  

BIAS = (CLOSE-SMA(CLOSE,N1))/SMA(CLOSE,N1)  
DIF = BIAS-BIAS\[N2\]  
DBCD = SMA(DIF,N3,1)  

SMA(X,N,M)为加权移动平均: Y = (X\*M+Y\*(N-M))/N  
默认: N1 = 5、N2 = 16、N3 = 17

#### 说明
DBCD的原理和构造方法与乖离率类似，用法也与乖离率相同；优点是能够保持指标的紧密同步，线条光滑，信号明确，能有效的过滤掉伪信号。


### 2.1.3 乖离率  

BIAS = 100*(CLOSE-SMA(CLOSE,N))/SMA(CLOSE,N)  
默认: N = 6、12、24等

#### 说明
BIAS通过计算股价在波动过程中与移动平均线出现的偏离程度来判断股价在剧烈波动时因偏离移动平均趋势可能形成的回档或反弹；正的乖离率越大，表示短期获利越大，获利回吐的可能性越高；负的乖离率越大，则空头回补的可能性越高。


### 2.1.4 动态买卖气指标  

DTM = IF(OPEN<OPEN\[1],0,MAX((HIGH-OPEN),(OPEN-OPEN\[1])))  
DBM = IF(OPEN≥OPEN\[1],0,MAX((OPEN-LOW), OPEN-OPEN\[1])))  
STM = SUM(DTM,N)  
SBM = SUM(DBM,N)  
ADTM = IF(STM>SBM,(STM-SBM)/STM,IF(STM=SBM,0, (STM-SBM)/SBM))  
ADTMMMA = SMA(ADTM,M)  
默认: N = 23、M = 8

#### 说明
该指标用开盘价的向上波动幅度和向下波动幅度的距离差值来描述人气高低；ADTM指标在+1到-1之间波动，低于-0.5时为低风险区，高于+0.5时为高风险区；ADTM上穿ADTMMMA时，买入股票；ADTM跌穿ADTMMMA时，卖出股票。


### 2.1.5 变动速率  

ROC = (CLOSE-CLOSE\[N])/CLOSE\[N]  
ROCMA = SMA(ROC,M)  
默认: N = 12、M = 6

#### 说明
反映价格变动的快慢程度；趋势明显的行情中，当ROC向下跌破零，卖出信号，ROC向上突破零，买入信号；震荡行情中，当ROC向下跌破ROCMA时，卖出信号，当ROC上穿ROCMA时，买入信号；股价创新高，ROC未配合上升，显示上涨动力减弱；股价创新低，ROC未配合下降，显示下跌动力减弱；股价与ROC从低位同时上升，短期反弹有望；股价与ROC从高位同时下降，警惕回落。


### 2.1.6 相对强弱指标  


UM = IF(CLOSE-CLOSE\[1]>0,CLOSE-CLOSE\[1],0)  
DM = IF(CLOSE-CLOSE\[1]<0,CLOSE\[1]-CLOSE,0)  
UA(N) = (UA\[1]\*(N-1)+UM)/N  
DA(N) = (DA\[1]\*(N-1)+DM)/N  
RSI = 100*(UA/(UA+DA))  
UA初始值 = SMA(UM,N)  
DA初始值 = SMA(DM,N)  
默认: N = 14

#### 说明
RSI指标多在30-70之间变动，通常80甚至90时被认为市场已到达超买状态，至此市场价格自然会回落调整。当价格低跌至30以下即被认为是超卖状态，市价将出现反弹回升。

### 2.1.7 货币流量指标  

TP = (HIGH+LOW+CLOSE)/3  
MF = TP\*VOL  
PF(N) = SUM(IF(TP>TP\[1],MF,0),N)  
NF(N) = SUM(IF(TP<=TP\[1],MF,0),N)  
MR(N) = PF(N)/NF(N)  
MFI = 100-(100/(1+MR))  
默认: N = 20

#### 说明
MFI指标是RSI扩展指标，MFI指标用成交金额代替的指数，是某一时间周期内上涨的成交量之和与下跌的成交量之和的比率。MFI>80为超买，当其回头向下跌破80时，为短线卖出时机。MFI<20为超卖，当其回头向上突破20时，为短线买入时机。


### 2.1.8 方向标准离差指标  

DMZ = IF(HIGH+LOW<=HIGH\[1]+LOW\[1],0,  
MAX(ABS(HIGH-HIGH\[1]),ABS(LOW-LOW\[1])))  
DMF = IF(HIGH+LOW>HIGH\[1]+LOW\[1],0,  
MAX(ABS(HIGH-HIGH\[1]),ABS(LOW-LOW\[1])))  
DIZ = SUM(DMZ,N)/(SUM(DMZ,N)+SUM(DMF,N))\*100  
DIF = SUM(DMF,N)/(SUM(DMZ,N)+SUM(DMF,N))\*100  
DDI = DIZ-DIF  
若分母为 0，则令 DDI = 0  
默认: N = 20

#### 说明
DDI指标，即方向标准离差指数，一般用于观察一段时间内股价相对于前一天向上波动和向下波动的比例，并对其进行移动平均分析。通过分析DDI柱状线，可以判断是买入信号还是卖出信号。


### 2.1.9 商品通道指标  

TP = (HIGH+LOW+CLOSE)/3  
CCI(N) = (TP-SMA(P,N))/(0.015\*MAD(TP,N))  
MAD 为求平均绝对离差  
默认: N = 20

#### 说明
CCI指标测量当前价格对近期平均价格的偏离程度。商品通道指标数值高则当前价格高于平均价格，反之亦然。作为超买超卖指标，商品通道指标能预测价格趋势的背离。



## 2.2 技术因子-动量反转型
### 2.2.1 人气指数  

AR(N) = SUM(HIGH-OPEN,N)/SUM(OPEN-LOW,N)\*100  
默认: N = 20

#### 说明
AR指标是反映市场当前情况下多空双方力量发展对比的结果。它是以当日的开盘价为基点，与当日最高价、最低价相比较，通过开盘价在股价中的地位反映市场买卖的人气。


### 2.2.2 心理线  

PSY = 100\*COUNT(CLOSE>CLOSE\[1],N)/N  
默认: N = 12

#### 说明
PSY将一定时期内投资者趋向买方或卖方的心理事实转化为数值，形成人气指数，从而判断股价的未来趋势；通常，指标小于25时关注做多机会，大于75时关注做空机会，小于10为极度超卖，大于90为极度超买，市场宽幅震荡时，PSY也会25～75区间反复突破，给出无效信号。

### 2.2.3 梅斯线  

MASSINDEX = EMA(HIGH-LOW,9)/EMA((EMA(HIGH-LOW,9),9)

#### 说明
Mass Index主要用于寻找飙涨股或者极度弱势股的重要趋势反转点；股价高低点之间的价差波带，忽而宽忽而窄，不断的重复循环变动，利用这种重复循环的波带，可以预测股价的趋势反转点。

### 2.2.4 随机动量指标
C(N) = (MAX(HIGH,N)+MAX(LOW,N))/2  
H = CLOSE-C(N)  
SH1 = EMA(H,N1)  
SH2 = EMA(SH1,N2)  
R = MAX(HIGH,N)-MAX(LOW,N)  
SR1 = EMA(R,N1)  
SR2 = EMA(SR1,N2)/2  
SMI = (SH2/SR2)\*100  
默认: N = 10、N1 = 3、N2 = 3

#### 说明
SMI随机动量指标基于股价变化与股价波动区间之间的联系，刻画出当前股价与近期股价波动的关系，对股价是否会反转或延续当前的趋势有一个基本的判读。

### 2.2.5 日内动量指标  

USUM(N) = SUM(IF(CLOSE>OPEN, CLOSE-OPEN,0),N)  
DSUM(N) = SUM(IF(CLOSE<=OPEN, OPEN-CLOSE,0),N)  
IMI(N) = USUM(N)/(USUM(N)+DSUM(N))\*100  
若分母为 0，则令 IMI = 100

#### 说明
IMI日内动量指标，通过计算过去一段周期内收盘价与开盘价的关系，来反应股票的买卖平衡。一般当IMI低于30，我们认为股票处于超卖状态；当IMI高于70，我们认为股票处于超买状态。

### 2.2.6 简易波动指标  

MM = ((HIGH-LOW)-(HIGH\[1]-LOW\[1]))/2  
BR = VOL/(HIGH-LOW)  
EMV = MM/BR

#### 说明
EMV指标是一个动量型指标，通过股价变动与成交量之间的关系去刻画股价的动量。当EMV大于0，说明股价有上升的动能；当EMV小于0，说明股价有下降的压力。


### 2.2.7 钱德动量摆动指标  

CZ1 = IF(CLOSE-CLOSE\[1]>0,CLOSE-CLOSE\[1],0)  
CZ2 = IF(CLOSE-CLOSE\[1]<0,ABS(CLOSE-CLOSE\[1]),0)  
SU(N) = SUM(CZ1,N)  
SD(N) = SUM(CZ2,N)  
CMO = (SU(N)-SD(N))/(SU(N)+SD(N))\*100  
默认: N = 20

#### 说明
CMO指标通过计算今收与昨收的价差来判断趋势的强弱，当CMO大于50时，处于超买状态；当CMO小于50时处于超卖状态。


### 2.2.8 阿隆指数  

AROON上升数 = (计算期天数-最高价后的天数)/计算期天数\*100  
AROON下降数 = (计算期天数-最低价后的天数)/计算期天数\*100  
AROON = AROON上升数 - AROON下降数  
默认: 计算期天数 N = 20

#### 说明
AROON指标计算自价格达到近期最高值和最低值以来所经过的期间数，帮助投资者预测证券价格趋势、强弱以及趋势的反转等。


### 2.2.9 买卖意愿指标  

BR(N) = SUM(MAX(0,HIGH-CLOSE\[1]),N)/SUM(MAX(0,CLOSE\[1]-LOW))  
默认: N = 20

#### 说明
BR指标是反映当前情况下多空双方力量争斗的结果。它以前一日的收盘价为基础，与当日的最高价、最低价相比较，通过昨日收盘价在股价中的地位反映市场买卖的人气。BR最好与AR结合使用，BR、AR均急跌，表明股价已到顶，反跌在即，投资者应尽快出货；BR比AR低，且AR<50，表明股价已经到底，投资者可吸纳低股；BR急速高升，而AR处在盘整或小跌时，表明股价正在上升；BR>AR，又转为BR<AR时，也可买进；BR攀至高峰，又以50的跌幅出现时，投资者也可低价进货，待股价升高再卖出。

## 2.3 技术因子-波动型

### 2.3.1 佳庆离散指标  

REM = EMA((HIGH-LOW),N)  
CV = 100*(REM-REM\[M])/REM\[M]  
默认: N = 10、M = 10

#### 说明
用于股价的波动情况；一个相对较短时间内的波动率上升意味着市场底部的到来，一段相对较长时间内波动率的下降意味着市场顶部的到来，可以根据波动率预测股票未来的趋势。


### 2.3.2 平均真实波幅  

TR = MAX(HIGH-LOW,ABS(HIGH-CLOSE\[1]),ABS(LOW-CLOSE\[1]))  
ATR = EMA(TR,N)  
默认: N = 20

#### 说明
用于表示价格的波动程度，价格波动幅度的突破通常也预示着价格的突破，该指标价值越高，趋势改变的可能性就越高；该指标的价值越低，趋势的移动性就越弱。

### 2.3.3 相对波动率指标  

UM(PRICE,N1) = IF(PRICE>PRICE\[1],STD(PRICE,N1),0)  
DM(PRICE,N1) = IF(PRICE<PRICE\[1],STD(PRICE,N1),0)  
UA(N2) = (UA\[1]\*(N-1)+UM)/N2  
DA(N2) = (DA\[1]\*(N-1)+DM)/N2  
RS(PRICE) = 100\*UA/(UA+DA)  
RVI = (RS(HIGH)+RS(LOW))/2  
UA初始值 = SMA(UM,N)  
DA初始值 = SMA(DM,N)  
若分母为 0，则令 RVI = 0  
默认: N = 5、N1 = 10、N2 = 20

#### 说明
RVI指标的计算方法与RSI类似，唯一的区别是RVI基于标准差，而RSI基于股价。RVI用于判断股价波动的方向，常与趋势型指标配合使用（如MA移动均线）。


### 2.3.4 区域指数  

TR = MAX(HIGH-LOW,ABS(CLOSE\[1]-HIGH),ABS(CLOSE\[1]-LOW))  
W = IF(CLOSE>CLOSE\[1],TR/(CLOSE-CLOSE\[1]),TR)  
SR(N1) = IF(MAX(W,N1)-MIN(W,N1)>0, (W-MIN(W,N1))/(MAX(W,N1)-MIN(W,N1))\*100, (W-MIN(W,N1))\*100)  
RI(N1,N2) = EMA(SR(N1),N2)  
默认: N1 = 20、N2 = 5

#### 说明
RI区域指标计算了期间内股票最高价到最低价的区域超过了期间之间收盘价到收盘价的区域的时间，可用于辨别趋势的开始与结束。


### 2.3.5 蔡金波动率指标  

CVI(N) = (EMA(HIGH-LOW,N)-EMA(HIGH-LOW,N)\[N])/EMA(HIGH-LOW,N)\*100  
默认: N = 20

#### 说明
CVI指标计算最高价与最低价的价差均值来衡量股价的波动率，与ATR不同的是CVI指标没有考虑周期间价格的跳空。在实际使用过程中，CVI指标结合用均线等其他趋势指标去增加趋势判断的准确率。


## 2.4 技术因子-趋势型

### 2.4.1 累计振动升降指标  

A = ABS(HIGH-CLOSE\[1])  
B = ABS(LOW-CLOSE\[1])  
C = ABS(HIGH-LOW\[1])  
D = ABS(CLOSE\[1]-OPEN\[1])  
E = CLOSE-CLOSE\[1]  
F = CLOSE-OPEN  
G = CLOSE\[1]-OPEN\[1]  
X = E+0.5F+G  
K = MAX(A,B)  
R = IF(A>B AND A>C, A+0.5B+0.25D, IF(B>A AND B>C, B+0.5A+0.25D, C+0.25D))  
SI = 16\*X/R\*K  
ASI(N) = SUM(SI,N)  
默认: N = 20

#### 说明
ASI累计振动升降指标通过比较过去一段时间股价开高低收的关系来判读股价的长期趋势。当ASI为正，说明趋势会继续；当ASI为负，说明趋势会终结。

### 2.4.2 多空指数  


BBI = (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4  
默认: M1 = 3、M2 = 6、M3 = 12、M4 = 20

#### 说明
BBI是一种将不同日数移动平均线加权平均之后的综合指标；计算BBI时，近期数据利用较多，远期数据利用次数较少，是一种变相的加权计算，既有短期移动平均线的灵敏，又有明显的中期趋势特征。


### 2.4.3 估波指标  


R(N1) = ((CLOSE-CLOSE\[N1])/CLOSE\[N1])\*100  
R(N2) = ((CLOSE-CLOSE\[N2])/CLOSE\[N2])\*100  
RC(N1,N2) = R(N1)+R(N2)  
COPPOCK(N1,N2,N3) = WMA(RC(N1,N2),N3)  
默认: N1 = 14、N2 = 11、N3 = 10

#### 说明
估波指标又称为“估波曲线”，通过计算月度价格的变化速率的加权平均值来测量市场的动量，属于长线指标。该指标适合在指数的月线图表中分析，当指标向上穿越零线，预示牛市来临，是中期买入信号，但其不适宜寻找卖出时机，需结合其它指标来进行分析。


### 2.4.4 成交量平滑异同均线指标  

SHORT = EMA(VOL,N1)  
LONG = EMA(VOL,N2)  
DIFF = SHORT-LONG  
DEA = EMA(DIFF,M)  
VMACD = DIFF-DEA  
默认: N1 = 12、N2 = 26、M = 9

#### 说明
VMACD指标与MACD指标的唯一区别是VMACD是基于成交量计算的，用来识别成交量的。


### 2.4.5 垂直水平过滤指标  

HCP = MAX(HIGH,N)  
LCP = MIN(LOW,N)  
A = ABS(HCP-LCP)  
B = SUM(ABS(CLOSE-CLOSE\[1]),N)  
VHF = A/B  
若分母为 0，则令 IMI = 0  
默认: N = 20

#### 说明
VHF指标用来判断当前行情处于趋势阶段还是震荡阶段，类似MACD等指标也能帮助识别行情趋势，但当行情处于盘整阶段时会经常发出错误信号，而RSI在震荡行情时能准确识别超买超卖的状态，但在趋势行情时总是发出错误信号。VHF指标通过识别趋势的强弱，来帮助投资者选择相应的指标（如MACD或RSI）。


### 2.4.6 三重指数移动平均指标  

EMA1 = EMA(REAL,N)  
EMA2 = EMA(EMA1,N)  
EMA3 = EMA(EMA2,N)  
TEMA = 3\*EMA1-3\*EMA2+EMA3  
默认: N = 5, REAL为任意序列，如CLOSE、VOL等

#### 说明
TEMA是对单一指数移动平均、双重指数移动平均和三重指数移动平均的综合，当存在趋势时，该指标的时滞与3个组成要素中的任何一个都要短。


### 2.4.7 平滑异同均线指标  

DIF = EMA(CLOSE,N1)-EMA(CLOSE,N2)  
DEA = EMA(DIF,N3)  
MACD = 2*(DIF-DEA)  
默认: N1 = 12、N2 = 26、N3 = 9

#### 说明
MACD称为指数平滑异同平均线，是从双指数移动平均线发展而来的，由快的指数移动平均线（EMA）减去慢的指数移动平均线。当MACD从负数转向正数，是买的信号；当MACD从正数转向负数，是卖的信号。

## 2.5 技术因子-成交量型
### 2.5.1 蔡金货币流量指标  

CLV = VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-CLOSE)  
CMF(N) = SUM(CLV,N)/SUM(VOL,N)\*100  
默认: N = 20

#### 说明
CMF指标基于这样的假设，即强势市场（处于上升趋势的市场）通常都伴随着位于日最高价与最低价之间上半部分的收盘价以及放大的成交量。与此相反，弱势市场（处于下跌趋势的市场）通常都伴随着位于日最高价与最低价之间的下半部分的收盘价以及放大的成交量。如果在成交量放大的同时，价格持续收于日最高价与最低价之间的上半部分，那么该指标将会是正值，表示该证券处于强势之中。相反，如果在成交量放大的同时，价格持续收于日最高价与最低价之间的下半部分，那么该指标将是负值，表示该证券处于弱势之中。

### 2.5.2 累积/派发线  

AD = SUM(VOL\*(2\*CLOSE-HIGH-LOW)/(HIGH+LOW),0)

#### 说明
AD指标将每日的成交量通过价格加权累计，用以计算成交量的动量；向上的AD表明买方占优势，而向下的AD表明卖方占优势，AD与价格的背离可视为买卖信号，即底背离考虑买入，顶背离考虑卖出。

### 2.5.3 佳庆指标

MID = SUM(VOL\*(2\*CLOSE-HIGH-LOW)/(HIGH+LOW),0)  
CHO = EMA(MID,N1)-EMA(MID,N2)  
默认: N1 = 10、N2 = 3

#### 说明
佳庆指标是对累积/派发线AD的改良；CHO曲线产生急促的凸起时，代表行情可能出现向上或向下反转；股价>90天平均线，CHO由负转正，买进参考；股价<90天平均线，CHO由正转负时，卖出参考。


### 2.5.4 成交量相对强弱指标  

U = IF(CLOSE>CLOSE\[1],VOL,IF(CLOSE=CLOSE\[1],VOL/2,0))  
D = IF(CLOSE<CLOSE\[1],VOL,IF(CLOSE=CLOSE\[1],VOL/2,0))  
UU = ((N-1)U\[1]+U)/N  
DD = ((N-1)D\[1]+D)/N  
VRSI = 100\*UU/(UU+DD)  
默认: N = 20

#### 说明
VRSI是市场成交量的相对强弱指标，通过动态分析成交量的变化，识破庄家的盘中对敲、虚假放量，虚假买卖盘等欺骗手段，从真实的量能变化中找出庄家的战略意图，从而达到安全跟庄、稳定获利的投资目标。

### 2.5.5 成交量比率指标  

A = IF(CLOSE>CLOSE\[1],VOL,0)  
B = IF(CLOSE<CLOSE\[1],VOL,0)  
VR = SUM(A,N)/SUM(B,N)\*100  
默认: N = 20

#### 说明
VR指标通过分析股价上升日成交额（或成交量，下同）与股价下降日成交额比值，从而掌握市场买卖气势的中期技术指标。主要用于个股分析，其理论基础是“量价同步”及“量须先于价”，以成交量的变化确认低价和高价，从而确定买卖时法。


### 2.5.6 价量趋势指标  

PVT = (CLOSE-CLOSE\[1])/CLOSE\[1]\*VOL + PVT\[1]  
PVT 初始值 = (CLOSE-CLOSE\[1])/CLOSE\[1]\*VOL

#### 说明
PVT指标与OBV指标类似，区别是计算PVT指标时，当收盘价大于前收盘价时只有部分成交量会累加到之前的PVT指标上。

### 2.5.7 能量指标  

OBV = OBV\[1]+IF(CLOSE>CLOSE\[1],VOL,IF(CLOSE<CLOSE\[1],-VOL,0))  
OBV 初始值为 0

#### 说明
OBV指标的理论基础是市场价格的有效变动必须有成交量配合，量是价的先行指标。利用OBV可以验证当前股价走势的可靠性，并可以得到趋势可能反转的信号。比起单独使用成交量来，OBV看得更清楚。


### 2.5.8 成交量摆动指标  

TR = IF(HIGH+LOW+CLOSE>HIGH\[1]+LOW\[1]+CLOSE\[1],1,-1)  
DM = HIGH-LOW  
CM = IF(TR=TR\[1],CM\[1]+DM,DM\[1]+DM)  
VF = VOL\*ABS(2\*(DM/CM-1))\*TR\*100  
KVO(N1,N2) = EMA(VF,N1)-EMA(VF,N2)  
默认: N1 = 34、N2 = 55

#### 说明
KVO指标的目的是为了观察短期和长期股票资金的流入和流出的情况。它的主要用途是确认股票价格趋势的方向和强度。用它来判断股价趋势方向，如果股价是上升趋势，它的摆动范围靠上（大于0的方向）；如果是下降趋势，它的摆动范围靠下（小于0的方向）。


# 3 另类因子
## 3.1 创新与专利类
### 3.1.1 审查期月数平均

$$
mean(\text{截止日前5年内新生成的有效的发明授权专利的审查期月数})
$$

#### 说明
对于高质量专利，其原创性、创新性较强，专利审查更细致，审查期可能更长；若审查时出现争议，对于高质宝贵的专利，企业也会坚持申请，双方拉锯也造成审查期较长；因此专利审核期越长，专利质量可能也越高。

#### 参考文献
郑兆磊, 2022, 专利研究系列四: 专利全解析, 兴业证券

### 3.1.2 独立权利要求项项数平均值

$$
mean(\text{截止日前5年内新生成的有效的发明公开审查中专利的独立权利要求项项数})
$$

#### 说明
独立权利要求项是指无需用其他权利要求来确定其范围和含义的完整权利要求；独立权利要求项越多的专利，其专利质量越高。

#### 参考文献
郑兆磊, 2022, 专利研究系列四: 专利全解析, 兴业证券

### 3.1.3 科技动量

基于各公司的专利分布向量计算各公司之间的科技关联度：

$$
TECH_{ijt} = \frac{T_{it}T'_{jt}}{(T_{it}T'_{it})^{1/2}(T_{jt}T'_{jt})^{1/2}}
$$

以科技关联度 $TECH_{ijt}$ 为权重，将与自身存在科技关联的公司的收益率进行加权，得到科技动量因子：

$$
TECHRET_{it} = \frac{\sum_{j \neq i} TECH_{ijt} \times RET_{jt}}{\sum_{j \neq i} TECH_{ijt}}
$$

其中：
1. $TECH_{ijt}$ 为公司 $i$ 和公司 $j$ 在第 $t$ 期的科技关联度，该公式中的 $T_{it}$ 和 $T_{jt}$ 为 $N$ 维的专利分布向量，$N$ 为专利商标局科技大类的数目，专利分布向量中的元素为公司 $i$ 在过去五年获取的各类科技专利在这 $N$ 类科技中的比例；
2. $RET_{jt}$ 为公司 $j$ 在 $t$ 期的收益率。

#### 说明
作者实证发现：目标公司的股票收益率与其科技关联度相近的公司前期的收益率之间有一种滞后-领先关系，以科技关联度为权重构建的科技动量因子对目标公司未来收益有预测性；一家公司科技的进步具有溢出效应，会影响到有科技关联的其他公司，改变这些关联公司的基本面，并最终反映到公司的股价中。

#### 参考文献
Lee, C. M. C., S. T. Sun, R. Wang, and R. Zhang, 2019, Technological Links and Return Predictability, Journal of Financial Economics 132(3), 76-96.

### 3.1.4 权利要求项项数平均值

$$
mean(\text{截止日前5年内新生成的有效的发明公开审查中专利的权利要求项项数})
$$

#### 说明
权利要求项是发明或者实用新型专利要求保护的内容，可以用于确定专利保护范围，具有直接的法律效力，是申请专利的核心；权利要求项越多的专利，其专利保护的项目也越多，表明专利的质量越高。

#### 参考文献
郑兆磊, 2022, 专利研究系列四: 专利全解析, 兴业证券

### 3.1.5 寿命月数平均值

$$
mean(\text{截止日前5年内新生成的有效的发明公开审查中专利的寿命月数})
$$

#### 说明
质量越高的专利，企业更有愿意维护这个专利，提升专利的寿命，使其发挥应有的价值；而低质量的专利，很难转化为企业效益，维护意义不大，企业也不会由于过度延长其寿命。

#### 参考文献
郑兆磊, 2022, 专利研究系列四: 专利全解析, 兴业证券

## 3.2 供应链与客户关系类
### 3.2.1 客户动量因子-基于销售占比

利用供应链的相关数据，计算如下客户动量因子：

$$
\text{cmom}_i^{1M} = \sum_{j=1}^{N_i} w_{ij}^{\text{sales}} \text{mom}_j^{1M}, i=1,2,\dots,N
$$

其中：
1. $\text{mom}_j^{1M}$ 为公司 $i$ 的客户 $j$ 过去一个月收益率；
2. $w_{ij}^{\text{sales}}$ 为销售占比。

#### 说明
客户股价变动会对供应商股价产生影响：在客户股价表现较好时，买入供应商；在客户股价表现较差时，卖出供应商，该类策略能获得高回报。可以从“投资者有限注意力”这个角度来解释客户动量。

#### 参考文献
Cohen, L., and A. Frazzini, 2008, Economic Links and Predictable Returns, Journal of Finance 63 (4), 1977-2011.

### 3.2.2 客户动量因子-基于单层客户关系

利用供应链的相关数据，结合图网络有关特征，计算如下客户动量因子：

$$
\text{cmom}_i^{1M} = \sum_{j=1}^{N_i} w_{ij}^{\text{centrality}} \text{mom}_j^{1M}, i=1,2,\dots,N
$$

其中：
1. $\text{mom}_j^{1M}$ 为公司 $i$ 的客户 $j$ 过去一个月收益率；
2. 权重 $w_{ij}^{\text{centrality}}$ 为：$w_{ij}^{\text{centrality}} = c_{ij} / \sum_{j=1}^{N_i} c_{ij}, i=1,2,\dots,N$；
3. 权重公式中的 $c_{ij}$ 为图网络中 $ij$ 边的中介中心度（即网络中经过某条边的最短路径的数量）。

#### 说明
常规的客户动量因子是以销售占比为权重，而该因子采用供应链图网络的边中介中心度 Edge Betweenness Centrality 作为权重，有效的避免了销售占比数据缺失问题，研究结果也表明 Edge Betweenness Centrality 与销售占比呈明显的正相关关系。

#### 参考文献
Hamuro, Y., and K. Okada, 2018, Predicting Stock Returns Based on the Time Lag in Information Diffusion through Supply Chain Networks, Special Interest Group on Financial Informatics Technical Reports 40-45, Japanese Society for Artificial Intelligence.

### 3.2.3 客户动量因子-基于多层客户关系

利用供应链的相关数据，结合图网络有关特征，计算如下客户动量因子：

$$
\text{cmom}_i^{TM} = \sum_{l=1}^{L} \sum_{(m,n) \in G_l} w_{mn}^{\text{centrality}} \text{mom}_n^{TM}, i=1,2,\dots,N
$$

$$
w_{mn}^{\text{centrality}} = c_{mn} / \sum_{l=1}^{L} \sum_{(m,n) \in G_l} c_{mn}
$$

其中：
1. $\text{mom}_j^{1M}$ 为 $i$ 的客户 $j$ 过去一个月收益率；
2. 权重 $w_{mn}^{\text{centrality}}$ 为考虑了多层（L层）客户关系的边中介中心度权重。

#### 说明
该因子同样采用了供应链图网络的边中介中心度 Edge Betweenness Centrality 作为权重，而且考虑了多层关系客户（客户的客户）；作者发现对比于只考虑直接的客户，客户的客户能带来更多的增量信息。

#### 参考文献
Yoshino, T., M. Morita, H. Tsuda, and T. Ohria, 2020, Customer Momentum Strategy: Empirical Study of the Japanese Stock Market, Securities Analysts Journal 58(5), 63-75;
Rei Yamamoto, Naoya Kawadai, and Hiroki Miyahara, 2021, Momentum Information Propagation through Global Supply Chain Networks, The Journal of Portfolio Management 47(8), 197-211.

### 3.2.4 供应链地位因子

$$
\text{客户集中度} = \frac{\text{前五大客户年度销售额}}{\text{公司年度销售总额}}
$$

$$
\text{供应商集中度} = \frac{\text{前五大供应商年度采购额}}{\text{公司年度采购总额}}
$$

$$
\text{占资水平} = \frac{\text{应付账款}+\text{预收账款}-\text{应收账款}-\text{预付账款}-\text{存货}}{\text{营业收入}}
$$

$$
\text{获现率} = \frac{\text{经营活动产生的现金净流量}}{\text{净利润}+\text{折旧摊销}+\text{财务费用}}
$$

构建上述4个细分因子，然后将这4个细分因子等权合成供应链地位因子。

#### 说明
测试结果表明：客户集中度因子与供应商集中度因子均为反向因子，表明市场对这认为上下游分散度较高公司，占资水平和获现率为正向因子；合成的供应链地位因子有较强的选股能力。

#### 参考文献
曹春晓, 2019, 上市公司供应链地位因子研究, 申万宏源

## 3.3 关联网络与另类数据类
### 3.3.1 基金关联网络牵引因子

涉及的股票关联网络的计算如下：
1. 选取是有权益持仓的公募基金，基金持仓数据为基金季报披露的前十大重仓股；
2. 取A基金最新季报中共同持有的股票 a 与股票 b 持仓市值，记作 $H\_a, H\_b$；
3. 计算 $H\_a/AMT\_a$，作为股票 a 的机构拥挤度 $I\_a$，股票 b 同理得 $I\_b$；
4. 定义A基金共同持仓股票 a 与股票 b 的关联度指标为：$J\_ab = \min(I\_a, I\_b)$；
5. 将所有共同持仓股票 a 与股票 b 的基金得到的关联度指标求和，得到股票 a 与股票 b 的关联度指标 $K\_ab$。

$$
Exp_{ave}^A = \frac{1}{N_A} \sum_{i=1}^{N_A} K_i^A * (chg_i - med)
$$

其中：
1. $N_A$ 与股票 A 一起被基金共同持有的关联股票数；
2. $K_i^A$ 为股票 A 与关联股票 i 的基金共同持仓关联度；
3. $chg_i$ 为关联股票 i 过去20日涨跌幅；
4. $med$ 为所有基金持仓股票过去20日涨跌幅截面中位数；
5. $chg_i - med$ 即为股票 A 的关联股票 i 的 Alpha 收益；
6. $K_i^A * (chg_i - med)$ 为股票 A 的关联股票 i 的 Alpha 锚定值；
7. 对所有股票的 $Exp_{ave}^A$ 做横截面回归，剔除自身 alpha 和行业因素，得到最终的因子。

#### 说明
基金共同持仓行为是股票关联关系的重要来源，由此构建股票关联网络来进一步刻画股票涨跌之间的牵引关系。股票 a 的关联股票的涨跌幅有锚定效应，若其关联股票普遍上涨，会提高市场对于股票 a 的涨幅预期；若本月其自身涨幅不高，则预期在次月出现补涨行情。

#### 参考文献
魏建榕, 王志豪, 2021, 从基金持仓行为到股票关联网络, 开源证券

### 3.3.2 消费指数变动

将每个公司一个月的销售额汇总得到每月的消费指数：

$$
SALESINDEX(i, t) = \sum_{\theta_t} DailyTotalSales(i, \theta_t)
$$

计算月度消费指数的年度同比变动：

$$
\Delta SALES(i, t) = \frac{SALESINDEX(i, t) - SALESINDEX(i, t-12)}{SALESINDEX(i, t-12)}
$$

其中：
1. $DailyTotalSales(i, \theta_t)$ 为消费板块上市公司 $i$ 的每日销售额；
2. $DailyAdjustmentIndex(i, \theta_t)$ 为消费板块上市公司 $i$ 的消费者数量调整指数。

#### 说明
消费数据与公司未来三个季度的盈收正相关，在非必需消费品板块中，这种关系不仅在大盘股中表现明显，在小盘股中表现的更加明显；与股票未来收益正相关。

#### 参考文献
Gupta Tarun, Leung Edward, Roscovan Viorel, 2022, Consumer Spending and The Cross-Section of Stock Returns, Journal of Portfolio Management 48 (7), 117-137.

### 3.3.3 新闻数量

$$
nc_{i,t} = \text{过去 } [t-d+1, t] \text{ 天提及过股票 } i \text{ 的新闻总量}
$$

#### 说明
新闻数量衡量了过去一段时间的新闻热度，负向因子，过去新闻热度越高，越容易使股价被高估，未来收益会变差。

#### 参考文献
从新闻情绪至新闻来源：多维度挖掘新闻个股因子, 2022.10 发布于 [数库Tech] 微信公众号

### 3.3.4 外资券商关联网络牵引因子

$$
Exp_{ave}^A = \frac{1}{N_A} \sum_{i=1}^{N_A} K_i^A * (chg_i - med)
$$

股票关联网络的计算：
1. 取外资券商A共同持有的股票a与股票b持仓市值数据，记作 $H\_a, H\_b$，以及相应的流通市值数据，记作 $MV\_a, MV\_b$；
2. 计算 $H\_a/MV\_a$，记作 $I\_a$，股票b记作 $I\_b$；
3. 定义外资券商A共同持仓股票a与股票b的关联强度指标为：$J\_ab = \min(I\_a, I\_b)$；
4. 将所有共同持仓股票a与股票b的外资券商得到的关联度指标求和，得到股票a与股票b的关联度指标 $K\_ab$。

其中：
1. $N_A$ 与股票A一起被外资券商共同持有的关联股票数；
2. $K_i^A$ 为股票A与关联股票 $i$ 的外资券商共同持仓关联度；
3. $chg_i$ 为关联股票 $i$ 过去20日涨跌幅；
4. $med$ 为所有外资券商持仓股票过去20日涨跌幅横截面中位数；
5. $chg_i - med$ 即为股票A的关联股票 $i$ 的 Alpha 收益；
6. $K_i^A * (chg_i - med)$ 为股票 A 的关联股票 $i$ 的 Alpha 锚定值；
7. 对所有股票的 $Exp_{ave}^A$ 做横截面回归，剔除自身 alpha 和行业因素，得到最终的因子。

#### 说明
被同一托管机构持仓的股票，其未来市场表现具有一定关联性，可以通过北向托管机构持仓行为来构建股票关联网络，进而刻画股票涨跌之间的牵引关系。

#### 参考文献
魏建榕, 王志豪, 2021, 北向关联持仓中的 Alpha, 开源证券

## 3.4 舆情与新闻类
### 3.4.1 隔夜新闻情绪

$$
senti_i^t = \frac{1}{N} \sum_{j=1}^{N} (pos_{i,j}^t - neg_{i,j}^t) * (1 - neu_{i,j}^t) * relevance_{i,j}^t
$$

其中：
1. 为计算 $t$ 交易日的隔夜情绪因子，选用的是 $[t-1 \text{日} 15:00, t \text{日} 9:25]$ 隔夜区间内的所有新闻情绪数据，新闻数量计为 $N$ 条；
2. $pos_{i,j}^t, neg_{i,j}^t, neu_{i,j}^t$ 分别股票 $i$ 在隔夜区间内第 $j$ 条新闻中属于正面、负面、中性情绪的概率；
3. $relevance_{i,j}^t$ 为股票与新闻的相关度。

#### 说明
隔夜情绪情绪因子衡量了股票收盘后到第二天收盘前这段时间内的情绪倾向，这段区间的新闻情绪还未被市场所反应，与未来的收益表现正相关，相关性也更强。

#### 参考文献
隔夜新闻情绪因子测试：如何有效结合动量因子？2022.10 发布于 [数库Tech] 微信公众号

### 3.4.2 Sentiment Beta (情绪贝塔)

$$
r_t^i = \alpha^i + \beta^i * (\frac{CSMS_t}{CSMS_{t-1}} - 1) + \epsilon_t^i
$$

$$
NewSentiBeta = -1 * abs(SentiBeta)
$$

将市场情绪指数变动与股票收益率进行时间序列回归，得到的 $\beta^i$ 即为 $SentiBeta$，再进一步计算得到 $NewSentiBeta$。

其中：
1. $r_t^i$ 为股票 $i$ 的日度收益率；
2. $CSMS_t$ 为市场情绪指数数值。

#### 说明
市场情绪指数对于大盘行情有一定的预测作用，通过时序回归得到的 SentiBeta 刻画了市场情绪对个股收益的影响。股票收益与市场情绪影响 SentiBeta 的绝对大小有关，越容易受市场情绪影响（即 NewSentiBeta 越小）的股票收益越小；越不容易受市场情绪影响（即 NewSentiBeta 越大）的股票收益越大。

#### 参考文献
Sentiment Beta: Risk or Alpha? 2021.05.10 发布于 [量化投资与机器学习]

### 3.4.3 新闻动量

$$
\text{将上市公司过去一个月内有新闻的交易日的收益率综合计算得到当月新闻动量因子}
$$

其中：
1. $w_{ij,t}$ 为 $t$ 交易日内，股票 $i$ 和股票 $j$ 共同出现过的那此新闻的数量；
2. $aw_{ij,t}$ 为 $t$ 交易日共现新闻数量的环比变动。

#### 说明
投资者是有限理性的，他们对信息的认知能力有限，进而导致市场反应不足；金融分析师也会根据新闻信息调整个股的收益预测，但这个过程通常会存在几天的时间差，由这些信息导致的股价变动也存在一定程度上的时间滞后，成为新闻动量效应的驱动力之一。

#### 参考文献
胡骧聪, 郑文才等, 2022, 量化投资新趋势(3): 驶向另类数据的信息蓝海, 中金公司

### 3.4.4 新闻股票共现

$$
w_{ij,t}^* = w_{ij,t} / w_{ii,t}
$$

$$
aw_{ij,t} = w_{ij,t}^* - w_{ij,t-1}^*
$$

$$
nco = \sum_{j=1}^N aw_{ij,t}^s
$$

其中：
1. $w_{ij,t}$ 为 $t$ 交易日内，股票 $i$ 和股票 $j$ 共同出现过的那此新闻的数量；
2. $aw_{ij,t}$ 为 $t$ 交易日共现新闻数量的环比变动。

#### 说明
当一新闻中提到多只股票时，投资者对单只股票的注意力就会转移到同时提到的其它股票上，从而增加了对所有提到的股票的关注。基于新闻网络的注意力溢出，在存在卖空限制的情况下，可能会导致对好消息的强烈反应，这反过来会导致估值过高，未来表现变差。

#### 参考文献
Li Guo, Lin Peng, Yubo Tao, Jun Tu, 2019, News Co-Occurrence, Attention Spillover and Return Predictability, Journal of Risk and Financial Management.

### 3.4.5 新闻股票行业共现

$$
nco_{i,t} = a_{i,t} - a_{i,t-1}
$$

其中：
1. $a_{i,t}$ 为 $t$ 区间内（如一个天、一个月），与股票 $i$ 共同出现在新闻中的行业数量；
2. 共现行业数量的环比变动即为股票行业共现因子。

#### 说明
在新闻中共同出现的行业和公司，存在跨行业或跨公司的关联关系，及显著的收益可预测性，在A股市场，股票行业共现因子与未来收益负相关。

#### 参考文献
Cohen, L. and Frazzini, A., 2008, Economic links and predictable returns, The Journal of Finance, 63 (4), 1977-2011.

## 3.5 行为与流量类
### 3.5.1 员工净流出因子

$$
\text{员工净流出} = \frac{\text{月末员工总流出数} - \text{月末员工总流入数}}{\text{月初总员工数}}
$$

作者利用 LinkedIn 中上市公司员工的简历数据，构建了上述月度员工净流出因子。

#### 说明
公司员工的流转信息对于股票的定价包含了其他常用因素中没有考虑到的信息。潜在的员工在加入公司前，一般会从多个渠道了解公司的信息并以此更新薪酬的期望；在职的员工，也会根据公司的经营状况选择是继续或离开当前所在的公式；所以员工净流出的因子能够从一定程度上反映公司的经营状况。而且员工总流出与股票收益之间的关系比员工总流入与股票收益之间的关系更为显著。

#### 参考文献
Ashwini Agrawal, Isaac Hacamo, Zhongchen Hu, Information Dispersion across Employees and Stock Returns 34(10), 2021.

### 3.5.2 自选股占比

$$
\text{自选股占比} = \frac{\text{个股自选股用户数}}{\text{自选股用户数合计}}
$$

#### 说明
自选股占比是投资者对个股内生关注度的度量；自选股票池在很大程度上代表了投资者对上市公司的“主动关注”，而非“被动接收”，是投资者偏好的内在反映，与股票未来收益负相关。

#### 参考文献
魏建榕, 胡亮勇, 2022, 自选股与点击量: 投资者关注度的选股能力, 开源证券

### 3.5.3 新闻点击量占比

$$
\text{个股的新闻点击量} / \text{A股所有上市公司新闻点击量合计值}
$$

#### 说明
新闻点击量占比是投资者对个股外驱关注度的度量；在浩如烟海的新闻信息中，公司相关新闻信息曝光度更高，更容易受到投资者的关注，被投资者点击的概率更大，与股票未来收益负相关。

#### 参考文献
魏建榕, 胡亮勇, 2022, 自选股与点击量: 投资者关注度的选股能力, 开源证券

### 3.5.4 个股点击量占比

$$
\text{个股点击量占比} = \frac{\text{个股当天点击量}}{\text{全市场所有个股当天点击量之和}}
$$

#### 说明
个股点击量占比是投资者对个股关注度的综合映射；当前受关注程度越高的个股通常具有更高的点击次数，比与股票未来收益负相关。

#### 参考文献
魏建榕, 胡亮勇, 2022, 自选股与点击量: 投资者关注度的选股能力, 开源证券

# 4 财务质量因子
## 4.1 每股指标

### 4.1.1 每股资本公积金

$$
\frac{\text{最近报告期资本公积金}}{\text{最近同期总股本}}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.2 每股未分配利润

$$
\frac{\text{最近报告期未分配利润总额}}{\text{最近同期总股本}}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.3 每股留存收益

$$
\frac{\text{最近报告期留存收益}}{\text{最近同期总股本}}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.4 每股货币资金

$$
\frac{\text{最近报告期货币资金}}{\text{最近同期总股本}}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.5 每股收益（摊薄）

$$
\frac{\text{最近12个月归属母公司净利润}(TTM)}{\text{平均总股本}}
$$

其中：
$$
\text{平均总股本} = \frac{\text{期初总股本} + \text{期末总股本}}{2}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.6 每股收益（扣除+摊薄）

$$
\frac{\text{最近12个月扣除非经常损益的净利润}(TTM)}{\text{平均总股本}}
$$

其中：
$$
\text{平均总股本} = \frac{\text{期初总股本} + \text{期末总股本}}{2}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.7 每股营业收入

$$
\frac{\text{最近12个月营业收入}(TTM)}{\text{平均总股本}}
$$

其中：
$$
\text{平均总股本} = \frac{\text{期初总股本} + \text{期末总股本}}{2}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.8 每股经营现金流

$$
\frac{\text{最近12个月经营活动产生的现金流量净额}(TTM)}{\text{平均总股本}}
$$

其中：
$$
\text{平均总股本} = \frac{\text{期初总股本} + \text{期末总股本}}{2}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.9 每股企业自由现金流

$$
\frac{\text{最近12个月企业自由现金流量}(TTM)}{\text{平均总股本}}
$$

其中：
$$
\text{平均总股本} = \frac{\text{期初总股本} + \text{期末总股本}}{2}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.10 每股净资产

$$
\frac{\text{最近报告期归属母公司股东权益合计}}{\text{最近同期总股本}}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

### 4.1.11 每股盈余公积金

$$
\frac{\text{最近报告期盈余公积金}}{\text{最近同期总股本}}
$$

#### 说明
每股指标通常由“基础财务指标/总股本”得到，衡量的是企业普通股股东每持有单位股票所享有的公司经营成果或是所承担的费用风险，可用于反映投资者购买股票的获利程度。

#### 参考文献
常规通用财务指标

## 4.2 盈利能力

### 4.2.1 净资产收益率 (ROE)

$$
\frac{\text{最近12个月归属母公司净利润}(TTM)}{\text{平均净资产}}
$$

其中：
$$
\text{平均净资产} = \frac{\text{期初归属母公司股东权益合计} + \text{期末归属母公司股东权益合计}}{2}
$$

#### 说明
反映股东权益的收益水平，用以衡量公司运用自有资本的效率，体现了自有资本获得净收益的能力；指标值越高，说明投资带来的收益越高；使用的是ROE的平均算法，反映的是报告期内单位净资产创造收益的平均水平。

#### 参考文献
Haugen, R., and N. Baker, 1996, Commonality in the determinants of expected stock returns. Journal of Financial Economics 41, 401-39.

### 4.2.2 摊薄净资产收益率 (Diluted ROE)

$$
\frac{\text{最近12个月归属母公司净利润}(TTM)}{\text{期末归属母公司股东权益合计}}
$$

#### 说明
反映股东权益的收益水平，用以衡量公司运用自有资本的效率，体现了自有资本获得净收益的能力；指标值越高，说明投资带来的收益越高；使用的是ROE的摊薄算法，反映的是期末单位净资产所创造的净利润。

#### 参考文献
Haugen, R., and N. Baker, 1996, Commonality in the determinants of expected stock returns. Journal of Financial Economics 41, 401-39.

### 4.2.3 单季度销售成本率

$$
\frac{\text{最近单季度营业成本}}{\text{同期单季度营业收入}}
$$

#### 说明
衡量公司为取得单位收入所要花费的成本，销售成本率越低，产品越容易销售，反映公司单季度盈利能力。

#### 参考文献
常规通用财务指标

### 4.2.4 所得税费用率

$$
\frac{\text{最近12个月所得税}(TTM)}{\text{最近12个月利润总额}(TTM)}
$$

#### 说明
反映公司的税负水平，体现企业的收益质量。

#### 参考文献
常规通用财务指标

### 4.2.5 盈利现金比率

$$
\frac{\text{最近12个月经营性活动产生的现金流净额}(TTM)}{\text{最近12个月净利润}(TTM)}
$$

#### 说明
反映企业净利润的收现情况，比例越大，盈利质量越高；当比率小于1，说明企业本期净利润中尚存在没有实现的现金收入，在这种情况下，即使企业盈利，也可能发生现金短缺，严重时会导致破产。

#### 参考文献
常规通用财务指标

### 4.2.6 销售现金比率

$$
\frac{\text{最近12个月经营性活动产生的现金流净额}(TTM)}{\text{最近12个月营业收入}(TTM)}
$$

#### 说明
反映企业营业收入的收现情况，比例越大，销售质量越高。

#### 参考文献
常规通用财务指标

### 4.2.7 全部资产现金回收率

$$
\frac{\text{最近12个月的经营性活动产生的现金流净额}(TTM)}{\text{平均总资产}}
$$

其中：
$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

#### 说明
用于衡量企业全部资产产生现金的能力，比值越高，说明企业利用资产创造的现金流入越多，资产的现金回收周期越短，企业获取现金能力越强。

#### 参考文献
Bouchaud, Jean-Philippe, Philipp Krueger, Augustin Landier, David Thesmar, 2019, Sticky Expectations and the Profitability Anomaly, Journal of Finance 74 (2), 639-674.

### 4.2.8 营业费用率

$$
\frac{\text{最近12个月销售费用}(TTM)}{\text{最近12个月营业收入}(TTM)}
$$

#### 说明
从事营业活动所需花费销售费用在总营业收入中的比重，该指标越低，说明营业过程中的销售费用支出越小，获利水平越高。

#### 参考文献
常规通用财务指标

### 4.2.9 财务费用率

$$
\frac{\text{最近12个月财务费用}(TTM)}{\text{最近12个月营业收入}(TTM)}
$$

#### 说明
衡量公司为取得单位收入所要花费的筹资费用，比率较高，说明公司财务负担较重，筹资成本较高。

#### 参考文献
常规通用财务指标

### 4.2.10 毛利率

$$
\frac{\text{最近12个月营业收入}(TTM) - \text{最近12个月营业成本}(TTM)}{\text{最近12个月营业收入}(TTM)}
$$

#### 说明
毛利率越高，意味着企业有可能存在较强盈利能力和成本控制能力，也可能意味着企业正处于新兴产业、市场竞争少，或者意味着企业具有某种壁垒核心竞争力。

#### 参考文献
Abarbanell, Jeffery S., and Brian J. Bushee, 1998, Abnormal returns to a fundamental analysis strategy, Accounting Review 73, 19-45.

### 4.2.11 净利率

$$
\frac{\text{最近12个月净利润}(TTM)}{\text{最近12个月营业收入}(TTM)}
$$

#### 说明
反映企业营业收入的收益水平，净利率越大，说明从单位收入中获得净利润的水平越高，企业的盈利能力越强。注：通常杜邦分析中的销售净利率=归属母公司股东的净利润/营业总收入。

#### 参考文献
Soliman, M, 2008, The use of DuPont analysis by market participants, Accounting Review 83, 823-53.

### 4.2.12 毛利润与净利润之比

$$
\frac{\text{最近12个月营业收入}(TTM) - \text{最近12个月营业成本}(TTM)}{\text{最近12个月净利润}(TTM)}
$$

#### 说明
通过税前毛利润和税后净利润之比，来衡量公司的税收基本面情况，该比值对公司未来盈利增长有正向预测能力。

#### 参考文献
Soliman, M, 2008, The use of DuPont analysis by market participants, Accounting Review 83, 823-53.

### 4.2.13 营业利润率

$$
\frac{\text{最近12个月营业利润}(TTM)}{\text{最近12个月营业收入}(TTM)}
$$

#### 说明
营业利润率相比毛利率，在计算利润时，剔除了营业费用等在营业收入中的占比，除了能够衡量公司盈利能力外，还能体现公司管理层的管理能力。注：通常杜邦分析中的经营利润率=EBIT/营业总收入。

#### 参考文献
Soliman, M, 2008, The use of DuPont analysis by market participants, Accounting Review 83, 823-53.

### 4.2.14 总营业费用率

$$
\frac{\text{最近12个月销售费用}(TTM) + \text{最近12个月管理费用}(TTM)}{\text{最近12个月营业收入}(TTM)}
$$

#### 说明
从事营业活动所需花非营业费用在总营业收入中的比重，该项指标越低，说明营业过程中的营业费用支出越小，获利水平越高。

#### 参考文献
常规通用财务指标

### 4.2.15 成本费用利润率

$$
\frac{\text{最近12个月净利润}(TTM)}{\text{最近12个月成本费用}(TTM)}
$$

其中：
$$
\text{成本费用} = \text{营业成本} + \text{销售费用} + \text{管理费用} + \text{财务费用}
$$

#### 说明
表明每付出1元成本费用可获得多少利润，体现了经营耗费所带来的经营成果；该项指标越高，利润越大，反映企业的经济效益越好。

#### 参考文献
常规通用财务指标

### 4.2.16 总资产毛利率

$$
\frac{\text{最近12个月营业收入}(TTM) - \text{最近12个月营业成本}(TTM)}{\text{平均总资产}}
$$

其中：
$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

#### 说明
反映公司盈利能力，即公司每占用1元的资产平均能获得多少元的毛利；指标取值越高，资产运营越有效。

#### 参考文献
Novy-Marx, Robert, 2013, The other side of value: The gross profitability premium, Journal of Financial Economics 108, 1-28.

## 4.3 投资回报类

### 4.3.1 普通股回报率

$$
\frac{\text{最近12个月净利润}(TTM) - \text{最近12个月优先股股利}(TTM)}{\text{平均普通股总股本}}
$$

其中：
$$
\text{平均总股本} = \frac{\text{本期普通股总股本} + \text{上年同期普通股总股本}}{2}
$$

$$
\text{优先股股利用 0 代替}
$$

#### 说明
衡量公司普通股资本的盈利能力。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 4.3.2 投入资本回报率 (ROIC)

$$
\frac{\text{最近12个月扣非后EBIT}(TTM) * (1 - \text{税率} 0.25)}{\text{期末资本总投入}}
$$

其中：
$$
\text{扣非后EBIT} = \text{营业利润} + \text{利息费用}
$$

$$
\text{资本总投入} = \text{股东权益合计} + \text{有息负债}
$$

$$
\text{有息负债} = \text{短期借款} + \text{长期借款} + \text{应付债券} + \text{一年内到期的非流动负债}
$$

#### 说明
用于衡量生产经营活动中所有投入资本赚取的收益，投入资本中既包含股东权益，也包含债权，能较为全面的反映企业的盈利能力。

#### 参考文献
常规通用财务指标

### 4.3.3 单季度投入资本回报率

$$
\frac{\text{最近单季度扣非后EBIT}(TTM) * (1 - \text{税率} 0.25)}{\text{期末资本总投入}}
$$

其中：
$$
\text{扣非后EBIT} = \text{营业利润} + \text{利息费用}
$$

$$
\text{资本总投入} = \text{股东权益合计} + \text{有息负债}
$$

$$
\text{有息负债} = \text{短期借款} + \text{长期借款} + \text{应付债券} + \text{一年内到期的非流动负债}
$$

#### 说明
在单季度时间区间上统计的投入资本回报率，反映公司单季度盈利能力。

#### 参考文献
常规通用财务指标

### 4.3.4 有形资本回报率

$$
\frac{\text{最近12个月息税前利润}(TTM)}{\text{期末有形资本}}
$$

其中：
$$
\text{有形资本} = \text{净营运资本} + \text{净固定资本}
$$

$$
\text{净营运资本} \approx \text{流动资产} - \text{流动负债}
$$

$$
\text{净固定资本} \approx \text{固定资产}
$$

#### 说明
有形资本是日常经营活动所需的资本投入，衡量有形资本的收益回报程度。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 4.3.5 扣非净资产收益率

$$
\frac{\text{最近12个月扣除非经常损益后归属母公司净利润}(TTM)}{\text{平均净资产}}
$$

其中：
$$
\text{平均净资产} = \frac{\text{期初归属母公司股东权益合计} + \text{期末归属母公司股东权益合计}}{2}
$$

#### 说明
反映股东权益的收益水平，用以衡量公司运用自有资本的效率，体现了自有资本获得净收益的能力；指标值越高，说明投资带来的收益越高；使用的是ROE的平均算法，反映的是报告期内单位净资产创造收益的平均水平；此处的利润扣除了与公司正常经营业务无直接关系那部分损益。

#### 参考文献
Haugen, R., and N. Baker, 1996, Commonality in the determinants of expected stock returns. Journal of Financial Economics 41, 401-39.

### 4.3.6 摊薄扣非净资产收益率

$$
\frac{\text{最近12个月归属母公司净利润}(TTM)}{\text{期末归属母公司股东权益合计}}
$$

#### 说明
反映股东权益的收益水平，用以衡量公司运用自有资本的效率，体现了自有资本获得净收益的能力；指标值越高，说明投资带来的收益越高；使用的是ROE的摊薄算法，反映的是期末单位净资产所创造的净利润。

#### 参考文献
Haugen, R., and N. Baker, 1996, Commonality in the determinants of expected stock returns. Journal of Financial Economics 41, 401-39.

### 4.3.7 净经营资产收益率

$$
\frac{\text{最近12个月的经营利润}(TTM)}{\text{平均净经营资产}}
$$

其中：
$$
\text{平均净经营资产} = \frac{\text{期初净经营资产} + \text{期末净经营资产}}{2}
$$

$$
\text{经营利润} = \text{净利润(含少数股东损益)} - \text{非经常性损益} + (\text{财务费用} - \text{利息净收入}) * (1 - 25\%)
$$

$$
\text{净经营资产} = \text{股东权益合计(含少数股东权益)} + \text{金融负债} - \text{金融资产} = \text{经营性资产} - \text{经营性负债}
$$

#### 说明
将计算传统ROE时的净利润和净资产拆分成经营活动部分和金融活动部分，只使用纯经营性业务对应的净利润和净资产来计算RNOA，剔除了金融活动的影响，更客观反映了企业的真实的盈利能力。

#### 参考文献
常规通用财务指标

### 4.3.8 总资产净利率

$$
\frac{\text{最近12个月净利润}(TTM)}{\text{平均总资产}}
$$

其中：
$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

#### 说明
反映公司总资产的收益水平，在衡量收益水平时，分母同时考虑了股东权益和负债，有利于避免举债经营的企业可能出现的ROE虚高的情况。

#### 参考文献
Balakrishnan, Karthik, Eli Bartov, and Lucile Faurel, 2010, Post loss/profit announcement drift, Journal of Accounting and Economics 50, 20-41.

### 4.3.9 总资产报酬率

$$
\frac{\text{最近12个月息税前利润}(TTM)}{\text{平均总资产}}
$$

其中：
$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

#### 说明
反映企业运用全部资产的总体获利能力和投入产出状况，取值越高，表明企业投入产出的水平越好，企业的资产运营越有效。

#### 参考文献
常规通用财务指标

### 4.3.10 总资产营业利润

$$
\frac{\text{最近12个月营业利润}(TTM)}{\text{平均总资产}}
$$

其中：
$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

#### 说明
反映股东权益的收益水平，用以衡量公司运用自有资本的效率，体现了自有资本获得净收益的能力；指标值越高，说明投资带来的收益越高；使用的是ROE的平均算法，反映的是报告期内单位净资产创造收益的平均水平。

#### 参考文献
Haugen, R., and N. Baker, 1996, Commonality in the determinants of expected stock returns. Journal of Financial Economics 41, 401-39.

### 4.3.11 净资产营业利润

$$
\frac{\text{最近12个月营业利润}(TTM)}{\text{平均净资产}}
$$

其中：
$$
\text{平均净资产} = \frac{\text{期初归属母公司股东权益合计} + \text{期末归属母公司股东权益合计}}{2}
$$

#### 说明
Fama-French 五因子模型中的盈利因子代理指标，用于衡量公司的盈利能力。

#### 参考文献
Fama, Eugene F., and Kenneth R. French, 2015, A five-factor asset pricing model, Journal of Financial Economics 116, 1-22.

## 4.4 盈余质量

### 4.4.1 非操纵性总应计盈余

分行业分年度进行如下截面回归：

$$
\frac{TA_{i,t}}{A_{i,t-1}} = \alpha_{i,1} \frac{1}{A_{i,t-1}} + \alpha_{i,2} \frac{\Delta REV_{i,t}}{A_{i,t-1}} + \alpha_{i,3} \frac{PPE_{i,t}}{A_{i,t-1}} + \epsilon_{i,t}
$$

将上述回归得到的回归系数代入下面的方程，即得到非操纵性总应计盈余：

$$
NDTAC_{i,t} = \alpha_{i,1} \frac{1}{A_{i,t-1}} + \alpha_{i,2} \frac{\Delta REV_{i,t} - \Delta REC_{i,t}}{A_{i,t-1}} + \alpha_{i,3} \frac{PPE_{i,t}}{A_{i,t-1}}
$$

其中：
1. $TA_{i,t}$ 为股票 $i$ 在 $t$ 期的总应计盈余，计算细节见相关页；
2. $A_{i,t-1}$ 为股票 $i$ 在 $t-1$ 期的总资产，用于标准化各财务指标；
3. $\Delta REV_{i,t}$ 为股票 $i$ 在 $t$ 期相对 $t-1$ 期的营业收入增加额；
4. $\Delta REC_{i,t}$ 为股票 $i$ 在 $t$ 期相对 $t-1$ 期的应收账款增加额；
5. $PPE_{i,t}$ 为股票 $i$ 在 $t$ 期的期末固定资产总额。

#### 说明
上述计算方法对应的是修正后Jones模型，只需去掉应收账款项就能得到原始的Jones模型；总应计盈余可以分为非操纵性总应计盈余和操纵性总应计盈余2部分，其中的非操纵性应计盈余对应的就是企业无法调整或操纵的那部分应计项目。

#### 参考文献
Dechow P., Sloan R., Sweeney A, 1995, Detecting Earnings Management, The Accounting Review 70(2), 193-225.
Jones J., 1991, Earnings Management During Import Relief Investigations. Journal of Accounting Research 29(2), 193-228.

### 4.4.2 盈余质量 (Accruals)

$$
Accruals = \frac{(\Delta CA - \Delta Cash) - (\Delta CL - \Delta STD - \Delta TP) - Dep}{AverageTotalAssets}
$$

其中：
1. $\Delta CA - \Delta Cash$ 对应流动资产部分；
2. $\Delta CL - \Delta STD - \Delta TP$ 对应流动负债部分；
3. $\Delta CA$ 为流动资产的变动、$\Delta Cash$ 为现金及现金等价物净增加额；
4. $\Delta CL$ 为流动负债的变动、$\Delta STD$ 为流动负债中的借款、$\Delta TP$ 为应付税款的变动；
5. $Dep$ 为折旧与摊销；
6. $AverageTotalAssets$ 为期初总资产和期末总资产的平均值。

#### 说明
从资产负债维度计算的总应计盈余，进而衡量公司盈余管理的程度；可靠性较低的应计盈余会导致较低的盈余持久性，投资者没有完全预期较低的盈余持久性，从而导致重大的证券错误定价。

#### 参考文献
Dechow P., Sloan R., Sweeney A, 1995, Detecting Earnings Management, The Accounting Review 70(2), 193-225.
Sloan R., 1996, Do Stock Prices Fully Reflect Information in Accruals and Cash Flows about Future Earnings?, The Accounting Review 71(3), 289-315.

### 4.4.3 总应计盈余（常规版）

$$
Accruals = \frac{(\Delta CA - \Delta Cash) - (\Delta CL - \Delta STD - \Delta TP) - Dep}{AverageTotalAssets}
$$

其中：
1. $\Delta CA - \Delta Cash$ 对应流动资产部分、$\Delta CL - \Delta STD - \Delta TP$ 对应流动负债部分；
2. $\Delta CA$ 流动资产的变动、$\Delta Cash$ 为现金及现金等价物净增加额；
3. $\Delta CL$ 为流动负债的变动、$\Delta STD$ 为流动负债中的借款、$\Delta TP$ 为应付税款的变动；
4. $Dep$ 为折旧与摊销；
5. $AverageTotalAssets$ 为期初总资产和期末总资产的平均值。

#### 说明
从资产负债维度计算的总应计盈余，进而衡量公司盈余管理的程度；可靠性较低的应计盈余会导致较低的盈余持久性，投资者没有完全预期较低的盈余持久性，从而导致重大的证券错误定价。

#### 参考文献
Dechow P., Sloan R., Sweeney A, 1995, Detecting Earnings Management, The Accounting Review 70(2), 193-225.
Sloan R., 1996, Do Stock Prices Fully Reflect Information in Accruals and Cash Flows about Future Earnings?, The Accounting Review 71(3), 289-315.

### 4.4.4 操纵性总应计盈余

分行业分年度进行如下截面回归：

$$
\frac{TA_{i,t}}{A_{i,t-1}} = \alpha_{i,1} \frac{1}{A_{i,t-1}} + \alpha_{i,2} \frac{\Delta REV_{i,t}}{A_{i,t-1}} + \alpha_{i,3} \frac{PPE_{i,t}}{A_{i,t-1}} + \epsilon_{i,t}
$$

将上述回归得到的回归系数代入下面的方程，即得到非操纵性总应计盈余：

$$
NDTAC_{i,t} = \alpha_{i,1} \frac{1}{A_{i,t-1}} + \alpha_{i,2} \frac{\Delta REV_{i,t} - \Delta REC_{i,t}}{A_{i,t-1}} + \alpha_{i,3} \frac{PPE_{i,t}}{A_{i,t-1}}
$$

将非操纵性总应计盈余从总应计盈余中扣除：

$$
DTAC_{i,t} = \frac{TA_{i,t}}{A_{i,t-1}} - NDTAC_{i,t}
$$

其中：
1. $TA_{i,t}$ 为股票 $i$ 在 $t$ 期的总应计盈余，计算细节见相关页；
2. $A_{i,t-1}$ 股票 $i$ 在 $t-1$ 期的总资产，用于标准化各财务指标；
3. $\Delta REV_{i,t}$ 为股票 $i$ 在 $t$ 期相对 $t-1$ 期的营业收入增加额；
4. $\Delta REC_{i,t}$ 为股票 $i$ 在 $t$ 期相对 $t-1$ 期的应收账款增加额；
5. $PPE_{i,t}$ 为股票 $i$ 在 $t$ 期的期末固定资产总额。

#### 说明
上述计算方法对应的是修正后Jones模型，只需去掉应收账款项就能得到原始的Jones模型；总应计盈余可以分为非操纵性总应计盈余和操纵性总应计盈余2部分，其中的操纵性应计盈余对应的就是企业出于某种动机而进行的盈余管理，其与股票未来收益负相关。

#### 参考文献
Dechow P., Sloan R., Sweeney A, 1995, Detecting Earnings Management, The Accounting Review 70(2), 193-225.
Jones J., 1991, Earnings Management During Import Relief Investigations. Journal of Accounting Research 29(2), 193-228.

### 4.4.5 盈余持续性

$$
X_{j,t} = \phi_{0,j} + \phi_{1,j}X_{j,t-1} + v_{j,t}
$$

其中：
1. 对年度收益进行如上的一阶自回归AR(1)，$\phi_{1,j}$ 即为盈余持续性衡量指标；
2. 年度收益 $X_{j,t}$ 可以取每股收益、净利润等指标；
3. 自回归滚动窗口可以取10个财年。

#### 说明
$\phi_{1,j}$ 取值越高，盈余持续性越强，如果 $\phi_{1,j}$ 接近1，意味着企业有高度持续的盈余；如果 $\phi_{1,j}$ 接近0，意味着企业的收益短暂而不可持续。

#### 参考文献
Francis, Jennifer, Ryan LaFond, Per M. Olsson, Katherine Schipper, 2004, Cost of equity and earnings attributes, Accounting Review 79, 967-1010.

### 4.4.6 盈余可预测性

$$
X_{j,t} = \phi_{0,j} + \phi_{1,j}X_{j,t-1} + v_{j,t}
$$

$$
Predictability = \sqrt{\hat{\sigma}^2(v_j)}
$$

其中：
1. 对年度收益进行一阶自回归AR(1)，回归的误差方差的平方根即为可预测性衡量指标；
2. 年度收益 $X_{j,t}$ 可以取每股收益、净利润等指标；
3. 自回归滚动窗口可以取10个财年。

#### 说明
Predictability 取值越小，盈余的可预测性越高。

#### 参考文献
冯佳睿等, 2020, 高频因子的现实与幻想, 海通证券.

## 4.5 营运能力

### 4.5.1 净经营资产周转率

$$
\frac{\text{最近12个月的营业收入}(TTM)}{\text{平均净经营资产}}
$$

其中：
$$
\text{平均净经营资产} = \frac{\text{期初净经营资产} + \text{期末净经营资产}}{2}
$$

$$
\text{净经营资产} = \text{股东权益合计(含少数股东权益)} + \text{金融负债} - \text{金融资产} = \text{经营性资产} - \text{经营性负债}
$$

#### 说明
相比传统的资产周转率，净经营资产周转率更准确的衡量了企业经营活动相关资产的收入转化效率，剔除了金融/财务方面的影响。

#### 参考文献
Soliman, M.T., 2008, The use of DuPont analysis by market participants. Accounting Review 83:823-53.

### 4.5.2 存货周转率

$$
\frac{\text{最近12个月的营业成本}(TTM)}{\text{平均存货}}
$$

其中：
$$
\text{平均存货} = \frac{\text{期初存货} + \text{期末存货}}{2}
$$

#### 说明
用于衡量产品从库存转移到销售的速度。较高的存货周转率通常说明公司产品销售较为迅速，存货占用水平低，存货转换为现金或应收账款的速度较快，存货流动性较强，进而增加企业流动性或短期偿债能力和获利能力。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 4.5.3 应收账款周转率

$$
\frac{\text{最近12个月营业收入}(TTM)}{\text{平均应收账款}}
$$

其中：
$$
\text{平均应收账款} = \frac{\text{期初应收账款} + \text{期末应收账款}}{2}
$$

#### 说明
衡量企业应收账款周转的速度，反映企业的收帐速度；取值较高，说明企业收款快，账龄短。

#### 参考文献
常规通用财务指标.

### 4.5.4 应收账款周转天数

$$
\frac{360}{\text{应收账款周转率}}
$$

#### 说明
企业收回应收账款从发生到收回周转一次的平均天数，周转天数越短，回款速度越快，资金被外单位占用的时间越短，管理工作的效率越高。

#### 参考文献
常规通用财务指标.

### 4.5.5 应付账款周转天数

$$
\frac{360}{\text{应付账款周转率}}
$$

#### 说明
衡量企业应付清供应商的欠款的平均天数，通常周转天数越长越好，说明公司可以更多的占用供应商货款来补充营运资本而无需向银行短期借款。

#### 参考文献
常规通用财务指标.

### 4.5.6 总资产周转率

$$
\frac{\text{最近12个月营业成本}(TTM)}{\text{平均总资产}}
$$

其中：
$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

#### 说明
综合评价企业全部资产的经营质量和利用效率的重要指标，总资产周转率高，说明单位资产产生的销售收入高，企业资源利用率高，企业管理层资产运用能力强。

#### 参考文献
Haugen, Robert A., and Nardin L. Baker, 1996, Commonality in the determinants of expected stock returns, Journal of Financial Economics 41, 401-439.

### 4.5.7 股东权益周转率

$$
\frac{\text{最近12个月营业收入}(TTM)}{\text{平均股东权益合计}}
$$

其中：
$$
\text{平均股东权益合计} = \frac{\text{期初股东权益合计} + \text{期末股东权益合计}}{2}
$$

#### 说明
衡量企业将股东权益转换为销售收入的程度，股东权益周转率越高，说明企业运用所有者的资产的效率越高，营运能力强。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 4.5.8 固定资产周转率

$$
\frac{\text{最近12个月营业收入}(TTM)}{\text{平均固定资产}}
$$

其中：
$$
\text{平均固定资产} = \frac{\text{期初固定资产} + \text{期末固定资产}}{2}
$$

#### 说明
衡量企业将固定资产转换为销售收入的程度，可用于分析对厂房、设备等固定资产的利用效率。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 4.5.9 应收周转率

$$
\frac{\text{最近12个月营业收入}(TTM)}{\text{期末应收账款} + \text{期末应收票据}}
$$

#### 说明
综合衡量了企业应收账款转换为销售收入的程度，取值越高，应收账款回款速度越快。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 4.5.10 应付账款周转率

$$
\frac{\text{最近12个月营业成本}(TTM)}{\text{平均应付账款}}
$$

其中：
$$
\text{平均应付账款} = \frac{\text{期初应付账款} + \text{期末应付账款}}{2}
$$

#### 说明
衡量企业应付账款周转的速度，反映企业占用供应商资金的状况；取值较低，说明企业较多占用供应商的货款，市场地位较高，但是还款压力较大。

#### 参考文献
常规通用财务指标.

### 4.5.11 现金循环周期

$$
\text{存货周转天数} + \text{应收账款周转天数} - \text{应付账款周转天数}
$$

其中：
$$
\text{应收账款周转天数} = \frac{360}{\text{应收账款周转率}}
$$

$$
\text{应付账款周转天数} = \frac{360}{\text{应付账款周转率}}
$$

$$
\text{存货周转天数} = \frac{360}{\text{存货周转率}}
$$

#### 说明
衡量现金支出和现金回收之间时间，当现金循环周期为负，表明该企业从客户获得资金比用资金支付应付款更快。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

## 4.6 偿债能力

### 4.6.1 利息保障倍数

$$
\frac{\text{最近12个月息税前利润}EBIT(TTM)}{\text{最近12个月利息费用}(TTM)}
$$

其中：
$$
\text{利息费用} = \text{财务费用中的利息支出} - \text{财务费用中的利息收入}
$$

#### 说明
反映了其盈利能力对到期偿还债务的保障能力。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 4.6.2 现金流覆盖率

$$
\frac{\text{最近12个月经营性活动产生的现金流量净额}(TTM)}{\text{最近12个月利息费用}(TTM)}
$$

其中：
$$
\text{利息费用} = \text{财务费用中的利息支出} - \text{财务费用中的利息收入}
$$

#### 说明
项目产生的现金流是否能覆盖相对应贷款本息的充分程度，揭示项目现金流对贷款的偿还能力。

#### 参考文献
常规通用财务指标

### 4.6.3 经营现金流量比率

$$
\frac{\text{最近12个月经营性活动产生的现金流量净额}(TTM)}{\text{平均流动负债合计}}
$$

其中：
$$
\text{平均流动负债合计} = \frac{\text{期初流动负债合计} + \text{期末流动负债合计}}{2}
$$

#### 说明
衡量企业经营活动所产生的现金流量可以抵偿流动负债的程度，比率越高，说明企业的财务弹性越好。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 4.6.4 有形净值债务率

$$
\frac{\text{最近报告期负债合计}}{\text{最近报告期有形净值}}
$$

其中：
$$
\text{有形净值} = \text{归属母公司股东权益} - \text{无形资产} - \text{开发支出} - \text{商誉} - \text{长期待摊费用} - \text{递延所得税资产}
$$

#### 说明
衡量企业的风险程度和长期偿债能力。有形净值债务率越大，表明风险越大，企业长期偿债能力越弱。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

## 4.7 资本结构

### 4.7.1 流动资产占比

$$
\frac{\text{最近报告期的流动资产合计}}{\text{最近同期的总资产}}
$$

#### 说明
反映流动资产在总资产中的占比，占比比较高说明公司资金周转快，有较强的变现能力，建议结合流动资产中细分项的分布情况做进一步分析。

#### 参考文献
常规通用财务指标

### 4.7.2 流动负债率

$$
\frac{\text{最近报告期的流动负债合计}}{\text{最近同期负债合计}}
$$

#### 说明
反映一个公司依赖短期债权人的程度，比率越高，公司对短期资金的依赖性越强。

#### 参考文献
常规通用财务指标

### 4.7.3 长期负债比率

$$
\frac{\text{最近报告期的非流动负债}}{\text{最近同期总资产}}
$$

#### 说明
该指标值越小，表明公司负债的资本化程度低，长期偿债压力小；反之，则表明公司负债的资本化程度高，长期偿债压力大。

#### 参考文献
常规通用财务指标

### 4.7.4 固定资产比例

$$
\frac{\text{最近报告期的固定资产}}{\text{最近同期总资产}}
$$

#### 说明
不同行业的固定资产比率存在较大差异，但固定资产比率越低企业资产才能更快的流动，从资金营运能力来看，固定资产比率越低企业营运能力越强。

#### 参考文献
常规通用财务指标

### 4.7.5 资本固定化率

$$
\frac{\text{最近报告期的总资产} - \text{最近同期的流动资产合计}}{\text{最近同期股东权益合计}}
$$

#### 说明
反映公司自有资本的固定化程度，该指标值越低，表明公司自有资本用于长期资产的数额相对较少；反之，则表明公司自有资本用于长期资产的数额相对较多。

#### 参考文献
常规通用财务指标

### 4.7.6 经营性现金流量净额占比

$$
\frac{\text{最近12个月经营性现金流量净额}(TTM)}{\text{最近12个月现金及现金等价物净增加额}(TTM)}
$$

#### 说明
用于体现公司现金流量的结构。

#### 参考文献
常规通用财务指标

## 4.8 改进型因子

### 4.8.1 盈利市值比（去杠杆后）

$$
\frac{\text{最近12个月的毛利润}(TTM)}{\text{当日经营性净资产市值}}
$$

其中：
$$
\text{经营性净资产市值} = \text{金融负债} - \text{金融资产} + \text{市值}
$$

#### 说明
改进后的市盈率倒数，不受企业杠杆的影响，对企业估值更合理准确，选股能力显著提高；计算时将“市值”替换成了“经营性净资产市值”，对债权和股权都做了市场定价，而且剥离了企业金融性活动的影响，能更准确反映企业核心资产；将“净利润”替换成“毛利润”，剔除了容易操纵的销售管理等费用的影响。

#### 参考文献
刘富兵, 李林井, 2019, 对价值因子的思考和改进, 国盛证券

### 4.8.2 账面市值比（去杠杆后）

$$
\frac{\text{最近报告期的经营性净资产}}{\text{当日经营性净资产市值}}
$$

其中：
$$
\text{经营性净资产} = \text{股东权益} + \text{净负债} = \text{股东权益合计(含少数股东权益)} + \text{金融负债} - \text{金融资产}
$$

$$
\text{经营性净资产市值} = \text{金融负债} - \text{金融资产} + \text{市值}
$$

#### 说明
改进后的市净率倒数，对企业估值更合理准确，选股能力显著提高；分子分母都使用了经营性资产相关指标，对债权和股权都做了市场定价，而且剥离了企业金融性活动的影响，更准确反映企业核心资产。

#### 参考文献
刘富兵, 李林井, 2019, 对价值因子的思考和改进, 国盛证券

### 4.8.3 现金流市值比（去杠杆后）

$$
\frac{\text{最近12个月的经营性活动产生的现金流净额}(TTM)}{\text{当日经营性净资产市值}}
$$

其中：
$$
\text{经营性净资产市值} = \text{金融负债} - \text{金融资产} + \text{市值}
$$

#### 说明
改进后的市现率倒数，对企业估值更合理准确，选股能力显著提高；计算时将“市值”替换成了“经营性净资产市值”，对债权和股权都做了市场定价，分子分母相匹配；而且剥离了企业金融性活动的影响，能更准确反映企业核心资产。

#### 参考文献
刘富兵, 李林井, 2019, 对价值因子的思考和改进, 国盛证券

### 4.8.4 标准化营业利润

$$
\frac{\text{营业利润}(TTM)\text{的当期值} - mean(\text{过去}T\text{期的营业利润}(TTM))}{std(\text{过去}T\text{期的营业利润}(TTM))}
$$

其中：
$$
\text{默认 } T = 6 \text{ 个季度。}
$$

#### 说明
通过对常规财务数据进行标准化构造的另类成长因子，标准化后的因子与成长和盈利大类因子相关性较高，可以通过回归中心化处理剔除相关性。

#### 参考文献
曹春晓, 2018, 基本面因子之经营性标准化成长因子, 申万宏源

### 4.8.5 规模调整后的ROE

连续8个季度的ROE(TTM)序列对相同区间的调整后的总资产序列进行OLS回归，取最新一期残差作为因子值，其中仅对总资产的年报进行调整，一季报、半年报、三季报沿用上年年报数据。

#### 说明
经规模调整后的盈利因子，更能反映企业真正的内生增长性。

#### 参考文献
古翔, 周萧潇, 刘均伟, 2020, 规模调整的盈利因子:从盈余公积谈起, 光大证券.

### 4.8.6 业绩趋势

$$
NetProfit_t = \alpha * t^2 + \beta * t + c
$$

其中：
1. 计算业绩增速：归母净利润TTM环比增速；
2. 计算业绩增长加速度：利用连续N个季度的单季度归母净利润对期数 $t$ 的二次方程进行回归，取二次项系数 $\alpha$ 作为业绩增长加速度的代理变量；
3. 先依据业绩增速指标将全市场股票均分成三组，打分1-3分；其次，每一组内依据业绩增长加速度指标进行标准分法打分；最后将两步得分相加得到综合得分即为业绩趋势因子。

#### 说明
业绩高速增长的公司往往能带来较高的超额收益，该因子综合了增速和加速度信息。

#### 参考文献
刘均伟, 周萧潇, 2019, 业绩趋势因子:捕捉业绩加速增长的超额收益, 光大证券.

### 4.8.7 营业能力改善

$$
Revenue_i = a_i + \beta_i Cost_i + \epsilon_i
$$

其中：
1. 取当天最近N个季度的营业收入 ($revenue$) 和营业成本 ($cost$) 数据，并进行Z-Score标准化处理；
2. 将标准化后的营业收入对营业成本进行OLS线性回归，回归得到的最近一个季度的残差 $\epsilon_0$ 即为RROC在当天的因子值；
3. $i \in \{0, 1, 2, ..., N-1\}$，默认 $N=8$。

#### 说明
通过营业收入与营业成本线性关系的变化来反映公司营业能力的变化，回归的当季残差项的正负方向代表营业能力的变好与变差，绝对值代表变化幅度有多大。

#### 参考文献
刘均伟, 2018, 创新基本面因子:捕捉产能利用率中的讯号, 光大证券.

### 4.8.8 产能利用率提升

$$
TotalOperationCost_i = a_i + \beta_i FixedAssets_i + \epsilon_i
$$

其中：
1. 以当天最近N个季度的营业总成本作为Y，以对应季度的固定资产作为X，进行OLS线性回归，回归前需对营业总成本和固定资产进行Z-Score标准化处理；
2. 回归得到的最近一个季度的残差 $\epsilon_0$ 即为OCFA在当天的因子值；
3. $i \in \{0, 1, 2, ..., N-1\}$，默认 $N=8$。

#### 说明
产能利用率反映企业营运效率，刻画了公司将生产设备用于生产工作的利用程度。生产设备作为一种固定成本，不会像原材料等变动成本那样随着产量而改变，所以对于设备等固定设施的利用率的高低一定程度上决定了最终实际平均成本的高低，利用率越高，摊销到单位产品上的平均实际成本就越低，企业的营运效率就越高。

#### 参考文献
刘均伟, 2018, 创新基本面因子:捕捉产能利用率中的讯号, 光大证券.

### 4.8.9 ROE增长减净资产增长

$$
ROE环比增速 = \frac{\text{本期}ROE(TTM) - \text{上季度}ROE(TTM)}{abs(\text{上季度}ROE(TTM))}
$$

$$
\text{净资产同比增速} = \frac{\text{最近报告期的归属母公司股东权益} - \text{上年同期值}}{\text{上年同期值}}
$$

综合因子：
$$
ROE\text{环比增速} - \text{净资产同比增速}
$$

#### 说明
经规模调整后的盈利因子，更能反映企业真正的内生增长性。

#### 参考文献
古翔, 周萧潇, 刘均伟, 2020, 规模调整的盈利因子:从盈余公积谈起, 光大证券.

## 4.9 异常与盈余质量类

### 4.9.1 异常毛利润

$$
\text{异常毛利润} = \frac{\text{当前季度毛利润} - \text{去年同期毛利润} \times \text{正常增长乘数}}{\text{当前季度末总资产}}
$$

其中：
$$
\text{正常增长乘数} = \frac{\text{最新季度销售商品、提供劳务收到的现金}}{\text{去年同期销售商品、提供劳务收到的现金}}
$$

#### 说明
异常毛利润增长通常是有利信号。毛利润增长大于销售增长，说明毛利率在提升，这可能是由于公司产品的市场竞争力提高了，也可能是由于公司成本控制的改进。

#### 参考文献
汪荣飞, 张然, 2018, 基本面分析在中国A股市场有用吗?——来自季度财务报表的证据, 金融学季刊;
曹春晓, 2018, 异常财务指标因子研究, 申万宏源.

### 4.9.2 异常预收款

$$
\text{异常预收款} = \text{当前季度预收款项} - \text{去年同期预收款项} \times \text{正常增长乘数}
$$

其中：
$$
\text{正常增长乘数} = \frac{\text{最新季度销售商品、提供劳务收到的现金}}{\text{去年同期销售商品、提供劳务收到的现金}}
$$

标准化处理：
$$
\text{当前季度末总资产}
$$

#### 说明
异常预收款增长通常是有利信号。一方面，预收账款的异常增长可能预示着公司在供应链中的话语权上升；另一方面，预收账款的异常增长可能是管理层向下盈余操纵的手段，公司延迟将销售收入确认到利润表，转而将其放在预收款作为下一期的盈余储备。

#### 参考文献
曹春晓, 2018, 异常财务指标因子研究, 申万宏源.

### 4.9.3 异常财务因子

$$
\text{异常}F = \frac{\text{当前季度末的}F - \text{去年同期}F \times \text{正常增长乘数}}{\text{当前季度末总资产}} \times (-1)
$$

其中：
$$
\text{正常增长乘数} = \frac{\text{最新季度销售商品、提供劳务收到的现金}}{\text{去年同期销售商品、提供劳务收到的现金}}
$$

计算步骤：
1. 财务指标 $F$ 分别取存货、应收款（应收账款+预付款项）、其他应收款、预收款项、销售管理费用（销售费用+管理费用）、毛利润，通过上述公式分别计算得到异常存货、异常应收款、异常其他应收款、异常预收款（正向）、异常销售管理费用、异常毛利润（正向）6个衍生指标；
2. 对这6个衍生指标分别进行截面标准化，然后加总求和得到异常财务因子。

#### 说明
异常财务因子是判断公司财务质量的综合指标；对每个季度横截面上异常财务因子标准化得分小于两倍标准差的股票，建议谨慎对待，这些公司可能存在存货、应收账款、其他应收款等异常激增等情况。

#### 参考文献
汪荣飞, 张然, 2018, 基本面分析在中国A股市场有用吗?——来自季度财务报表的证据, 金融学季刊;
曹春晓, 2018, 异常财务指标因子研究, 申万宏源.

### 4.9.4 标准化调整的营业利润

$$
\frac{\text{调整的营业利润}(TTM)\text{的当期值} - mean(\text{过去}T\text{期调整的营业利润}(TTM))}{std(\text{过去}T\text{期调整的营业利润}(TTM))}
$$

其中：
$$
\text{调整的营业利润} = \text{营业利润} + \text{预收账款}
$$

$$
\text{默认 } T = 6 \text{ 个季度。}
$$

#### 说明
通过对常规财务数据进行标准化构造的另类成长因子，标准化后的因子与成长和盈利大类因子相关性较高，可以通过回归中心化处理剔除相关性。

#### 参考文献
曹春晓, 2018, 基本面因子之经营性标准化成长因子, 申万宏源.

### 4.9.5 财务质量因子

分别计算下面4个指标在所处行业内的百分位数作为各自的因子值，然后再分别进行横截面上标准化，最后加总求和得到财务质量因子：

$$
\text{单季度所得税费用}
$$
$$
\text{单季度营业收入}
$$

$$
\text{单季度营业费用}
$$
$$
\text{单季度营业收入}
$$

$$
\text{应收账款周转率同比增量}
$$

$$
\frac{\text{单季度所得税费用}}{\text{单季度营业收入}} \text{ 的同比增量}
$$

#### 参考文献
曹春晓, 2019, 财务造假启示录: 财务质量因子研究, 申万宏源.



# 5 高频因子
## 5.1 资金流因子

### 5.1.1 残差资金流强度因子

计算小单资金流强度：分子为（小单买额 - 小单卖额）之和，分母为其绝对值；计算大单资金流强度：分子为（大单买额 - 大单卖额）之和，分母为其绝对值：

$$
S_t = \frac{\sum_{t-T}^t (buy_t - sell_t)}{\sum_{t-T}^t |buy_t - sell_t|}
$$

分别将大小单资金流强度与过去20日涨跌幅做回归，得到残差，即为大小单残差资金流因子：

$$
S_t = a + bRet20_t + \varepsilon_t
$$

#### 说明
残差资金流因子剥离了涨跌幅对资金流强度的影响，衡量的是在同等涨跌幅 (Ret20) 的情况下，资金流强度的选股能力。

#### 参考文献
魏建榕, 高鹏, 2021, 大单与小单资金流的 alpha 能力, 开源证券.

### 5.1.2 聪明钱因子

对选定股票，回溯取其过去10个交易日的分钟行情数据，构造指标 $S_t = \frac{|R_t|}{V_t^{0.25}}$，其中 $R_t$ 为第 $t$ 分钟涨跌幅，$V_t$ 为第 $t$ 分钟成交量；

将分钟数据按照指标 $S_t$ 从大到小进行排序，取成交量累积占比20%的分钟，视为聪明钱交易；

计算聪明钱交易的成交量加权平均价 $VWAP_{smart}$；计算所有交易的成交量加权平均价 $VWAP_{all}$

计算如下聪明钱因子：

$$
\text{聪明钱因子} = \frac{VWAP_{smart}}{VWAP_{all}}
$$

#### 说明
聪明钱因子是通过在分钟行情数据的价量信息中识别机构参与交易的多寡而构建的跟踪聪明钱交易的选股因子，而聪明钱的交易是基于“单笔订单数量更大、订单报价更为激进”的交易特征来识别的。

#### 参考文献
魏建榕, 2020, 聪明钱因子模型的2.0版本, 开源证券.

### 5.1.3 开盘后大单净买入占比

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{\sum_{j=1}^N (\text{大买单成交额}_{i,j,n} - \text{大卖单成交额}_{i,j,n})}{\sum_{j=1}^N \text{成交额}_{i,j,n}}
$$

其中：
1. 基于逐笔成交数据中的叫买与叫卖单号将逐笔成交数据合成为买卖单数据；
2. 根据买卖单分布界定大小单，如可将多日买卖单成交单数据对数调整后的“均值+1倍标准差”作为大单筛选阈值；
3. $i$、$j$、$n$ 分别表示第 $i$ 只股票在第 $n$ 个交易日内的第 $j$ 分钟的成交额；
4. 开盘后的大单净买入占比用的是 9:30~10:00 范围内的数据；
5. 月度选股下，$T=20$个交易日；周度选股下，$T=5$个交易日。

#### 说明
大单类因子刻画了大资金的交易行为，大资金往往具有信息优势，一般受到大资金关注的股票，未来通常具有更好的表现。

#### 参考文献
1. 冯佳睿, 袁林青, 2021, 大单的精细化处理与大单因子重构, 海通证券.
2. 冯佳睿, 袁林青, 2019, 买卖单数据中的 Alpha, 海通证券.

### 5.1.4 开盘后大单净买入强度

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{mean(\text{大买单成交额}_{i,j,n} - \text{大卖单成交额}_{i,j,n})}{std(\text{大买单成交额}_{i,j,n} - \text{大卖单成交额}_{i,j,n})}
$$

其中：
1. 基于逐笔成交数据中的叫买与叫卖单号将逐笔成交数据合成为买卖单数据；
2. 根据买卖单分布特征界定大小单，比如可将多日买卖单成交单数据对数调整后的“均值+1倍标准差”作为大单筛选阈值；
3. $i$、$j$、$n$ 分别表示第 $i$ 只股票在第 $n$ 个交易日内的第 $j$ 分钟的成交额；
4. 开盘后的大单净买入强度用的是 9:30~10:00 范围内的数据；
5. 月度选股下，$T=20$个交易日；周度选股下，$T=5$个交易日。

#### 说明
大单类因子刻画了大资金的交易行为，大资金往往具有信息优势，一般受到大资金关注的股票，未来通常具有更好的表现；大单净买入强度因子同时衡量了大单买入强度和大单买入稳健性两方面的信息。

#### 参考文献
冯佳睿, 袁林青, 2021, 大单的精细化处理与大单因子重构, 海通证券.

### 5.1.5 小单和小单的错位相关性

$$
RankCorr(S_t, S_{t+1})
$$

其中：
$S_t$ 和 $S_{t+1}$ 都为小单（<4万元）过去20个交易日同期的资金净流入序列，但是双方日期错开1个交易日。

#### 说明
在资金流错位相关性因子中，上述小单和小单的错误相关性因子表现最好；该因子的alpha来源是“散户追涨杀跌的羊群效应”，其中，散户羊群效应可以通过股票“过去N日收益率”与“小单未来N日净流入”的秩相关系数 $RankCorr(S_t, S_{t+1})$ 来衡量。

#### 参考文献
魏建榕, 盛少成, 2022, 资金流动办学与散户羊群效应, 开源证券.

### 5.1.6 开盘后买入意愿强度

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{mean(\text{买入意愿}_{i,j,n})}{std(\text{买入意愿}_{i,j,n})}
$$

$$
\text{买入意愿}_{i,j,n} = \text{净主买成交额}_{i,j,n} - \text{净委买增额}_{i,j,n}
$$

$$
\text{净主买成交额}_{i,j,n} = \text{主动买入成交额}_{i,j,n} - \text{主动卖出成交额}_{i,j,n}
$$

$$
\text{净委买增额}_{i,j,n} = \text{委托买单增加量}_{i,j,n} - \text{委托卖单增加量}_{i,j,n}
$$

其中：
1. 净主买成交额由逐笔成交数据计算得到，具体可参考开盘后净主买占比/强度等因子；
2. 净委买增额由盘口委托快照数据计算得到，具体可参考开盘后净委买增额占比因子；
3. $i, j, n$ 分别表示第 $i$ 只股票在第 $n$ 个交易日内的第 $j$ 分钟的数据；
4. 开盘后的买入意愿占比用的是 9:30~10:00 范围内的数据；
5. 月度选股下，T=20个交易日；周度选股下，T=5个交易日。

#### 说明
净委买变化刻画了投资者还未释放的买入意愿，而净主买成交额则体现刻画了投资者已经释放的买入意愿，两者结合则得到广义的投资者主动买入意愿；开盘后30分钟内的买入意愿强度越高，投资者的买入意愿越稳健。

#### 参考文献
冯佳睿, 袁林青, 2020, 基于直观逻辑和机器学习的高频数据低频化应用, 海通证券.

### 5.1.7 开盘后净委买增额占比

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{\sum_{j=1}^N (\text{净委买增额}_{i,j,n})}{\sum_{j=1}^N \text{成交额}_{i,j,n}}
$$

$$
\text{净委买增额}_{i,j,n} = \text{委托买单增加量}_{i,j,n} - \text{委托卖单增加量}_{i,j,n}
$$

其中：
1. 盘口委托快照数据包含买一至买十的委买量、卖一至卖十的委卖量等指标；
2. 委托买单增加量为日内 t 至 t+1 时刻各档的委买变化量之和，委托卖单增加量同理；
3. 一般用前1档位数据即可；使用的档位数量越多，因子的选股能力越弱；
4. $i, j, n$ 分别表示第 $i$ 只股票在第 $n$ 个交易日内的第 $j$ 分钟的数据；
5. 开盘后的净委买增额占比用的是 9:30~10:00 范围内的数据；
6. 月度选股下，T=20个交易日；周度选股下，T=5个交易日。

#### 说明
刻画了投资者的买入意愿，委买量的增加代表了投资者买入意愿的增强，委卖量的增加代表了投资者卖出意愿的增强；开盘后30分钟的数据更是包含了投资者对于前一天收盘后股票信息的集中反馈，集中反馈越正面，体现出的买入意愿越强，股票未来的超额收益表现越好。

#### 参考文献
1. 冯佳睿, 袁林青, 2019, 捕捉投资者的交易意愿, 海通证券.
2. 冯佳睿等, 2020, 高频因子的现实与幻想, 海通证券.

### 5.1.8 开盘后净主买占比

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{\sum_{j=1}^N \text{净主买成交额}_{i,j,n}}{\sum_{j=1}^N \text{成交额}_{i,j,n}}
$$

$$
\text{净主买成交额}_{i,j,n} = \text{主动买入成交额}_{i,j,n} - \text{主动卖出成交额}_{i,j,n}
$$

其中：
1. 基于逐笔成交数据中的BS标志将逐笔成交数据合成为分钟级别的主动买卖金额，B为主动买入（卖出方先挂单，买入方主动触碰卖单并成交）；S为主动卖出（买入方先挂单，卖出方主动触碰买单并成交）；
2. 剔除了处于涨跌停分钟上的主动买入金额和主动卖出金额数据；
3. $i, j, n$ 分别表示第 $i$ 只股票在第 $n$ 个交易日内的第 $j$ 分钟的成交额；
4. 开盘后的净主买占比用的是 9:30~10:00 范围内的数据；
5. 月度选股下，T=20个交易日；周度选股下，T=5个交易日。

#### 说明
刻画了投资者在开盘后30分钟内净买入行为的强度，开盘后净主买占比越高，投资者的主动买入行为越强烈。

#### 参考文献
冯佳睿, 袁林青, 2020, 基于主动买入行为的选股因子, 海通证券.

### 5.1.9 开盘后净主买强度

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{mean(\text{净主买成交额}_{i,j,n})}{std(\text{净主买成交额}_{i,j,n})}
$$

$$
\text{净主买成交额}_{i,j,n} = \text{主动买入成交额}_{i,j,n} - \text{主动卖出成交额}_{i,j,n}
$$

其中：
1. 基于逐笔成交数据中的BS标志将逐笔成交数据合成为分钟级别的主动买卖金额，B为主动买入（卖出方先挂单，买入方主动触碰卖单并成交）；S为主动卖出（买入方先挂单，卖出方主动触碰买单并成交）；
2. 剔除了处于涨跌停分钟上的主动买入金额和主动卖出金额数据；
3. $i, j, n$ 分别表示第 $i$ 只股票在第 $n$ 个交易日内的第 $j$ 分钟的成交额；
4. 开盘后的买入意愿占比用的是 9:30~10:00 范围内的数据；
5. 月度选股下，T=20个交易日；周度选股下，T=5个交易日。

#### 说明
同时刻画了投资者在开盘后30分钟内主动买入行为的强度和稳健性，开盘后净主买强度越高，投资者的主动买入行为越稳健。

#### 参考文献
冯佳睿, 袁林青, 2020, 基于主动买入行为的选股因子, 海通证券.

### 5.1.10 平均单笔流出金额占比

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{\sum_{j=1}^N Amt_{i,j,n} \cdot I_{r_{i,j,n}<0} / \sum_{j=1}^N TrdNum_{i,j,n} \cdot I_{r_{i,j,n}<0}}{\sum_{j=1}^N Amt_{i,j,n} / \sum_{j=1}^N TrdNum_{i,j,n}}
$$

其中：
1. $Amt_{i,j,n}$ 为股票 $i$ 在第 $n$ 个交易日内第 $j$ 分钟的成交额序列；
2. $r_{i,j,n}$ 为股票 $i$ 在第 $n$ 个交易日内第 $j$ 分钟的收益率序列；
3. $TrdNum_{i,j,n}$ 为股票 $i$ 在第 $n$ 个交易日内第 $j$ 分钟的成交笔数序列；
4. 月度选股下，T=20个交易日；周度选股下，T=5个交易日；
5. 还可计算平均单笔流入金额占比，只需将 $I_{r_{i,j,n}<0}$ 改为 $I_{r_{i,j,n}>0}$ 即可。

#### 说明
平均单笔成交金额占比类因子可以捕捉日内大单的交易行为，大单交易会使得股票在当日的平均单笔成交金额占比在全市场中处于相对较高水平。平均单笔流出金额占比因子具有较强的选股能力，与股票未来收益正相关；在股票下跌时，如果单笔成交金额大，说明委买有大单，是一种抄底行为。

#### 参考文献
冯佳睿, 姚石, 2019, 日内分时成交中的玄机, 海通证券.

### 5.1.11 主力交易情绪

$$
TE = RankCorr(A_{order}, C_{minute})
$$

其中：
1. $A_{order}$ 为某只股票某个交易内的分钟单笔成交金额序列；
2. $C_{minute}$ 为某只股票某个交易内的分钟收盘价序列；
3. 滚动20个交易日计算TE指标的均值，即主力交易情绪因子 MTE。

#### 说明
MTE因子实质上反映了主力参与交易的相对价位，因子值越大，表明主力交易更倾向于出现在高价位，这是“逢高出货”的表现，反映了主力悲观态度；而因子值越小，则表明主力交易更多出现在低价位，这是“逢低吸筹”的表现，属于乐观情绪。

#### 参考文献
魏建榕, 苏良, 2022, 高频因子: 分钟单笔金额序列中的主力行为刻画, 开源证券.

### 5.1.12 主力交易强度

$$
TS = RankCorr(A_{order}, A_{minute})
$$

其中：
1. $A_{order}$ 为某只股票某个交易内的分钟单笔成交金额序列；
2. $A_{minute}$ 为某只股票某个交易内的分钟成交金额序列；
3. 滚动20个交易日计算TS指标的均值，即主力交易强度因子 MTS。

#### 说明
单笔成交金额与成交额之间的相关性强弱，描述的是代表主力的“相对大单”对分钟成交额的影响，从资金行为学的角度来看，单笔成交金额与成交额的相关性越强，主力资金主导成交节奏的能力也越强。

#### 参考文献
魏建榕, 苏良, 2022, 高频因子: 分钟单笔金额序列中的主力行为刻画, 开源证券.

### 5.1.13 单笔成交金额分位数

$$
S = \frac{A_{0.1} - A_{min}}{A_{max} - A_{min}}
$$

#### 说明
① A为某只股票某个交易内的分钟单笔成交金额序列，下标 0.1、max、min分别表示该序列的10%分位数、最大值、最小值；
② 需要将A序列从小到大排序，剔除最大的10个样本值（极端值）；
③ 滚动20个交易日计算S指标的均值，即可得到QUA因子。

单笔成交金额分位数QUA本质上刻画了大单相对小单的成交金额偏离程度，QUA取值越小，双方偏离程度越高，而大单相对小单的成交金额偏离越大，主力（“相对大单”）对于该股票的关注度越高，主力资金参与交易的意愿越强，未来股价表现会越好；单笔成交金额分布形态除了计算分位数外，还可以计算标准差STD、峰度KURT、偏度SKEW等。

#### 参考文献
魏建榕, 苏良, 2022, 高频因子: 分钟单笔金额序列中的主力行为刻画, 开源证券.

### 5.1.14 散户羊群效应

$$
RankCorr(R_t, S_{t+1})
$$

#### 说明
① $R_t$ 为过去20个交易日的日度收益率序列，用于表示市场涨跌；
② $S_{t+1}$ 为小单（<4万元）过去20个交易日同期的资金净流入序列（前置1期），用于表示散户行为。

衡量了散户追涨杀跌的程度，与股票未来收益负相关，即股票的散户羊群效应越高，未来预期收益越差。

#### 参考文献
魏建榕, 盛少成, 2022, 资金流动力学与散户羊群效应, 开源证券.

### 5.1.15 开盘后买入意愿占比

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{\sum_{j=1}^N (\text{买入意愿}_{i,j,n})}{\sum_{j=1}^N \text{成交额}_{i,j,n}}
$$

$$
\text{买入意愿}_{i,j,n} = \text{净主买成交额}_{i,j,n} - \text{净委买增额}_{i,j,n}
$$

$$
\text{净主买成交额}_{i,j,n} = \text{主动买入成交额}_{i,j,n} - \text{主动卖出成交额}_{i,j,n}
$$

$$
\text{净委买增额}_{i,j,n} = \text{委托买单增加量}_{i,j,n} - \text{委托卖单增加量}_{i,j,n}
$$

#### 说明
① 净主买成交额由逐笔成交数据计算得到，具体可参考开盘后净主买占比/强度等因子；
② 净委买增额由盘口委托快照数据计算得到，具体可参考开盘后净委买增额占比因子；
③ $i$、$j$、$n$ 分别表示第 $i$ 只股票在第 $n$ 个交易日内的第 $j$ 分钟的数据；
④ 开盘后的买入意愿占比用的是9:30~10:00范围内的数据；
⑤ 月度选股下，$T=20$个交易日；周度选股下，$T=5$个交易日。

净委买变化刻画了投资者还未释放的买入意愿，而净主买成交额则体现刻画了投资者已经释放的买入意愿，两者结合则得到广义的投资者主动买入意愿；开盘后30分钟内买入意愿占比越高，投资者的买入意愿越强。

#### 参考文献
冯佳睿, 袁林青, 2020, 基于直观逻辑和机器学习的高频数据低频化应用, 海通证券.

### 5.1.16 主动买卖因子

逐日计算大单和中单总的主动买卖因子 $ACT_{正向,t}$，以及小单的主动买卖因子 $ACT_{负向,t}$：

$$
ACT_{正向,t} = \frac{\text{主动买入金额(大单+中单)} - \text{主动卖出金额(大单+中单)}}{\text{主动买入金额(大单+中单)} + \text{主动卖出金额(大单+中单)}}
$$

$$
ACT_{负向,t} = \frac{\text{主动买入金额(小单)} - \text{主动卖出金额(小单)}}{\text{主动买入金额(小单)} + \text{主动卖出金额(小单)}}
$$

回溯过去20个交易日，取收益率最高 $\lambda$ 比例的交易日，称为高收益日；取收益率最低 $\lambda$ 例的交易日，称为低收益日；

对高收益日的 $ACT_{正向,t}$ 因子取平均，记为 $ACT_{正向}$；
对低收益日的 $ACT_{负向,t}$ 因子取平均，记为 $ACT_{负向}$。

#### 说明
切割后的主动买卖因子，细致的刻画了在市场上涨和下跌的不同情境下，不同交易者主动买卖意愿的差异；测试发现以大户和中户投资者为主的主动买卖因子，在高收益端呈现正向选股效应；而以小户为代表的主动买卖因子，在低收益端呈现负向选股效应。

#### 参考文献
魏建榕, 2020, 主动买卖因子的正确用法, 开源证券.


## 5.2 价格行为与动量类

### 5.2.1 APM因子

对选定股票，回溯取其过去20日数据，记逐日上午的股票收益率为 $r_t^{am}$，指数收益率为 $R_t^{am}$；逐日下午的股票收益率为 $r_t^{pm}$，指数收益率为 $R_t^{pm}$；

将得到的40组上午与下午 $(r, R)$ 的收益率数据进行回归：$r_i = \alpha + \beta R_i + \varepsilon_i$，得到残差项 $\varepsilon_i$；

以上得到的40个残差 $\varepsilon_i$ 中，上午残差记为 $\varepsilon_t^{am}$，下午残差记为 $\varepsilon_t^{pm}$，进一步计算每日上午与下午残差的差 $\delta_t = \varepsilon_t^{am} - \varepsilon_t^{pm}$；

构造统计量 stat 来衡量上午与下午残差的差异程度，计算公式如下（$\mu$ 为均值，$\sigma$ 为标准差）：

$$
\text{stat} = \frac{\mu(\delta_t)}{\sigma(\delta_t)/\sqrt{N}}
$$

为了消除动量因子影响，将统计量 stat 对动量因子进行横截面回归：$\text{stat}_j = b \text{Ret}20_j + \varepsilon_j$，其中 $\text{Ret}20$ 为股票过去20日的收益率，代表动量因子；回归得到的残差值 $\varepsilon$ 即为 APM 因子。

#### 说明
基于“知情交易者更加倾向于在每日上午进行交易，上午的价格行为蕴藏了更多可用于选股的信息量”的想法，构建了刻画了股票上午与下午的价格行为差异程度的 APM 因子；改进的 APM 因子是将个股和指数的上午收益 $r_t^{am}$ 替换成了隔夜收益 $r_t^{overnight}$。

#### 参考文献
魏建榕, 2020, APM因子模型的进阶版, 开源证券.

### 5.2.2 大单推动涨幅

$$
\sum_{n=t}^{t-T+1} \prod (prod(1 + r_{i,j,n} \cdot I_{\{j \in IdxSet\}})) - 1
$$

$$
\text{平均单笔成交金额} = \sum_{j=1}^N Amt_{i,j,n} / \sum_{j=1}^N TrdNum_{i,j,n}
$$

其中：
1. $r_{i,j,n}$ 为股票 $i$ 在第 $n$ 个交易日内第 $j$ 分钟的收益率序列；
2. $Amt_{i,j,n}$ 为股票 $i$ 在第 $n$ 个交易日内第 $j$ 分钟的成交额序列；
3. $TrdNum_{i,j,n}$ 为股票 $i$ 在第 $n$ 个交易日内第 $j$ 分钟的成交笔数序列；
4. $IdxSet$ 为第 $n$ 个交易日内平均单笔成交金额最大的30%的分钟K线的序号；
5. 月度选股下，$T=20$个交易日；周度选股下，$T=5$个交易日。

#### 说明
平均单笔成交金额较大的K线多空博弈激烈，未来的反转效应更强；该因子与股票未来收益负相关。

#### 参考文献
冯佳睿, 姚石, 2019, 日内分时成交中的玄机, 海通证券.

### 5.2.3 理想反转因子

1. 对单只股票，回溯其过去20日的数据，计算每日日内逐笔成交金额分布的13/16分位值；
2. 13/16分位值高的10个交易日，涨跌幅加总，记作 $M_{high}$；13/16分位值低的10个交易日，涨跌幅加总，记作 $M_{low}$；
3. 计算理想反转因子：$M = M_{high} - M_{low}$。

#### 说明
早期版本的理想反转因子是以“每日的平均单笔成交金额（成交金额/成交笔数）”作为切割标准；作者分析得出“反转之力的微观来源，是大单成交”，由高分位值较高（即大单成交较多）的那些交易日得到的反转因子，具有更强的反转特性。

#### 参考文献
魏建榕, 2019, A 股反转之力的微观来源, 开源证券.

### 5.2.4 价格自相关性合成因子 (双序列差分)

$$
CDPDP = \frac{dP^+dP^+Corr - mean(dP^+dP^+Corr)}{std(dP^+dP^+Corr)} + \frac{dP^-dP^-Corr - mean(dP^-dP^-Corr)}{std(dP^-dP^-Corr)}
$$

1. 每月月底，回溯每只股票过去20个交易日，每日先将该股票的分钟盘价序列做一阶差分，整理得到两个序列，$\Delta P_t$ 与 $\Delta P_{t+1}$ 序列，其中，$\Delta P_t = P_t - P_{t-1}$；
2. 分别取两个序列中，$\Delta P_t > 0$ 且 $\Delta P_{t+1} > 0$ 的部分，计算相关系数同时取20日的平均值，得到子因子 $dP^+dP^+Corr$；取 $\Delta P_t > 0$ 且 $dP^-dP^-Corr$ 的部分，按相同逻辑计算得到子因子；
3. 将两个子因子各自横截面标准化，再等权线性组合得到价格自相关性合成因子（双序列差分）CDPDP。

#### 说明
价格自相关性合成因子(双序列差分)与价格自相关性因子(单序列差分)的逻辑相通，即：不希望股价连续出现大幅上涨或大幅下跌的变动，因子取值越低的股票，未来收益表现越好。

#### 参考文献
高子剑, 2021, CPV 因子移位版, 价格自相关性中蕴藏的选股信息, 东吴证券.

### 5.2.5 时间加权平均的股票相对价格位置

计算股票相对价格位置，即股票当期价格相对区间最高最低价的分位数：

$$
RPP_{i,t} = \frac{(P_{i,t} - L_i)}{(H_i - L_i)}
$$

将股票相对价格位置对时间进行积分，得到时间加权平均的相对价格位置ARPP：

$$
ARPP_i = \int_0^T RPP_{i,t} \cdot dt \Rightarrow ARPP_{i,t} = \frac{(\int_0^T P_{i,t} \cdot dt - L_i)}{(H_i - L_i)}
$$

其中：
1. $RPP_{i,t}$ 表示股票 $i$ 在某一时间区间内的时刻 $t$ 的相对价格位置；
2. $H_i$ 和 $L_i$ 为该时间区间内股票 $i$ 的最高价和最低价；
3. $\int_0^T P_{i,t} \cdot dt$ 表示股票 $i$ 在时间区间0到T内的时间加权平均价格（即 TWAP），可以用分钟高开低收均值在时序上的均值作为区间的 TWAP。

#### 说明
衡量股票在价格相对高位停留的时间长短，股票在价格相对高位停留的时间越长，因子取值越大。

#### 参考文献
朱剑涛, 2020, 基于时间尺度度量的日内买卖压力, 东方证券.

### 5.2.6 价量相关性趋势因子

1. 每月月底，回溯每只股票过去20个交易日的价量信息，每日计算该股票分钟收盘价与分钟成交量的相关系数；
2. 将每只股票的20个相关系数 $\rho_t$ 对时间 $t$ 回归，取回归系数 $\beta$，即 $\rho_t = \beta t + \varepsilon_t$，其中 $t=1,2,3,...,20$；
3. 将所有股票的回归系数 $\beta$ 在横截面上剔除市值、传统价量类因子（20日反转、20日换手率、20日波动率因子），得到的结果即为价量相关性趋势因子 PV_corr_trend。

#### 说明
PV_corr_trend越小，即价量相关系数随时间推移变小的股票，未来收益倾向于越高。

#### 参考文献
高子剑, 2020, 高频价量相关性, 意想不到的选股因子, 东吴证券.

### 5.2.7 残差反转因子

计算小单资金流强度：分子为（小单买额-小单卖额）之和，分母为其绝对值；计算大单资金流强度：分子为（大单买额-大单卖额）之和，分母为其绝对值，即：

$$
S_t = \frac{\sum_{t-T}^t (buy_t - sell_t)}{\sum_{t-T}^t |buy_t - sell_t|}
$$

将反转因子（过去20日涨跌幅）与大单资金流强度和小单资金流强度各自做回归，得到的残差即为得到大单和小单的残差反转因子：

$$
Ret20_t = a + b * S_t + \varepsilon_t
$$

#### 说明
残差反转因子剥离了资金流强度对传统反转因子的影响，选股效果得到进一步提升。

#### 参考文献
魏建榕, 高鹏, 2021, 大单与小单资金流的 alpha 能力, 开源证券.


## 5.3 交易行为与微观结构因子

### 5.3.1 非流动性

不考虑日内价格变动方向，直接进行无向叠加，求得股价变动的最短路径：

$$
\text{shortcut} = 2 * (high - low) - |close - open|
$$

利用高频分钟 K 线，每日的 K 线最短路径非流动性因子：

$$
ILLIQ_t = \frac{1}{d} \sum_{i=1}^d \left( \sum_{j=1}^p \frac{\text{ShortCut}_j}{\text{Value}_j} \right)_{t-i}
$$

其中：
1. $\text{ShortCut}_j$ 表示日内频率下第 $j$ 根K线的最短路径；
2. $\text{Value}_j$ 表示日内频率下第 $j$ 根K线内的成交额；
3. $p$ 表示日内频率分段个数；
4. $d$ 表示日间移动平均的周期参数。

#### 说明
通过使用更高频的K线数据提升K线最短路径对股票交易时市场冲击的代理精度，大大提升了K线最短路径非流动性因子的有效性与预测能力。同时随着数据频率的提高，K线最短路径定义对经典定义下的非流动性因子提升程度愈加显著。比起经典的定义方式，K线最短路径能更充分地利用高频数据新引入的信息。

#### 参考文献
刘均伟, 2017, 基于 K 线最短路径构造的非流动性因子, 光大证券.

### 5.3.2 换手率分布均匀度因子

每月月底，回溯所有股票过去20个交易日，计算当日分钟换手率的标准差：

$$
TurnVolDaily = std(\text{分钟换手率})
$$

$$
\text{分钟换手率} = \frac{\text{分钟成交量}}{\text{当日流通股本}}
$$

基于这20个交易日的分钟换手率序列，计算标准差和均值，然后进行标准化和市值中性化处理得到换手率分布均匀度 UTD：

$$
UTD = \frac{std(TurnVolDaily)}{mean(TurnVolDaily)}
$$

#### 说明
换手率分布均匀度因子衡量了股票换手率在日内不同交易时段的分布均匀程度，以及该均匀程度在日间是否平稳，股票换手率在日内及日间越平稳，即因子值越小越好。

#### 参考文献
高子剑, 2021, 换手率分布均匀度, 基于分钟成交量的选股因子, 东吴证券.

### 5.3.3 买单非流动性

$$
r_{i,t} = \alpha + \beta_1 * S_{i,t} + \beta_2 * B_{i,t} + \epsilon_{it}
$$

其中：
1. $\beta_1$ 为卖出非流动性系数；
2. $\beta_2$ 为买入非流动性系数；
3. $S_{i,t}$ 为股票 i 在 t 时间区间内的主动卖出金额；
4. $B_{i,t}$ 为股票 i 在 t 时间区间内的主动买入金额。

#### 说明
买单非流动性衡量了高频数据下主动买入的交易金额对于股票价格变动的影响；买单非流动性对收益率的预测效果不及卖单非流动性，主要是由于投资者存在亏损厌恶的心理。

#### 参考文献
1. Brennan, M. J., Chordia, T., Subrahmanyam, A., & Tong, Q., 2012, Sell-order liquidity and the cross-section of expected stock returns. Journal of Financial Economics, 105(3), 523-541.
2. 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 5.3.4 一致卖出交易

通过定义实体K线，将交易日中的每根5分钟K线划分为集体一致交易与非集体一致交易：

$$
|close - open| <= \alpha |high - low|
$$

利用下跌的实体 K 线构造一致卖出交易因子：

$$
NCV_t = \frac{1}{d} \sum_{i=1}^d (\frac{ConsistentVolume_{fall}}{Volume})_{t-i}
$$

其中：
1. $\alpha$ 为一致参数，$\alpha$ 越大，K线一致性越强（上下引线越短，K线越实体）；
2. $ConsistentVolume_{fall}$ 为下跌的5分钟实体K线的总成交量；
3. $Volume$ 为当日总成交量；
4. $d$ 表示移动平均的周期参数。

#### 说明
集体一致交易行为预示交易机会，如果一支股票在一段时间内大部分的交易属于集体一致交易行为，趋势性显著，则很可能这支股票在这段时间内有新信息进入，正在经历市场消化过程 (price-in)，未来股价很可能产生大幅变化，为交易者创造交易机会。

#### 参考文献
刘均伟, 2018, 一致交易因子:挖掘集体行为背后的收益, 光大证券.

### 5.3.5 一致买入交易

通过定义实体K线，将交易日中的每根5分钟K线划分为集体一致交易与非集体一致交易：

$$
|close - open| <= \alpha |high - low|
$$

利用上涨的实体K线构造一致买入交易因子：

$$
PCV_t = \frac{1}{d} \sum_{i=1}^d (\frac{ConsistentVolume_{rise}}{Volume})_{t-i}
$$

其中：
1. $\alpha$ 为一致参数，$\alpha$ 越大，K线一致性越强（上下引线越短，K线越实体）；
2. $ConsistentVolume_{rise}$ 为上涨的5分钟实体K线的总成交量；
3. $Volume$ 为当日总成交量；
4. $d$ 表示移动平均的周期参数。

#### 说明
集体一致交易行为预示交易机会，如果一支股票在一段时间内大部分的交易属于集体一致交易行为，趋势性显著，则很可能这支股票在这段时间内有新信息进入，正在经历市场消化过程 (price-in)，未来股价很可能产生大幅变化，为交易者创造交易机会。

#### 参考文献
刘均伟, 2018, 一致交易因子:挖掘集体行为背后的收益, 光大证券.

### 5.3.6 一致交易

通过定义实体K线，将交易日中的每根5分钟K线划分为集体一致交易与非集体一致交易：

$$
|close - open| <= \alpha |high - low|
$$

利用属于集体一致交易的K线成交量，计算一致交易类因子：

$$
TCV_t = \frac{1}{d} \sum_{i=1}^d (\frac{ConsistentVolume}{Volume})_{t-i}
$$

其中：
1. $\alpha$ 为一致参数，$\alpha$ 越大，K线一致性越强（上下引线越短，K线越实体）；
2. $ConsistentVolume$ 为5分钟实体K线的总成交量；
3. $Volume$ 为当日总成交量；
4. $d$ 表示移动平均的周期参数。

#### 说明
集体一致交易行为预示交易机会，如果一支股票在一段时间内大部分的交易属于集体一致交易行为，趋势性显著，则很可能这支股票在这段时间内有新信息进入，正在经历市场消化过程 (price-in)，未来股价很可能产生大幅变化，为交易者创造交易机会。

#### 参考文献
刘均伟, 2018, 一致交易因子:挖掘集体行为背后的收益, 光大证券.

### 5.3.7 成交量占比

开盘集合竞价成交量占比因子 OCVP：

$$
OCVP_t = \frac{1}{d} \sum_{i=1}^d (\frac{\text{每日开盘集合竞价阶段成交量} VOL_{call}}{\text{日内个股总成交量} VOL_{total}})_{t-i}
$$

收盘集合竞价成交量占比因子 BCVP：

$$
BCVP_t = \frac{1}{d} \sum_{i=1}^d (\frac{\text{收盘前5分钟内个股成交量} VOL_{call}}{\text{日内个股总成交量} VOL_{total}})_{t-i}
$$

成交量占比复合因子：

$$
OBCVP_t = \alpha OCVP_t + (1 - \alpha) BCVP_t
$$

其中：
$w_i$ 为开盘集合竞价时间不同时间点上的成交量加权权重，可以是等权，也可以是时间衰减权重；$\alpha$ 是2大成交量占比因子的复合权重。

#### 说明
集合竞价阶段是反映投资者行为信息的重要时点，集合竞价成交量因子能反映多空双方之间的博弈，集合竞价成交量占比越低，股票次月的收益率越高。

#### 参考文献
刘均伟, 2017, 见微知著:成交量占比高频因子解析, 光大证券.


### 5.3.8 尾盘成交占比

$$
\frac{1}{T} \sum_{n=t}^{t-T+1} \frac{Vol_{i, 14:30-15:00, n}}{Vol_{i,n}}
$$

其中：
1. $Vol_{i, 14:30-15:00, n}$ 为第 $i$ 只股票在第 $n$ 个交易日内 14:30-15:00 尾盘的成交量；
2. $Vol_{i,n}$ 为第 $i$ 只股票在第 $n$ 个交易日的总成交量；
3. 月度选股下，T=20个交易日；周度选股下，T=5个交易日。

#### 说明
尾盘成交量占比因子和股票未来收益负相关，其较好的选股效果可能源于：1. 尾盘投机度高，容易出现价格操纵行为；2. 非知情交易者（散户）不愿承担日内波动，更倾向于尾盘交易，而知情交易者（机构）则倾向于在早盘交易。

#### 参考文献
冯佳睿等, 2020, 高频因子的现实与幻想, 海通证券.


### 5.3.9 卖单非流动性

$$
r_{i,t} = \alpha + \beta_1 * S_{i,t} + \beta_2 * B_{i,t} + \epsilon_{it}
$$

#### 说明
① $\beta_1$ 为卖出非流动性系数；
② $\beta_2$ 为买入非流动性系数；
③ $S_{i,t}$ 为股票 $i$ 在 $t$ 时间区间内的主动卖出金额；
④ $B_{i,t}$ 为股票 $i$ 在 $t$ 时间区间内的主动买入金额。

卖单非流动性衡量了高频数据下主动卖出的交易金额对于股票价格变动的影响；卖单非流动性在控制风险后的 Fama-MacBeth 截面回归对收益率显著，且预测效果要好于买单非流动性，主要是由于投资者存在亏损厌恶的心理。

#### 参考文献
① Brennan, M. J., Chordia, T., Subrahmanyam, A., & Tong, Q., 2012, Sell-order liquidity and the cross-section of expected stock returns. Journal of Financial Economics 105(3), 523-541.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 5.3.10 价格冲击偏差

利用股票每日的5分钟线，以成交额作为权重进行如下加权最小二乘回归，再计算价格冲击偏差：

$$
return_i = gamma^{up} \cdot I_i \cdot MF_i + gamma^{down} \cdot (1 - I_i) \cdot MF_i
$$

$$
MF_i = MoneyFlow_i / Amount_i
$$

$$
gammabias = \frac{gamma_{up} - gamma_{down}}{(gamma_{up} - gamma_{down})\text{的估计标准差}}
$$

#### 说明
① $return_i$ 分别为第 $i$ 个5分钟线的收益率；
② $MoneyFlow_i$、$Amount_i$ 分别为第 $i$ 个5分钟内的主动净流入金额和成交额；
③ $MF_i$ 为主动净买入占比；
④ $I_i$ 为示性函数，当大于0时取1，否则取0。

价格冲击偏差主要表征了股票在某一时间区间上涨或者下跌的难易程度，价格冲击偏差较大（正值）的股票，相同比例的主动订单对其股价向下的冲击小于向上的冲击，股票容易上涨，反之，股票容易下跌。

#### 参考文献
朱剑涛, 2016, 非对称价格冲击带来的超额收益, 东方证券.


## 5.4 波动率与分布类

### 5.4.1 高频上行波动占比

$$
\text{高频上行波动占比} = \frac{\sum_t (r_t^i I_{\{r_t^i > 0\}})^2}{\sum_t (r_t^i)^2}
$$

其中：
1. $r_t^i$ 为分钟频率下的股票收益序列，如1分钟、5分钟、10分钟等；
2. 在任意选股时刻，因子值为前N日指标的均值，如月度选股，计算因子值时使用的是股票过去20个交易日的均值；
3. 还可计算高频下行波动占比，只需将 $I_{\{r_t^i > 0\}}$ 改为 $I_{\{r_t^i < 0\}}$ 即可。

#### 说明
股票高频收益的上行波动衡量了股票价格拉升的特征。假设有两只股票在过去一段时间有着相同的涨幅，其中一只股票的涨幅由持续稳定的小幅上涨累计带来，而另一只股票的上涨源自股票短期的大幅拉升，那么后者更有可能在收益上出现反转，而后在因子值上也会体现出较高的上行波动率。

#### 参考文献
冯佳睿, 袁林青, 2017, 高频因子之已实现波动分解, 海通证券.

### 5.4.2 高频收益偏度

$$
RSkew_i = \frac{\sqrt{N} \sum_{j=1}^N (r_{ij} - \bar{r}_i)^3}{RVar_i^{3/2}}
$$

$$
RVar_i = \sum_{j=1}^N (r_{ij} - \bar{r}_i)^2
$$

其中：
1. $r_{ij}$ 为股票 $i$ 的1分钟对数收益率序列或5分钟对数收益率序列；
2. 在任意选股时刻，因子值为前N日指标的均值，如月度选股，计算因子值时使用的是股票过去20个交易日的均值。

#### 说明
股票日内的价格形态分布特征对于股票未来收益具有一定预测作用，高频收益偏度就是刻画日内价格形态的特征之一；高频偏度因子与股票未来收益负相关，前期偏度越小的股票未来表现越好。

#### 参考文献
冯佳睿, 袁林青, 2017, 高频因子之股票收益分布特征, 海通证券.

### 5.4.3 理想振幅因子

对单只股票，回溯取其最近N个交易日（默认N=20）的数据，计算每日的振幅（最高价/最低价-1）；

选择收盘价较高的 $\lambda$（默认25%）有效交易日，计算振幅均值得到高价振幅因子 $V_{high}(\lambda)$；选择收盘价较低的 $\lambda$（默认25%）有效交易日，计算振幅均值得到低价振幅因子数学公式：$V_{low}(\lambda)$；

计算理想振幅因子：$V(\lambda) = V_{high}(\lambda) - V_{low}(\lambda)$。

#### 说明
基于股价将振幅因子进行切割得到的理想振幅因子，考虑了不同价格区间的振幅分布信息差异（高价振幅因子具有更强的负向选股能力），选股能力更强。

#### 参考文献
魏建榕, 2020, 振幅因子的隐藏结构, 开源证券.

# 6 流动性因子
## 6.1 北向资金类
### 6.1.1 净流入强度

$$
\text{净流入强度} = \frac{\text{当日陆股通净流入金额}}{\text{当日陆股通买入成交额} + \text{当日陆股通卖出成交额}}
$$

#### 说明
净流入强度与未来一段时间市场涨跌幅呈正相关；总体来看，净流入强度越大，后市上涨概率越高，净流出强度越大，后市下跌的概率越高。

#### 参考文献
魏建榕, 傅开波, 2021, 北上资金: 怎样才是真正的强流入?, 开源证券.

### 6.1.2 北上持仓因子

北上持仓因子为如下HoldPER和MHoldPER因子的等权合成：

$$
HoldPER = \frac{\text{月末持仓市值}}{\text{月末流通市值}}
$$

$$
MHoldPER = mean(\text{过去20个交易日的} \frac{\text{月末持仓市值}}{\text{月末流通市值}})
$$

#### 说明
衡量了北上持仓日均占比和月末占比情况，与股票未来收益呈正比，在横截面选股时，能较好的区分出优质股（多头收益贡献很大）。

#### 参考文献
朱剑涛, 张惠澍, 2020, 从北上资金中提取的系列 alpha 因子, 东方证券.

### 6.1.3 北上交易行为因子

北上交易行为因子为如下MHold、DV2DV_STD、DV2Hold_MAXCP 和 DV2ABSDV_MAXVOL 的等权合成：

$$
MHold = mean(\text{过去20交易日的日度持仓市值})
$$

$$
DV2DV\_STD = \frac{mean(\text{过去20个交易日的日度净流入量})}{std(\text{过去20个交易日的日度净流入量})}
$$

$$
DV2Hold\_MAXCP = \frac{mean(\text{股价最高3个交易日的日度净流入量})}{mean(\text{过去20交易日日度持仓量})}
$$

$$
DV2ABSDV\_MAXVOL = \frac{mean(\text{交易量最高3个交易日的日度净流入量})}{mean(\text{过去20交易日的日度净流入量绝对值})}
$$

#### 说明
北上交易行为主要是指北上资金的流入流出情况。如果股票的北上资金持续流入较多，且流入量较为稳定，说明该股票受到了理性的北上资金的持续追捧，未来收益可期（DV2DV_STD）；如果在过去股价最高的几天还有北上资金加速流入，说明当前股价处于一个较有吸引力的位置，未来收益可期（DV2Hold_MAXCP）；如果北上资金在股票放量时加速进入，说明北上资金看好该股票（DV2ABSDV_MAXVOL）。

#### 参考文献
朱剑涛, 张惠澍, 2020, 从北上资金中提取的系列 alpha 因子, 东方证券.

## 6.2 换手率与成交类
### 6.2.1 月均换手率

换手率月平均值，即：

$$
mean(\text{当月各交易日度换手率序列})
$$

其中
$$
\text{日度换手率} = \frac{\text{成交量}}{\text{流通股本}}
$$

#### 说明
股票流动性衡量指标，股票过去一段时间换手率较高，意味着区间成交量大，单位成交对股价的冲击就较小。

#### 参考文献
朱剑涛, 2016, 非流动性的度量及其横截面溢价, 东方证券.

### 6.2.2 市值调整换手率

进行如下横截面回归得到的残差 $\varepsilon$ 即为市值调整换手率：

$$
ln(Turnover_{i,t}) = \alpha_t + \beta_t ln(MktValue_{i,t}) + \varepsilon_t
$$

#### 说明
① $ln(Turnover)$ 为过去一个月的日均换手率的对数；
② $ln(MktValue)$ 为流通市值的对数。

日均换手率和流通市值的对数存在较高的负相关性，剔除市值影响后有助于提高换手率在多头端的表现，带来明显的正向超额收益。

#### 参考文献
朱剑涛, 2015, 投机、交易行为与股票收益(上), 东方证券.

### 6.2.3 成交额

最近K个月日度成交额的均值，即：

$$
mean(\text{最近K个月日度成交额序列})
$$

#### 说明
衡量股票交易流动性的重要指标，如果股票流动性较差，价格相对较低，投资者对其预期收益就会较高，存在流动性溢价。

#### 参考文献
Brennan, Michael J., Tarun Chordia, and Avanidhar Subrahmanyam, 1998, Alternative factor specifications, security characteristics, and the cross-section of expected stock returns, Journal of Financial Economics 49, 345-373.

### 6.2.4 成交额波动

最近K个月日度成交额的标准差，即：

$$
std(\text{最近K个月日度成交额序列})
$$

#### 说明
衡量了过去一段时间股票成交额变化的稳定程度，取值高，说明成交额波动大，股票的交易流动性风险较大。

#### 参考文献
Chordia, Tarun, Avanidhar Subrahmanyam, and V. Ravi Anshuman, 2001, Trading activity and expected stock returns, Journal of Financial Economics 59, 3-32.

### 6.2.5 成交额波动系数

$$
\frac{mean(\text{最近K个月日度成交额序列})}{std(\text{最近K个月日度成交额序列})}
$$

#### 说明
在衡量成交额高低水平的同时还考虑了成交额的波动情况，是衡量交易流动性的综合性指标。

#### 参考文献
Kewei Hou, Chen Xue, and Lu Zhang, 2020, Replicating anomalies, Review of Financial Studies 32(5), 2019-2133.

### 6.2.6 月度异常换手率

$$
\frac{mean(\text{过去20个交易日的日度换手率序列})}{mean(\text{过去250个交易日的日度换手率序列})}
$$

#### 说明
衡量了过去一个月换手率相对过去一年换手率的变化程度，如果近期换手率有所上升，说明受情绪驱动的非理性投资者对股票保持高度乐观，抬高了股价，降低了未来收益。

#### 参考文献
Liu, Jianan, Robert F. Stambaugh, and Yu Yuan, 2019, Size and value in china, Journal of Financial Economics forthcoming 134(1): 48-69.

### 6.2.7 换手率

最近K个月日度换手率的均值，即：

$$
mean(\text{最近K个月日度换手率序列})
$$

其中
$$
\text{日度换手率} = \frac{\text{成交量}}{\text{流通股本}}
$$

#### 说明
换手率通常被用来衡量股票的投机程度，换手率高，说明投资者主要是为了博取短期收益，交易频繁，存在高投机性。

#### 参考文献
Datar, Vinay T., Narayan Y. Naik, and Robert Radcliffe, 1998, Liquidity and stock returns: An alternative test, Journal of Financial Markets 1, 203-219.

### 6.2.8 换手波动

最近K个月日度换手率的标准差，即：

$$
std(\text{最近K个月日度换手率序列})
$$

其中
$$
\text{日度换手率} = \frac{\text{成交量}}{\text{流通股本}}
$$

#### 说明
衡量了换手率的稳定性，换手率波动越小的股票，其未来收益越高。

#### 参考文献
Chordia, Tarun, Avanidhar Subrahmanyam, and V. Ravi Anshuman, 2001, Trading activity and expected stock returns, Journal of Financial Economics 59(1), 3-32.

### 6.2.9 换手率变异系数

$$
\frac{std(\text{最近K个月日度换手率序列})}{mean(\text{相同时间区间的换手率序列})}
$$

其中
$$
\text{日度换手率} = \frac{\text{成交量}}{\text{流通股本}}
$$

#### 说明
如果换手率变异系数较高，说明换手率的波动较大，也就是说对于持有该股票的投资者在未来卖出股票有着更高的交易成本不确定性，所以这类股票就需要有更高的风险溢价来补偿这些不确定性。

#### 参考文献
Chordia, Tarun, Avanidhar Subrahmanyam, and V. Ravi Anshuman, 2001, Trading activity and expected stock returns, Journal of Financial Economics 59(1), 3-32.

### 6.2.10 股票交易周转率

$$
\text{一段时间内的股票交易总额} \over \text{平均总市值}
$$

其中
$$
\text{平均总市值} = \frac{\text{期初总市值} + \text{期末总市值}}{2}
$$

#### 说明
用于衡量在给定时期内流通在外股票的百分比，反映股票交易活跃程度以及股票交易的相对流量或是股票交易的难易程度。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

## 6.3 冲击与其它
### 6.3.1 带条件的流动性冲击

$$
ILLIQ_{i,t} = \alpha_{0,i} + \alpha_{1,i}ILLIQ_{i,t-1} + \alpha_{2,i}\varepsilon_{i,t-1} + \varepsilon_{i,t}
$$

#### 说明
① $ILLIQ_{i,t}$ 为股票 $i$ 在 $t$ 月的 Amihud 非流动性因子；
② 每个月滚动过去60个月的 Amihud 非流动性因子序列拟合上述 ARMA(1,1) 模型，估计方法采用极大似然法，至少需要包含24个月的样本数据；
③ ARMA(1,1) 估计得到的 $-\varepsilon_{i,t}$ 即为带条件的流动性冲击。

带条件的流动性冲击为真实的非流动性与估计得到的非流动性条件值之差再取相反数，是流动性冲击的一种参数的衡量方法，与股票未来收益呈反比。

#### 参考文献
Turan G. Bali, Lin Peng, Yannan Shen, Yi Tang, 2014, Liquidity Shocks and Stock Market Reactions, The Review of Financial Studies 27(5): 1434-1485.

### 6.3.2 流动性冲击

$$
LIQU_{i,t} = - (ILLIQ_{i,t} - AVGILLIQ_{i|t-12,t-1})
$$

#### 说明
① $ILLIQ_{i,t}$ 为股票 $i$ 在 $t$ 月的 Amihud 非流动性因子；
② $AVGILLIQ_{i|t-12,t-1}$ 为 Amihud 非流动因子在过去12个月的均值。

LIQU衡量了流动性在时间维度上的相对变动。正的流动性冲击（LIQU取值为正，流动性提高）对应较低的未来收益，负的流动性冲击（LIQU取值为负，流动性降低）对应较高的未来收益，因为投资者想要得到更高的风险溢价来补偿非流动性风险，市场对流动性冲击也反应不足。

#### 参考文献
Turan G. Bali, Lin Peng, Yannan Shen, Yi Tang, 2014, Liquidity Shocks and Stock Market Reactions, The Review of Financial Studies 27(5): 1434-1485.

### 6.3.3 Amihud 非流动性因子

$$
ILLIQ_{i,t} = Avg \left[ \frac{|R_{i,d}|}{VOLD_{i,d}} \right]
$$

#### 说明
① $R_{i,d}$ 为 $t$ 月内股票 $i$ 的日度收益率；
② $VOLD_{i,d}$ 为 $t$ 月内股票 $i$ 的日度成交额；
③ 通常要求月内正常交易的交易日天数不少于15天。

预期市场流动性不足会对事前股票超额收益产生积极影响，带来非流动性溢价；取值越高，流动性风险越大，承担风险带来的未来收益越高。

#### 参考文献
Amihud, Yakov, 2002, Illiquidity and stock returns: Cross-section and time-series effects, Journal of Financial Markets 5, 31-56.

### 6.3.4 负收益非流动性

负的日收益率/负收益的当天的日成交额，按月取平均，即：

$$
ILLIQ_{i,t} = \frac{1}{n_d} \sum_{k \in d} \frac{|r_{d_{i,t-k}}|}{Amount_{i,t-k}}
$$

#### 说明
① $Amount_{i,t-k}$ 为负收益当天的日成交额；
② $r_{i,t}$ 是股票 $i$ 在 $t$ 时刻收益率；
③ $n_d$ 为下跌的天数。

负收益非流动性度量了股票收益率为负的时候的流动性，若股票负的非流动性较大，则需要更高的风险溢价来补偿非流动性风险。

#### 参考文献
① Akbas, F., 2011, The volatility of liquidity and expected stock returns. Doctoral dissertation, Texas A&M University.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 6.3.5 非流动性的变异系数

$$
ILLIQ_{i,t} = \frac{|r_{i,t}|}{Amount_{i,t}}
$$

$$
CVILLIQ_i = \frac{\sigma(ILLIQ_i)}{\overline{ILLIQ_i}}
$$

#### 说明
① $Amount_{i,t}$ 为股票 $i$ 在 $t$ 时刻的交易金额；
② $r_{i,t}$ 是股票 $i$ 在 $t$ 时刻收益率；
③ 时间窗口期为过去20个交易日。

股票的非流动性的变异系数较高，说明非流动性的波动较大，对于持有该类股票的投资者来说，未来卖出该类股票有着更高的交易成本不确定性，所以由此能够获得更高的风险溢价。

#### 参考文献
① Akbas, F., 2011, The volatility of liquidity and expected stock returns. Doctoral dissertation, Texas A&M University.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 6.3.6 散户的买卖非平衡性

$$
BSI = \frac{B - S}{B + S}
$$

#### 说明
① $B$ 是散户的买单金额；
② $S$ 是散户的卖单金额。

散户的交易情绪对于股票价格的影响较大，若BSI较大，说明散户在过去持续的买入，股票未来的短期收益率也会较好，反之亦然。

#### 参考文献
① Barber, B. M., Odean, T., & Zhu, N., 2009, Do retail trades move markets?, Review of Financial Studies 22(1), 151-186.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

# 7 规模因子
## 7.1 规模类
### 7.1.1 特异市值

$$
m_{i,t} = \alpha_{0jt} IND_{it} + \alpha_{1jt} b_{it} + \alpha_{2jt} ln(NI)^+_{it} + \alpha_{3jt} I_{<0} ln(NI)^+_{it} + \alpha_{4jt} LEV_{it} + \varepsilon_{it}
$$

#### 说明
① $m_{i,t}$ 为对数市值、$IND_{it}$ 为行业哑变量（如中信一级行业）、$b_{it}$ 为对数净资产；
② $NI^+$ 为净利润绝对值，将净利润拆分成正负两个变量放入模型中；
③ $LEV_{it}$ 为杠杆率，如资产负债率；
④ 上述截面回归得到的残差 $\varepsilon_{it}$ 即为特异市值。

上述市值解释模型得到的特异市值代表了市值不可被公司基本面或市场解释的部分，衡量了当前市值与内生市值的偏离程度；因子值越大，说明当前市值越偏离内生市值越多，市值被高估，股票未来收益越有可能向下跌。

#### 参考文献
Rhodes-Kropf, M., Robinson, D. T., Viswanathan, S., 2005, Valuation waves and merger activity: The empirical evidence, Journal of Financial Economics, 77(3), 561-603.

### 7.1.2 总市值

$$
\text{总市值} = \text{当日收盘价} \times \text{当日总股本}
$$

#### 说明
总市值是反映上市公司规模大小的指标，是最早被发现的、有效的、也最具代表性的风格因子，即小市值股票收益率高于大市值股票。

#### 参考文献
Banz, Rolf W., 1981, The relationship between return and market value of common stocks, Journal of Financial Economics 9, 3-18.

### 7.1.3 自由流通股市值

$$
\text{自由流通股市值} = \text{当日收盘价} \times \text{当日自由流通股本}
$$

#### 说明
流通股是指上市公司股份中可以在交易所交易的股份数量，而自由流通股是在流通股中剔除了持股比例超过5%的股东的持股量，因为这部分股东持股通常是为了控制公司业务，不会产生频繁的交易行为。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 7.1.4 对数总市值

$$
ln(\text{当日收盘价} \times \text{当日总股本})
$$

#### 说明
A股市值分布存在严重厚尾特征，使用对数市值可有效改善因子分布，使其更接近正态分布，增加选股的稳健性。

#### 参考文献
Banz, Rolf W., 1981, The relationship between return and market value of common stocks, Journal of Financial Economics 9, 3-18.

### 7.1.5 非线性市值

$$
LNCAP^3_{i,t} = \alpha_t + \beta_t LNCAP_{i,t} + \epsilon_{i,t}
$$

#### 说明
① $LNCAP_{i,t}^3$ 为对数市值的3次方；
② 进行横截面上的WLS加权回归，回归权重为根号总市值；
③ 回归得到的残差 $\epsilon_{i,t}$，对残差 $\epsilon_{i,t}$ 进行去极值和标准化处理后，即为中市值因子。

非线性市值因子也称为中市值因子，用于捕捉到市值与股票收益的非线性关系；A股中长期存在小市值效应，但市值与股票收益之间并不是线性的，随着市值的上升，因子对股票收益的影响会下降，进而高估中盘股的收益；该因子对市值因子的值更加，大小市值股票的因子值更低。

#### 参考文献
MSCI, 2012, The Barra China Equity Model (CNE5).

### 7.1.6 对数流通市值

$$
ln(\text{当日收盘价} \times \text{当日自由流通股本})
$$

#### 说明
A股市值分布存在严重厚尾特征，使用对数市值可有效改善因子分布，使其更接近正态分布，增加选股的稳健性。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.


# 8 投资因子
## 8.1 成长与投资类
### 8.1.1 资产增长率-单季度

$$
\frac{\text{最近报告期总资产} - \text{上一报告期总资产}}{\text{上一报告期总资产}}
$$

#### 说明
即总资产环比增速，用于衡量总资产的季度增长情况；作者发现，总资产增长越多的公司获得的后续回报就越低；A股回测发现，增长率统计区间长达5年时，因子与未来收益存在负相关关系；统计区间为年度或季度时，存在正相关关系。

#### 参考文献
Cooper, Michael J., Huseyin Gulen, and Michael J. Schill, 2008, Asset growth and the cross-section of stock returns, Journal of Finance 63, 1609-1651.

### 8.1.2 异常资本投资

$$
CI_{t-1} = \frac{CE_{t-1}}{(CE_{t-2} + CE_{t-3} + CE_{t-4})/3} - 1
$$

其中
$$
CE = \frac{\text{资本性支出}}{\text{营业收入}}
$$

#### 说明
资本性支出 = 购建固定资产、无形资产和其他长期资产支付的现金 - 处置固定资产、无形资产和其他长期资产收回的现金净额。

异常资本投资为企业最近一年的资本性支出CE相对此前三年平均值的变化。异常资本投资高的企业，未来收益会更低，而且对于那些有着更大投资自主权的企业（比如有着更高现金流量和更低负债率的企业），这种投资效应更加显著。

#### 参考文献
Titman, Sheridan, KC John Wei, and Feixue Xie, 2004, Capital Investments and Stock Returns, Journal of financial and Quantitative Analysis 39(4), 677-700.

### 8.1.3 账面价值增长率-单季度

$$
\frac{\text{最近报告期的股东权益合计} - \text{上一报告期的股东权益合计}}{\text{上一报告期的股东权益合计}}
$$

#### 说明
反映企业资本规模的扩张速度，是衡量企业总量规模变动和成长状况的重要指标。A股回测发现，增长率统计区间长达5年时，因子与未来收益存在负相关关系；统计区间为年度或季度时，存在正相关关系。

#### 参考文献
Richardson, Scott A., Richard G. Sloan, Mark T. Soliman, and Irem Tuna, 2005, Accrual reliability, earnings persistence and stock prices, Journal of Accounting and Economics 39, 437-485.

### 8.1.4 股票发行数量增长率

$$
issuance_t = \alpha + \beta t + \varepsilon_t
$$

$$
IGRO = - mean(issuance_{t \in \{1,2,3,4,5\}})
$$

其中
$$
\beta
$$
即为股票发行数量增长率。

#### 说明
① $issuance_t$ 为年度流通股本；
② $t=\{1,2,3,4,5\}$。

使用最近5年的年度流通股本对时间 $t$ 进行回归，取时间 $t$ 前的系数，除以最近5年的年度流通股本的平均值，最后取相反数即得到股票发行数量增长率；若使用过去 $n$ 个季度指标计算因子，得到的是季度增长率。

#### 参考文献
MSCI, 2018, Barra China A Total Market Equity Model for Long-Term Investors.

### 8.1.5 负债增长率

$$
\frac{\text{最近报告期负债合计} - \text{上年同期负债合计}}{\text{上年同期负债合计}}
$$

#### 说明
即负债合计同比增速，用于衡量公司负债的年度增长情况；A股中测试发现，增长率统计区间长达5年时，因子与未来收益存在负相关关系；统计区间为年度或季度时，存在正相关关系。

#### 参考文献
Richardson, Scott A., Richard G. Sloan, Mark T. Soliman, and Irem Tuna, 2005, Accrual reliability, earnings persistence and stock prices, Journal of Accounting and Economics 39, 437-485.

### 8.1.6 存货变动

$$
\frac{\text{最近报告期存货} - \text{上年同期存货}}{\text{平均总资产}}
$$

其中
$$
\text{平均总资产} = \frac{\text{本期总资产} + \text{上年同期值}}{2}
$$

#### 说明
作者发现存货增加（减少）的企业在过去5年经历了更高（更低）的盈利水平、成长和异常回报，而这些趋势在存货变化后会立即逆转，以上这些变化与需求变动有关。

#### 参考文献
Thomas, Jacob K., and Huai Zhang, 2002, Inventory changes and future returns, Review of Accounting Studies 7, 163-187.

### 8.1.7 非流动性经营资产变动

$$
\frac{\text{最近报告期的非流动性经营资产} - \text{上年同期值}}{\text{平均总资产}}
$$

其中
$$
\text{非流动性经营资产} = \text{非流动资产} \text{剔除金融资产相关科目后剩余的部分} \approx \text{非流动资产} - \text{长期投资}
$$
(注：图片原文公式描述为“非流动资产中剔除金融资产相关科目后剩余的部分”)

#### 说明
非流动性经营资产主要包括无形资产等分项，会计核算时也带有较大的主观性，容易涉及盈余操纵，可靠性较低；非流动性经营资产的变动与企业未来盈利负相关，与股票未来收益也负相关，容易造成错误定价。

#### 参考文献
Richardson, Scott A., Richard G. Sloan, Mark T. Soliman, and Irem Tuna, 2005, Accrual reliability, earnings persistence and stock prices, Journal of Accounting and Economics 39, 437-485.

### 8.1.8 存货增长率

$$
\frac{\text{最近报告期存货} - \text{上年同期值}}{\text{上年同期值}}
$$

#### 说明
即存货同比增速，低存货增长的公司表现优于高存货增长的公司。

#### 参考文献
Belo, Frederico, and Xiaoji Lin, 2011, The inventory growth spread, Review of Financial Studies 25, 278-313.

### 8.1.9 净经营资产变动

$$
\frac{\text{最近报告期的净经营资产} - \text{上年同期值}}{\text{最近报告期的总资产}}
$$

其中
$$
\text{净经营资产} = \text{股东权益合计(含少数股东权益)} + \text{金融负债} - \text{金融资产} = \text{经营性资产} - \text{经营性负债}
$$

#### 说明
净经营资产增长过快的企业未来盈利能力会有所下降，净经营资产变动与股票未来收益负相关。

#### 参考文献
D. A. Hirshleifer, K. Hou, S. H. Teoh and Y. L. Zhang, 2004, Do Investors Overvalue Firms with Bloated Balance Sheets? Journal of Financial Economics 38(1), 297-331.

### 8.1.10 总应计盈余

$$
\frac{\text{最近12个月总应计盈余(TTM)}}{\text{平均总资产}}
$$

其中
$$
\text{总应计盈余} = \text{净利润} - \text{经营活动产生的现金流量净额}
$$

$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

#### 说明
可靠性较低的应计盈余会导致较低的盈余持久性，投资者没有完全预期较低的盈余持久性，从而导致重大的证券错误定价，是应计盈余管理的衡量指标。

#### 参考文献
Richardson, Scott A., Richard G. Sloan, Mark T. Soliman, and Irem Tuna, 2005, Accrual reliability, earnings persistence and stock prices, Journal of Accounting and Economics 39, 437-485.

### 8.1.11 总应计盈余占比

$$
\frac{\text{最近12个月总应计盈余(TTM)}}{\text{最近12个月净利润(TTM)}}
$$

其中
$$
\text{总应计盈余} = \text{净利润} - \text{经营活动产生的现金流量净额}
$$

#### 说明
应计盈余管理的衡量指标，相比常规计算方式，总应计盈余占比更偏向于反映收益的构成情况，即收益中有多少是应计收益，有多少是现金形式。

#### 参考文献
Hafzalla, Nader, Russell Lundholm, and Matthew E. Van Winkle, 2011, Percent accruals, Accounting Review 86, 209-236.

### 8.1.12 股票发行

$$
ln \left( \frac{\text{当前最新拆分调整后的总股本}}{\text{上年同期拆分调整后的总股本}} \right)
$$

#### 说明
衡量股票发行年度变动情况，作者发现，股票发行与横截面股票收益预测有较强的负相关关系。

#### 参考文献
Pontiff, Jeffrey, and Artemiza Woodgate, 2008, Share issuance and cross-sectional returns, Journal of Finance 63, 921-945.

### 8.1.13 综合股权发行-年度

$$
ln \left( \frac{\text{当前最新总市值}}{\text{上年同期的总市值}} \right) - \text{相同时间区间内的对数累计收益率}
$$

#### 说明
在年度区间内，衡量不归因于该区间内股票收益的股权发行的年度变动情况，与股票未来收益负相关，存在综合股权发行效应；除年度区间外，还可计算单季度、过去5年等时间上的变动。

#### 参考文献
Daniel, Kent D., and Sheridan Titman, 2006, Market reactions to tangible and intangible information, Journal of Finance 61, 1605-1643.

### 8.1.14 综合债务发行-年度

$$
\text{最近1年的总负债对数增长率} = ln \left( \frac{\text{最近报告期总负债}}{\text{上年同期总负债}} \right)
$$

#### 说明
衡量债务发行的年度变动情况；作者发现，过去一段时间的公司债务发行对数增长率与股票未来收益负相关，存在综合债务发行效应；A股中测试发现，统计区间长达5年时，因子与未来收益存在负相关关系；统计区间为年度或季度时，存在正相关关系。

#### 参考文献
Lyandres, Evgeny, Le Sun, and Lu Zhang, 2008, The new issues puzzle: Testing the investment-based explanation, Review of Financial Studies 21, 2825-2855.

### 8.1.15 员工总数增长率-年度

$$
\frac{\text{最近报告期的员工总数} - \text{上年同期的员工总数}}{\text{平均员工人数}}
$$

其中
$$
\text{平均员工人数} = \frac{\text{最近报告期的员工总数} + \text{上年同期的员工总数}}{2}
$$

#### 说明
企业雇佣增长率可以预测未来的股票收益，雇佣增长率较高的企业，未来股票收益相对较低。

#### 参考文献
Belo, F., Lin, X., & Bazdresch, S., 2014, Labor Hiring, Investment, and Stock Return Predictability in the Cross Section, Journal of Political Economy 122(1), 129-177.

### 8.1.16 资本支出增长率-n年

$$
\frac{\text{最近12个月的资本性支出(TTM)} - \text{前置n年的同期值}}{\text{前置n年的同期值}}
$$

其中
$$
n = 2\text{年}(2 \times 12\text{个月}), 3\text{年}(3 \times 12\text{个月})
$$

$$
\text{资本性支出} = \text{购建固定资产、无形资产和其他长期资产支付的现金} - \text{处置固定资产、无形资产和其他长期资产收回的现金净额}
$$

#### 说明
资本性支出的增长与股票未来收益呈反比，而且公司增加资本性支出的行为还会影响传统基本面特征（如市值、账面市值比）与股票收益的关系，降低风险暴露。

#### 参考文献
Anderson, C., and L. Garcia-Feijoo, 2006, Empirical evidence on capital investment, growth options, and security returns, Journal of Finance 61, 171-194.

### 8.1.17 金融负债变动

$$
\frac{\text{最近报告期的金融负债} - \text{上年同期值}}{\text{平均总资产}}
$$

其中
$$
\text{金融负债} = \text{短期借款} + \text{交易性金融负债} + \text{应付票据} + \text{一年内到期的非流动负债} + \text{长期借款} + \text{应付债券}
$$

#### 说明
金融负债主要包含一些计息债务，会计核算时简洁明确，可靠性高，盈余操纵空间很小；金融负债的变动与企业未来盈利正相关，与股票未来收益也正相关。

#### 参考文献
Richardson, Scott A., Richard G. Sloan, Mark T. Soliman, and Irem Tuna, 2005, Accrual reliability, earnings persistence and stock prices, Journal of Accounting and Economics 39, 437-485.

### 8.1.18 流动性经营资产变动

$$
\frac{\text{最近报告期的流动性经营资产} - \text{上年同期值}}{\text{平均总资产}}
$$

其中
$$
\text{流动性经营资产} = \text{流动资产中剔除金融资产相关科目后的剩余部分} \approx \text{应收账款} + \text{应收票据} + \text{预付账款} + \text{其他应收款} + \text{存货} + \text{待摊费用}
$$

#### 说明
流动性经营资产中的分项在会计核算时带有较大的主观性，其中的应收款项、存货等都容易涉及盈余操纵，可靠性较低；流动性经营资产的变动与企业未来盈利负相关，与股票未来收益也负相关，容易造成错误定价。

#### 参考文献
Richardson, Scott A., Richard G. Sloan, Mark T. Soliman, and Irem Tuna, 2005, Accrual reliability, earnings persistence and stock prices, Journal of Accounting and Economics 39, 437-485.

### 8.1.19 流动性经营负债变动

$$
\frac{\text{最近报告期的流动性经营负债} - \text{上年同期值}}{\text{平均总资产}}
$$

其中
$$
\text{流动性经营负债} = \text{流动负债中剔除金融负债相关科目后的剩余部分} \approx \text{流动负债} - (\text{短期借款} + \text{交易性金融负债} + \text{应付票据} + \text{一年内到期的非流动负债})
$$

#### 说明
流动性经营负债主要指流动性负债中的各类应付款项，对应的是企业的财务义务，可靠性高，盈余操作空间小，与股票未来收益正相关。

#### 参考文献
Richardson, Scott A., Richard G. Sloan, Mark T. Soliman, and Irem Tuna, 2005, Accrual reliability, earnings persistence and stock prices, Journal of Accounting and Economics 39, 437-485.

# 9 动量因子
## 9.1 动量类
### 9.1.1 最近52周的最高价

$$
\frac{\text{当前收盘价}}{\text{最近52周最高价}}
$$

#### 说明
股价最接近52周高点的公司平均股价要比离52周高点远的公司能够获得更高的因子的回报。

#### 参考文献
George, T. J., & Hwang, C. Y., 2004, The 52-week high and momentum investing, The Journal of Finance 59(5), 2145-2176.

### 9.1.2 残差动量-基于FF3

$$
residualmom_{i,t} = \frac{\sum_{t=T-12}^{T-2} \epsilon_{i,t}}{\sqrt{\sum_{t=T-12}^{T-2} \epsilon_{i,t}^2}}
$$

其中 $\epsilon_{i,t}$ 为 FF3 模型回归残差：
$$
r_{i,t} = \alpha_i + \beta_{1,i} RMRF_t + \beta_{2,i} SMB_t + \beta_{3,i} HML_t + \epsilon_{i,t}
$$

#### 说明
使用过去36个月（即：T-36至T-1）的收益率数据，构Fama-French三因子模型，得到残差序列，利用T-12到T-2这11个月的残差计算残差动量。
基于渐进的信息扩散假设（gradual-information-diffusion），信息在投资者之间的扩散是缓慢的，而且相比于影响所有公司的公共事件，投资者对于那些针对公司的特定事件的反应则更加迟缓，这就形成了残差动量。

#### 参考文献
Blitz, D., M. X. Hanauer, and M. Vidojevic, 2020, The idiosyncratic momentum anomaly, International Review of Economics & Finance 69, 932-957.
Blitz, David, Joop, Huij, and Martin Martens, 2011, Residual Momentum, Journal of Empirical Finance 18(3), 506-521.

### 9.1.3 动量加速度

$$
\text{最近后6个月(t-6至t-1)的累计日收益率} - \text{最近前6个月(t-12至t-7)的累计日收益率}
$$

#### 说明
在横截面上基于动量加速度进行排名构建多空动量策略能获得显著的异常收益。

#### 参考文献
Gettleman, Eric, and Joseph M. Marks, 2006, Acceleration strategies, working paper.

### 9.1.4 残差动量-基于CAPM

$$
IMOM_{i,t} = \prod_{j=2}^{12} (1 + \epsilon_{i,t-j})
$$

其中
$$
r_{i,t} - r_{f,t} = \alpha + \beta(r_{m,t} - r_{f,t}) + \epsilon_{i,t}
$$

#### 说明
基于 CAPM 的回归残差直接计算残差动量。
基于渐进的信息扩散假设（gradual-information-diffusion），信息在投资者之间的扩散是缓慢的，而且相比于影响所有公司的公共事件，投资者对于那些对公司的特定事件的反应则更加迟缓，这就形成了残差动量。

#### 参考文献
Chaves, Denis B, 2012, Eureka! A Momentum Strategy that Also Works in Japan, Social Science Electronic Publishing, Available at SSRN: https://ssrn.com/abstract=1982100.

### 9.1.5 趋势动量

$$
MA_{j,t,L} = \frac{Pt}{\frac{P_{j,d-L+1} + P_{j,d-L+2} + ... + P_{j,d-1} + P_{j,d}}{L}}
$$

#### 说明
以月度因子为例，在每月t最后一个交易日d计算每只股票在不同时间尺度L的移动平均价格作为趋势信号。
由于不同股票的价格在量级上存在差异，为了使趋势信号在截面上具有可比性，将移动平均价格序列除以d日最新收盘价，做标准化处理。

#### 参考文献
Chan, Louis K. C., Josef Lakonishok, and Theodore Sougiannis, 2001, The stock market valuation of research and development expenditures, Journal of Finance 56, 2431-2456.

### 9.1.6 标准化的未预期盈余

$$
\text{盈余预差} = \text{当前第t季度的每股收益} - \text{第(t-4)季度的每股收益}
$$

$$
\text{标准化的未预期盈余} = \frac{\text{最近1个季度盈余预差}}{std(\text{最近8个季度盈余预差})}
$$

#### 说明
盈余公告后价格漂移效应PEAD的衡量指标，当实际盈利高于预期盈利时，在盈余公告发布后3-6个月内能给股价带来正面效应；反之亦然。

#### 参考文献
Foster, George, Chris Olsen, and Terry Shevlin, 1984, Earnings releases, anomalies, and the behavior of security returns, Accounting Review 59, 574-603.

### 9.1.7 时间序列动量

$$
TSMOM_{mi} = sign \left( \frac{1}{N} \sum_{j=1}^{12} r_{m-j,i} \right) \times \frac{r_{m,i}}{\sigma_{m,i}}
$$

其中
$$
\text{① m为月份，i代表股票；}
$$
$$
\text{② } r_{m,i} \text{表示股票i第m月相对时间序列上的超额收益，计算方法为个股收益率减去之前月收益率的指数移动平均值；}
$$

#### 说明
时序动量考虑的是个股本身过去一段时间的收益表现，与未来收益有显著的负相关关系。

#### 参考文献
① Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H., 2012, Time series momentum, Journal of Financial Economics 104(2), 228-250.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 9.1.8 收益季节性

$$
\text{在 t 月的月末，计算过去2-5年(即 t-12 月)的月超额收益率，即得到 2-5 年期的收益季节性。}
$$

$$
\text{在 t 月的月末，计算过去2-5年(t-12月)的月超额收益率，即得到 2-5 年期的收益季节性。}
$$
(注：图片中文字重复，整理为“计算去年同月(t-12月)的月超额收益率，即得到1年期的收益季节性”及“计算滞后2-5年的相同日历月超额收益率的均值，即得到2-5年期的收益季节性”)

#### 说明
作者发现股票收益存在季节性，未来收益可以用过去同期（差）的收益来预测；某只股票在过去某个月份的表现优（差）于其他股票，在下一个相同日历月的表现通常是继续优（差）于其他股票。
滞后的月份数需满足12的倍数。

#### 参考文献
Heston, S. L. and R. Sadka, 2008, Seasonality in the cross-section of stock returns, Journal of Financial Economics 87(2), 418-445.

### 9.1.9 趋势因子

$$
M\tilde{A}_{j,t,L} = \frac{MA_{j,t,L}}{P_{t}^{jd}}
$$

$$
MA_{j,t,L} = \frac{Pt}{\frac{P_{j,d-L+1} + \dots + P_{j,d}}{L}}
$$

$$
E[r_{i,t+1}] = \sum_{m=1}^{12} \frac{1}{12} \beta_{i,t+1-m} M\tilde{A}_{j,t,L}
$$

#### 说明
趋势因子综合考量了在不同时间点上，不同时间跨度的动量或者反转因子对股票未来收益的贡献。
使用预期收益率和个股j在t期的最新均线指标值计算得出每股预期收益率。
对各趋势信号，使用自过去12个月的 $\beta_{i,t}$ 来计算下个月的预期因子收益率。

#### 参考文献
Han, Y., Zhou, G., Y. Zhu, 2016, A trend factor: any economic gains from using information over investment horizons? Journal of Financial Economics, 122(2), 352-375.

### 9.1.10 收益季节性反转

$$
\text{在 t 月的月末，计算滞后 2-5 年区间内所有非同月的月超额收益均值（不包含同期日历月的收益），取均值，即得到 2-5 年期的收益率季节性反转。}
$$

#### 说明
收益季节性反转对应的是剔除相同日历月后“其他月份”的收益，同样包含（了未来收益的相关信息；这些过去的其他的月份的收益负向的预测未来收益，比如：如果其他非12月的收益较高，预示着下个12月份的收益较低，反之亦然。

#### 参考文献
① Heston, S. L. and R. Sadka, 2008, Seasonality in the cross-section of stock returns, Journal of Financial Economics 87(2), 418-445.
② M. Keloharju, J.T. Linnainmaa, P.M. Nyberg, 2021, Are return seasonalities due to risk or mispricing?, Journal of Financial Economics 139(1), 138-161.

### 9.1.11 价差偏离度

$$
PriceSpead_{i,t} = ln(StockPrice_{i,t}) - ln(ReferencePrice_{i,t})
$$

$$
SpreadBias_{i,t} = \frac{PriceSpead_{i,t} - mean(PriceSpead_{i,t})}{std(PriceSpead_{i,t})}
$$

#### 说明
每月月底在全市场搜索与股票 i 距离最近（相似度最高）的 N=10 只股票等权构成该股票的特征组合，特征组合的净值价格我们称之为参考价格：$ReferencePrice_{i,t}$，其中，股票间的距离 = 1 - 两股票过去250个交易日涨跌幅的 pearson 相关系数；
利用高频分钟 K 线，每日的 K 线最短路径非流动性因子：
价差偏离度衡量了个股相对其特征组合（相似的股票）价格的偏离程度。在基本面没有发生重大变化的前提下，价差偏离度低，股票近期跑输其特征组合，股票相对估值偏低，有向上回复的动力，有正的预期超额收益，价差偏离度越高，股票处于相对高估状态，后期有回调的压力。

#### 参考文献
朱剑涛, 2016, 投机、交易行为与股票收益(下), 东方证券.

### 9.1.12 长期反转

$$
\text{t月末，计算第 (t-59) 月至第 (t-12) 月的日度累计收益率}
$$

#### 说明
投资者在短期内对某些消息会有过度反应，使得股票被严重高估或低估，形成错误定价，在均值回复作用下，后期会出现反转；所以，做多（做空）前期表现差（好）的股票，能获得可观的超额收益。

#### 参考文献
De Bondt, Werner F. M., and Richard Thaler, 1985, Does the stock market overreact?, Journal of Finance 40, 793-805.

### 9.1.13 标准化的未预期收入

$$
\text{收入预差} = \text{当前第t季度的营业收入} - \text{第(t-4)季度的营业收入}
$$

$$
\text{标准化的未预期收入} = \frac{\text{最近1个季度收入预差}}{std(\text{最近8个季度收入预差})}
$$

#### 说明
盈余公告后价格漂移效应PEAD的衡量指标，当实际收入高于预期收入时，在公告发布后3-6个月内能给股价带来正面效应；反之亦然。

#### 参考文献
Jegadeesh, Narasimhan, and Joshua Livnat, 2006, Revenue surprises and stock returns, Journal of Accounting and Economics 41, 147-171.

### 9.1.14 未预期税费支出

$$
\text{最近单季度所得税费用} - \text{上年同期单季度所得税费用}
$$

#### 说明
税收支出的季节性差异与未来收益成正相关。税收支出包含有关核心盈利能力的信息，这些信息会延迟反映股价上。

#### 参考文献
Thomas, Jacob, and Frank X. Zhang, 2011, Tax expense momentum, Journal of Accounting Research 49, 791-821.

### 9.1.15 资本收益过剩

$$
CGO_t = \frac{P_{t-1} - RP_t}{P_{t-1}}
$$

$$
RP_t = \frac{1}{k} \sum_{n=1}^{T} (V_{t-n} \prod_{\tau=1}^{n-1} (1-V_{t-n+\tau})) P_{t-n}
$$

#### 说明
参考价格代表了当前的平均成交价格（价格前基于换手率的权重衡量了股票持有至今仍未被交易的概率）。CGO衡量了当前时点投资者的平均盈亏情况；CGO越大，与股票未来收益正相关，CGO越小，表示投资者平均盈利越高，卖出股票的倾向越强烈，股票更有可能被低估。

#### 参考文献
Huijun W., Jinghua Y., Jianfeng Y., 2016, Reference-dependent preferences and the risk-return trade-off, Journal of Financial Economics 123(2): 395-414.

### 9.1.16 加速度动量

$$
P_{i,t} = \alpha + \beta t + \gamma t^2
$$

其中
$$
\text{① } P_{i,t} \text{为股票i在t日的价格，如收盘价等；}
$$
$$
\text{② } t \text{为时间等差序列，} t=1,2,3,...,n \text{，表示过去的n, ..., 3,2,1天；}
$$
$$
\text{③ } n \text{估计得到的 } \gamma \text{ 即为加速度动量，} \beta \text{ 为价格涨跌强弱。}
$$

#### 说明
加速度动量衡量了股票历史价格上涨或下跌的速度（对历史价格的visual pattern的刻画）；当股票呈现“价格处于上涨（下跌）趋势且价格走势加速上涨（下跌）”模式时，更能吸引投资者注意力并引起过度反应，进而获得比传统动量更高的收益。

#### 参考文献
Chen, L. W., & Yu, H. Y., 2014, Investor Attention, Visual Price Pattern, and Momentum Investing, In 27th Australasian Finance and Banking Conference.

### 9.1.17 盈余公告回报

$$
\text{预期累计收益率} = \text{大盘指数的累计收益率}
$$

$$
\text{公告当天及前后交易日共3天的个股累计收益} - \text{对应时间段上的预期累计收益}
$$

#### 说明
盈余公告回报捕捉了市场对公司盈余公告中包含的意外信息的反应，是PEAD效应的衡量指标。

#### 参考文献
Brandt, M. W., Kishore, R., Santa-Clara, P., & Venkatachalam, M., 2008, Earnings announcements are full of surprises, SSRN elibrary.

### 9.1.18 财务动量

$$
F\text{-}Momen \ it = \sum_{j \neq i} F\text{-}link \ ijt \cdot Ret \ jt
$$

$$
F\text{-}link \ ijt = \frac{C_{it} \cdot C_{jt}}{\sqrt{C_{it} \cdot C_{it}} \sqrt{C_{jt} \cdot C_{jt}}}
$$

#### 说明
财务动量衡量的是财务能力相似度高的公司之间的股价协同效应，是通过加工方法创新寻找的新Alpha。

### 9.1.19 行业动量-横向切割

$$
\text{① 对各行业，回溯过去20日的成分股数据；}
$$
$$
\text{② 将成分股按近20日成交金额从大到小排序，逐一累积成交金额；}
$$
$$
\text{③ 取累计成交金额占比达到 } \lambda \% \text{ (如 } \lambda=60 \text{ )，认定为龙头股，其余下的为普通股；}
$$
$$
\text{④ 分别计算龙头股、普通股的近20日平均涨幅：} R\_\text{龙头} \text{，} R\_\text{普通} \text{；}
$$
$$
\text{⑤ 可以进一步构建牵引力因子 } G = R\_\text{龙头} - R\_\text{普通} \text{。}
$$

#### 说明
测试结果表明：龙头股呈现动量效应，普通股呈现反转效应。

#### 参考文献
① Tobias J. Moskowitz, Mark Grinblatt, 1999, Do industries explain momentum?, Journal of Finance 54(4), 1249-1290.
② 聂建梅, 2020, A股行业动量的精细结构, 开源证券.

### 9.1.20 信息离散度

$$
ID = sign(PRET) \times [\%neg - \%pos]
$$

其中
$$
\text{① } PRET \text{ 为过去12个月的累计收益率，剔除最近1个月的收益数据；}
$$
$$
\text{② } \%neg \text{、} \%pos \text{ 分别为这段时间内下跌交易日的占比、上涨交易日的占比。}
$$

#### 说明
作者认为“一系列频繁但微小的变化对于人的吸引力远不如少数却显著的变化对人的吸引力大”。投资者对于连续信息造成的股价变化反应不足；信息离散度越低（信息连续性越强）越好。

#### 参考文献
Da, Zhi, Umit G. Gurun, and Mitch Warachka, 2014, Frog in the pan: Continuous information and momentum, Review of Financial Studies 27(7), 2171-2218.

### 9.1.21 分析师共同覆盖关联股票动量

$$
CS_{it} = \sum_{j=1}^{N} \frac{N_{i,j}}{N_i} Ret_{j,20}
$$

其中
$$
\text{① } N_{i,j} \text{ 为共同覆盖股票 i 和股票 j 的分析师数量；}
$$
$$
\text{② } Ret_{j,20} \text{ 为股票 j 过去1个月 (20个交易日) 的收益率；}
$$
$$
\text{③ N 为股票池中的股票数量。}
$$

#### 说明
作者发现分析师共同覆盖是动量溢出效应（经济上有关联或基本面相似的公司，信息是会相互影响的，如果投资者或分析师对关联公司的相关信息存在反应不足，那么就会产生跨公司的收益可预测性）的根源，以“分析师共同覆盖”关系构建的关联股票动量因子可以解释其他关联关系（如行业、产业链、地域等）构建的关联动量。

#### 参考文献
Ali, U., Hirshleifer, D. A., 2020, Shared analyst coverage: Unifying momentum spillover effects, Journal of Financial Economics 136(3), 649-675.

### 9.1.22 基于排序的动量

$$
rank_{i,d} = \frac{y(R_{i,d}) - \frac{N_d+1}{2}}{\sqrt{\frac{(N_d+1)(N_d-1)}{12}}}
$$

$$
rank_{i,t}(N, M) = \frac{1}{M} \sum_{j=t-M}^{t} rank_{i,j}
$$

#### 说明
① $R_{i,d}$ 为股票 $i$ 在第 $d$ 天的收益率，$y(R_{i,d})$ 为其在截面上的排名，$N_d$ 为第 $d$ 天的股票总数；
② $rank_{i,d}$ 为第 $d$ 天的标准化排名得分；
③ 因子值为过去一段时间（如 $M$ 天）内标准化排名得分的均值。

相比于传统的动量构建方式（只考虑区间头尾的价格），RANK动量考虑了区间内每日的收益率情况，并且采用rank排名代替绝对收益率大小，能更稳健地衡量股价波动的加权效应，降低异常值的影响。

#### 参考文献
① 朱剑涛, 刘静远, 2019, 存在于全市场范围内的稳健动量效应, 东方证券.
② Wright, J. H., 2000, Alternative variance-ratio tests using ranks and signs, Journal of Business and Economic Statistics 18, 1-9.

### 9.1.23 短期反转

$$
\text{计算上个月（如过去20个交易日）的累计收益率}
$$

#### 说明
上个月的月度回报与下个月的月度股票回报呈负相关；过去一段时间表现好的股票，未来表现较差，存在短期反转效应。

#### 参考文献
Jegadeesh, Narasimhan, 1990, Evidence of predictable behavior of security returns, Journal of Finance 45, 881-898.

### 9.1.24 盈余公告股票跳动

$$
\text{公告披露后下一日开盘价} - \text{公告日收盘价}
$$

#### 说明
盈余公告跳动反映了盈余公告对股价带来的冲击，正向跳空说明盈余超预期，负向跳空说明盈余低预期；该指标向跳空说明盈余低预期。

#### 参考文献
Zhou, H., & Zhu, J. Q., 2012, Jump on the Post-Earnings Announcement Drift (corrected), Financial Analysts Journal 68(3), 63-80.

### 9.1.25 行业动量-纵向切割

$$
\text{日内收益率} = \text{今收盘价} - \text{今开盘价}
$$

$$
\text{隔夜收益率} = \text{今开盘价} - \text{昨收盘价}
$$

#### 说明
① 对每个行业，回溯过去20日（或其他窗口）的行情数据；
② 将20个交易日的日内收益率加总，得到日内因子；将过去20个交易日的隔夜收益率加总，得到隔夜因子；
③ 测试结果表明：日内因子呈现动量效应，隔夜因子呈现反转效应（即数值越大，年化收益率倾向于越低）。

#### 参考文献
① 魏建榕, 2020, A股行业动量的精细结构, 开源证券.
② Tobias J. Moskowitz, Mark Grinblatt, 1999, Do industries explain momentum?, Journal of Finance 54(4), 1249-1290.

### 9.1.26 基本面隐含收益

$$
FIR_{i,t} = \sum_{k=1}^{K} \sum_{L \in \{1,2,4,8\}} E_t[\beta_{i,t+1}^{k,L}] MA_{i,t,L}^k
$$

其中
$$
MA_{i,t,L}^k = \frac{1}{L} (F_{i,t}^k + F_{i,t-1}^k + \dots + F_{i,t-L+1}^k)
$$

#### 说明
① $F$ 为选取的7个基本面因子：ROE、ROA、总资产收益率、每股盈利、CPA（现金流资产比）、GPA（毛利润资产比）、Net payout ratio（净派息率）；
② $MA$ 为基本面因子的移动均值；
③ 利用回归模型估计基本面因子的系数 $\beta$，并以此预测未来的基本面隐含收益 $FIR$。
FIR综合了多方面的基本面信息，又考虑基本面数据的变化趋势，能获得比传统动量更高的收益。

#### 参考文献
① Huang Dashan, Zhang Huacheng, Zhou Guofu, 2019, Twin Momentum: Fundamental Trends Matter, Available at SSRN: https://ssrn.com/abstract=2894068.
② 广发证券, 2019, FIR 既综合了多方面的基本面信息, 又考虑基本面数据的变化趋势.

### 9.1.27 基于振幅切割的动量

$$
\text{以160个交易日作为窗口期，将振幅较低的 } \lambda \% \text{ 个交易日的涨跌幅加总，得到动量因子。}
$$

#### 说明
① $\lambda \%$ 取值范围在 $[50\%, 70\%]$；
② 作者从交易行为维度出发，利用日度振幅，从长端涨跌幅中切割出了有效的动量因子；
③ 而且测试发现低振幅水平下涨跌幅因子呈现动量效应，高振幅水平下涨跌幅因子呈现反转效应，并且动量效应的分布和强度具有不对称性。

#### 参考文献
魏建榕, 高鹏, 王志豪, 2020, A股市场中如何构造动量因子?, 开源证券.

# 10 波动率因子
## 10.1 波动率与风险类
### 10.1.1 异常尾部概率E

$$
E_{\varphi} = \int_{k}^{+\infty} f(x)dx - \int_{-\infty}^{-k} f(x)dx = P(x \ge k) - P(x \le -k)
$$

#### 说明
① x 为特质收益率 $\varepsilon_{i,d}$，通过 $R_{i,d} = \alpha_i + \beta_i R_{m,d} + \gamma_i R_{m,d}^2 + \varepsilon_{i,d}$ 估计得到；
② k 为左、右异尾的阈值，如 k=1.5；
③ 可以使用过去3个月的日度收益数据计算上述因子。

优于偏度的新的收益非对称性的度量方式之一；若该因子大于零，则表示该股票在过去出现大涨的概率胜于大跌的概率，而投资者出于“追涨”的心理往往乐于购买这类股票，从而使得这类股票在未来下跌的概率增大。

#### 参考文献
① 朱剑涛, 2021, 收益率的非对称分布与尾部蕴含的 Alpha, 东方证券.
② Jiang, L., Wu, K., Zhou, G., & Zhu, Y., 2020, Stock return asymmetry: beyond skewness, Journal of Financial and Quantitative Analysis, 1-65.

### 10.1.2 异常尾部概率S

基于熵原理构造 $S_{\varphi}$：
$$
S_{\varphi} = Sign(E_{\varphi}) * \frac{1}{2} \left\{ \int_{-\infty}^{-k} (f_1^{\frac{1}{2}} - f_2^{\frac{1}{2}})^2 dx + \int_{k}^{+\infty} (f_1^{\frac{1}{2}} - f_2^{\frac{1}{2}})^2 dx \right\}
$$

采用非参的核密度估计法估计未知分布 $f_1(x), f_2(x)$，核密度函数选取高斯核函数：
$$
\widehat{f(x)} = \frac{1}{nh} \sum_{i=1}^{n} \kappa \left( \frac{r_i - x}{h} \right)
$$
$$
\kappa(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}z^2}
$$

#### 说明
① x 为特质收益率 $\varepsilon_{i,d}$，通过 $R_{i,d} = \alpha_i + \beta_i R_{m,d} + \gamma_i R_{m,d}^2 + \varepsilon_{i,d}$ 估计得到；
② k 为左、右异尾的阈值，如 k=1.5；
③ 可以使用过去3个月的日度收益数据计算上述因子；
④ h 为带宽，估计采用 Silverman(1986) 经验法则，即 $h \approx 1.06 \widehat{\sigma} n^{-1/5}$。

优于偏度的新的收益非对称性的度量方式之一；横截面上股票过去收益正向非对称性越高，未来收益越低；偏度对于未来收益的影响为正或者负与市场风险、股票特质波动率、投资者信心指数、市场流动性以及投资者未实现盈利值有关，如在投资者信心高涨的时期，偏度与未来收益负相关，而在投资者信心低落的时期，偏度与未来收益正相关。

#### 参考文献
① 朱剑涛, 2021, 收益率的非对称分布与尾部蕴含的 Alpha, 东方证券.
② Jiang, L., Wu, K., Zhou, G., & Zhu, Y., 2020, Stock return asymmetry: beyond skewness, Journal of Financial and Quantitative Analysis, 1-65.

### 10.1.3 最小收益率

最近K个月的最低日度收益率或是排名倒数n名的最低日度收益率的平均值，即：

$$
- min(\text{最近K个月日度收益率序列})
$$

#### 说明
作者用最小收益率来检查极端回报和波动率之间关系，如果异常收益是由波动率推动的，MIN应该存在与MAX相反的效应，比如，如果投资者有偏斜偏好，那么回报为负偏的股票应该要求更高的回报。

#### 参考文献
Bali, Turan G., Nusret Cakici, and Robert F. Whitelaw, 2011, Maxing out: Stocks as lotteries and the cross-section of expected returns, Journal of Financial Economics 99, 427-446.

### 10.1.4 尾部Beta

尾部Beta为市场超额收益超出其VaR时CAPM模型中的 $\beta_j^T$：
$$
R_j^e = \beta_j^T R_m^e + \varepsilon_j, R_m^e < -VaR_m(\bar{p})
$$

基于极值理论的估计方法得到 $\widehat{\beta_j^T}$：
$$
\widehat{\beta_j^T} = \tau_j(k/n)^{1/\widehat{\alpha_m}} \frac{\widehat{VaR}_j(k/n)}{\widehat{VaR}_m(k/n)}
$$

$$
\frac{1}{\widehat{\alpha_m}} = \frac{1}{k} \sum_{i=1}^{k} \log X_{n,n-i+1}^{(m)} - \log X_{n,n-k}^{(m)}
$$

$$
\tau_j(k/n) = \frac{1}{k} \sum_{t=1}^{n} 1_{\{ x_t^{(j)} > X_{n,n-k}^{(j)} \text{ and } X_t^{(m)} > X_{n,n-k}^{(m)} \}}
$$

#### 说明
① $\bar{p}$ 为显著性水平，常取值为 5%，即有 $P(R_m^e < -VaR_m(\bar{p})) = \bar{p}$；
② $\widehat{VaR}_j(k/n), \widehat{VaR}_m(k/n)$ 分别为股票 j 和市场的第 k+1 大损失；
③ k 为 n 个交易日中股票或市场收益损失超出其 VaR 值的天数，所以 $k \approx 0.05n$；
④ 令 $X_t^{(m)} = -R_{m,t}^e$，其中 $t=1,2...,n$，n 为计算期间内的交易日天数；将市场超额收益的相反数 $X_t^{(m)}$ 从小到大排序，得到 $X_{n,1}^{(m)} \le X_{n,2}^{(m)} \le ... \le X_{n,n}^{(m)}$；$X_t^{(j)}, X_{n,i}^{(j)}$ 的定义方式同上。

尾部Beta表示当市场出现极端下跌行情时，个股收益对市场收益的敏感性。若尾部Beta越大，表示个股收益对市场的极端负收益越敏感，使得收益率低于尾部Beta偏小的股票。

#### 参考文献
① 朱剑涛, 2021, 收益率的非对称分布与尾部蕴含的 Alpha, 东方证券.
② Oordt, M. V., & Zhou, C., 2016, Systematic tail risk. Ssrn Electronic Journal, 51.

### 10.1.5 尾部风险

先计算每个月整个市场的尾部风险 $\lambda$：
$$
\lambda_t = \frac{1}{K_t} \sum_{k=1}^{K_t} \frac{r_{kt}}{\mu_t}
$$

再通过时序回归计算个股在 $\lambda$ 上的暴露 $\beta_i$，即为个股的尾部风险因子：
$$
r_{i,t} = \alpha + \beta_i \lambda_t + \varepsilon_i
$$

#### 说明
① $\mu_t$ 为第 t 月所有股票日收益率放在一起计算得到的 25% 分位数；
② $r_{kt}$ 为第 t 月所有股票日收益率中第 k 个小于 $\mu_t$ 的日度收益率；
③ $K_t$ 为第 t 月所有股票日收益率小于 $\mu_t$ 的个数；
④ 时序回归的时间窗口可以为 120 个月，$\mu_t$ 比 $r_{i,t}$ 滞后一个月。

尾部风险通常指的是发生负面罕见事件的风险，尾部风险对总市场回报具有很强的预测能力，而且尾部风险负载较高的股票比尾部风险负载较低的股票能获得更高的未来收益。

#### 参考文献
Kelly, B. T., & Jiang, H., 2014, Tail Risk and Asset Prices, Review of Financial Studies 27(10), 2841-2871.

### 10.1.6 在险价值

$$
Prob(\Delta P < -VaR) = 1 - \alpha
$$

#### 说明
① $\Delta P$ 为某一金融资产在一定持有期 $\Delta t$ 内的价值损失额；
② $\alpha$ 为置信水平，通常取 95%。

VaR指的是在一定置信水平 $\alpha$ 下，某一金融资产或证券组合价值在未来特定时期内的最大可能损失。

#### 参考文献
G30, G. D. S. G., 1993, Derivatives: Practices and Principles. G30.

### 10.1.7 条件风险价值

$$
CVaR_{\alpha} = - \frac{\int_{0}^{\alpha} VaR_{\gamma}(X)d\gamma}{\alpha}
$$

$$
CVaR_{\alpha} = E[X|X < VaR_{\alpha}]
$$

#### 说明
① $\alpha$ 为置信水平，通常取 95%。

CVaR指的是在一定的置信水平 $\alpha$ 下，某一金融资产或证券组合价值在未来特定时期内超出VaR的那部分损失的条件期望；CVaR不仅考虑了不利情况（损失超过VaR）发生的概率，而且考虑了不利情况发生时的损失程度（超过VaR的那部分损失的条件期望值），有效的改善了VaR在处理损失分布的后尾现象时存在的问题。

#### 参考文献
R. T. Rockafellar, S. Uryasev, 2000, Optimization of conditional value-at-risk, Journal of Risk 2(3), 21-41.

### 10.1.8 极端下行风险

利用个股过去两年Fama-French三因子回归的残差收益率月极小值数据，通过极值分布和极大似然估计得到刻画分布尾部厚度的参数，极值分布函数可表示为：

$$
H(x) = 1 - \exp \left[ - \left( 1 - \gamma * \frac{x - \mu}{\sigma} \right)^{- \frac{1}{\gamma}} \right]
$$

$$
1 - \gamma * \frac{x - \mu}{\sigma} > 0, \quad \gamma \ne 0
$$

#### 说明
① $\gamma$ 度量尾部厚度；
② $\mu$ 为均值；
③ $\sigma$ 为标准差。

极端下行风险度量了股票收益率分布尾部厚度，作者在美股市场验证极端下行风险与期望收益率有着显著的正相关关系，但在A股市场表现并不理想。

#### 参考文献
① Huang, W., Liu, Q., Rhee, S. G., & Wu, F., 2012, Extreme downside risk and expected stock returns. Journal of Banking & Finance 36(5), 1492-1502.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 10.1.9 最大收益率

最近 K 个月的最高日度收益率，或是排名前 n 名的最高日度收益率的平均值，即：

$$
max(\text{最近K个月日度收益率序列})
$$

#### 说明
最大收益率用于衡量股票异常正收益，表示潜在的收益可能性，研究表明过去一个月的最大日回报与预期股票回报之间存在着显著的负相关关系。

#### 参考文献
Bali, Turan G., Nusret Cakici, and Robert F. Whitelaw, 2011, Maxing out: Stocks as lotteries and the cross-section of expected returns, Journal of Financial Economics 99, 427-446.

### 10.1.10 价格时滞

在每月月底利用本月交易数据回归如下方程：

$$
r_{i,t} = \alpha_i + \beta_i MKT_t + \sum_{k=1}^{n} \delta_i^{(k)} MKT_{t-k} + \varepsilon_{i,t}
$$

$$
PriceDelay = 1 - \frac{R^2_{\delta_i^{(k)}=0}}{R^2}
$$

#### 说明
① $MKT_t$ 为 t 期的市场收益率；
② $MKT_{t-k}$ 为滞后 k 期的市场收益率；
③ n 表示滞后的期数，通常 n=3、5；
④ $R^2_{\delta_i^{(k)}=0}$ 表示令 $\delta_i^{(k)}=0$ 时回归方程的拟合度；
⑤ $R^2$ 表示上述回归方程的拟合度。

价格时滞衡量了个股是否及时反映了市场公共信息，如果价格时滞较大，表明投资者对市场公共信息反应不足（相对滞后），过度关注个股层面的信息，刺激过度投机，从而导致股价被高估。

#### 参考文献
① Kewei Hou, Tobias J. Moskowitz. 2005. Market frictions, price delay, and the cross-section of expected returns. Review of Financial Studies 18, 981-1020.
② 朱剑涛, 2015, 投机、交易行为与股票收益(上), 东方证券.

### 10.1.11 协偏度 (朱剑涛版)

$$
CS = \frac{\sum_{t=1}^{n} [(r_{1,t} - \bar{r}_1)(r_{2,t} - \bar{r}_2)^2]}{\sum_{t=1}^{n} [(r_{2,t} - \bar{r}_2)^3]}
$$

#### 说明
① $r_{1,t}$ 为股票在 $t$ 时刻的收益率；
② $r_{2,t}$ 为市场基准在 $t$ 时刻的收益率；
③ $n$ 为过去的交易日数，一般 $n=20$，计算期间内至少有15个日度收益率数据。

动量效应与系统偏度有关，低预期回报动量投资组合的偏度高于高预期回报动量投资组合，买入低系统偏度的组合能获得超额收益，系统偏度能带来风险溢价。

#### 参考文献
朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 10.1.12 总偏度

最近K个月的日度收益率的偏度，K=6、12等，即：

$$
Skew_i = \frac{\frac{1}{T} \sum_{t=1}^{T} (r_{it} - \bar{r}_i)^3}{\left( \frac{1}{T} \sum_{t=1}^{T} (r_{it} - \bar{r}_i)^2 \right)^{\frac{3}{2}}}
$$

#### 说明
总偏度与股票收益负相关，由于投资者追求具有正偏度的股票，导致其价格容易被高估，使得预期收益率较低，属于低风险异象。

#### 参考文献
Ang, Andrew, Robert J. Hodrick, Yuhang Xing, and Xiaoyan Zhang, 2006, The cross-section of volatility and expected returns, Journal of Finance 61, 259-299.

### 10.1.13 负偏度系数

$$
NCSKEW_i = \frac{-(n(n-1))^{1.5} \sum (r_{it} - \bar{r}_i)^3}{(n-1)(n-2) (\sum (r_{it} - \bar{r}_i)^2)^{1.5}}
$$

#### 说明
① $r_{it}$ 是股票 $i$ 在 $t$ 时刻收益率；
② $n$ 为过去的交易日数，默认取 $n=20$。

该指标用于衡量股价暴跌的可能性，学界通常认为负偏度系数高的股票有着更高的暴跌可能，也就有着期望更高的风险溢价。

#### 参考文献
① Chen, J., Hong, H., & Stein, J. C., 2001, Forecasting crashes: Trading volume, past returns, and conditional skewness in stock prices. Journal of financial Economics 61(3), 345-381.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 10.1.14 市场Beta

$$
\beta_i = \frac{cov(r_i, r_m)}{var(r_m)}
$$

$$
r_i - r_f = \alpha + \beta_i (r_m - r_f) + \epsilon_i
$$

#### 说明
① $r_i$ 和 $r_m$ 分别为股票和市场在过去K个月的月度收益率；
② 一般 K=60 个月，至少包含24个月度收益率；
③ 如果使用日度收益率做计算，一般 K=12 个月，至少包含50个日度收益率。

市场贝塔系数是股票收益对市场收益的敏感度，CAPM模型表明市场贝塔系数应该与股票收益呈正相关。

#### 参考文献
Fama, Eugene F., and James D. MacBeth, 1973, Risk, return, and equilibrium: Empirical tests, Journal of Political Economy 81, 607-636.

### 10.1.15 下行贝塔

$$
\beta^{-} = \frac{Cov(r_i, r_m | r_m < \mu_m)}{Var(r_m | r_m < \mu_m)}
$$

#### 说明
① $r_i$ 和 $r_m$ 分别为股票和市场在过去K个月的月度收益率；
② $\mu_m$ 为这段时间市场日度收益率的平均值；
③ 一般 K=12 个月，至少包含50个日度收益率。

由于投资者往往更关注下行风险，在计算市场Beta时，剔除了市场收益向上的交易日，只考虑下跌的交易日，下行Beta能较好地度量个股的系统性风险，获得明显的下行风险溢价。

#### 参考文献
Ang, A., Chen, J., & Xing, Y., 2006, Downside risk. Review of Financial Studies 19(4), 1191-1239.

### 10.1.16 特质波动率

最近K个月CAPM或FF3-factor model回归的残差的标准差：

$$
std(\varepsilon_{i,t})
$$

#### 说明
① CAPM回归：
$$
r_{i,t} - r_{f,t} = \alpha_i + \beta_i (r_{m,t} - r_{f,t}) + \varepsilon_{i,t}
$$
② FF3-factor model回归：
$$
r_{i,t} = \alpha_i + \beta_{1,i} MKT_t + \beta_{2,i} SMB_t + \beta_{3,i} HML_t + \varepsilon_{i,t}
$$

特质波动率与股票收益率负相关，特质波动率低的股票回报率较高，属于低风险异象。

#### 参考文献
Ang, Andrew, Robert J. Hodrick, Yuhang Xing, and Xiaoyan Zhang, 2006, The cross-section of volatility and expected returns, Journal of Finance 61, 259-299.

### 10.1.17 总波动率

最近K个月的日度收益率的标准差，即：

$$
std(\text{最近K个月的日度收益率序列})
$$

#### 说明
总波动率与股票收益率负相关，总波动率低的股票回报率较高，属于低风险异象。

#### 参考文献
Ang, Andrew, Robert J. Hodrick, Yuhang Xing, and Xiaoyan Zhang, 2006, The cross-section of volatility and expected returns, Journal of Finance 61, 259-299.

### 10.1.18 上下行波动率

$$
DUVOL_i = \log \left( \frac{(n_u - 1) \sum_d (r_{it} - \bar{r}_i)^2}{(n_d - 1) \sum_u (r_{it} - \bar{r}_i)^2} \right)
$$

#### 说明
① $r_{it}$ 是股票 $i$ 在 $t$ 时刻收益率；
② $n_u$ 为大于平均复合收益率的天数；
③ $n_d$ 为小于平均复合收益率的天数。

与负偏度系数类似，上下行波动率也用于衡量股价暴跌的可能性，学界通常认为上下行波动率高的股票有着更高的暴跌可能，也就有着期望更高的风险溢价。

#### 参考文献
① Chen, J., Hong, H., & Stein, J. C., 2001, Forecasting crashes: Trading volume, past returns, and conditional skewness in stock prices. Journal of financial Economics 61(3), 345-381.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

### 10.1.19 协偏度 (系统偏度)

$$
CS = \frac{E[\varepsilon_i, \varepsilon_m^2]}{\sqrt{E[\varepsilon_i^2]} E[\varepsilon_m^2]}
$$

#### 说明
① $\varepsilon_i$ 为过去K个月内，股票日度超额收益对市场日度超额收益进行回归后的残差；
② $\varepsilon_m$ 为同一时期内，通过均值中心化后的市场日度超额收益 ($\varepsilon_m = r_m - \bar{r}_m$)；
③ K=1, 6, 12，计算期间内至少有15个日度收益率数据。

动量效应与系统偏度有关，低预期回报动量投资组合的偏度高于高预期回报动量投资组合，买入低系统偏度的组合能获得超额收益，系统偏度能带来风险溢价。

#### 参考文献
Harvey, Campbell R., and Akhtar Siddique, 2000, Conditional skewness in asset pricing tests, Journal of Finance 55(3), 1263-1295.

### 10.1.20 特异度

在每月月底基于上月的日交易数据进行FF3-factor model回归，得到：

$$
\text{特异度} = 1 - \text{上述回归模型的拟合优度} R^2
$$

#### 说明
FF3-factor model 回归：
$$
r_{i,t} = \alpha_i + \beta_{1,i} MKT_t + \beta_{2,i} SMB_t + \beta_{3,i} HML_t + \varepsilon_{i,t}
$$

特异度反映了股票收益中不能被市场、规模、估值三种常见的风格因子解释的成分占比，特异度越高说明个股涨跌与市场风格的相关性越小，交易中市场层面的信息占比越低，过去一段时间内被过度投机的可能性和程度越大。

#### 参考文献
朱剑涛, 2015, 投机、交易行为与股票收益(上), 东方证券.

### 10.1.21 特质偏度

最近K个月CAPM/FF3-factor model回归的残差的偏度：

$$
IdoSkew_i = \frac{\frac{1}{T} \sum_{t=1}^{T} \varepsilon_{it}^3}{\left( \frac{1}{T} \sum_{t=1}^{T} \varepsilon_{it}^2 \right)^{\frac{3}{2}}}
$$

其中
① CAPM 回归：
$$
r_{i,t} - r_{f,t} = \alpha_i + \beta_i (r_{m,t} - r_{f,t}) + \varepsilon_{i,t}
$$

② FF3-factor model 回归：
$$
r_{i,t} = \alpha_i + \beta_{1,i} MKT_t + \beta_{2,i} SMB_t + \beta_{3,i} HML_t + \varepsilon_{i,t}
$$

③ 偏度计算公式为：
$$
IdoSkew_i = \frac{\frac{1}{T} \sum_{t=1}^{T} \varepsilon_{it}^3}{\left( \frac{1}{T} \sum_{t=1}^{T} \varepsilon_{it}^2 \right)^{\frac{3}{2}}}
$$

#### 说明
特质偏度为特质收益的偏度，其与股票收益负相关，属于低风险异象。

#### 参考文献
Boyer, Brian, Todd Mitton, and Keith Vorkink, 2009, Expected idiosyncratic skewness, Review of Financial Studies 23, 169-202.

### 10.1.22 Frazzini-Pedersen贝塔

$$
betaFP = \hat{\rho} \frac{\widehat{\sigma}_i}{\widehat{\sigma}_m}
$$

其中
① $\widehat{\sigma}_i$和$\widehat{\sigma}_m$ 分别为过去K个月股票和市场的对数收益标准差；
② $\hat{\rho}$ 为过去Y年的股票和市场日度收益的相关系数，其中日度收益率为重叠了3天的对数收益率 $r_{i,t}^{3d} = \sum_{k=0}^{2} \log(1 + R_{i,t+k}^i)$，一般Y=5年，至少包含750个日度收益率；
③ 一般K=12个月，至少包含120个日度收益率。

#### 说明
对传统CAPM模型下的Beta做了改进，考虑了波动率估计误差的影响。

#### 参考文献
Frazzini, Andrea, and Lasse Heje Pedersen, 2014, Betting against beta, Journal of Financial Economics 111, 1-25.

### 10.1.23 Dimson贝塔

$$
r_{id} - r_{fd} = \alpha_i + \hat{\beta}_{i1} (r_{md-1} - r_{fd-1}) + \hat{\beta}_{i2} (r_{md} - r_{fd}) + \hat{\beta}_{i3} (r_{md+1} - r_{fd+1}) + \epsilon_{id}
$$

$$
betaDM = \hat{\beta}_{i1} + \hat{\beta}_{i2} + \hat{\beta}_{i3}
$$

其中
① $r_{id}, r_{md}, r_{fd}$ 分别为K个月内第 d 日的股票收益率、市场收益率、无风险利率；
② 一般K=1、6、12个月，至少包含15个日度收益率。

#### 说明
作者通过同时引入领先和滞后的收益率做回归的形式来处理股票非频繁交易带来的贝塔估计偏差问题。

#### 参考文献
Dimson, Elroy, 1979, Risk measurement when shares are subject to infrequent trading, Journal of Financial Economics 7, 197-226.

### 10.1.24 月度特质波动率

过去 24-60 个月的 Fama-French 三因子回归的残差收益率加权平方和，即：

$$
(IV_{i,t}^{WM})^2 = \frac{1}{\sum w_k} \sum_{k=1}^{T} w_k (\epsilon_{i,t+1-k})^2
$$

其中
① 权重 $w_k = 0.9^k$；
② $\epsilon_{i,t+1-k}$ 为第 i 个股票在第 t+1-k 月的三因子回归残差。

#### 说明
月度特质波动率衡量了股票过去的投机程度，与股票未来收益呈现显著的负相关。

#### 参考文献
① Cao, X., & Xu, Y., 2010, Long-run idiosyncratic volatilities and cross-sectional stock returns.
② 朱剑涛, 2017, 技术类新 Alpha 因子的批量测试, 东方证券.

# 11 杠杆因子
## 11.1 杠杆类
### 11.1.1 市场杠杆

$$
\frac{\text{最近报告期的总资产}}{\text{总市值}}
$$

#### 说明
表示资产总额是总市值的多少倍，作者测试结果表明平均回报率与市场杠杆正相关；
Barra模型中的市场杠杆 = (最近报告期的非流动负债合计 + 总市值) / 总市值。

#### 参考文献
Fama, Eugene F., and Kenneth R. French, 1992, The cross-section of expected stock returns, Journal of Finance 47, 427-465.

### 11.1.2 资产负债率

$$
\frac{\text{最近报告期的负债合计}}{\text{最近同期的总资产}}
$$

#### 说明
衡量企业利用债权人提供资金进行经营活动的能力，以及反映债权人发放贷款的安全程度的指标。

#### 参考文献
常规通用财务指标.

### 11.1.3 权益乘数

$$
\frac{\text{最近报告期平均总资产}}{\text{最近同期平均归属母公司股东权益合计}}
$$

其中
$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

$$
\text{平均归属母公司股东权益合计} = \frac{\text{期初归属母公司股东权益合计} + \text{期末归属母公司股东权益合计}}{2}
$$

#### 说明
表示总资产对净资产倍数，权益乘数反映了企业财务杠杆的大小，权益乘数越大，说明股东投入的资本在资产中所占的比重越小，财务杠杆越大。

#### 参考文献
Soliman, M, 2008, The use of DuPont analysis by market participants. Accounting Review 83, 823-53.

### 11.1.4 负债权益比

$$
\frac{\text{最近报告期负债合计}}{\text{最近报告期股东权益合计}}
$$

#### 说明
也称为产权比率，反映了债权人所提供的资金与股东所提供的资金的对比关系，该比率越低，说明企业长期财务状况越好，债权人的权益越有保障。

#### 参考文献
常规通用财务指标.

### 11.1.5 债务权益比

$$
\frac{\text{最近报告期债务总额}}{\text{最近报告期股东权益合计}}
$$

其中
$$
\text{总债务} = \text{短期借款} + \text{长期借款}
$$

#### 说明
反映公司股权融资和债务融资的比例，公司财务杠杆衡量指标，一般来说，比例越低，公司财务状况越好，面临的财务风险越低。

#### 参考文献
常规通用财务指标.

### 11.1.6 股东权益比率

$$
\frac{\text{最近报告期股东权益合计}}{\text{最近同期总资产}}
$$

#### 说明
反映股东权益在总资产中的占比，如果占比过小，表明企业过度负债，容易削弱公司抵御外部冲击的能力；而占比过大，意味着企业没有积极地利用财务杠杆作用来扩大经营规模。

#### 参考文献
常规通用财务指标.


### 11.1.7 流动比率

$$
\frac{\text{最近报告期流动资产合计}}{\text{最近同期流动负债合计}}
$$

#### 说明
反映企业的短期偿债能力。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 11.1.8 速动比率

$$
\frac{\text{最近报告期速动资产}}{\text{最近同期流动负债合计}}
$$

其中
$$
\text{速动资产} = \text{流动资产} - \text{存货} - (\text{应付账款} + \text{待摊费用})
$$

#### 说明
反映企业的短期偿债能力。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 11.1.9 现金比率

$$
\frac{\text{最近报告期货币资金} + \text{最近报告期交易性金融资产}}{\text{最近同期流动负债合计}}
$$

#### 说明
现金比率反映企业的即刻变现能力，或是企业立即偿还到期债务的能力。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 11.1.10 账面杠杆

$$
\frac{\text{最近报告期总资产}}{\text{最近同期股东权益合计}}
$$

#### 说明
表示资产总额是股东权益总额的多少倍，作者测试结果表明平均回报率与账面杠杆负相关；Barra中的账面杠杆 = 最近报告期的 (非流动负债合计 + 优先股账面价值 + 普通股账面价值) / 最近报告期的普通股账面价值。

#### 参考文献
Fama, Eugene F., and Kenneth R. French, 1992, The cross-section of expected stock returns, Journal of Finance 47, 427-465.

# 12 估值因子
## 12.1 估值类
### 12.1.1 估值趋势偏离度

假设个股的估值水平VR由长期趋势项和短期偏离项共同决定：

$$
VR_t^i = Trend_t^i + Deviation_t^i
$$

长期趋势项 $Trend$ 可以看作是行业基本趋势和个股特异性因素共同决定，其中：
$$
Trend_t^i \triangleq C^i \times SVR_t^i
$$

引入误差修正模型来对上述关系进行建模：
$$
\Delta VR_t^i = \alpha^i \times \Delta SVR_t^i + \lambda^i \times ECM_{t-1} + \varepsilon_t^i
$$

$$
ECM_{t-1} = (VR_{t-1}^i - C^i \times SVR_{t-1}^i)
$$

决定，其中：
$$
DR \triangleq (VR_t^i - C^i \times SVR_t^i) / VR_t^i
$$

其中
① $SVR_t^i$ 为股票 $i$ 所属行业所有成分股在 $t$ 时刻的估值中位数；
② $C^i$ 是股票 $i$ 的特异性因素，取值与时间无关；
③ $\lambda^i$ 为短期偏离的回复速度，$\lambda^i \in [-1, 0]$；
④ $VR_t^i$ 估值水平可以选择市净率、市销率的倒数等估值指标；
⑤ 每日计算，样本为过去每间隔20个交易日的36个样本点。

#### 说明
该因子从时序维度刻画了估值趋势的短期偏离程度；因子的绝对值越大，说明当期估值对长期趋势的偏离越明显，潜在的投资获利空间也越大；因子的正负方向代表着估值水平相较于当前趋势来说是低估还是高估。

#### 参考文献
徐寅, 2020, 基于误差修正模型的估值趋势偏离因子研究, 兴业证券.

### 12.1.2 营运资本同比增速

$$
\frac{\text{最近报告期营运资本} - \text{上年同期营运资本}}{abs(\text{上年同期营运资本})}
$$

其中
$$
\text{营运资本} = \text{流动资产合计} - \text{货币资金} - (\text{流动负债合计} - \text{应付票据} - \text{一年内到期的非流动负债})
$$

#### 说明
取值为正，说明营运资本的增加加快于营运负债的增加，企业使用了较多资金积累营运资产（如增加存货、延长回收应收账款等），使得企业可利用的流动资金减少，进而减少公司的估值；反之，企业以“负债形式”积累了较多资金（如压缩存货、收取预收账款、延期支付应付账款等），增加了企业的可用资金，进而增加公司估值。

#### 参考文献
常规通用财务指标.

### 12.1.3 账面市值比

$$
\frac{\text{最近报告期归属于母公司股东权益合计}}{\text{总市值}}
$$

#### 说明
也称为市净率的倒数；用于衡量股价高低和公司内在价值的估值因子，账面市值比高的股票，投资价值相对更高。
(注：图片标题为“账面市值比”，对应英文 Book-to-Market Ratio)

#### 参考文献
Basu, Sanjoy, 1983, The relationship between earnings' yield, market value and return for NYSE common stocks: Further evidence, Journal of Financial Economics 12, 129-156.

### 12.1.4 企业价值

$$
\text{总市值} + \text{总负债} + \text{优先股账面价值} - (\text{货币资金} + \text{短期投资})
$$

#### 说明
衡量公司的总价值，是理论上购买整个公司需要支付的价格，通常用来评估一公司的潜在收购价值，是用于计算估值指标的基础财务指标。

#### 参考文献
Loughran, Tim, and Jay W. Wellman, 2011, New evidence on the relation between the enterprise multiple and average stock returns, Journal of Financial and Quantitative Analysis 46, 16-29.

### 12.1.5 企业价值倍数

$$
\text{息税折旧摊销前利润EBITDA}
$$

其中
$$
\text{企业价值} = \text{总市值} + \text{总负债} + \text{优先股账面价值} - (\text{货币资金} + \text{短期投资})
$$
(注：此处公式整理自图片内容，图片中主公式未完整显示除号结构，但根据定义应为 企业价值/EBITDA)

$$
\frac{\text{企业价值}}{\text{息税折旧摊销前利润EBITDA}}
$$

#### 说明
衡量企业总价值（股权+债权）创造全部收益的能力。计算是用的息税折旧摊销前利润，消除了利息、所得税、折旧和摊销的影响，能够衡量整个企业的盈利能力；企业价值则同时考虑了股权价值和债权价值对企业经营的影响。

#### 参考文献
Loughran, Tim, and Jay W. Wellman, 2011, New evidence on the relation between the enterprise multiple and average stock returns, Journal of Financial and Quantitative Analysis 46, 16-29.

### 12.1.6 现金流市值比

$$
\frac{\text{最近12个月经营性活动产生的现金流净额(TTM)}}{\text{总市值}}
$$

#### 说明
为市现率的倒数；从现金流层面反映公司的内在价值，一般具有较高现金流市值的公司，往往具有较高的投资价值。

#### 参考文献
Desai, Hemang, Shivaram Rajgopal, and Mohan Venkatachalam, 2004, Value-glamour and accruals mispricing: One anomaly or two?, Accounting Review 79, 355-385.




### 12.1.7 盈利市值比

$$
\frac{\text{最近12个月归属母公司净利润(TTM)}}{\text{总市值}}
$$

#### 说明
为市净率的倒数；用于衡量股价高低和公司内在价值的估值因子，盈利市值比高的股票，说明公司盈利能力强，当前股价对应的投资回报率高，股价相对便宜。
(注：说明中原文应为“市盈率的倒数”)

#### 参考文献
Basu, Sanjoy, 1977, Investment performance of common stocks in relation to their price-earnings ratios: A test of the efficient market hypothesis, Journal of Finance 32, 663-682.

### 12.1.8 销售收入市值比

$$
\frac{\text{最近12个月营业收入(TTM)}}{\text{总市值}}
$$

#### 说明
为市销率的倒数；由于营业收入相对净利润等盈利指标更为稳定，所以常用于对盈利低或未盈利的成长股的估值；取值越高，股价越有可能被低估。

#### 参考文献
Barbee, William C., Sandip Mukherji Jr., and Gary A. Raines, 1996, Do sales-price and debt-equity explain stock returns better than book-market and firm size?, Financial Analysts Journal 52, 56-60.

### 12.1.9 股息率

$$
\frac{\text{最近12个月现金红利(TTM)}}{\text{总市值}}
$$

#### 说明
相当于投资股票所能获得的获利率，是常用的估值指标，股息率高的股票，相对而言，股票价格被低估，有较高的投资价值。

#### 参考文献
Litzenberger, Robert H., and Krishna Ramaswamy, 1979, The effect of personal taxes and dividends on capital asset prices: Theory and empirical evidence, Journal of Financial Economics 7, 163-195.

### 12.1.10 市盈率增长率

$$
\frac{\text{最近报告期市盈率(TTM)}}{\text{归属母公司净利润(TTM)同比增速}}
$$

#### 说明
相比传统估值因子，盈利市值比PEG包含了成长的视角，衡量企业总价值的同时增长表现出色，即由该股票当前有较高的市盈率，该股票相对其他股票也可能被低估的。
(注：该因子即 PEG 指标的变形或相关指标，通常 PEG = PE / Growth)

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 12.1.11 股东盈余市值比

$$
\frac{\text{最近12个月的股东盈余(TTM)}}{\text{总市值}}
$$

其中
$$
\text{股东盈余} = \text{净利润} + \text{折旧摊销} + \text{资产减值准备} + \text{研发费用} + \text{递延所得税费用} - \text{维护性资本支出}
$$

#### 说明
① 维护性资本支出：暂以“购建固定资产、无形资产和其他长期资产支付的现金”代替。
以巴菲特提出的“股东盈余”为基础，结合国内会计准则，对净利润进行调整，由此计算得到股东盈余市值比；在净利润调整过程中，规避了权责发生制下粉饰财务报表问题，同时将公司致力于长期发展的成本支出加回利润能更好地反映公司的盈利质量和发展潜力。

#### 参考文献
① 徐寅, 2020, 股东盈余视角下的价值因子研究, 兴业证券.
② Buffett's 1986 shareholder letter.

### 12.1.12 EBIT/企业价值

$$
\frac{\text{息税前利润EBIT}}{\text{企业价值}}
$$

其中
$$
\text{企业价值} = \text{总市值} + \text{总负债} + \text{优先股账面价值} - (\text{货币资金} + \text{短期投资})
$$

#### 说明
衡量企业总价值（股权+债权）创造全部收益的能力。计算是用的是息税前利润，消除了利息、所得税的影响，能够衡量整个企业的盈利能力；企业价值则同时考虑了股权价值和债权价值对企业经营的影响。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 12.1.13 盈利市值比60日变动

$$
\text{最新盈利市值比} EP_t - 60 \text{个交易日前盈利市值比} EP_{t-60}
$$

#### 说明
相比传统估值因子，盈利市值比变动属于动态价值投资的理念，关注的是“哪些股票的估值正在变得越来越便宜？”。

#### 参考文献
魏建榕, 2020, 估值因子坏了怎么修？开源证券.

### 12.1.14 Sale/企业价值

$$
\frac{\text{最近12个月营业收入(TTM)}}{\text{企业价值}}
$$

#### 说明
常规企业价值倍数倒数的衍生，从最原始的销售收入角度，衡量企业总价值的营收能力，更适用于无盈利能力、自由现金流为负等公司的估值。

#### 参考文献
Loughran, Tim, and Jay W. Wellman, 2011, New evidence on the relation between the enterprise multiple and average stock returns, Journal of Financial and Quantitative Analysis 46, 16-29.

### 12.1.15 总资产市值比

$$
\frac{\text{最近报告期总资产}}{\text{总市值}}
$$

#### 说明
常用的估值因子，用于衡量股票价格是否被高估或低估，与账面市值比类似。

#### 参考文献
Fama E. and French K, 1992, The cross-section of expected stock returns, Journal of Finance 47, 427-465.

### 12.1.16 负债市值比

$$
\frac{\text{最近报告期负债合计}}{\text{总市值}}
$$

#### 说明
反映公司建立资产的资金来源中股本与债务比例，是公司财务杠杆的衡量指标。

#### 参考文献
Bhandari, Laxmi Chand, 1988, Debt/equity ratio and expected common stock returns: Empirical evidence, Journal of Finance 43, 507-528.

### 12.1.17 市盈率增长率/股息率

$$
\text{EPGY} = \frac{\text{市盈率增长率(PEG)}}{\text{股息率}}
$$

#### 说明
用于表明股票的估值在预期增长率和股息率之间有多大吸引力。在PEG相差不多的情况下，EPGY越低的股票，更具有吸引力，因为其股息率更高。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management (1st edition), McGraw-Hill Library of Investment and Finance.

# 13 无形资产因子
## 13.1 无形资产类
### 13.1.1 无形资产强度

$$
IAI_{it} = \frac{KC_{it} + OC_{it}}{TA_{it} + KC_{it} + OC_{it} - GW_{it}}
$$

#### 说明
① 知识资本 $KC_{it}$ + 组织资本 $OC_{it}$ 为内部创造的无形资产（计算细节参考相关因子）；
② $TA_{it}$ 为 t 期末的总资产；
③ $GW_{it}$ 为 t 期末的公司商誉。

无形资产强度与股票收益有着非常强的正相关，比传统的基本面因子更具有解释力；而且是未来毛利率增长率有力预测指标，投资者可能会低估无形资产密集型企业的未来增长和盈利能力，从而导致错误定价。

#### 参考文献
冯佳睿, 姚石, 2019, 日内分时成交中的玄机, 海通证券.

### 13.1.2 销售收入增长减存货增长

$$
\text{最近单季度营业收入的同比增速} - \text{同期单季度存货的同比增速}
$$

#### 说明
销售收入的增长快于存货的增长，通常被视为公司销售状况良好，库存管理有效的信号。

#### 参考文献
Abarbanell, Jeffery S., and Brian J. Bushee, 1998, Abnormal returns to a fundamental analysis strategy, Accounting Review 73, 19-45.

### 13.1.3 毛利润增长减销售收入增长

$$
\text{最近单季度毛利润的同比增速} - \text{同期单季度营业收入的同比增速}
$$

#### 说明
毛利润的增长快于销售收入的增长，说明公司的成本控制能力增强，或者产品定价能力提升，盈利质量改善。

#### 参考文献
Abarbanell, Jeffery S., and Brian J. Bushee, 1998, Abnormal returns to a fundamental analysis strategy, Accounting Review 73, 19-45.

### 13.1.4 企业年龄

$$
\text{当前时刻与公司IPO日之间的月份数量}
$$

#### 说明
上市时间久的公司比上市时间短的公司能获得更高的投资收益。

#### 参考文献
Jiang, Guohua, Charles M. C. Lee, and Yi Zhang, 2005, Information uncertainty and expected returns, Review of Accounting Studies 10, 185-221.

### 13.1.5 创新能力因子

将下面3个细分因子分别进行横截面上标准化，然后加总求和得到创新能力因子：

$$
\text{① 最近12个月研发费用(TTM)}
$$
$$
\text{② 最近12个月营业收入(TTM)}
$$
$$
\text{③ 最近一年申请发明专利数量}
$$

#### 说明
综合了公司的研发投入、营收规模及专利产出，多维度衡量公司的创新能力。

#### 参考文献
郑兆磊, 2022, 专利研究系列四: 专利全解析, 兴业证券.

### 13.1.6 研发费用市值比

$$
\frac{\text{最近12个月的研发费用(TTM)}}{\text{总市值}}
$$

#### 说明
如果研发费用较为空缺，可用管理费用代替。

#### 参考文献
Chan, Louis K. C., Josef Lakonishok, and Theodore Sougiannis, 2001, The stock market valuation of research and development expenditures, Journal of Finance 56, 2431-2456.

### 13.1.7 内部创造的无形资产

t 期末的内部创造的无形资产由知识资本 $KC_{it}$ 和组织资本 $OC_{it}$ 构成：

$$
INT_{i,t} = KC_{i,t} + OC_{i,t}
$$

知识资本通过累加公司的研发支出 $R\&D$ 估算得到：
$$
KC_{i,t} = (1 - \delta_{R\&D}) KC_{i,t-1} + R\&D_{i,t}
$$

组织资本通过累加销售管理费用 $SG\&A$ 估算得到：
$$
OC_{i,t} = (1 - \delta_{SG\&A}) OC_{i,t-1} + \theta * SG\&A_{i,t}
$$

其中
$$
OC_{i,0} = \frac{SG\&A_{i,1}}{g + \delta_{SG\&A}} \quad , \quad KC_{i,0} = \frac{R\&D_{i,1}}{g + \delta_{R\&D}}
$$

#### 说明
① $\delta_{R\&D}$ 和 $\delta_{SG\&A}$ 分别为研发支出、销售管理费用的折旧率，可取经验值 30%, 20%；
② $\theta$ 为将销售管理费用算作组织资本的比例，可取经验值 30%；
③ $g$ 为研发支出和销售管理费用的平均增速，可取经验值 10% ~ 20% 等。
“无形资产”、“商誉”等会计科目更多的是对公司外部获取的各种权利（专利权、商标权等）的资本化；而公司内部创造的无形资产主要指的是公司通过在研发、销售、管理等方面的支出而逐步沉淀下来的研究成果、品牌、人力资源、企业文化等方面的无形资产；公司内部创造的无形资产资本化后，可以用于改进传统估值因子。

#### 参考文献
魏建榕, 2021, 分析师自标价的Alpha信息, 开源证券.

### 13.1.8 研发费用收入比

$$
\frac{\text{最近12个月研发费用(TTM)}}{\text{最近12个月营业收入(TTM)}}
$$

#### 说明
如果研发费用较为稀疏，可用管理费用代替。

#### 参考文献
Chan, Louis K. C., Josef Lakonishok, and Theodore Sougiannis, 2001, The stock market valuation of research and development expenditures, Journal of Finance 56, 2431-2456.


### 13.1.9 现金与总资产比

$$
\frac{\text{最近12个月的现金及现金等价物(TTM)}}{\text{平均总资产}}
$$

其中
$$
\text{平均总资产} = \frac{\text{期初总资产} + \text{期末总资产}}{2}
$$

#### 说明
与现金流和总需求冲击相关性更强的高风险公司有更强的预防性现金储蓄动机，而这种预防性现金储蓄动机意味着预期股票收益与现金持有量之间存在正相关关系。

#### 参考文献
Palazzo, Berardino, 2012, Cash holdings, risk, and expected returns, Journal of Financial Economics 104, 162-185.

# 14 股东因子
## 14.1 股东类
### 14.1.1 机构持股比例

$$
\frac{\text{机构持股数量}}{\text{总股本}}
$$

#### 说明
即除自然人、高管外的机构持股数占总股数的比例。

#### 参考文献
常规通用财务指标.


### 14.1.2 十大股东占比分散度

$$
std(\text{十大股东各自的持仓占比序列})
$$

#### 说明
刻画了股票十大股东各自持仓的不均衡程度，属于正向因子；十大股东持股越不均衡、股票越集中在前几位大股东手中，该股票后续相对走势会更好。

#### 参考文献
胡骥聪, 刘均伟, 2020, 股东类大类因子:拆解股东数据中的多元信息, 光大证券.

### 14.1.3 持仓机构个数

$$
\text{持仓机构个数}
$$

#### 说明
持仓机构个数统计了持有该股票的机构总数量，有越多机构持仓的股票，越受机构投资偏爱，未来收益会更强。

#### 参考文献
胡骥聪, 刘均伟, 2020, 股东类大类因子:拆解股东数据中的多元信息, 光大证券.

### 14.1.4 股权集中度

$$
\frac{\text{前三大流通股东持股数}}{\text{流通股股本}}
$$

#### 说明
衡量公司的股权分布状态的主要指标，也是衡量公司稳定性强弱的重要指标，同时也是衡量公司结构的重要指标。

#### 参考文献
常规通用财务指标.

### 14.1.5 股东户数

$$
\text{股东户数}
$$

#### 说明
持有该股票的股东人数越多，股票更易流通；如果股票只集中在少数机构手中，该股的流通性会变差，会提高该股票的交易难度。

#### 参考文献
Ludwig B. Chincarini, Daehwan Kim, 2006, Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management (1st edition), McGraw-Hill Library of Investment and Finance.

### 14.1.6 股权集中度（流通）

$$
\frac{\text{前三大流通股东持股数}}{\text{流通股股本}}
$$

#### 说明
衡量公司的股权分布状态的主要指标，也是衡量公司稳定性强弱的重要指标，同时也是衡量公司结构的重要指标。

#### 参考文献
常规通用财务指标.

### 14.1.7 户均持股比例

$$
\frac{\text{总股本} / \text{股东户数}}{\text{总股本}}
$$

#### 说明
为股东的平均持股数占总股本的比例，一般来说，比例大的股票筹码集中度高，筹码锁定性强，浮筹少，上涨较容易。

#### 参考文献
常规通用财务指标.

# 15 成长因子
## 15.1 成长类
### 15.1.1 单季度销售收入同比增长率

$$
\frac{\text{最近单季度营业收入} - \text{上年同期值}}{abs(\text{上年同期值})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.2 单季度净利润同比增长率

$$
\frac{\text{最近单季度净利润} - \text{上年同期值}}{abs(\text{上年同期值})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.3 单季度营业利润同比增长率

$$
\frac{\text{最近单季度营业利润} - \text{上年同期值}}{abs(\text{上年同期值})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.4 单季度营业总成本同比增长率

$$
\frac{\text{最近单季度营业总成本} - \text{上年同期值}}{\text{上年同期值}}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.5 归属母公司股东权益环比增长率

$$
\frac{\text{最近报告期归属母公司股东权益} - \text{上一报告期归属母公司股东权益}}{abs(\text{上一报告期归属母公司股东权益})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.6 单季度经营性活动产生的现金流净额同比增长率

$$
\frac{\text{最近单季度经营性活动产生的现金流净额}(TTM) - \text{上年同期值}}{abs(\text{上年同期值})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.7 单季度摊薄净资产收益率变动

$$
\text{最近报告期单季度摊薄净资产收益率}(TTM) - \text{上年同期单季度摊薄净资产收益率}(TTM)
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.8 单季度总资产净利率变动

$$
\text{最近报告期单季度总资产净利率} - \text{上年同期单季度总资产净利率}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.9 单季度总资产报酬率变动

$$
\text{最近报告期单季度总资产报酬率} - \text{上年同期单季度总资产报酬率}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.10 投入资本回报率变动

$$
\text{最近报告期单季度投入资本回报率} - \text{上年同期单季度投入资本回报率}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.11 单季度成本费用利润率的同比增长率

$$
\frac{\text{最近单季度成本费用利润率} - \text{上年同期单季度成本费用利润率}}{abs(\text{上年同期单季度成本费用利润率})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.12 单季度营业利润率同比增长率

$$
\frac{\text{最近单季度营业利润率} - \text{上年同期单季度营业利润率}}{abs(\text{上年同期单季度营业利润率})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.13 单季度净利率同比增长率

$$
\frac{\text{最近单季度净利率} - \text{上年同期单季度净利率}}{abs(\text{上年同期单季度净利率})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.14 单季度毛利率同比增长率

$$
\frac{\text{最近单季度毛利率} - \text{上年同期单季度毛利率}}{abs(\text{上年同期单季度毛利率})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.15 有形资本回报率环比增长率

$$
\frac{\text{最近报告期有形资本回报率}(TTM) - \text{上一报告期有形资本回报率}(TTM)}{abs(\text{上一报告期有形资本回报率}(TTM))}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.16 单季度销售成本率同比增长率

$$
\frac{\text{最近单季度销售成本率} - \text{上年同期单季度销售成本率}}{abs(\text{上年同期单季度销售成本率})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.17 每股净资产同比增长率

$$
\frac{\text{最近报告期每股净资产} - \text{上年同期每股净资产}}{abs(\text{上年同期每股净资产})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.18 基本每股收益同比增长率

$$
\frac{\text{最近报告期基本每股收益} - \text{上年同期基本每股收益}}{abs(\text{上年同期基本每股收益})}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.19 稀释每股收益同比增长率

$$
\frac{\text{最近报告期稀释每股收益}(TTM) - \text{上年同期稀释每股收益}(TTM)}{abs(\text{上年同期稀释每股收益}(TTM))}
$$

#### 说明
成长因子是基于股票过去一段时间的各项估值或盈利指标计算得到的增长量或增长率指标，用于衡量公司各个维度的成长性；计算成长因子时，最常用的统计维度有：环比增长率（或环比增量）、同比增长率（或同比增量），计算用的基础财务数据也有“最近12个月TTM”和“单季度”2类，经测试发现：“单季度同比”和“TTM环比”这2个维度下计算得到的因子表现通常相对更优；至于增长率和增量，表现相差不大，增长率更具可比性，增量更不容易受基础指标取值异常的影响。

#### 参考文献
常规通用财务指标.

### 15.1.20 劳动力效应

$$
\text{最近12个月的人均销售收入}(TTM) - \text{上年同期值}
$$

其中：
$$
\text{人均销售收入}(TTM) = \frac{\text{最近12个月的营业收入}(TTM)}{\text{平均员工总数}}
$$

$$
\text{平均员工总数} = \frac{\text{期初员工总数} + \text{期末员工总数}}{2}
$$

#### 说明
存在劳动力效应，给定销售收入，如果所需的员工人数有所减少，即人均销售收入有所增长，在投资者看来是一个积极信号。

#### 参考文献
Abarbanell, J. S. and Bushee, B. J., 1998, Abnormal Returns to a Fundamental Analysis Strategy, Accounting Review 73, 19-45.

# 16 分析师预期类

### 16.1.1 一致预期收益率

$$
TR = \sum_{i=1}^N \frac{P_i^e}{P}
$$

其中：
1. $P_i^e$ 为第 $i$ 家机构发布的股票目标价格；
2. $P$ 为月末计算因子的股票收盘价；
3. $N$ 为发布分析师预期目标价格的机构数量。

#### 说明
目标价反映了分析师对股票未来走势的看好程度，如果股票的目标价对应涨跌幅越大，说明分析师对该股票越看好。

#### 参考文献
魏建榕, 2021, 分析师目标价的 Alpha 信息, 开源证券.

### 16.1.2 预期EPS调整

$$
\frac{\text{当前预期EPS} - \text{三个月前预期EPS}}{abs(\text{三个月前预期EPS})}
$$

#### 说明
由于PE=股价/EPS，预期PE调整因子可以看作是股价的上涨幅度相对EPS上调幅度的偏差百分比。

#### 参考文献
曹春晓, 2018, 基于分析师一致预期的选股研究, 申万宏源.

### 16.1.3 分析师盈利预测上调的概率

$$
FOM = \frac{K - M}{N}
$$

其中：
1. $N$ 为分析师针对个股当期年报业绩给出的预测报告篇数；保证预测结果的准确性，测试时仅纳入公告前半年（180天）内的预测结果，且 $N>=3$，否则视为缺失值；
2. $K$ 为分析师预测的净利润比当前最新的分析师预测净利润高的报告篇数；
3. $M$ 为分析师预测的净利润比当前最新的分析师预测净利润低的报告篇数；
4. 如果一天内发布了多篇预测报告，则对每个最新预测值都计算FOM指标，再取均值；
5. 可以将业绩预告、业绩快报、实际财报披露的净利润值也视为一种盈利预测，用于计算FOM，从而可以更加及时全面地反映公司的盈利情况的预期变化。

#### 说明
FOM取值范围为[-1,1]，FOM越大说明盈余预测上调的程度越高；FOM=1意味着过去所有分析师预测的净利润都比当前最新的分析师预测净利润高，盈利预测下调的程度最高；FOM=0意味着过去分析师预测的净利润相比当前最新的分析师预测净利润高低各半，盈利预测与过去一致。

#### 参考文献
朱剑涛, 刘静涵, 2021, 更稳健易算的分析师盈利用调因子, 东方证券.

### 16.1.4 预期EPS上调比例

$$
EpsDiffusion = \frac{up - down}{up + down}
$$

对于个股，提取过去三个月所有对应的公司报告中的预期数据计算：
1. $up$ = 给出上调 eps 的机构数；
2. $down$ = 给出下调 eps 的机构数。

#### 说明
衡量分析师对EPS的上调比例，反映分析师现在与过去对于市场的预期差。

#### 参考文献
曹春晓, 2018, 基于分析师一致预期的选股研究, 申万宏源.

### 16.1.5 预期PE调整

$$
\frac{\text{当前预期PE} - \text{三个月前预期PE}}{abs(\text{三个月前预期PE})}
$$

#### 说明
由于PE=股价/EPS，预期PE调整因子可以看作是股价的上涨幅度相对EPS上调幅度的偏差百分比。

#### 参考文献
曹春晓, 2018, 基于分析师一致预期的选股研究, 申万宏源.

### 16.1.6 分析师覆盖

过去一段时间（如过去3个月）有报告覆盖的券商数量

#### 说明
分析师覆盖可以看作是对公司信息不确定性的度量，覆盖率高的公司信息确定性越强，越不容易被高估；分析师覆盖也能反映了公司的交易热度，覆盖率高的公司可能被过度关注，短期过热，未来会有回调风险。

#### 参考文献
1. Charles M.C. Lee, Eric C. So, 2016, Uncovering expected returns: Information in analyst coverage proxies, Journal of Financial Economics 124(2), 331-348.
2. 朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.7 加权预期收益率

$$
WTR = \sum_i \frac{P_i^e}{P} w_i = \sum_i \frac{P_i^e}{P} \times \frac{P_i^e \times P/P_i^0}{\sum P_i^e \times P/P_i^0}
$$

其中：
1. $P_i^e$ 为第 $i$ 家机构发布的股票目标价格；
2. $P_i^0$ 表示在第 $i$ 家机构发布价格预测前一个交易日的收盘价；
3. $P$ 为月末计算因子的股票收盘价；
4. $N$ 为发布分析师预期目标价格的机构数量。

#### 说明
加权后的目标收益率，加入了利用后续行情验证目标价格的权重 $w_i$，如果股票价格走势能够验证分析师的目标收益率 ($w_i > 1$)，则应给予目标价格更大的权重；反之，给予更小的权重。

#### 参考文献
魏建榕, 2021, 分析师目标价的 Alpha 信息, 开源证券.

### 16.1.8 加权的盈余调整幅度

1. 计算过去3个月每个机构最后一次预测净利润相对于1个月前6个月内的同向报告期最新一次预测的百分比；
2. 调整幅度为本次预测净利润相对上一次预测净利润的调整幅度；
3. 按 ACCwt2 方法加权过去3个月每家机构的最后一次预测净利润的调整幅度，得到加权的盈余调整幅度。

#### 说明
相比传统的盈余调整度量方法（一般一致预期净利润的变化率作为盈余调整），加权的盈余调整幅度，考虑了各分析师预测的异质性。

#### 参考文献
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.9 调整预期收益率

$$
\text{当前最新加权预期收益率} - \text{去年同期值}
$$

#### 说明
反映的是分析师预期的变化在短期对市场的冲击影响，调整预期收益率的空头端收益比多端收益更明显。

#### 参考文献
魏建榕, 2021, 分析师目标价的 Alpha 信息, 开源证券.

### 16.1.10 分析师预测偏差同向的概率

$$
FOM = \frac{K - M}{N}
$$

其中：
1. $N$ 为分析师针对个股当期年报业绩给出的预测报告篇数；保证预测结果的准确性，仅纳入公告前半年（180天）内的预测结果，且 $N>=3$，否则视为缺失值；
2. $K$ 为分析师预测净利润低于实际公告净利润的报告篇数；
3. $M$ 为分析师预测净利润高于实际公告净利润的报告篇数。

#### 说明
衡量分析师对EPS的上调比例，反映分析师现在与过去对于市场的预期差。

#### 参考文献
朱剑涛, 刘静涵, 2020, 来自年报季报业绩超预期事件的异动收益, 东方证券.

### 16.1.11 总分析师覆盖率

过去一段时间（如取过去3个月）券商-报告数据对的数量。

#### 说明
分析师覆盖可以看作是对公司信息不确定性的度量，覆盖率高的公司信息确定性越强，越不容易被高估；分析师覆盖也能反映了公司的交易热度，覆盖率高的公司可能被过度关注，短期过热，未来会有回调风险。

#### 参考文献
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.12 一致预期EP

一致预期净利润是通过过去一段时间各机构在研究报告上披露的预测值进行加权得到，加权方式有算术平均（Wind的一致预期）、时间与机构双重加权（朝阳永续）、按预测精度加权（东方证券）等。

#### 说明
估值因子，使用的一致预期净利润是通过对过去一段时间各机构在研究报告上披露的预测值进行加权得到，相比于历史财报数据，一致预期数据包含了分析师对于未来的展望，具有一定的前瞻性。

#### 参考文献
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.13 一致预期BP

一致预期净资产是通过过去一段时间各机构在研究报告上披露的预测值进行加权得到。

#### 说明
估值因子，使用的一致预期净资产是通过对过去一段时间各机构在研究报告上披露的预测值进行加权得到，相比于历史财报数据，一致预期数据包含了分析师对于未来的展望，具有一定的前瞻性。

#### 参考文献
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.14 关注度修正预期收益率

$$
CTR = Rank(WTR) * Rank(C)
$$

其中：
1. $Rank(WTR)$ 为横截面上 $WTR$ 加权预期收益率因子的排序值；
2. $Rank(C)$ 为横截面上关注度因子 $C$ 的排序值；
3. 关注度因子定义为：回看过去一段时间的分析师预期，分别统计覆盖不同股票的分析师数量，结果按机构保留唯一值。

#### 说明
关注度修正预期收益率综合了股票关注度对分析师预测的影响：分析师给出的目标价格还会受到分析师情绪的影响，而关注度高的股票，会导致分析师情绪的同质化，进而削弱分析师情绪的影响。

#### 参考文献
魏建榕, 2021, 分析师目标价的 Alpha 信息, 开源证券.

### 16.1.15 一致预期EPS

$$
\text{最终的一致预期EPS因子是通过对过去一段时间各机构在研究报告上披露的预测数据进行加权得到。}
$$

#### 说明
每月月初，会直接取过去6个月（180天）内，卖方分析师针对个股当年报业绩给出的预测报告篇数；保证预测结果的准确性，仅纳入公告前半年（180天）内的预测结果，且 $N \ge 3$，否则视为缺失值。

#### 参考文献
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.16 异常分析师覆盖

每月月底对全市场（中证全指成分股）以分析师覆盖的对数为Y，同期对市值对数、换手的对数、动量因子为X，进行横截面回归，回归残差即为异常分析师覆盖：

$$
\ln(1 + COV_{i,m}) = \beta_0 + \beta_1 SIZE_{i,m} + \beta_2 LNTO_{i,m} + \beta_3 MOM_{i,m} + \varepsilon_{i,m}
$$

其中：
1. $COV_{i,m}$ 为第 $i$ 只股票第 $m$ 月底的简单分析师覆盖或分析师总覆盖；
2. $SIZE_{i,m}$ 为第 $m$ 月底的总市值对数；
3. $LNTO_{i,m}$ 为第 $m$ 月底过去3个月日均换手率对数；
4. $MOM_{i,m}$ 为截止 $m$ 月底过去3个月收益率。

#### 说明
分析师覆盖可以看作是对公司信息不确定性的度量，覆盖率高的公司信息确定性越强，越不容易被高估；分析师覆盖也能反映了公司的交易热度，覆盖率高的公司可能被过度关注，短期过热，未来会有回调风险。

#### 参考文献
Charles M.C. Lee, Eric C. So, 2016, Uncovering expected returns: Information in analyst coverage proxies, Journal of Financial Economics 124(2), 331-348.
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.17 一致预期PEG

$$
\text{预期EPS上调比例} = \frac{\text{一致预期FY1净利润}}{\text{预期净利润增长率}}
$$

$$
\text{一致预期FY1净利润}
$$

$$
\text{预期净利润增长率}
$$

其中：
1. 一致预期净利润增长率 = 一致预期净利润FY2相对于一致预期净利润FY0的年化增长率。

#### 说明
PEG指标（市盈率相对盈利增长比率）是用公司的市盈率除以公司的盈利增长速度。当时市盈率越高，意味着投资者愿意为公司未来的获利支付更高的成本，但如果公司业绩增长无法支撑，则股价可能高估。

#### 参考文献
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.18 一致预期ROE

$$
\text{使用的一致预期基础指标是通过对过去一段时间各机构在研究报告上披露的预测数据进行加权得到，加权方式有算术平均（Wind的一致预期）、时间与机构双重加权（朝阳永续）、按预测精度加权（东方证券）等。}
$$

#### 说明
质量因子，使用的一致预期ROE是通过一致预期净利润和一致预期净资产计算得到，相比于历史财报数据，一致预期数据包含了分析师对于未来的展望，具有一定的前瞻性。

#### 参考文献
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.19 目标价隐含收益率

$$
\frac{\text{隐含目标价} - \text{最近股价}}{\text{最近股价}}
$$

$$
\text{目标价隐含收益率} = \text{一致预期目标价} - \text{最近股价}
$$

#### 说明
隐含目标价收益越高，预期未来的收益率也更高。

#### 参考文献
朱剑涛, 2017, 分析师研报的数据特征与 alpha, 东方证券.

### 16.1.20 业绩超预期

$$
\text{EarningSurprise} = \frac{A - F}{\hat{\sigma}}
$$

$$
\hat{\sigma} = k * MAD
$$

$$
MAD = median(|X_i - median_j(X_j)|)
$$

其中：
1. $A - F$ = 实际公告净利润A - 分析师一致预期净利润F，即分析师一致预期偏差；
2. $\hat{\sigma}$ 为过去5年的年报净利润标准差，但考虑到样本少导致标准差估计不准确的问题，改为采用中位数绝对偏差的稳健估计方法 $\hat{\sigma}$；
3. $k$ 为比例因子常量，取决于分布类型，对于正态分布数据，k $\approx$ 1.4826。

#### 说明
如果上市公司财报披露的实际盈利超过公告前的分析师一致预期值，就认为公司发生了业绩超预期，业绩超预期事件公告后存在显著的异常收益。

#### 参考文献
朱剑涛, 刘静涵, 2020, 来自年报季报业绩超预期事件的异常收益, 东方证券.

# 17 综合质量类

## 17.1 质量因子

通过 Gordon 成长模型分解为3个维度；计算各维度细分指标的横截面排序并进行z-score标准化，然后加总各指标z-score并进行标准化，得到各维度的得分；加总3个维度的得分并再次标准化，得到各股票的质量因子。

$$
\frac{Price}{BookValue} = \frac{(Profit/BookValue) * (Dividend/Profit)}{RequiredReturn - Growth}
$$

$$
= \frac{Profitability * PayoutRatio}{RequiredReturn - Growth}
$$

其中3个维度分别为：
1. Profitability 表示盈利能力，包括毛利总资产 (gross profit over assets, GPOA)、ROE、ROA、现金流总资产 (cash flow over assets, CFOA)、毛利率 (gross margin) 和盈利中现金的占比 (即盈利减去应计项目，ACC) 这6个指标；
2. Growth 表示成长能力，用于反映盈利的可持续性与增长变化率；
3. RequiredReturn 表示公司的稳定性或者安全性，包括低beta (BAB)、低杠杆率、低破产概率 (O-score) 和低ROE波动率；
4. PayoutRatio 表示股东所得分红在总利润中的占比，主要用于衡量公司的偿付能力 (作者在2019年发布的质量因子未使用该维度)。

#### 说明
质量因子在衰退期和波动率突然大涨的时期表现最好，在大熊市期间表现也不错，这表明质量因子有助于控制组合的风险。

#### 参考文献
Asness, Clifford S., Andrea Frazzini, Lasse Heje Pedersen, 2019, Quality Minus Junk, Review of Accounting Studies 24(1), 34-112.

## 17.2 质量增长

$$
\text{质量增长} = \text{当前最新质量因子} - \text{去年同期值}
$$

或者质量因子的同比增量：

$$
\text{质量增长} = \frac{\text{当前最新质量因子} - \text{去年同期值}}{\text{去年同期值}}
$$

#### 说明
中国股市存在显著的正质量增长溢价，即具有高质量增长公司产生的回报显著高于质量增长公司产生的回报。质量增长因子源自投资者对公司质量变化反应不足。

#### 参考文献
Yin Libo, Huiyi Liao, Firm's Quality Increases and the Cross-Section of Stock Returns: Evidence from China, International Review of Economics and Finance 2019, forthcoming.
