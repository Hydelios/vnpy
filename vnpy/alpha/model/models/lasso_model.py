import numpy as np
import polars as pl
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

from vnpy.alpha import (
    AlphaDataset,
    AlphaModel,
    Segment,
    logger
)


class LassoCVModel(AlphaModel):
    """
    LASSO regression learning algorithm with Time-Series Cross-Validation.
    自动寻找最优惩罚力度(alpha)，并防止时序未来函数泄露。
    """

    def __init__(
        self,
        cv_folds: int = 5,
        max_iter: int = 5000,  # LassoCV 寻优时可能需要更多迭代次数以保证收敛
        random_state: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        cv_folds : int
            时间序列交叉验证的折数
        max_iter : int
            最大迭代次数
        random_state : int
            随机种子
        """
        self.cv_folds: int = cv_folds
        self.max_iter: int = max_iter
        self.random_state: int | None = random_state

        self.model: LassoCV = None
        self.feature_names: list[str] = []

    def fit(self, dataset: AlphaDataset) -> None:
        """
        Fit the model with dataset using Cross-Validation

        Parameters
        ----------
        dataset : AlphaDataset
            The dataset used for training
        """
        # 1. 仅获取训练集，将 VALID 留作严格的纯样本外验证
        df_train: pl.DataFrame = dataset.fetch_learn(Segment.TRAIN)

        # 2. 数据清洗：去重并排序
        df_train = df_train.unique(subset=["datetime", "vt_symbol"])
        df_train = df_train.sort(["datetime", "vt_symbol"])

        # 3. 提取特征名称
        self.feature_names = df_train.columns[2:-1]

        # 4. 转换为 numpy 数组
        X: np.ndarray = df_train.select(self.feature_names).to_numpy()
        y: np.ndarray = np.array(df_train["label"])

        # 5. 核心逻辑：设置时间序列交叉验证，防止未来数据泄露到训练集
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        # 6. 创建并训练 LassoCV 模型
        self.model = LassoCV(
            cv=tscv,               # 使用时序交叉验证
            max_iter=self.max_iter,
            random_state=self.random_state,
            fit_intercept=True,    # 开启截距项，吸收无法被因子解释的整体截面 Beta 漂移
            copy_X=False,          # 节省内存
            n_jobs=-1              # 开启所有CPU核心并行加速 CV 搜索
        )
        
        self.model.fit(X, y)

        # 打印由数据自动选出的最优 alpha 值
        logger.info(f"LassoCV 交叉验证完成，最佳惩罚系数 (Alpha): {self.model.alpha_:.6f}")

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """
        Make predictions using the fitted model

        Parameters
        ----------
        dataset : AlphaDataset
            The dataset used for prediction
        segment : Segment
            The segment of data to use for prediction

        Returns
        -------
        np.ndarray
            Prediction results

        Raises
        ------
        ValueError
            If the model has not been fitted yet
        """
        # 检查模型是否已训练
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        # 获取预测数据并排序
        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        # 提取特征矩阵
        data: np.ndarray = df.select(df.columns[2: -1]).to_numpy()

        # 返回预测结果
        result: np.ndarray = self.model.predict(data)

        return result

    def detail(self) -> None:
        """
        Output detailed information about the model

        Displays feature importance based on the coefficients
        of the LASSO model, showing only non-zero features
        sorted by absolute value.
        """
        if self.model is None:
            logger.info("模型尚未训练，无法查看特征细节。")
            return

        # 获取特征系数
        coef: np.ndarray = self.model.coef_

        # 组合特征名与系数
        data: list[tuple[str, float]] = list(zip(self.feature_names, coef, strict=False))

        # 过滤掉被 LASSO 压缩为 0 的噪音特征
        data = [x for x in data if x[1]]

        # 按绝对值大小（重要性）降序排序
        data.sort(key=lambda x: abs(x[1]), reverse=True)

        # 过滤掉浮点数精度级别的极小值
        data = [x for x in data if round(x[1], 6) != 0]

        # 打印特征重要性
        logger.info(f"LASSO模型最终保留的特征总数量: {len(data)}")

        for name, importance in data:
            logger.info(f"{name}: {importance:.6f}")