from typing import cast, Any

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel


class LgbModel(AlphaModel):
    """
    LightGBM Model (Quant Optimized Version)
    
    优化点：
    1. 加入 colsample_bytree 防止多因子共线性过拟合。
    2. 加入 reg_alpha/lambda 正则化。
    3. 统一数据输入格式，确保预测安全。
    4. 改进特征重要性输出。
    """

    def __init__(
        self,
        learning_rate: float = 0.05,        # [优化] 调低学习率，更稳
        num_leaves: int = 31,
        max_depth: int = -1,                # [新增] 限制树深
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        feature_fraction: float = 0.6,      # [关键] 每次只用 60% 的因子，抗过拟合神器
        bagging_fraction: float = 0.7,      # [新增] 样本采样，增加随机性
        bagging_freq: int = 1,
        reg_alpha: float = 0.1,             # [新增] L1 正则化
        reg_lambda: float = 0.1,            # [新增] L2 正则化
        log_evaluation_period: int = 20,
        seed: int = 42,
        use_gpu: bool = True  # 默认尝试开启 GPU
    ):
        self.params: dict[str, Any] = {
            "objective": "regression",
            "metric": ["mse", "mae"],
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": reg_alpha,
            "lambda_l2": reg_lambda,
            "verbosity": -1,
            "seed": seed,
            "num_threads": -1,                 # 建议用 num_threads（而不是 n_jobs）
        }

        if use_gpu:
            # 走 OpenCL GPU（你已验证 clinfo 平台/设备存在）
            self.params.update({
                "device_type": "gpu",           # 关键：OpenCL GPU
                "gpu_platform_id": 0,           # NVIDIA CUDA 平台
                "gpu_device_id": 1,             # V100 是 Device #1
            })
        else:
            self.params.update({
                "device_type": "cpu",
            })

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds
        self.log_evaluation_period: int = log_evaluation_period
        
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    def _prepare_data(self, dataset: AlphaDataset) -> list[lgb.Dataset]:
        ds: list[lgb.Dataset] = []
        
        # 记录特征名，用于后续一致性校验
        sample_df = dataset.fetch_learn(Segment.TRAIN)
        exclude = {"datetime", "vt_symbol", "label"}
        self.feature_names = [c for c in sample_df.columns if c not in exclude]

        for segment in [Segment.TRAIN, Segment.VALID]:
            df: pl.DataFrame = dataset.fetch_learn(segment)
            df = df.sort(["datetime", "vt_symbol"])

            # 统一转为 numpy，去除 Pandas 依赖，提高速度
            # 注意：LightGBM 对 NaN 友好，不需要像 MLP 那样填充 0
            data = df.select(self.feature_names).to_numpy()
            label = np.array(df["label"])
            lgb_data = lgb.Dataset(data, label=label, feature_name=self.feature_names)
            ds.append(lgb_data)

        return ds

    def fit(self, dataset: AlphaDataset) -> None:
        ds: list[lgb.Dataset] = self._prepare_data(dataset)

        self.model = lgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round,
            valid_sets=ds,
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(self.log_evaluation_period)
            ]
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        data: np.ndarray = df.select(self.feature_names).to_numpy()
        
        # 这里的 predict 会自动利用训练时记录的 feature_name 结构（如果输入是 dataframe）
        # 因为我们输入是 numpy，只要保证列顺序一致即可 (AlphaDataset 保证了这点)
        result: np.ndarray = cast(np.ndarray, self.model.predict(data))
        return result

    def detail(self) -> pd.DataFrame | None:
        """
        返回特征重要性 DataFrame，方便做特征筛选
        """
        if not self.model:
            return None

        # 获取重要性 (split: 被选为分裂节点的次数, gain: 带来的增益)
        importance_split = self.model.feature_importance(importance_type='split')
        importance_gain = self.model.feature_importance(importance_type='gain')
        
        feature_name = self.model.feature_name()

        df = pd.DataFrame({
            'Feature': feature_name,
            'Importance_Split': importance_split,
            'Importance_Gain': importance_gain
        })
        
        # 按 Gain 排序，通常 Gain 对量化更有意义
        df = df.sort_values('Importance_Gain', ascending=False).reset_index(drop=True)
        
        # 简单画个图 (可选)
        try:
            lgb.plot_importance(self.model, max_num_features=20, importance_type='gain', title='Feature Importance (Gain)')
            plt.show()
        except Exception:
            pass # 防止无界面环境下报错

        return df