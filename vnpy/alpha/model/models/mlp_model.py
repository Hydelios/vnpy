import copy
from collections import defaultdict
from typing import Literal, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error       # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim

# 假设这些是你 vnpy 环境中的原始引用，保持不变
from vnpy.alpha import (
    AlphaDataset,
    AlphaModel,
    Segment,
    logger
)


class MlpModel(AlphaModel):
    """
    Multi-Layer Perceptron Model (Optimized Version)

    [优化点说明]:
    1. Deep Network: 支持多层深层网络结构，挖掘非线性因子。
    2. Regularization: 集成了 Dropout 和 Weight Decay 防止过拟合。
    3. Stabilization: 使用 Batch Normalization 和 SiLU 激活函数，打破训练瓶颈。
    4. Early Stopping: 更加灵敏的早停机制。
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int] = (512, 256, 128),  # 默认使用深层结构
        lr: float = 0.001,
        n_epochs: int = 5000,             # 给足够的轮数，依靠早停来结束
        batch_size: int = 2048,           # 中等大小的 Batch，平衡速度与随机性
        early_stop_rounds: int = 50,
        eval_steps: int = 20,
        optimizer: Literal["sgd", "adam"] = "adam",
        weight_decay: float = 0.01,       # [关键修改] 默认增强正则化，防止后期反弹
        dropout_rate: float = 0.3,        # [关键修改] 默认提高 Dropout，增加泛化能力
        activation: str = "SiLU",         # [关键修改] 默认使用 SiLU (Swish)
        device: str = "cuda",             # 默认尝试使用 cuda，没有则会自动回退或报错(取决于环境)
        seed: int | None = None
    ) -> None:
        """
        初始化 MLP 模型参数
        """
        self.input_size: int = input_size
        self.hidden_sizes: tuple[int] = hidden_sizes
        self.lr: float = lr
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.early_stop_rounds: int = early_stop_rounds
        self.eval_steps: int = eval_steps
        self.device: str = device if torch.cuda.is_available() else "cpu" # 自动检测设备
        self.fitted: bool = False
        self.feature_names: list[str] = []
        self.best_step: int | None = None

        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # 设置损失函数 (MSE)
        self._scorer = mean_squared_error

        # [初始化核心网络]
        self.model: nn.Module = MlpNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_rate=dropout_rate
        )

        # 移动模型到指定设备
        self.model = self.model.to(self.device)

        # 设置优化器
        optimizer_name = optimizer.lower()
        if optimizer_name == "adam":
            self.optimizer: optim.Optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        # 设置学习率调度器 (ReduceLROnPlateau)
        # 当 loss 不再下降时，自动减小学习率
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0.00001,
            eps=1e-08,
        )

    def fit(
        self,
        dataset: AlphaDataset,
        evaluation_results: dict | None = None,
    ) -> None:
        """训练模型"""
        if evaluation_results is None:
            evaluation_results = {}

        train_valid_data: dict[str, dict] = defaultdict(dict)

        # 数据准备
        for segment in [Segment.TRAIN, Segment.VALID]:
            df: pl.DataFrame = dataset.fetch_learn(segment)
            df = df.sort(["datetime", "vt_symbol"])

            features = df.select(df.columns[2: -1]).to_numpy()
            labels = np.array(df["label"])

            # [安全检查] 处理 NaN，防止 Loss 变成 nan
            if np.isnan(features).any():
                logger.warning(f"{segment} 数据集包含 NaN 值，已自动填充为 0")
                features = np.nan_to_num(features)
            
            # 转换为 Tensor 并移动到设备
            train_valid_data["x"][segment] = torch.from_numpy(features).float().to(self.device)
            train_valid_data["y"][segment] = torch.from_numpy(labels).float().to(self.device)
            evaluation_results[segment] = []

        # 记录特征名称
        df = dataset.fetch_learn(Segment.TRAIN)
        self.feature_names = df.columns[2:-1]

        # 训练状态追踪
        early_stop_count: int = 0
        train_loss: float = 0
        best_valid_score: float = np.inf
        best_params = None

        train_samples: int = train_valid_data["y"][Segment.TRAIN].shape[0]

        # 主训练循环
        for step in range(1, self.n_epochs + 1):
            if early_stop_count >= self.early_stop_rounds:
                logger.info("达到早停条件, 训练结束")
                break

            # 训练一个 Batch
            batch_loss = self._train_step(train_valid_data, train_samples)
            train_loss += batch_loss

            # 定期评估
            if step % self.eval_steps == 0 or step == self.n_epochs:
                early_stop_count, best_valid_score, best_params = self._evaluate_step(
                    train_valid_data,
                    evaluation_results,
                    step,
                    train_loss,
                    early_stop_count,
                    best_valid_score
                )
                train_loss = 0 # 重置累计 Loss

        self.fitted = True
        # 恢复最佳模型参数
        if best_params:
            self.model.load_state_dict(best_params)
            logger.info(f"已恢复至最佳 Step {self.best_step} 的参数")

    def _train_step(
        self,
        train_valid_data: dict[str, dict[Segment, torch.Tensor]],
        train_samples: int
    ) -> float:
        batch_loss = AverageMeter()
        self.model.train() # 启用 Dropout 和 BN 的训练模式
        self.optimizer.zero_grad()

        # 随机采样 Batch
        batch_indices = np.random.choice(train_samples, self.batch_size)
        batch_features = train_valid_data["x"][Segment.TRAIN][batch_indices]
        batch_labels = train_valid_data["y"][Segment.TRAIN][batch_indices]

        # 前向传播
        predictions = self.model(batch_features)
        cur_loss = self._loss_fn(predictions, batch_labels)
        
        # 反向传播
        cur_loss.backward()
        self.optimizer.step()
        
        batch_loss.update(cur_loss.item())

        return batch_loss.val

    def _evaluate_step(
        self,
        train_valid_data: dict[str, dict[Segment, torch.Tensor]],
        evaluation_results: dict[Segment, list[float]],
        step: int,
        train_loss: float,
        early_stop_count: int,
        best_valid_score: float
    ) -> tuple[int, float, dict[str, torch.Tensor] | None]:
        early_stop_count += 1
        train_loss /= self.eval_steps # 计算平均训练 Loss

        with torch.no_grad():
            self.model.eval() # 切换到评估模式 (关闭 Dropout)
            data: torch.Tensor = train_valid_data["x"][Segment.VALID]
            # 验证集可能很大，分批预测以防爆显存
            pred: torch.Tensor = cast(torch.Tensor, self._predict_batch(data, return_cpu=False))
            valid_loss = self._loss_fn(pred, train_valid_data["y"][Segment.VALID])
            loss_val = valid_loss.item()

        logger.info(f"[Step {step}]: train_loss {train_loss:.6f}, valid_loss {loss_val:.6f}")
        evaluation_results[Segment.TRAIN].append(train_loss)
        evaluation_results[Segment.VALID].append(loss_val)

        best_params = None
        # 如果验证集 Loss 创新低
        if loss_val < best_valid_score:
            # 这里的 1e-7 是为了防止极其微小的数值抖动误判
            if best_valid_score - loss_val > 1e-7:
                logger.info(f"\t验证集损失从 {best_valid_score:.6f} 降低到 {loss_val:.6f}")
                best_valid_score = loss_val
                self.best_step = step
                early_stop_count = 0
                best_params = copy.deepcopy(self.model.state_dict())
            
        # 更新学习率 (如果 Loss 不降，自动降低学习率)
        if self.scheduler is not None:
            self.scheduler.step(metrics=valid_loss, epoch=step)

        return early_stop_count, best_valid_score, best_params

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = pred.reshape(-1), target.reshape(-1)
        loss: torch.Tensor = nn.MSELoss()(pred, target)
        return loss

    def _predict_batch(self, data: torch.Tensor, return_cpu: bool = True) -> np.ndarray | torch.Tensor:
        data = data.to(self.device)
        predictions: list[torch.Tensor] = []
        self.model.eval()

        with torch.no_grad():
            # 推理时的 Batch Size 可以设置大一些，加速预测
            inference_batch_size: int = 8192
            for i in range(0, len(data), inference_batch_size):
                x: torch.Tensor = data[i: i + inference_batch_size]
                predictions.append(self.model(x).detach().reshape(-1))

        if return_cpu:
            return cast(np.ndarray, np.concatenate([pr.cpu().numpy() for pr in predictions]))
        else:
            return torch.cat(predictions, dim=0)

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model has not been trained yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])
        data: np.ndarray = df.select(df.columns[2: -1]).to_numpy()
        
        # 预测时也处理 NaN
        if np.isnan(data).any():
            data = np.nan_to_num(data)
            
        return cast(np.ndarray, self._predict_batch(torch.Tensor(data)))

    def detail(self) -> pd.DataFrame | None:
        """输出模型训练详情和特征重要性"""
        if not self.fitted:
            logger.info("模型尚未训练，无法显示详细信息")
            return None

        logger.info("-" * 30)
        logger.info(f"输入特征维度: {self.input_size}")
        logger.info(f"隐藏层结构: {self.hidden_sizes}")
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型总参数量: {total_params:,}")
        logger.info(f"训练设备: {self.device}")
        logger.info(f"最佳 Step: {self.best_step}")
        logger.info("-" * 30)
        
        return self._calculate_feature_importance()

    def _calculate_feature_importance(self) -> pd.DataFrame:
        """使用 Permutation Importance 计算特征重要性"""
        self.model.eval()
        importance_dict = {}
        
        # 增加样本量以获得更稳定的评估
        n_samples = 2000
        test_data = torch.randn(n_samples, self.input_size).to(self.device)
        base_pred = self.model(test_data).detach()

        noise_level = 0.1
        logger.info("正在计算特征重要性...")
        
        for i, feature_name in enumerate(self.feature_names):
            perturbed_data = test_data.clone()
            # 加入噪声干扰该特征
            perturbed_data[:, i] += torch.randn(n_samples).to(self.device) * noise_level

            with torch.no_grad():
                new_pred = self.model(perturbed_data)
                # 计算预测值的偏离程度 (RMSE)
                importance = torch.sqrt(torch.mean((new_pred - base_pred) ** 2)).item()
                importance_dict[feature_name] = importance

        df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        df = df.sort_values('Importance', ascending=False)
        df = df.set_index('Feature')
        return df


class AverageMeter:
    """计算滑动平均值的工具类"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MlpNetwork(nn.Module):
    """
    [Optimized] 深度神经网络结构
    
    使用 nn.Sequential 封装，包含：
    Linear -> BatchNorm -> Activation -> Dropout
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_sizes: tuple[int] = (256,),
        activation: str = "SiLU",
        dropout_rate: float = 0.3  # 默认 Dropout 率
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        
        # 1. 输入层 Dropout (保留，轻微干扰原始输入)
        layers.append(nn.Dropout(0.05))

        # 构建网络层级
        layer_sizes = [input_size] + list(hidden_sizes)

        # 2. 构建隐藏层循环
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),           # [核心] 加速收敛，平滑梯度
                self._get_activation(activation),   # [核心] 动态激活函数
                nn.Dropout(dropout_rate)            # [核心] 防止过拟合
            ])

        # 3. 输出层 (直接线性输出)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # 封装为 Sequential
        self.network = nn.Sequential(*layers)

        # 执行权重初始化
        self._initialize_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        if name == "LeakyReLU":
            return nn.LeakyReLU(negative_slope=0.1)
        elif name == "SiLU":
            return nn.SiLU()  # 量化金融推荐
        elif name == "Tanh":
            return nn.Tanh()
        elif name == "ReLU":
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function type: {name}")

    def _initialize_weights(self) -> None:
        """Kaiming 初始化 (适配 ReLU/SiLU 系列)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.1,
                    mode="fan_in",
                    nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)