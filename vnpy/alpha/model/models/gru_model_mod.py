from __future__ import annotations

import copy
import os
import tempfile
import uuid
from typing import Literal, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel
from vnpy.alpha import logger


class _LazySequenceDataset(Dataset):
    """
    GRU 懒加载序列数据集。

    只保存：
    - features: 原始二维特征矩阵 [num_rows, num_features]
    - end_positions: 每个样本窗口终点在 features 中的行号 [num_samples]
    - labels/sample_dates/row_positions: 终点行上的元数据

    不保存物化后的三维窗口矩阵 [num_samples, lookback, num_features]。
    DataLoader 取到 batch 时，才临时从 features 中切出 [batch, lookback, num_features]。
    """

    def __init__(
        self,
        *,
        features: np.ndarray,
        end_positions: np.ndarray,
        lookback: int,
        input_size: int,
        labels: np.ndarray | None = None,
        sample_dates: np.ndarray | None = None,
        row_positions: np.ndarray | None = None,
        feature_path: str | None = None,
    ) -> None:
        self.features = features
        self.end_positions = np.asarray(end_positions, dtype=np.int64)
        self.lookback = int(lookback)
        self.input_size = int(input_size)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.sample_dates = np.asarray(sample_dates, dtype=object) if sample_dates is not None else np.empty(0, dtype=object)
        self.row_positions = np.asarray(row_positions, dtype=np.int64) if row_positions is not None else np.empty(0, dtype=np.int64)
        self.feature_path = feature_path
        self.closed = False

        if self.features.ndim != 2:
            raise ValueError(f"features 必须是二维矩阵，实际 shape={self.features.shape}")
        if self.features.shape[1] != self.input_size:
            raise ValueError(f"features 列数应为 {self.input_size}，实际为 {self.features.shape[1]}")
        if self.labels is not None and len(self.labels) != len(self.end_positions):
            raise ValueError("labels 长度必须等于 end_positions 长度")
        if len(self.sample_dates) not in {0, len(self.end_positions)}:
            raise ValueError("sample_dates 长度必须等于 end_positions 长度")
        if len(self.row_positions) not in {0, len(self.end_positions)}:
            raise ValueError("row_positions 长度必须等于 end_positions 长度")

    def __len__(self) -> int:
        return int(len(self.end_positions))

    def __getitem__(self, index: int) -> int:
        # 返回样本编号，而不是窗口本身。窗口由 collate_fn 按 batch 统一切片。
        return int(index)

    def close(self) -> None:
        """释放 memmap 句柄并尝试删除临时文件。"""
        if self.closed:
            return

        mmap_obj = getattr(self.features, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass

        if self.feature_path and os.path.exists(self.feature_path):
            try:
                os.remove(self.feature_path)
            except Exception:
                # Windows 或多进程 DataLoader 场景下，文件可能仍被占用；不影响模型结果。
                pass

        self.closed = True


class GruModel(AlphaModel):
    """
    内存安全版 GRU Alpha 模型。

    与 eager 版本的核心区别：
    - 不再预先构造 `x = [num_samples, lookback, input_size]`。
    - 训练/评估/预测阶段只保存二维特征矩阵和样本终点索引。
    - 每个 batch 临时切出三维张量，batch 用完即可释放。

    空间复杂度：
    - eager 版本：O(N * lookback * F)
    - 本版本：O(N * F + N)

    参数说明中，N 为原始二维面板行数，F 为特征数量。
    """

    def __init__(
        self,
        input_size: int,
        lookback: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        n_epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 2000,
        early_stop_rounds: int = 20,
        optimizer: Literal["sgd", "adam", "adamw"] = "adam",
        device: str = "cpu",
        seed: Optional[int] = None,
        grad_clip: float = 3.0,
        metric: Literal["neg_mse", "ic", "rank_ic"] = "rank_ic",
        min_ic_samples: int = 5,
        feature_fill_value: float | None = 0.0,
        sample_stride: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_memmap: bool = False,
        memmap_dir: str | None = None,
        feature_dtype: Literal["float32", "float64"] = "float32",
        sanitize_chunk_rows: int = 200_000,
        keep_data_for_detail: bool = False,
        importance_sample_size: int = 0,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size 必须为正整数")
        if lookback <= 0:
            raise ValueError("lookback 必须为正整数")
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须为正整数")
        if num_layers <= 0:
            raise ValueError("num_layers 必须为正整数")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if early_stop_rounds <= 0:
            raise ValueError("early_stop_rounds 必须为正整数")
        if metric not in {"neg_mse", "ic", "rank_ic"}:
            raise ValueError("metric 只能是 'neg_mse', 'ic', 'rank_ic'")
        if sample_stride <= 0:
            raise ValueError("sample_stride 必须为正整数")
        if num_workers < 0:
            raise ValueError("num_workers 不能为负数")
        if feature_dtype not in {"float32", "float64"}:
            raise ValueError("feature_dtype 只能是 'float32' 或 'float64'")
        if sanitize_chunk_rows <= 0:
            raise ValueError("sanitize_chunk_rows 必须为正整数")
        if importance_sample_size < 0:
            raise ValueError("importance_sample_size 不能为负数")

        self.input_size = int(input_size)
        self.lookback = int(lookback)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.early_stop_rounds = int(early_stop_rounds)
        self.optimizer_name = optimizer.lower()
        self.device = device
        self.seed = seed
        self.grad_clip = float(grad_clip)
        self.metric = metric
        self.min_ic_samples = int(min_ic_samples)
        self.feature_fill_value = feature_fill_value
        self.sample_stride = int(sample_stride)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.use_memmap = bool(use_memmap)
        self.memmap_dir = memmap_dir
        self.feature_dtype = np.float32 if feature_dtype == "float32" else np.float64
        self.sanitize_chunk_rows = int(sanitize_chunk_rows)
        self.keep_data_for_detail = bool(keep_data_for_detail)
        self.importance_sample_size = int(importance_sample_size)

        self.fitted = False
        self.valid_loss = float("inf")
        self.best_epoch = -1
        self.best_valid_score = -float("inf")
        self.feature_names: list[str] = []
        self._importance_dataset: _LazySequenceDataset | None = None
        self._tempdir: tempfile.TemporaryDirectory | None = None
        self._warned_workers = False

        self._set_seed()
        self.model = GruNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer_obj = self._make_optimizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dataset: AlphaDataset, evals_result: dict | None = None) -> None:
        """训练模型。"""
        if evals_result is None:
            evals_result = {}

        self._set_seed()
        self.model.to(self.device)

        if self._importance_dataset is not None:
            self._importance_dataset.close()
            self._importance_dataset = None

        train_df = dataset.fetch_learn(Segment.TRAIN).sort(["datetime", "vt_symbol"])
        self.feature_names = self._infer_feature_names(train_df, require_label=True)
        self._validate_feature_count()

        train_ds: _LazySequenceDataset | None = None
        valid_ds: _LazySequenceDataset | None = None

        try:
            train_ds = self._build_sequence_dataset(
                df=train_df,
                require_label=True,
                segment_name="train",
            )
            valid_df = dataset.fetch_learn(Segment.VALID).sort(["datetime", "vt_symbol"])
            valid_ds = self._build_sequence_dataset(
                df=valid_df,
                require_label=True,
                segment_name="valid",
            )

            evals_result[Segment.TRAIN] = []
            evals_result[Segment.VALID] = []

            logger.info(
                f"train: rows={train_df.height}, sequence_samples={len(train_ds)}, "
                f"lookback={self.lookback}, features={len(self.feature_names)}"
            )
            logger.info(
                f"valid: rows={valid_df.height}, sequence_samples={len(valid_ds)}, "
                f"lookback={self.lookback}, features={len(self.feature_names)}"
            )

            if len(train_ds) == 0:
                raise ValueError(
                    "训练集没有可用的 GRU 序列样本。请检查 lookback 是否过大、label 是否全为 NaN、"
                    "或 feature_fill_value=None 时特征窗口是否存在 NaN/Inf。"
                )

            best_params = copy.deepcopy(self.model.state_dict())
            best_valid_score = -float("inf")
            best_epoch = -1
            early_stop_count = 0
            last_valid_loss = float("inf")

            for epoch in range(self.n_epochs):
                logger.info(f"Epoch {epoch}:")
                logger.info("training...")
                self._train_step(train_ds, epoch)

                logger.info("evaluating...")
                train_loss, train_score = self._evaluate_step(train_ds)
                valid_loss, valid_score = self._evaluate_step(valid_ds)
                last_valid_loss = valid_loss

                logger.info(
                    f"train_loss={train_loss:.6f}, train_score={train_score:.6f}, "
                    f"valid_loss={valid_loss:.6f}, valid_score={valid_score:.6f}"
                )

                evals_result[Segment.TRAIN].append(train_score)
                evals_result[Segment.VALID].append(valid_score)

                # 正常用 valid_score；若 valid 没有可用样本，则退回 train_score，避免 best_params 留在随机初始化。
                selection_score = valid_score if np.isfinite(valid_score) else train_score

                if np.isfinite(selection_score) and selection_score > best_valid_score:
                    best_valid_score = float(selection_score)
                    best_epoch = int(epoch)
                    early_stop_count = 0
                    best_params = copy.deepcopy(self.model.state_dict())
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.early_stop_rounds:
                        logger.info("early stop")
                        break

            self.model.load_state_dict(best_params)
            self.fitted = True
            self.valid_loss = float(last_valid_loss)
            self.best_epoch = int(best_epoch)
            self.best_valid_score = float(best_valid_score)

            logger.info(f"best score: {best_valid_score:.6f} @ {best_epoch}")

            # 默认不保留训练/验证二维特征矩阵，避免 fit 后常驻大内存。
            # 如需 detail() 中计算 permutation importance，请设置 keep_data_for_detail=True 且 importance_sample_size>0。
            if self.keep_data_for_detail and self.importance_sample_size > 0:
                if len(valid_ds) > 0:
                    self._importance_dataset = valid_ds
                    valid_ds = None
                else:
                    self._importance_dataset = train_ds
                    train_ds = None

        finally:
            if train_ds is not None:
                train_ds.close()
            if valid_ds is not None:
                valid_ds.close()

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """
        对指定 segment 预测。

        返回数组长度等于：
            dataset.fetch_infer(segment).sort(["datetime", "vt_symbol"]).height

        无法形成完整 lookback 历史窗口的行返回 NaN。
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用 fit()")
        if not self.feature_names:
            raise ValueError("feature_names 为空，模型状态不完整")

        df = dataset.fetch_infer(segment).sort(["datetime", "vt_symbol"])
        if df.height == 0:
            return np.array([], dtype=np.float32)

        pred_ds: _LazySequenceDataset | None = None
        try:
            pred_ds = self._build_sequence_dataset(
                df=df,
                require_label=False,
                segment_name=f"predict_{segment}",
            )

            full_pred = np.full(df.height, np.nan, dtype=np.float32)
            if len(pred_ds) == 0:
                return full_pred

            pred = self._predict_dataset(pred_ds)
            full_pred[pred_ds.row_positions] = pred.astype(np.float32, copy=False)
            return full_pred
        finally:
            if pred_ds is not None:
                pred_ds.close()

    def detail(self) -> pd.DataFrame:
        """输出模型信息，并在启用时返回流式 permutation importance。"""
        if not self.fitted:
            logger.info("模型尚未训练，无法显示详细信息")
            return pd.DataFrame(columns=["Importance"]).rename_axis("Feature")

        logger.info(f"输入特征维度: {self.input_size}")
        logger.info(f"lookback窗口: {self.lookback}")
        logger.info(f"GRU层数: {self.num_layers}")
        logger.info(f"GRU隐藏层维度: {self.hidden_size}")
        logger.info(f"dropout: {self.dropout}")
        logger.info(f"metric: {self.metric}")
        logger.info(f"best score: {self.best_valid_score:.6f} @ {self.best_epoch}")
        logger.info(f"valid_loss: {self.valid_loss:.6f}")
        logger.info(f"训练设备: {self.device}")
        logger.info(f"学习率: {self.lr}")
        logger.info(f"批次大小: {self.batch_size}")
        logger.info(f"sample_stride: {self.sample_stride}")
        logger.info(f"use_memmap: {self.use_memmap}")

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型总参数量: {total_params:,}")

        return self._calculate_feature_importance()

    def close_cached_data(self) -> None:
        """手动释放 detail() 重要性分析保留的数据。"""
        if self._importance_dataset is not None:
            self._importance_dataset.close()
            self._importance_dataset = None

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_sequence_dataset(
        self,
        *,
        df: pl.DataFrame,
        require_label: bool,
        segment_name: str,
    ) -> _LazySequenceDataset:
        """
        从二维面板数据构造懒加载序列 Dataset。

        注意：这里不构造三维 x，只构造 sample_end_positions。
        """
        if df.height == 0:
            empty_features = np.empty((0, self.input_size), dtype=self.feature_dtype)
            return _LazySequenceDataset(
                features=empty_features,
                end_positions=np.empty(0, dtype=np.int64),
                lookback=self.lookback,
                input_size=self.input_size,
                labels=np.empty(0, dtype=np.float32) if require_label else None,
                sample_dates=np.empty(0, dtype=object),
                row_positions=np.empty(0, dtype=np.int64),
            )

        missing_features = [c for c in self.feature_names if c not in df.columns]
        if missing_features:
            raise ValueError(f"数据缺少训练时使用的特征列: {missing_features}")
        if require_label and "label" not in df.columns:
            raise ValueError("学习数据缺少 label 列")

        # row_pos 对齐外部可见顺序：sort(["datetime", "vt_symbol"])
        # 之后再按 vt_symbol/datetime 排序，保证同一标的历史窗口在 features 中连续。
        output_df = df.sort(["datetime", "vt_symbol"])
        output_df = output_df.with_columns(pl.Series("__row_pos__", np.arange(output_df.height, dtype=np.int64)))
        work_df = output_df.sort(["vt_symbol", "datetime"])

        features, feature_path = self._make_feature_array(work_df, segment_name)

        if self.feature_fill_value is None:
            row_bad = self._row_has_nonfinite(features)
        else:
            self._fill_nonfinite_features_inplace(features, float(self.feature_fill_value))
            row_bad = np.zeros(features.shape[0], dtype=bool)

        labels_by_row: np.ndarray | None = None
        if require_label:
            labels_by_row = self._series_to_numpy(work_df["label"], dtype=np.float32)

        datetimes = work_df["datetime"].to_numpy()
        row_positions_by_row = self._series_to_numpy(work_df["__row_pos__"], dtype=np.int64)
        symbols = work_df["vt_symbol"].to_numpy()

        candidate_ends = self._make_candidate_end_positions(symbols)
        if len(candidate_ends) == 0:
            return _LazySequenceDataset(
                features=features,
                end_positions=np.empty(0, dtype=np.int64),
                lookback=self.lookback,
                input_size=self.input_size,
                labels=np.empty(0, dtype=np.float32) if require_label else None,
                sample_dates=np.empty(0, dtype=object),
                row_positions=np.empty(0, dtype=np.int64),
                feature_path=feature_path,
            )

        mask = np.ones(len(candidate_ends), dtype=bool)

        if require_label:
            assert labels_by_row is not None
            mask &= np.isfinite(labels_by_row[candidate_ends])

        if self.feature_fill_value is None:
            bad_prefix = np.empty(len(row_bad) + 1, dtype=np.int64)
            bad_prefix[0] = 0
            np.cumsum(row_bad.astype(np.int64), out=bad_prefix[1:])
            starts = candidate_ends - self.lookback + 1
            window_bad_count = bad_prefix[candidate_ends + 1] - bad_prefix[starts]
            mask &= window_bad_count == 0

        end_positions = candidate_ends[mask]
        sample_dates = datetimes[end_positions]
        row_positions = row_positions_by_row[end_positions]

        labels = labels_by_row[end_positions].astype(np.float32, copy=False) if require_label else None

        return _LazySequenceDataset(
            features=features,
            end_positions=end_positions,
            lookback=self.lookback,
            input_size=self.input_size,
            labels=labels,
            sample_dates=sample_dates,
            row_positions=row_positions,
            feature_path=feature_path,
        )

    def _make_candidate_end_positions(self, symbols: np.ndarray) -> np.ndarray:
        n_rows = len(symbols)
        if n_rows < self.lookback:
            return np.empty(0, dtype=np.int64)

        is_start = np.empty(n_rows, dtype=bool)
        is_start[0] = True
        is_start[1:] = symbols[1:] != symbols[:-1]

        group_starts = np.flatnonzero(is_start)
        group_ends = np.r_[group_starts[1:], n_rows]

        pieces: list[np.ndarray] = []
        for start, end in zip(group_starts, group_ends):
            first_end = int(start) + self.lookback - 1
            if first_end >= int(end):
                continue
            pieces.append(np.arange(first_end, int(end), self.sample_stride, dtype=np.int64))

        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(pieces)

    def _make_feature_array(self, df: pl.DataFrame, segment_name: str) -> tuple[np.ndarray, str | None]:
        n_rows = df.height
        n_features = len(self.feature_names)

        if self.use_memmap:
            directory = self._get_memmap_dir()
            path = os.path.join(directory, f"gru_{segment_name}_{uuid.uuid4().hex}.dat")
            features = np.memmap(path, dtype=self.feature_dtype, mode="w+", shape=(n_rows, n_features))

            # 列式写入 memmap，避免 df.select(...).to_numpy() 一次性产生完整二维临时矩阵。
            for j, name in enumerate(self.feature_names):
                col = self._series_to_numpy(df[name], dtype=self.feature_dtype)
                features[:, j] = col
            features.flush()
            return features, path

        # 非 memmap 模式会持有一份 C 连续二维矩阵。
        # 这仍是 O(N*F)，但不会乘以 lookback。
        arr = df.select(self.feature_names).to_numpy()
        features = np.asarray(arr, dtype=self.feature_dtype, order="C")
        if not features.flags["C_CONTIGUOUS"]:
            features = np.ascontiguousarray(features, dtype=self.feature_dtype)
        return features, None

    def _get_memmap_dir(self) -> str:
        if self.memmap_dir is not None:
            os.makedirs(self.memmap_dir, exist_ok=True)
            return self.memmap_dir
        if self._tempdir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="gru_lazy_")
        return self._tempdir.name

    def _series_to_numpy(self, series: pl.Series, dtype: np.dtype) -> np.ndarray:
        arr = series.to_numpy()
        return np.asarray(arr, dtype=dtype)

    def _row_has_nonfinite(self, features: np.ndarray) -> np.ndarray:
        row_bad = np.zeros(features.shape[0], dtype=bool)
        for start in range(0, features.shape[0], self.sanitize_chunk_rows):
            end = min(start + self.sanitize_chunk_rows, features.shape[0])
            block = features[start:end]
            row_bad[start:end] = ~np.isfinite(block).all(axis=1)
        return row_bad

    def _fill_nonfinite_features_inplace(self, features: np.ndarray, fill_value: float) -> None:
        for start in range(0, features.shape[0], self.sanitize_chunk_rows):
            end = min(start + self.sanitize_chunk_rows, features.shape[0])
            block = features[start:end]
            bad = ~np.isfinite(block)
            if bad.any():
                # 对 memmap view 原地写入；对 ndarray 也是原地写入。
                block[bad] = fill_value
                features[start:end] = block

        if isinstance(features, np.memmap):
            features.flush()

    # ------------------------------------------------------------------
    # Training / evaluation / prediction
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        seq_ds: _LazySequenceDataset,
        *,
        shuffle: bool,
        include_label: bool,
        epoch: int | None = None,
        subset_indices: np.ndarray | None = None,
        perturb_feature: int | None = None,
        perm_lookup: dict[int, int] | None = None,
    ) -> DataLoader:
        if self.num_workers > 0 and not self._warned_workers:
            logger.info(
                "num_workers > 0 时，不同平台/启动方式可能复制大数组或 memmap 句柄；"
                "若目标是最小内存占用，建议保持 num_workers=0。"
            )
            self._warned_workers = True

        loader_dataset: Dataset
        if subset_indices is not None:
            loader_dataset = Subset(seq_ds, [int(i) for i in subset_indices])
            # Subset 已经规定顺序；这里一般不再 shuffle。
            effective_shuffle = False
        else:
            loader_dataset = seq_ds
            effective_shuffle = shuffle

        generator = None
        if self.seed is not None and effective_shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(self.seed) + int(epoch or 0))

        kwargs = {
            "dataset": loader_dataset,
            "batch_size": self.batch_size,
            "shuffle": effective_shuffle,
            "drop_last": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": self._make_collate_fn(
                seq_ds,
                include_label=include_label,
                perturb_feature=perturb_feature,
                perm_lookup=perm_lookup,
            ),
        }
        if generator is not None:
            kwargs["generator"] = generator
        if self.num_workers > 0:
            kwargs["persistent_workers"] = False

        return DataLoader(**kwargs)

    def _make_collate_fn(
        self,
        seq_ds: _LazySequenceDataset,
        *,
        include_label: bool,
        perturb_feature: int | None = None,
        perm_lookup: dict[int, int] | None = None,
    ):
        lookback = self.lookback
        input_size = self.input_size
        features = seq_ds.features
        end_positions = seq_ds.end_positions
        labels = seq_ds.labels

        def collate(batch_indices: list[int]):
            sample_indices = np.asarray(batch_indices, dtype=np.int64)
            x = np.empty((len(sample_indices), lookback, input_size), dtype=np.float32)

            for i, sample_idx in enumerate(sample_indices):
                end_pos = int(end_positions[sample_idx])
                start_pos = end_pos - lookback + 1
                x[i] = features[start_pos:end_pos + 1]

            if perturb_feature is not None:
                if perturb_feature < 0 or perturb_feature >= input_size:
                    raise ValueError(f"perturb_feature 越界: {perturb_feature}")
                if perm_lookup is None:
                    raise ValueError("perturb_feature 不为 None 时必须提供 perm_lookup")

                for i, sample_idx in enumerate(sample_indices):
                    permuted_idx = int(perm_lookup.get(int(sample_idx), int(sample_idx)))
                    end_pos = int(end_positions[permuted_idx])
                    start_pos = end_pos - lookback + 1
                    x[i, :, perturb_feature] = features[start_pos:end_pos + 1, perturb_feature]

            x_tensor = torch.from_numpy(x)

            if include_label:
                if labels is None:
                    raise ValueError("include_label=True 但 Dataset 没有 labels")
                y = labels[sample_indices].astype(np.float32, copy=False)
                return x_tensor, torch.from_numpy(y)

            return x_tensor

        return collate

    def _train_step(self, train_ds: _LazySequenceDataset, epoch: int) -> None:
        self.model.train()
        if len(train_ds) == 0:
            return

        loader = self._make_loader(
            train_ds,
            shuffle=True,
            include_label=True,
            epoch=epoch,
        )

        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(self.device, non_blocking=self.pin_memory)
            batch_labels = batch_labels.to(self.device, non_blocking=self.pin_memory)

            predictions = self.model(batch_features)
            loss = self._loss_fn(predictions, batch_labels)
            if not torch.isfinite(loss):
                continue

            self.optimizer_obj.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer_obj.step()

    def _evaluate_step(self, seq_ds: _LazySequenceDataset) -> tuple[float, float]:
        if len(seq_ds) == 0 or seq_ds.labels is None:
            return float("inf"), -float("inf")

        predictions = self._predict_dataset(seq_ds)
        labels = seq_ds.labels
        mask = np.isfinite(predictions) & np.isfinite(labels)
        if not mask.any():
            return float("inf"), -float("inf")

        mse = float(np.mean((predictions[mask] - labels[mask]) ** 2))

        if self.metric == "neg_mse":
            score = -mse
        else:
            score = self._daily_ic_score(
                predictions=predictions,
                labels=labels,
                sample_dates=seq_ds.sample_dates,
                rank=(self.metric == "rank_ic"),
            )
            if not np.isfinite(score):
                score = -mse

        return mse, float(score)

    def _predict_dataset(
        self,
        seq_ds: _LazySequenceDataset,
        *,
        subset_indices: np.ndarray | None = None,
        perturb_feature: int | None = None,
        perm_lookup: dict[int, int] | None = None,
    ) -> np.ndarray:
        if len(seq_ds) == 0:
            return np.empty(0, dtype=np.float32)

        self.model.eval()
        loader = self._make_loader(
            seq_ds,
            shuffle=False,
            include_label=False,
            subset_indices=subset_indices,
            perturb_feature=perturb_feature,
            perm_lookup=perm_lookup,
        )

        preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch_features in loader:
                batch_features = batch_features.to(self.device, non_blocking=self.pin_memory)
                batch_pred = self.model(batch_features).detach().cpu().numpy()
                preds.append(batch_pred.astype(np.float32, copy=False))

        if not preds:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(preds).reshape(-1)

    def _loss_fn(self, predictions: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(predictions) & torch.isfinite(label)
        if mask.sum() == 0:
            return torch.tensor(float("nan"), device=predictions.device)
        return torch.mean((predictions[mask] - label[mask]) ** 2)

    def _daily_ic_score(
        self,
        *,
        predictions: np.ndarray,
        labels: np.ndarray,
        sample_dates: np.ndarray,
        rank: bool,
    ) -> float:
        if len(sample_dates) != len(predictions):
            return float("nan")

        df = pd.DataFrame({
            "datetime": sample_dates,
            "pred": predictions,
            "label": labels,
        })

        ic_values: list[float] = []
        for _dt, g in df.groupby("datetime", sort=False):
            p = g["pred"].to_numpy(dtype=float)
            y = g["label"].to_numpy(dtype=float)
            mask = np.isfinite(p) & np.isfinite(y)
            if mask.sum() < self.min_ic_samples:
                continue

            p = p[mask]
            y = y[mask]

            if rank:
                p = pd.Series(p).rank(method="average").to_numpy(dtype=float)
                y = pd.Series(y).rank(method="average").to_numpy(dtype=float)

            if np.std(p) <= 1e-12 or np.std(y) <= 1e-12:
                continue

            ic = float(np.corrcoef(p, y)[0, 1])
            if np.isfinite(ic):
                ic_values.append(ic)

        if not ic_values:
            return float("nan")
        return float(np.mean(ic_values))

    # ------------------------------------------------------------------
    # Importance / metadata
    # ------------------------------------------------------------------

    def _calculate_feature_importance(self) -> pd.DataFrame:
        """
        流式 permutation importance。

        默认关闭，因为对高维因子逐列扰动本身就是重计算任务。
        启用方式：
            keep_data_for_detail=True, importance_sample_size > 0
        """
        if self._importance_dataset is None or self.importance_sample_size <= 0:
            return pd.DataFrame(columns=["Importance"]).rename_axis("Feature")

        seq_ds = self._importance_dataset
        if len(seq_ds) == 0:
            return pd.DataFrame(columns=["Importance"]).rename_axis("Feature")

        rng = np.random.default_rng(self.seed)
        n = min(len(seq_ds), self.importance_sample_size)
        if n < len(seq_ds):
            sample_indices = rng.choice(len(seq_ds), size=n, replace=False).astype(np.int64)
        else:
            sample_indices = np.arange(len(seq_ds), dtype=np.int64)

        base_pred = self._predict_dataset(seq_ds, subset_indices=sample_indices)
        importance_dict: dict[str, float] = {}

        for feature_idx, feature_name in enumerate(self.feature_names):
            permuted = sample_indices.copy()
            rng.shuffle(permuted)
            perm_lookup = {int(src): int(dst) for src, dst in zip(sample_indices, permuted)}

            new_pred = self._predict_dataset(
                seq_ds,
                subset_indices=sample_indices,
                perturb_feature=feature_idx,
                perm_lookup=perm_lookup,
            )
            diff = np.abs(new_pred - base_pred)
            importance_dict[feature_name] = float(np.nanmean(diff)) if len(diff) else float("nan")

        out = pd.DataFrame({
            "Feature": list(importance_dict.keys()),
            "Importance": list(importance_dict.values()),
        })
        return out.sort_values("Importance", ascending=False).set_index("Feature")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _infer_feature_names(self, df: pl.DataFrame, require_label: bool) -> list[str]:
        columns = list(df.columns)
        required = {"datetime", "vt_symbol"}
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"数据缺少必要列: {sorted(missing)}")
        if require_label and "label" not in columns:
            raise ValueError("学习数据缺少 label 列")

        excluded = {"datetime", "vt_symbol", "label"}
        feature_names = [c for c in columns if c not in excluded]
        if not feature_names:
            raise ValueError("没有找到特征列")
        return feature_names

    def _validate_feature_count(self) -> None:
        if len(self.feature_names) != self.input_size:
            raise ValueError(
                f"input_size={self.input_size} 与实际特征数={len(self.feature_names)} 不一致。"
                f"实际特征列: {self.feature_names}"
            )

    def _set_seed(self) -> None:
        if self.seed is None:
            return
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _make_optimizer(self) -> optim.Optimizer:
        if self.optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        if self.optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.lr)
        if self.optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.lr)
        raise ValueError(f"不支持的优化器: {self.optimizer_name}")

    def __del__(self) -> None:
        try:
            self.close_cached_data()
            if self._tempdir is not None:
                self._tempdir.cleanup()
        except Exception:
            pass


class GruNetwork(nn.Module):
    """标准 GRU 回归网络。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, lookback, input_size]
        return: [batch_size]
        """
        if x.dim() != 3:
            raise ValueError(f"GRU 输入必须是 3D: [batch, lookback, input_size]，实际为 {tuple(x.shape)}")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"最后一维特征数应为 {self.input_size}，实际为 {x.shape[-1]}")

        out, _ = self.gru(x)
        pred = self.fc(out[:, -1, :])
        return pred.squeeze(-1)
