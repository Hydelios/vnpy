from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vnpy.alpha import logger
from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel


class RidgeModel(AlphaModel):
    """
    Ridge regression model for alpha research.

    线性模型对 NaN/inf 和特征量纲比较敏感，因此模型内部固定封装
    imputer/scaler/ridge pipeline，便于在 Notebook、回测脚本和实盘
    信号脚本中复用同一个 pickle 产物。
    """

    def __init__(
        self,
        alpha: float = 1.0,
        alphas: Sequence[float] | np.ndarray | None = None,
        cv_folds: int = 5,
        fit_intercept: bool = True,
        standardize: bool = True,
        impute_strategy: str = "median",
        impute_fill_value: float | None = None,
        solver: str = "auto",
        positive: bool = False,
        max_iter: int | None = None,
        tol: float = 1e-4,
        random_state: int | None = None,
        scoring: str | None = None,
        sample_weight_col: str | None = None,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha 必须大于等于 0")
        if cv_folds < 2:
            raise ValueError("cv_folds 必须大于等于 2")
        if sample_weight_col is not None and not str(sample_weight_col).strip():
            raise ValueError("sample_weight_col 不能为空字符串")

        alpha_candidates: list[float] | None = None
        if alphas is not None:
            alpha_candidates = [float(value) for value in list(alphas)]
            if not alpha_candidates:
                raise ValueError("alphas 不能为空")
            if any(value <= 0 for value in alpha_candidates):
                raise ValueError("alphas 必须全部大于 0")

        self.alpha: float = float(alpha)
        self.alphas: list[float] | None = alpha_candidates
        self.cv_folds: int = int(cv_folds)
        self.fit_intercept: bool = fit_intercept
        self.standardize: bool = standardize
        self.impute_strategy: str = impute_strategy
        self.impute_fill_value: float | None = impute_fill_value
        self.solver: str = solver
        self.positive: bool = positive
        self.max_iter: int | None = max_iter
        self.tol: float = float(tol)
        self.random_state: int | None = random_state
        self.scoring: str | None = scoring
        self.sample_weight_col: str | None = sample_weight_col

        self.params: dict[str, Any] = {
            "alpha": self.alpha,
            "alphas": self.alphas,
            "cv_folds": self.cv_folds,
            "fit_intercept": self.fit_intercept,
            "standardize": self.standardize,
            "impute_strategy": self.impute_strategy,
            "impute_fill_value": self.impute_fill_value,
            "solver": self.solver,
            "positive": self.positive,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "scoring": self.scoring,
            "sample_weight_col": self.sample_weight_col,
        }

        self.model: Pipeline | None = None
        self.feature_names: list[str] = []
        self.input_size: int = 0
        self.selected_alpha: float | None = None

    def _resolve_feature_names(self, df: pl.DataFrame) -> list[str]:
        exclude = {"datetime", "vt_symbol", "label"}
        if self.sample_weight_col is not None:
            exclude.add(self.sample_weight_col)

        feature_names = [name for name in df.columns if name not in exclude]
        if not feature_names:
            raise ValueError("训练数据缺少特征列")
        return feature_names

    def _to_matrix(self, df: pl.DataFrame) -> np.ndarray:
        data = np.asarray(df.select(self.feature_names).to_numpy(), dtype=np.float64)
        return np.where(np.isfinite(data), data, np.nan)

    def _to_label(self, df: pl.DataFrame) -> np.ndarray:
        if "label" not in df.columns:
            raise ValueError("训练数据缺少 label 列")
        return np.asarray(df["label"], dtype=np.float64)

    def _to_weight(self, df: pl.DataFrame) -> np.ndarray | None:
        if self.sample_weight_col is None:
            return None
        if self.sample_weight_col not in df.columns:
            raise ValueError(f"缺少 sample_weight_col: {self.sample_weight_col}")

        weight = np.asarray(df[self.sample_weight_col], dtype=np.float64)
        return np.where(np.isfinite(weight) & (weight >= 0), weight, 0.0)

    def _prepare_fit_data(self, dataset: AlphaDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        df_train = dataset.fetch_learn(Segment.TRAIN).sort(["datetime", "vt_symbol"])
        if df_train.is_empty():
            raise ValueError("训练集为空，无法训练 RidgeModel")

        self.feature_names = self._resolve_feature_names(df_train)
        self.input_size = len(self.feature_names)

        data = self._to_matrix(df_train)
        label = self._to_label(df_train)
        weight = self._to_weight(df_train)

        valid_label = np.isfinite(label)
        if not np.all(valid_label):
            data = data[valid_label]
            label = label[valid_label]
            if weight is not None:
                weight = weight[valid_label]

        if label.size == 0:
            raise ValueError("训练集 label 全部无效，无法训练 RidgeModel")
        if weight is not None and float(np.sum(weight)) <= 0:
            raise ValueError("sample_weight_col 全部为 0，无法训练 RidgeModel")

        return data, label, weight

    def _build_estimator(self) -> Ridge | RidgeCV:
        if self.alphas is not None:
            return RidgeCV(
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
                scoring=self.scoring,
                cv=TimeSeriesSplit(n_splits=self.cv_folds),
            )

        return Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            copy_X=False,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
            positive=self.positive,
            random_state=self.random_state,
        )

    def _build_pipeline(self) -> Pipeline:
        steps: list[tuple[str, object]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy=self.impute_strategy,
                    fill_value=self.impute_fill_value,
                    keep_empty_features=True,
                ),
            )
        ]
        if self.standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("ridge", self._build_estimator()))
        return Pipeline(steps)

    def _ridge_step(self) -> Ridge | RidgeCV:
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        return cast(Ridge | RidgeCV, self.model.named_steps["ridge"])

    def fit(self, dataset: AlphaDataset) -> None:
        data, label, weight = self._prepare_fit_data(dataset)
        self.model = self._build_pipeline()

        fit_kwargs: dict[str, np.ndarray] = {}
        if weight is not None:
            fit_kwargs["ridge__sample_weight"] = weight
        self.model.fit(data, label, **fit_kwargs)

        ridge = self._ridge_step()
        self.selected_alpha = float(getattr(ridge, "alpha_", self.alpha))
        if self.alphas is not None:
            logger.info(f"RidgeCV 交叉验证完成，最佳惩罚系数 Alpha: {self.selected_alpha:.6f}")

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        df = dataset.fetch_infer(segment).sort(["datetime", "vt_symbol"])
        data = self._to_matrix(df)
        return np.asarray(self.model.predict(data), dtype=np.float64)

    def detail(self) -> pd.DataFrame | None:
        if self.model is None:
            logger.info("模型尚未训练，无法查看特征细节。")
            return None

        ridge = self._ridge_step()
        coef = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
        if len(coef) != len(self.feature_names):
            raise ValueError("Ridge 系数数量与特征数量不一致")

        df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Coef": coef,
                "Abs_Coef": np.abs(coef),
            }
        )
        return df.sort_values("Abs_Coef", ascending=False).reset_index(drop=True)


class CorrPrunedRidgeModel(AlphaModel):
    """
    Ridge model with greedy high-correlation feature pruning.

    特征先由上游数据集完成截面标准化，本模型只在训练集样本上估计特征
    相关矩阵，并在高度相关的一组特征里优先保留对 label 线性相关更高
    的特征。
    """

    def __init__(
        self,
        alpha: float = 1.0,
        corr_threshold: float = 0.9,
        selection_max_rows: int = 300_000,
        fit_intercept: bool = True,
        impute_strategy: str = "constant",
        impute_fill_value: float | None = 0.0,
        solver: str = "auto",
        max_iter: int | None = None,
        tol: float = 1e-4,
        random_state: int | None = 42,
        sample_weight_col: str | None = None,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha 必须大于等于 0")
        if not 0 <= corr_threshold < 1:
            raise ValueError("corr_threshold 必须位于 [0, 1) 区间")
        if selection_max_rows <= 0:
            raise ValueError("selection_max_rows 必须大于 0")

        self.alpha = float(alpha)
        self.corr_threshold = float(corr_threshold)
        self.selection_max_rows = int(selection_max_rows)
        self.fit_intercept = fit_intercept
        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value
        self.solver = solver
        self.max_iter = max_iter
        self.tol = float(tol)
        self.random_state = random_state
        self.sample_weight_col = sample_weight_col
        self.params: dict[str, Any] = {
            "alpha": self.alpha,
            "corr_threshold": self.corr_threshold,
            "selection_max_rows": self.selection_max_rows,
            "fit_intercept": self.fit_intercept,
            "impute_strategy": self.impute_strategy,
            "impute_fill_value": self.impute_fill_value,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "sample_weight_col": self.sample_weight_col,
        }

        self.model: Pipeline | None = None
        self.all_feature_names: list[str] = []
        self.feature_names: list[str] = []
        self.dropped_feature_names: list[str] = []
        self.feature_relevance_: dict[str, float] = {}
        self.selected_alpha: float | None = None

    def _resolve_feature_names(self, df: pl.DataFrame) -> list[str]:
        exclude = {"datetime", "vt_symbol", "label"}
        if self.sample_weight_col is not None:
            exclude.add(self.sample_weight_col)
        feature_names = [name for name in df.columns if name not in exclude]
        if not feature_names:
            raise ValueError("训练数据缺少特征列")
        return feature_names

    def _to_matrix(self, df: pl.DataFrame, names: list[str]) -> np.ndarray:
        data = np.asarray(df.select(names).to_numpy(), dtype=np.float64)
        return np.where(np.isfinite(data), data, np.nan)

    def _to_label(self, df: pl.DataFrame) -> np.ndarray:
        if "label" not in df.columns:
            raise ValueError("训练数据缺少 label 列")
        return np.asarray(df["label"], dtype=np.float64)

    def _to_weight(self, df: pl.DataFrame) -> np.ndarray | None:
        if self.sample_weight_col is None:
            return None
        if self.sample_weight_col not in df.columns:
            raise ValueError(f"缺少 sample_weight_col: {self.sample_weight_col}")
        weight = np.asarray(df[self.sample_weight_col], dtype=np.float64)
        return np.where(np.isfinite(weight) & (weight >= 0), weight, 0.0)

    def _sample_for_selection(self, n_rows: int) -> np.ndarray:
        if n_rows <= self.selection_max_rows:
            return np.arange(n_rows)
        rng = np.random.default_rng(self.random_state)
        return np.sort(rng.choice(n_rows, size=self.selection_max_rows, replace=False))

    def _select_features(self, data: np.ndarray, label: np.ndarray) -> list[str]:
        sample_idx = self._sample_for_selection(data.shape[0])
        x = np.nan_to_num(data[sample_idx], nan=0.0, posinf=0.0, neginf=0.0, copy=True)
        y = label[sample_idx].astype(np.float64, copy=True)

        x_centered = x - x.mean(axis=0, keepdims=True)
        x_std = x_centered.std(axis=0, ddof=0)
        usable = x_std > 1e-12
        x_scaled = np.zeros_like(x_centered)
        x_scaled[:, usable] = x_centered[:, usable] / x_std[usable]

        y_centered = y - float(np.mean(y))
        y_std = float(np.std(y_centered))
        if y_std <= 1e-12:
            relevance = np.zeros(x.shape[1], dtype=np.float64)
        else:
            relevance = np.abs(x_scaled.T @ (y_centered / y_std)) / max(len(y), 1)
        relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_relevance_ = {
            name: float(value) for name, value in zip(self.all_feature_names, relevance, strict=True)
        }

        corr = (x_scaled.T @ x_scaled) / max(x_scaled.shape[0], 1)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)

        order = np.argsort(-relevance)
        selected: list[int] = []
        for idx in order:
            if not usable[idx]:
                continue
            if selected and float(np.max(np.abs(corr[idx, selected]))) > self.corr_threshold:
                continue
            selected.append(int(idx))

        if not selected:
            selected = [int(order[0])]

        selected_set = set(selected)
        self.dropped_feature_names = [
            name for i, name in enumerate(self.all_feature_names) if i not in selected_set
        ]
        return [self.all_feature_names[i] for i in selected]

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy=self.impute_strategy,
                        fill_value=self.impute_fill_value,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=self.alpha,
                        fit_intercept=self.fit_intercept,
                        copy_X=False,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        solver=self.solver,
                    ),
                ),
            ]
        )

    def fit(self, dataset: AlphaDataset) -> None:
        df_train = dataset.fetch_learn(Segment.TRAIN).sort(["datetime", "vt_symbol"])
        if df_train.is_empty():
            raise ValueError("训练集为空，无法训练 CorrPrunedRidgeModel")

        self.all_feature_names = self._resolve_feature_names(df_train)
        data_all = self._to_matrix(df_train, self.all_feature_names)
        label = self._to_label(df_train)
        weight = self._to_weight(df_train)

        valid_label = np.isfinite(label)
        data_all = data_all[valid_label]
        label = label[valid_label]
        if weight is not None:
            weight = weight[valid_label]
        if label.size == 0:
            raise ValueError("训练集 label 全部无效，无法训练 CorrPrunedRidgeModel")

        self.feature_names = self._select_features(data_all, label)
        selected_idx = [self.all_feature_names.index(name) for name in self.feature_names]
        data = data_all[:, selected_idx]

        self.model = self._build_pipeline()
        fit_kwargs: dict[str, np.ndarray] = {}
        if weight is not None:
            fit_kwargs["ridge__sample_weight"] = weight
        self.model.fit(data, label, **fit_kwargs)
        self.selected_alpha = self.alpha

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        df = dataset.fetch_infer(segment).sort(["datetime", "vt_symbol"])
        data = self._to_matrix(df, self.feature_names)
        return np.asarray(self.model.predict(data), dtype=np.float64)

    def detail(self) -> pd.DataFrame | None:
        if self.model is None:
            logger.info("模型尚未训练，无法查看特征细节。")
            return None
        ridge = cast(Ridge, self.model.named_steps["ridge"])
        coef = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
        df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Coef": coef,
                "Abs_Coef": np.abs(coef),
                "Relevance": [self.feature_relevance_.get(name, 0.0) for name in self.feature_names],
            }
        )
        return df.sort_values("Abs_Coef", ascending=False).reset_index(drop=True)


class PCARidgeModel(AlphaModel):
    """Ridge model on PCA components estimated from standardized features."""

    def __init__(
        self,
        alpha: float = 1.0,
        n_components: float | int = 0.95,
        pca_max_rows: int = 500_000,
        fit_intercept: bool = True,
        impute_fill_value: float = 0.0,
        solver: str = "auto",
        max_iter: int | None = None,
        tol: float = 1e-4,
        random_state: int | None = 42,
        sample_weight_col: str | None = None,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha 必须大于等于 0")
        if isinstance(n_components, float) and not 0 < n_components <= 1:
            raise ValueError("float n_components 必须位于 (0, 1] 区间")
        if isinstance(n_components, int) and n_components <= 0:
            raise ValueError("int n_components 必须大于 0")
        if pca_max_rows <= 0:
            raise ValueError("pca_max_rows 必须大于 0")

        self.alpha = float(alpha)
        self.n_components = n_components
        self.pca_max_rows = int(pca_max_rows)
        self.fit_intercept = fit_intercept
        self.impute_fill_value = float(impute_fill_value)
        self.solver = solver
        self.max_iter = max_iter
        self.tol = float(tol)
        self.random_state = random_state
        self.sample_weight_col = sample_weight_col
        self.params: dict[str, Any] = {
            "alpha": self.alpha,
            "n_components": self.n_components,
            "pca_max_rows": self.pca_max_rows,
            "fit_intercept": self.fit_intercept,
            "impute_fill_value": self.impute_fill_value,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "sample_weight_col": self.sample_weight_col,
        }

        self.model: Ridge | None = None
        self.feature_names: list[str] = []
        self.pca_mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.selected_alpha: float | None = None

    def _resolve_feature_names(self, df: pl.DataFrame) -> list[str]:
        exclude = {"datetime", "vt_symbol", "label"}
        if self.sample_weight_col is not None:
            exclude.add(self.sample_weight_col)
        feature_names = [name for name in df.columns if name not in exclude]
        if not feature_names:
            raise ValueError("训练数据缺少特征列")
        return feature_names

    def _to_matrix(self, df: pl.DataFrame) -> np.ndarray:
        data = np.asarray(df.select(self.feature_names).to_numpy(), dtype=np.float64)
        return np.nan_to_num(
            data,
            nan=self.impute_fill_value,
            posinf=self.impute_fill_value,
            neginf=self.impute_fill_value,
            copy=False,
        )

    def _to_label(self, df: pl.DataFrame) -> np.ndarray:
        if "label" not in df.columns:
            raise ValueError("训练数据缺少 label 列")
        return np.asarray(df["label"], dtype=np.float64)

    def _to_weight(self, df: pl.DataFrame) -> np.ndarray | None:
        if self.sample_weight_col is None:
            return None
        if self.sample_weight_col not in df.columns:
            raise ValueError(f"缺少 sample_weight_col: {self.sample_weight_col}")
        weight = np.asarray(df[self.sample_weight_col], dtype=np.float64)
        return np.where(np.isfinite(weight) & (weight >= 0), weight, 0.0)

    def _sample_for_pca(self, n_rows: int) -> np.ndarray:
        if n_rows <= self.pca_max_rows:
            return np.arange(n_rows)
        rng = np.random.default_rng(self.random_state)
        return np.sort(rng.choice(n_rows, size=self.pca_max_rows, replace=False))

    def _fit_pca(self, data: np.ndarray) -> None:
        sample_idx = self._sample_for_pca(data.shape[0])
        x = data[sample_idx]
        self.pca_mean_ = x.mean(axis=0)
        centered = x - self.pca_mean_
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        total = float(eigvals.sum())
        if total <= 0:
            raise ValueError("PCA 协方差矩阵方差为 0")
        explained = eigvals / total
        if isinstance(self.n_components, float):
            n_keep = int(np.searchsorted(np.cumsum(explained), self.n_components) + 1)
        else:
            n_keep = min(int(self.n_components), eigvecs.shape[1])
        self.components_ = eigvecs[:, :n_keep].T
        self.explained_variance_ratio_ = explained[:n_keep]

    def _transform(self, data: np.ndarray) -> np.ndarray:
        if self.pca_mean_ is None or self.components_ is None:
            raise ValueError("PCA 尚未拟合")
        return (data - self.pca_mean_) @ self.components_.T

    def fit(self, dataset: AlphaDataset) -> None:
        df_train = dataset.fetch_learn(Segment.TRAIN).sort(["datetime", "vt_symbol"])
        if df_train.is_empty():
            raise ValueError("训练集为空，无法训练 PCARidgeModel")
        self.feature_names = self._resolve_feature_names(df_train)
        data = self._to_matrix(df_train)
        label = self._to_label(df_train)
        weight = self._to_weight(df_train)

        valid_label = np.isfinite(label)
        data = data[valid_label]
        label = label[valid_label]
        if weight is not None:
            weight = weight[valid_label]
        if label.size == 0:
            raise ValueError("训练集 label 全部无效，无法训练 PCARidgeModel")

        self._fit_pca(data)
        z = self._transform(data)
        self.model = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            copy_X=False,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
        )
        fit_kwargs: dict[str, np.ndarray] = {}
        if weight is not None:
            fit_kwargs["sample_weight"] = weight
        self.model.fit(z, label, **fit_kwargs)
        self.selected_alpha = self.alpha

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        df = dataset.fetch_infer(segment).sort(["datetime", "vt_symbol"])
        data = self._to_matrix(df)
        return np.asarray(self.model.predict(self._transform(data)), dtype=np.float64)

    @property
    def n_components_(self) -> int:
        return 0 if self.components_ is None else int(self.components_.shape[0])

    @property
    def explained_variance_sum_(self) -> float:
        if self.explained_variance_ratio_ is None:
            return 0.0
        return float(np.sum(self.explained_variance_ratio_))

    def detail(self) -> pd.DataFrame | None:
        if self.model is None or self.components_ is None:
            logger.info("模型尚未训练，无法查看特征细节。")
            return None
        coef_pc = np.asarray(self.model.coef_, dtype=np.float64).reshape(-1)
        coef = self.components_.T @ coef_pc
        df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Coef": coef,
                "Abs_Coef": np.abs(coef),
            }
        )
        return df.sort_values("Abs_Coef", ascending=False).reset_index(drop=True)


class ElasticNetModel(AlphaModel):
    """ElasticNet/Lasso sparse linear model for standardized alpha features."""

    def __init__(
        self,
        alpha: float = 1e-3,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        impute_strategy: str = "constant",
        impute_fill_value: float | None = 0.0,
        max_iter: int = 2_000,
        tol: float = 1e-4,
        random_state: int | None = 42,
        selection: str = "cyclic",
        sample_weight_col: str | None = None,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha 必须大于等于 0")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio 必须位于 [0, 1] 区间")
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = fit_intercept
        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.selection = selection
        self.sample_weight_col = sample_weight_col
        self.params: dict[str, Any] = {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.fit_intercept,
            "impute_strategy": self.impute_strategy,
            "impute_fill_value": self.impute_fill_value,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "selection": self.selection,
            "sample_weight_col": self.sample_weight_col,
        }
        self.model: Pipeline | None = None
        self.feature_names: list[str] = []
        self.selected_alpha: float | None = None
        self.selected_l1_ratio: float | None = None

    def _resolve_feature_names(self, df: pl.DataFrame) -> list[str]:
        exclude = {"datetime", "vt_symbol", "label"}
        if self.sample_weight_col is not None:
            exclude.add(self.sample_weight_col)
        feature_names = [name for name in df.columns if name not in exclude]
        if not feature_names:
            raise ValueError("训练数据缺少特征列")
        return feature_names

    def _to_matrix(self, df: pl.DataFrame) -> np.ndarray:
        data = np.asarray(df.select(self.feature_names).to_numpy(), dtype=np.float64)
        return np.where(np.isfinite(data), data, np.nan)

    def _to_label(self, df: pl.DataFrame) -> np.ndarray:
        if "label" not in df.columns:
            raise ValueError("训练数据缺少 label 列")
        return np.asarray(df["label"], dtype=np.float64)

    def _to_weight(self, df: pl.DataFrame) -> np.ndarray | None:
        if self.sample_weight_col is None:
            return None
        if self.sample_weight_col not in df.columns:
            raise ValueError(f"缺少 sample_weight_col: {self.sample_weight_col}")
        weight = np.asarray(df[self.sample_weight_col], dtype=np.float64)
        return np.where(np.isfinite(weight) & (weight >= 0), weight, 0.0)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy=self.impute_strategy,
                        fill_value=self.impute_fill_value,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "elasticnet",
                    ElasticNet(
                        alpha=self.alpha,
                        l1_ratio=self.l1_ratio,
                        fit_intercept=self.fit_intercept,
                        copy_X=False,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=self.random_state,
                        selection=self.selection,
                    ),
                ),
            ]
        )

    def fit(self, dataset: AlphaDataset) -> None:
        df_train = dataset.fetch_learn(Segment.TRAIN).sort(["datetime", "vt_symbol"])
        if df_train.is_empty():
            raise ValueError("训练集为空，无法训练 ElasticNetModel")
        self.feature_names = self._resolve_feature_names(df_train)
        data = self._to_matrix(df_train)
        label = self._to_label(df_train)
        weight = self._to_weight(df_train)

        valid_label = np.isfinite(label)
        data = data[valid_label]
        label = label[valid_label]
        if weight is not None:
            weight = weight[valid_label]
        if label.size == 0:
            raise ValueError("训练集 label 全部无效，无法训练 ElasticNetModel")

        self.model = self._build_pipeline()
        fit_kwargs: dict[str, np.ndarray] = {}
        if weight is not None:
            fit_kwargs["elasticnet__sample_weight"] = weight
        self.model.fit(data, label, **fit_kwargs)
        self.selected_alpha = self.alpha
        self.selected_l1_ratio = self.l1_ratio

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        df = dataset.fetch_infer(segment).sort(["datetime", "vt_symbol"])
        data = self._to_matrix(df)
        return np.asarray(self.model.predict(data), dtype=np.float64)

    def detail(self) -> pd.DataFrame | None:
        if self.model is None:
            logger.info("模型尚未训练，无法查看特征细节。")
            return None
        estimator = cast(ElasticNet, self.model.named_steps["elasticnet"])
        coef = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
        df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Coef": coef,
                "Abs_Coef": np.abs(coef),
                "Nonzero": np.abs(coef) > 1e-10,
            }
        )
        return df.sort_values("Abs_Coef", ascending=False).reset_index(drop=True)
