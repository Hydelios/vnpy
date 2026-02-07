from __future__ import annotations

import time
from multiprocessing import get_context
from typing import Mapping, Sequence

import polars as pl
from tqdm import tqdm  # type: ignore

from .feature_spec import BaseFeatureSpec, InputRequirements
from .utility import calculate_by_expression, calculate_by_polars


class FeatureComputer:
    def __init__(
        self,
        *,
        id_cols: Sequence[str] = ("datetime", "vt_symbol"),
        join_how: str = "left",
        sort_keys: bool = True,
    ):
        self.id_cols = list(id_cols)
        self.join_how = join_how
        self.sort_keys = sort_keys

    def compute(
        self,
        df: pl.DataFrame,
        expressions: Mapping[str, str | pl.Expr] | None = None,
        *,
        spec: BaseFeatureSpec | None = None,
        features: Sequence[str] | None = None,
        exposures_df: pl.DataFrame | None = None,
        external_inputs: Mapping[str, pl.DataFrame] | None = None,
        external_outputs: Mapping[str, pl.DataFrame] | None = None,
        max_workers: int | None = None,
        return_requirements: bool = False,
        strict: bool = True,
    ) -> pl.DataFrame | tuple[pl.DataFrame, InputRequirements]:
        if spec is not None and expressions is not None:
            raise ValueError("spec and expressions are mutually exclusive")

        req = InputRequirements()
        if spec is not None:
            expressions = dict(spec.export_features())
            if features is not None:
                missing = [f for f in features if f not in expressions]
                if missing:
                    raise ValueError(f"unknown features: {missing}")
                expressions = {k: expressions[k] for k in features}
            req = spec.requirements.subset(features=features)

        if expressions is None:
            raise ValueError("expressions or spec must be provided")

        need_cols = set(self.id_cols)
        if req.bars_cols:
            need_cols |= set(req.bars_cols)
        if strict:
            miss = [c for c in need_cols if c not in df.columns]
            if miss:
                raise ValueError(f"missing base columns: {miss}")

        work = df
        if exposures_df is not None:
            if req.exposure_cols:
                miss = [c for c in req.exposure_cols if c not in exposures_df.columns]
                if miss:
                    raise ValueError(f"missing exposure columns: {miss}")
                cols = list(self.id_cols)
                for col in sorted(req.exposure_cols):
                    if col not in cols:
                        cols.append(col)
                exposures_df = exposures_df.select(cols)
            work = work.join(exposures_df, on=self.id_cols, how=self.join_how)
        elif strict and req.exposure_cols:
            raise ValueError(f"missing exposures_df for: {sorted(req.exposure_cols)}")

        if external_inputs:
            for _, ext in external_inputs.items():
                work = work.join(ext, on=self.id_cols, how=self.join_how)

        if self.sort_keys and all(c in work.columns for c in self.id_cols):
            work = work.sort(self.id_cols)

        expr_items = list(expressions.items())
        if expr_items:
            args = [(work, name, expression) for name, expression in expr_items]
            if max_workers and max_workers > 0:
                context = get_context("spawn")
                with context.Pool(processes=max_workers) as pool:
                    it = pool.imap(calculate_feature, args)
                    series_list = [result for result in tqdm(it, total=len(args))]
            else:
                series_list = []
                for arg in tqdm(args):
                    series_list.append(calculate_feature(arg))
            work = work.with_columns(series_list)

        if external_outputs:
            for _, ext in external_outputs.items():
                work = work.join(ext, on=self.id_cols, how=self.join_how)
            if self.sort_keys and all(c in work.columns for c in self.id_cols):
                work = work.sort(self.id_cols)

        if return_requirements:
            return work, req
        return work


def calculate_feature(args: tuple[pl.DataFrame, str, str | pl.expr.expr.Expr]) -> pl.Series:
    start = time.time()

    df, name, expression = args

    try:
        if isinstance(expression, pl.expr.expr.Expr):
            result = calculate_by_polars(df, expression)["data"].alias(name)
        else:
            result = calculate_by_expression(df, expression)["data"].alias(name)
    except Exception as e:
        expr_text = str(expression)
        print(f"[FeatureError] name={name} | expr={expr_text} | err={e}")
        raise RuntimeError(f"Error calculating feature '{name}' with expression: {expr_text}") from e

    end = time.time()
    print(f"Feature calculation {name} took: {end - start} seconds | {expression}")

    return result
