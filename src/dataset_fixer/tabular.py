"""Canonical pandas boundaries for report and metric tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


TableLike = pd.DataFrame | Iterable[Mapping[str, Any]]


def frame(
    values: TableLike | None = None,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return an independent, range-indexed table with optional fixed columns."""

    if isinstance(values, pd.DataFrame):
        result = values.copy(deep=True)
    else:
        result = pd.DataFrame.from_records(list(values or ()), columns=columns)
    if columns is not None:
        result = result.reindex(columns=columns)
    return result.convert_dtypes().reset_index(drop=True)


def records(values: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a table to JSON-ready row records without leaking pandas NA."""

    clean = values.astype(object).where(values.notna(), None)
    return clean.to_dict(orient="records")


def chart_data(values: TableLike) -> pd.DataFrame:
    """Prepare tabular values for Altair while preserving rows and columns."""

    result = frame(values).replace([np.inf, -np.inf], np.nan)
    return result.astype(object).where(result.notna(), None)


def stable_sort(
    values: pd.DataFrame,
    by: str | Sequence[str],
    *,
    ascending: bool | Sequence[bool] = True,
) -> pd.DataFrame:
    """Sort deterministically and restore the public RangeIndex contract."""

    return values.sort_values(
        by=list(by) if not isinstance(by, str) else by,
        ascending=ascending,
        kind="stable",
    ).reset_index(drop=True)
