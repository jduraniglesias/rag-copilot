from __future__ import annotations
from typing import Dict, Any
import pandas as pd

def _apply_where(df: pd.DataFrame, where: Dict[str, Any] | None) -> pd.DataFrame:
    if not where:
        return df
    out = df
    for col, val in where.items():
        out = out[out[col] == val]
    return out

def sum_range(df: pd.DataFrame, col: str, where: Dict[str, Any] | None = None) -> float:
    df2 = _apply_where(df, where)
    return float(df2[col].sum())

def pivot_table(df: pd.DataFrame, index: str, values: str, aggfunc: str = "sum") -> pd.DataFrame:
    tbl = pd.pivot_table(df, index=index, values=values, aggfunc=aggfunc).reset_index()
    # ensure plain types for JSON
    return tbl.astype(object)
