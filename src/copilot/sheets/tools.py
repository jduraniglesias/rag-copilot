from __future__ import annotations
from typing import Dict, Any
import pandas as pd

def sum_range(df: pd.DataFrame, col: str, where: Dict[str, Any] | None = None) -> float:
    data = df
    if where:
        for k, v in where.items():
            data = data[data[k] == v]
    return float(data[col].sum())

def pivot_table(df: pd.DataFrame, index: str, values: str, aggfunc: str = "sum") -> pd.DataFrame:
    return pd.pivot_table(df, index=index, values=values, aggfunc=aggfunc).reset_index()
