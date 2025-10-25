# minimal intent router + dispatcher for Sheets actions
from __future__ import annotations
from typing import Any, Dict, Tuple
import pandas as pd

from .tools import sum_range, pivot_table

def classify_intent(q: str) -> str:
    ql = q.lower()
    if "pivot" in ql: return "pivot"
    if "sum" in ql or "total" in ql: return "sum"
    return "ask"  # fallback to retrieval QA

def dispatch(intent: str, df: pd.DataFrame, **kwargs) -> Tuple[str, Any]:
    """
    Returns (intent, result). For 'ask' return ('ask', None) and let the QA path handle it.
    kwargs can carry params like col=, where=, index=, values=, aggfunc=...
    """
    if intent == "sum":
        col = kwargs.get("col")
        where = kwargs.get("where")
        return "sum", sum_range(df, col=col, where=where)
    if intent == "pivot":
        index = kwargs.get("index")
        values = kwargs.get("values")
        aggfunc = kwargs.get("aggfunc", "sum")
        return "pivot", pivot_table(df, index=index, values=values, aggfunc=aggfunc)
    return "ask", None
