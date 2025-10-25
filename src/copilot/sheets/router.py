from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Literal
import re
import pandas as pd

from .tools import sum_range, pivot_table

Intent = Literal["sum", "pivot", "ask"]

# --- Keyword detectors ---
_SUM    = re.compile(r"\b(sum|total|add up)\b", re.I)
_PIVOT  = re.compile(r"\b(pivot|group by|breakdown)\b", re.I)
# e.g., "where region=west", `where dept = 'Hardware'`
_WHERE  = re.compile(r"\bwhere\s+([A-Za-z0-9_]+)\s*=\s*(['\"]?)([^'\",]+)\2", re.I)
# e.g., "sum sales", "sum of amount"
_SUM_COL= re.compile(r"\b(?:sum|total)(?:\s+of)?\s+([A-Za-z0-9_]+)\b", re.I)
# e.g., "pivot by region", "group by category"
_BY     = re.compile(r"\b(?:pivot|group by|by)\s+([A-Za-z0-9_]+)\b", re.I)
# e.g., "values sales", "value amount"
_VALUES = re.compile(r"\b(values?|measure|metric)\s+([A-Za-z0-9_]+)\b", re.I)
# e.g., "using avg", "aggfunc mean"
_AGG    = re.compile(r"\b(?:using|aggfunc)\s+(sum|avg|mean|count|min|max)\b", re.I)

# Common column name hints
LIKELY_VALUE_COLS = ("sales", "amount", "revenue", "value", "total", "count")
LIKELY_INDEX_COLS = ("region", "category", "dept", "department", "type", "status")

@dataclass
class ParsedIntent:
    kind: Intent
    args: Dict[str, Any]

def classify_intent(q: str) -> Intent:
    if _PIVOT.search(q): return "pivot"
    if _SUM.search(q):   return "sum"
    return "ask"

def _guess_value_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for name in LIKELY_VALUE_COLS:
        for c in cols:
            if c.lower() == name: 
                return c
    # numeric fallback
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else (cols[0] if cols else None)

def _guess_index_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for name in LIKELY_INDEX_COLS:
        for c in cols:
            if c.lower() == name:
                return c
    # categorical-ish fallback
    for c in cols:
        if df[c].nunique() <= max(50, len(df) // 5):
            return c
    return cols[0] if cols else None

def parse(q: str, df: Optional[pd.DataFrame] = None) -> ParsedIntent:
    """Parse instruction -> (intent, args) with light defaults based on df."""
    k = classify_intent(q)
    ql = q.lower()
    args: Dict[str, Any] = {}

    if k == "sum":
        m_col = _SUM_COL.search(q)
        if m_col:
            args["col"] = m_col.group(1)
        # where clause (supports a single equality for now)
        m_where = _WHERE.search(q)
        if m_where:
            col, _, val = m_where.groups()
            args["where"] = {col: val}
        # defaults if not provided
        if df is not None:
            args.setdefault("col", _guess_value_col(df))

    elif k == "pivot":
        m_by = _BY.search(q)
        if m_by:
            args["index"] = m_by.group(1)
        m_val = _VALUES.search(q)
        if m_val:
            args["values"] = m_val.group(2)
        m_agg = _AGG.search(q)
        if m_agg:
            agg = m_agg.group(1)
            args["aggfunc"] = "mean" if agg == "avg" else agg
        if df is not None:
            args.setdefault("index", _guess_index_col(df))
            args.setdefault("values", _guess_value_col(df))
            args.setdefault("aggfunc", "sum")

    # 'ask' passes through
    return ParsedIntent(k, args)

def dispatch(intent: str, df: pd.DataFrame, **kwargs) -> Tuple[str, Any, str]:
    """
    Returns (intent, result, explanation).
    For 'ask' return ('ask', None, 'routed to QA').
    kwargs may include: col, where, index, values, aggfunc...
    """
    if intent == "sum":
        col   = kwargs.get("col")
        where = kwargs.get("where")
        if not col:
            raise ValueError("sum: 'col' is required (e.g., 'sum sales ...').")
        res = sum_range(df, col=col, where=where)
        expl = f"Summed '{col}'" + (f" with filter {where}" if where else "")
        return "sum", res, expl

    if intent == "pivot":
        index   = kwargs.get("index")
        values  = kwargs.get("values")
        aggfunc = kwargs.get("aggfunc", "sum")
        if not index or not values:
            raise ValueError("pivot: 'index' and 'values' are required (e.g., 'pivot by region values sales').")
        tbl = pivot_table(df, index=index, values=values, aggfunc=aggfunc)
        expl = f"Pivoted by '{index}' with values '{values}' using '{aggfunc}'."
        return "pivot", tbl, expl

    return "ask", None, "routed to QA"
