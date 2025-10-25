from __future__ import annotations
import json, os, time
from typing import Any, Dict

def _stamp() -> str:
    # e.g., 2025-10-25T15-04-12
    return time.strftime("%Y-%m-%dT%H-%M-%S")

def _ensure_dir(path_dir: str) -> None:
    os.makedirs(path_dir, exist_ok=True)

def _sanitize(obj: Any):
    """
    Make args/config JSON-safe: convert Paths, sets, Namespace-like objects, etc.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    # Common non-serializable types:
    try:
        import pathlib, argparse
        if isinstance(obj, pathlib.Path):
            return str(obj)
        if isinstance(obj, argparse.Namespace):
            return {k: _sanitize(v) for k, v in vars(obj).items() if k not in ("func",)}
    except Exception:
        pass
    # Fallback: try plain JSON conversion
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def log_run(config: Dict[str, Any], metrics: Dict[str, Any], dir_path: str = "runs") -> str:
    """
    Write {config, metrics, ts} as JSON to runs/eval-<timestamp>.json
    Returns full filepath.
    """
    _ensure_dir(dir_path)
    payload = {
        "ts": _stamp(),
        "config": _sanitize(config),
        "metrics": _sanitize(metrics),
    }
    fname = f"eval-{payload['ts']}.json"
    fpath = os.path.join(dir_path, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return fpath
