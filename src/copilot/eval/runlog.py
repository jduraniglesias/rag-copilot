import json, time, os
from typing import Dict
def log_run(metrics: Dict, path_dir="runs"):
    os.makedirs(path_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    with open(f"{path_dir}/eval-{ts}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
