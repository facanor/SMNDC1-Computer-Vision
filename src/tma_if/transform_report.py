from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from .naming import normalize_core

def find_transform_csvs(base_dir: Path, prev_results_dir: Path, keywords: List[str]) -> List[Path]:
    keys = [k.lower() for k in keywords]
    out: List[Path] = []
    for root in [base_dir, prev_results_dir]:
        if not root.exists():
            continue
        # search shallow-ish (depth 4) for speed
        for p in root.glob("**/*.csv"):
            name_l = p.name.lower()
            if any(k in name_l for k in keys):
                out.append(p)
    # deduplicate
    uniq = []
    seen = set()
    for p in out:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq

def summarize_transform_csv(csv_path: Path) -> Dict:
    info: Dict = {"path": str(csv_path), "n_rows": 0, "n_unique_cores": 0, "has_dice": False, "dice_stats": {}, "columns": []}
    try:
        df = pd.read_csv(csv_path)
        info["n_rows"] = int(df.shape[0])
        info["columns"] = [str(c) for c in df.columns]
        # core_name column guess
        core_col = None
        for c in df.columns:
            cl = str(c).lower()
            if cl in ["core_name", "core", "corename"]:
                core_col = c
                break
        if core_col is None:
            # fallback: first column
            core_col = df.columns[0]

        core_norms = []
        for v in df[core_col].astype(str).tolist():
            cn, _ = normalize_core(v)
            if cn is not None:
                core_norms.append(cn)
        info["n_unique_cores"] = int(len(set(core_norms)))

        # dice column guess
        dice_col = None
        for c in df.columns:
            if str(c).lower() in ["dice_final", "dice", "dice_score"]:
                dice_col = c
                break
        if dice_col is not None:
            info["has_dice"] = True
            s = pd.to_numeric(df[dice_col], errors="coerce").dropna()
            if len(s) > 0:
                info["dice_stats"] = {
                    "min": float(s.min()),
                    "p25": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "p75": float(s.quantile(0.75)),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                }
    except Exception as e:
        info["error"] = str(e)
    return info
