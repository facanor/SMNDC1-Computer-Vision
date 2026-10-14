from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

def find_excels(base_dir: Path, names: List[str]) -> List[Path]:
    out: List[Path] = []
    for n in names:
        p = base_dir / n
        if p.exists() and p.is_file():
            out.append(p)
    # Also: search shallow (depth 2) for any xlsx with those exact names
    for p in base_dir.glob("*.xlsx"):
        if p.name in names and p not in out:
            out.append(p)
    for p in base_dir.glob("*/*.xlsx"):
        if p.name in names and p not in out:
            out.append(p)
    return out

def xlsx_overview(xlsx_path: Path) -> Dict:
    info: Dict = {"path": str(xlsx_path), "sheets": [], "sheet_shapes": {}, "columns_preview": {}}
    try:
        xl = pd.ExcelFile(xlsx_path)
        info["sheets"] = list(xl.sheet_names)
        for sh in xl.sheet_names:
            try:
                df = xl.parse(sh, header=None)
                info["sheet_shapes"][sh] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
                # try headered parse too (best-effort)
                dfh = xl.parse(sh)
                cols = [str(c) for c in dfh.columns[:30]]
                info["columns_preview"][sh] = cols
            except Exception:
                continue
    except Exception as e:
        info["error"] = str(e)
    return info

def build_core_to_cpid_map(tma_array_xlsx: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Uses the pair of sheets like:
      - TMA_Num: grid with strings like '10-1'
      - TMA_ID : same grid coordinates contain CPID
    Returns:
      core_pos ('10-1') -> CPID
      core_norm ('10_1') -> CPID
    """
    pos_to_cpid: Dict[str, str] = {}
    norm_to_cpid: Dict[str, str] = {}

    xl = pd.ExcelFile(tma_array_xlsx)
    if "TMA_Num" not in xl.sheet_names or "TMA_ID" not in xl.sheet_names:
        return norm_to_cpid, pos_to_cpid

    num = xl.parse("TMA_Num", header=None).astype(str)
    cid = xl.parse("TMA_ID", header=None).astype(str)

    # find cells in num matching pattern r-c
    for r in range(num.shape[0]):
        for c in range(num.shape[1]):
            v = str(num.iat[r, c]).strip()
            if "-" in v and v.replace("-", "").isdigit():
                # v like "10-1"
                cpid = str(cid.iat[r, c]).strip()
                if cpid and cpid.lower() not in ["nan", "none", ""]:
                    pos_to_cpid[v] = cpid
                    norm_to_cpid[v.replace("-", "_")] = cpid
    return norm_to_cpid, pos_to_cpid

def tumor_annotations_overview(tumor_xlsx: Path) -> Dict:
    info: Dict = {"path": str(tumor_xlsx), "sheets": [], "guesses": {}}
    try:
        xl = pd.ExcelFile(tumor_xlsx)
        info["sheets"] = list(xl.sheet_names)
        # heuristic: take first sheet
        sh = xl.sheet_names[0]
        df = xl.parse(sh)
        cols = {str(c).lower(): str(c) for c in df.columns}
        # CPID column guess
        cpid_col = None
        for k in ["cpid", "case", "case_id", "patient", "id"]:
            if k in cols:
                cpid_col = cols[k]; break
        # OS guess
        os_col = None
        for k in ["os_days", "os", "overall_survival_days", "survival_days", "days"]:
            if k in cols:
                os_col = cols[k]; break
        # EVENT guess
        ev_col = None
        for k in ["event_bin", "event", "status", "dead", "vital_status"]:
            if k in cols:
                ev_col = cols[k]; break

        info["guesses"] = {"sheet": sh, "cpid_col": cpid_col, "os_col": os_col, "event_col": ev_col, "n_rows": int(df.shape[0])}
    except Exception as e:
        info["error"] = str(e)
    return info
