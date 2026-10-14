from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .naming import normalize_core

def group_files_by_core(files: List[Path]) -> Dict[str, List[Path]]:
    m: Dict[str, List[Path]] = {}
    for p in files:
        cn, _ = normalize_core(p.name)
        if cn is None:
            continue
        m.setdefault(cn, []).append(p)
    return m

def _score_file(p: Path) -> Tuple[int, int, int, int]:
    """
    Higher is better. Tuple sorting descending.
    Heuristics:
      - Prefer filenames containing '8' (common in your naming), then shorter names, then underscore over dash.
      - Prefer files with '.tif/.tiff' implicitly handled by upstream filter.
    """
    name = p.name.lower()
    s8 = 1 if ("_8" in name or "8.tif" in name or "8.tiff" in name) else 0
    has_underscore = 1 if "_" in name else 0
    has_dash = 1 if "-" in name else 0
    # prefer underscore, penalize dash (but both can exist)
    ud = has_underscore - has_dash
    # prefer shorter filename (invert length)
    inv_len = -len(name)
    return (s8, ud, inv_len, 0)

def select_best_file(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    # sort by score (descending)
    return sorted(paths, key=_score_file, reverse=True)[0]

def build_channel_index(channel_dir: Path, img_exts: List[str]) -> Dict[str, Dict]:
    """
    Returns core_norm -> { 'best': <path>, 'n': <count>, 'candidates': [..first few..] }
    """
    exts = set(e.lower() for e in img_exts)
    files = [p for p in channel_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    grouped = group_files_by_core(files)

    out: Dict[str, Dict] = {}
    for cn, plist in grouped.items():
        best = select_best_file(plist)
        out[cn] = {
            "best": str(best) if best else None,
            "n": len(plist),
            "candidates": [str(x) for x in sorted(plist)[:5]]
        }
    return out
