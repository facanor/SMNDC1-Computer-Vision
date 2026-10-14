from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterable
from .naming import normalize_core

def is_dir(p: Path) -> bool:
    try:
        return p.exists() and p.is_dir()
    except Exception:
        return False

def find_first_existing_dir(base: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = base / name
        if is_dir(p):
            return p
    return None

def list_immediate_subdirs(p: Path) -> List[Path]:
    if not is_dir(p):
        return []
    return [x for x in p.iterdir() if x.is_dir()]

def pick_channel_dir(group_root: Path, channel_candidates: List[str]) -> Optional[Path]:
    # direct match first
    for c in channel_candidates:
        p = group_root / c
        if is_dir(p):
            return p
    # fallback: search subdirs by keyword containment (case-insensitive)
    subs = list_immediate_subdirs(group_root)
    for sd in subs:
        name = sd.name.lower()
        for c in channel_candidates:
            if c.lower() in name:
                return sd
    return None

def list_files_nonrecursive(d: Path) -> List[Path]:
    if not is_dir(d):
        return []
    return [x for x in d.iterdir() if x.is_file()]

def filter_by_ext(paths: Iterable[Path], exts: List[str]) -> List[Path]:
    exts_l = set([e.lower() for e in exts])
    out = []
    for p in paths:
        if p.suffix.lower() in exts_l:
            out.append(p)
    return out

def cores_from_files(paths: List[Path]) -> Dict[str, List[str]]:
    """
    Returns: core_norm -> list of example raw filenames
    """
    d: Dict[str, List[str]] = {}
    for p in paths:
        core_norm, _ = normalize_core(p.name)
        if core_norm is None:
            continue
        d.setdefault(core_norm, [])
        if len(d[core_norm]) < 5:
            d[core_norm].append(p.name)
    return d

def scan_group_channels(group_root: Path, channels_cfg: Dict[str, List[str]], img_exts: List[str]) -> Dict[str, Dict]:
    """
    Scans B/G/R channels if present (non-recursive), counts files and cores.
    """
    out: Dict[str, Dict] = {}
    for ch_key, cand_list in channels_cfg.items():
        ch_dir = pick_channel_dir(group_root, cand_list)
        if ch_dir is None:
            out[ch_key] = {"dir": None, "n_files": 0, "cores": {}, "files": []}
            continue
        files = filter_by_ext(list_files_nonrecursive(ch_dir), img_exts)
        core_map = cores_from_files(files)
        out[ch_key] = {
            "dir": str(ch_dir),
            "n_files": len(files),
            "cores": core_map,
            "files": [str(x) for x in files[:50]],  # sample only
        }
    return out

def find_mask_like_assets(group_root: Path, img_exts: List[str], mask_dir_keywords: List[str], mask_file_keywords: List[str]) -> List[Path]:
    """
    Controlled recursive scan: only descends into directories whose names contain mask_dir_keywords,
    and collects files with mask_file_keywords in filename (plus allowed extensions).
    """
    if not is_dir(group_root):
        return []
    kw_dirs = [k.lower() for k in mask_dir_keywords]
    kw_files = [k.lower() for k in mask_file_keywords]
    exts_l = set([e.lower() for e in img_exts])

    mask_files: List[Path] = []

    # Walk only into keyword dirs to avoid heavy full-tree walk
    for root, dirs, files in os.walk(group_root):
        root_p = Path(root)
        root_name = root_p.name.lower()

        # Decide whether to continue descending:
        # Keep only dirs containing keywords OR if we're at group_root (to discover such dirs).
        if root_p != group_root and not any(k in root_name for k in kw_dirs):
            # prune descent
            dirs[:] = []
            continue

        # prune child dirs to only likely ones
        pruned = []
        for d in dirs:
            dn = d.lower()
            if any(k in dn for k in kw_dirs):
                pruned.append(d)
        if root_p == group_root:
            # allow discovering candidate dirs from root level
            dirs[:] = pruned
        else:
            dirs[:] = pruned

        for fn in files:
            fpl = fn.lower()
            if Path(fn).suffix.lower() not in exts_l:
                continue
            if any(k in fpl for k in kw_files):
                mask_files.append(root_p / fn)

    return mask_files
