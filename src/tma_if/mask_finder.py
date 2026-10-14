from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

MASK_HINTS = [
    "_cp_masks", "_cyto_masks", "_cytoplasm_masks",
    "_nuc_masks", "_nucleus_masks", "_nuclei_masks",
    "_prot_masks", "_protein_masks",
    "mask", "masks",
]

@dataclass
class MaskHit:
    chosen: Optional[Path]
    candidates: List[Path]
    reason: str


def _is_mask_path(p: Path) -> bool:
    name = p.name.lower()
    return any(h in name for h in ["_cp_masks", "_nuc_masks", "_prot_masks", "mask", "masks"])


def _core_stem(p: Path) -> str:
    # Keep full stem (includes .ome_* inside name). Example: 1_1.ome_R_citoplasma8
    return p.stem


def find_mask_for_image(
    image_path: str,
    prefer: str = "label",
    extra_roots: Optional[List[str]] = None,
) -> Tuple[Optional[str], List[str], str]:
    """
    Returns (chosen_mask_path, all_candidates, reason).

    Behavior:
    - If image_path already looks like a mask => returns itself.
    - Else searches same directory for mask-like variants.
    - Optionally searches extra_roots (recursively) for the same filename variants.
    """
    p = Path(image_path)
    if not p.exists():
        return None, [], f"image_not_found: {image_path}"

    if _is_mask_path(p):
        return str(p), [str(p)], "input_is_mask"

    stem = _core_stem(p)
    parent = p.parent

    # Common explicit variants
    explicit = [
        parent / f"{stem}_cp_masks.tif",
        parent / f"{stem}_nuc_masks.tif",
        parent / f"{stem}_prot_masks.tif",
        parent / f"{stem}_cp_masks.tiff",
        parent / f"{stem}_nuc_masks.tiff",
        parent / f"{stem}_prot_masks.tiff",
    ]

    candidates: List[Path] = []
    for q in explicit:
        if q.exists():
            candidates.append(q)

    # Keyword glob in same dir (NO '**' inside filename pattern)
    for kw in ["_cp_masks", "_nuc_masks", "_prot_masks", "mask", "masks"]:
        for ext in ["tif", "tiff", "png"]:
            for cand in parent.glob(f"{stem}*{kw}*.{ext}"):
                if cand.exists():
                    candidates.append(cand)

    # Search in extra roots (mask cache, etc.)
    if extra_roots:
        wanted_names = set([c.name for c in candidates] + [f"{stem}_cp_masks.tif", f"{stem}_nuc_masks.tif", f"{stem}_prot_masks.tif"])
        for r in extra_roots:
            rr = Path(r)
            if not rr.exists():
                continue
            # rglob on filename patterns (safe)
            for ext in ["tif", "tiff", "png"]:
                for kw in ["_cp_masks", "_nuc_masks", "_prot_masks", "mask", "masks"]:
                    pattern = f"{stem}*{kw}*.{ext}"
                    for cand in rr.rglob(pattern):
                        if cand.exists():
                            candidates.append(cand)

            # Also exact name hits
            for name in list(wanted_names):
                hit = next(rr.rglob(name), None)
                if hit is not None and hit.exists():
                    candidates.append(hit)

    # Deduplicate
    uniq = []
    seen = set()
    for c in candidates:
        s = str(c.resolve())
        if s not in seen:
            uniq.append(c)
            seen.add(s)
    candidates = uniq

    if not candidates:
        return None, [], "no_mask_found"

    # Preference: label masks if prefer == "label"
    if prefer == "label":
        # crude heuristic: prefer nuc masks for B channel, cp masks for R channel, otherwise any
        name = p.name.lower()
        if "_b_" in name or "nucleos" in name:
            for c in candidates:
                if "_nuc_masks" in c.name.lower():
                    return str(c), [str(x) for x in candidates], "picked_nuc_masks"
        if "_r_" in name or "citoplasma" in name:
            for c in candidates:
                if "_cp_masks" in c.name.lower():
                    return str(c), [str(x) for x in candidates], "picked_cp_masks"

    return str(candidates[0]), [str(x) for x in candidates], "picked_first_candidate"
