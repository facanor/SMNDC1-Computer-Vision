#!/usr/bin/env python3
"""Build the canonical MAPK3/SMNDC1 TMA core manifest.

The manifest contains cores for which all three IF channels (B/G/R) are
available in all four experimental datasets:

    MAPK3 Exon4
    MAPK3 NoExon4
    SMNDC1 Exon4
    SMNDC1 NoExon4

When available, core-to-patient and overall-survival information are added
from TMA_Array.xlsx and Tumor_Annotations.xlsx.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import pandas as pd
import yaml

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from tma_if.excel_report import build_core_to_cpid_map
from tma_if.manifest import build_channel_index
from tma_if.scan import find_first_existing_dir, pick_channel_dir


REQUIRED_DATASETS = (
    "mapk3_exon",
    "mapk3_noexon",
    "smndc1_exon",
    "smndc1_noexon",
)
CHANNELS = ("B", "G", "R")


def _column_ci(df: pd.DataFrame, requested: str) -> Optional[str]:
    if requested in df.columns:
        return requested
    lookup = {str(c).strip().lower(): c for c in df.columns}
    return lookup.get(str(requested).strip().lower())


def _clean_cpid(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "na", "n/a"}:
        return None
    if s.endswith(".0"):
        left = s[:-2]
        if left.isdigit():
            s = left
    return s


def load_clinical_survival(
    tumor_annotations_xlsx: Path,
    clinical_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Return one survival row per CPID.

    The returned columns are CPID, OS_days, OS_censor and OS_event.
    OS_event is derived as 1 - OS_censor only for rows where OS_censor is
    explicitly coded as 0 or 1.
    """
    xl = pd.ExcelFile(tumor_annotations_xlsx)
    if not xl.sheet_names:
        raise ValueError(f"No worksheets found in {tumor_annotations_xlsx}")

    sheet = clinical_cfg.get("sheet") or xl.sheet_names[0]
    if sheet not in xl.sheet_names:
        raise ValueError(
            f"Clinical worksheet {sheet!r} was not found in "
            f"{tumor_annotations_xlsx}. Available sheets: {xl.sheet_names}"
        )

    df = xl.parse(sheet)

    cpid_col = _column_ci(df, clinical_cfg.get("cpid_col", "CPID"))
    followup_col = _column_ci(
        df,
        clinical_cfg.get(
            "followup_or_death_col", "Date_merged_last_followup_death"
        ),
    )
    diagnosis_col = _column_ci(
        df, clinical_cfg.get("diagnosis_date_col", "Date_of_diagnosis")
    )
    censor_col = _column_ci(df, clinical_cfg.get("censor_col", "OS_censor"))

    if cpid_col is None:
        raise ValueError("Could not find the configured CPID column in clinical data.")
    if followup_col is None:
        raise ValueError(
            "Could not find the configured follow-up/death column in clinical data."
        )

    out = pd.DataFrame(index=df.index)
    out["CPID"] = df[cpid_col].map(_clean_cpid)

    if bool(clinical_cfg.get("followup_is_days", False)):
        out["OS_days"] = pd.to_numeric(df[followup_col], errors="coerce")
    else:
        if diagnosis_col is None:
            raise ValueError(
                "diagnosis_date_col is required when followup_is_days is false."
            )
        diagnosis = pd.to_datetime(df[diagnosis_col], errors="coerce")
        followup = pd.to_datetime(df[followup_col], errors="coerce")
        out["OS_days"] = (followup - diagnosis).dt.days.astype("Float64")

    if censor_col is not None:
        out["OS_censor"] = pd.to_numeric(df[censor_col], errors="coerce")
    else:
        out["OS_censor"] = pd.Series(pd.NA, index=df.index, dtype="Float64")

    out["OS_event"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    binary = out["OS_censor"].isin([0, 1])
    out.loc[binary, "OS_event"] = 1 - out.loc[binary, "OS_censor"]

    out.loc[pd.to_numeric(out["OS_days"], errors="coerce") < 0, "OS_days"] = pd.NA
    out = out.dropna(subset=["CPID"]).copy()

    # If duplicate clinical rows exist for a patient, keep the row with the
    # longest valid follow-up while preserving deterministic ordering.
    out["_os_sort"] = pd.to_numeric(out["OS_days"], errors="coerce")
    out = out.sort_values(
        ["CPID", "_os_sort"], ascending=[True, False], na_position="last"
    )
    out = out.drop_duplicates("CPID", keep="first").drop(columns="_os_sort")

    return out[["CPID", "OS_days", "OS_censor", "OS_event"]].reset_index(drop=True)


def _cores_with_all_channels(channel_indices: Dict[str, Dict[str, Dict]]) -> Set[str]:
    sets = [set(channel_indices[ch].keys()) for ch in CHANNELS]
    return set.intersection(*sets) if sets else set()


def _require_mapping(config: Dict[str, Any], *keys: str) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            dotted = ".".join(keys)
            raise KeyError(f"Missing configuration key: {dotted}")
        cur = cur[key]
    return cur


def _resolve_dataset_roots(base_dir: Path, groups_cfg: Dict[str, Any]) -> Dict[str, Path]:
    candidates = {
        "mapk3_exon": _require_mapping(groups_cfg, "mapk3", "exon_candidates"),
        "mapk3_noexon": _require_mapping(groups_cfg, "mapk3", "noexon_candidates"),
        "smndc1_exon": _require_mapping(groups_cfg, "smndc1", "exon_candidates"),
        "smndc1_noexon": _require_mapping(groups_cfg, "smndc1", "noexon_candidates"),
    }

    roots: Dict[str, Path] = {}
    missing = []
    for name, names in candidates.items():
        root = find_first_existing_dir(base_dir, list(names))
        if root is None:
            missing.append(name)
        else:
            roots[name] = Path(root).resolve()

    if missing:
        raise FileNotFoundError(
            "Could not resolve dataset directories for: " + ", ".join(missing)
        )
    return roots


def _resolve_input_xlsx(
    explicit: Optional[str],
    base_dir: Path,
    filename: str,
    required: bool,
) -> Optional[Path]:
    path = Path(explicit).expanduser().resolve() if explicit else (base_dir / filename)
    if path.exists() and path.is_file():
        return path
    if required:
        raise FileNotFoundError(f"Required input file not found: {path}")
    return None


def build_manifest(
    cfg: Dict[str, Any],
    tma_array_xlsx: Optional[Path],
    tumor_annotations_xlsx: Optional[Path],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    base_dir = Path(_require_mapping(cfg, "base_dir")).expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base_dir does not exist: {base_dir}")

    groups_cfg = _require_mapping(cfg, "groups")
    channels_cfg = _require_mapping(cfg, "channels")
    img_exts = list(cfg.get("img_exts", [".tif", ".tiff"]))

    roots = _resolve_dataset_roots(base_dir, groups_cfg)

    channel_dirs: Dict[str, Dict[str, Path]] = {}
    indices: Dict[str, Dict[str, Dict[str, Dict]]] = {}

    for dataset in REQUIRED_DATASETS:
        channel_dirs[dataset] = {}
        indices[dataset] = {}
        for channel in CHANNELS:
            candidates = channels_cfg.get(channel)
            if not candidates:
                raise KeyError(f"No directory candidates configured for channel {channel}")
            channel_dir = pick_channel_dir(roots[dataset], list(candidates))
            if channel_dir is None:
                raise FileNotFoundError(
                    f"Could not resolve channel {channel} under {roots[dataset]}"
                )
            channel_dir = Path(channel_dir).resolve()
            channel_dirs[dataset][channel] = channel_dir
            indices[dataset][channel] = build_channel_index(channel_dir, img_exts)

    dataset_core_sets = {
        dataset: _cores_with_all_channels(indices[dataset])
        for dataset in REQUIRED_DATASETS
    }
    all4 = set.intersection(*(dataset_core_sets[d] for d in REQUIRED_DATASETS))

    norm_to_cpid: Dict[str, str] = {}
    if tma_array_xlsx is not None:
        norm_to_cpid, _ = build_core_to_cpid_map(tma_array_xlsx)
        norm_to_cpid = {
            str(k): _clean_cpid(v)
            for k, v in norm_to_cpid.items()
            if _clean_cpid(v) is not None
        }

    clinical_map: Dict[str, Dict[str, Any]] = {}
    if tumor_annotations_xlsx is not None:
        clinical = load_clinical_survival(
            tumor_annotations_xlsx, cfg.get("clinical", {})
        )
        clinical_map = clinical.set_index("CPID").to_dict(orient="index")

    rows = []
    for core_norm in sorted(
        all4,
        key=lambda x: tuple(int(v) for v in x.split("_", maxsplit=1)),
    ):
        cpid = _clean_cpid(norm_to_cpid.get(core_norm))
        clinical = clinical_map.get(cpid, {}) if cpid is not None else {}

        row: Dict[str, Any] = {
            "core_norm": core_norm,
            "core_pos": core_norm.replace("_", "-"),
            "CPID": cpid,
            "OS_days": clinical.get("OS_days", pd.NA),
            "OS_censor": clinical.get("OS_censor", pd.NA),
            "OS_event": clinical.get("OS_event", pd.NA),
        }

        for dataset in REQUIRED_DATASETS:
            for channel in CHANNELS:
                row[f"{dataset}_{channel}"] = indices[dataset][channel][core_norm]["best"]

        rows.append(row)

    columns = [
        "core_norm",
        "core_pos",
        "CPID",
        "OS_days",
        "OS_censor",
        "OS_event",
        *[
            f"{dataset}_{channel}"
            for dataset in REQUIRED_DATASETS
            for channel in CHANNELS
        ],
    ]
    manifest = pd.DataFrame(rows, columns=columns)

    summary: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "dataset_roots": {k: str(v) for k, v in roots.items()},
        "channel_dirs": {
            d: {c: str(p) for c, p in ch.items()} for d, ch in channel_dirs.items()
        },
        "cores_with_BGR_by_dataset": {
            k: len(v) for k, v in dataset_core_sets.items()
        },
        "all4_cores": int(len(all4)),
        "cpid_mapped": int(manifest["CPID"].notna().sum()) if not manifest.empty else 0,
        "OS_days_mapped": int(
            pd.to_numeric(manifest["OS_days"], errors="coerce").notna().sum()
        ) if not manifest.empty else 0,
        "OS_censor_mapped": int(
            pd.to_numeric(manifest["OS_censor"], errors="coerce").notna().sum()
        ) if not manifest.empty else 0,
        "OS_event_mapped": int(
            pd.to_numeric(manifest["OS_event"], errors="coerce").notna().sum()
        ) if not manifest.empty else 0,
        "tma_array_xlsx": str(tma_array_xlsx) if tma_array_xlsx else None,
        "tumor_annotations_xlsx": (
            str(tumor_annotations_xlsx) if tumor_annotations_xlsx else None
        ),
    }
    return manifest, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical manifest containing cores with B/G/R channels "
            "in MAPK3 Exon4, MAPK3 NoExon4, SMNDC1 Exon4 and SMNDC1 NoExon4."
        )
    )
    parser.add_argument("--config", required=True, help="Path to paths.yaml.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--tma_array",
        default=None,
        help="Optional path to TMA_Array.xlsx. Defaults to <base_dir>/TMA_Array.xlsx.",
    )
    parser.add_argument(
        "--tumor_annotations",
        default=None,
        help=(
            "Optional path to Tumor_Annotations.xlsx. Defaults to "
            "<base_dir>/Tumor_Annotations.xlsx."
        ),
    )
    parser.add_argument(
        "--allow_missing_metadata",
        action="store_true",
        help=(
            "Build the image manifest even if TMA_Array.xlsx or "
            "Tumor_Annotations.xlsx is unavailable."
        ),
    )
    parser.add_argument(
        "--manifest_name",
        default="manifest_all4.csv",
        help="Output CSV filename (default: manifest_all4.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_dir = Path(_require_mapping(cfg, "base_dir")).expanduser().resolve()

    require_metadata = not args.allow_missing_metadata
    tma_array = _resolve_input_xlsx(
        args.tma_array,
        base_dir,
        "TMA_Array.xlsx",
        required=require_metadata,
    )
    tumor_annotations = _resolve_input_xlsx(
        args.tumor_annotations,
        base_dir,
        "Tumor_Annotations.xlsx",
        required=require_metadata,
    )

    manifest, summary = build_manifest(cfg, tma_array, tumor_annotations)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = outdir / args.manifest_name
    summary_path = outdir / "manifest_summary.json"

    manifest.to_csv(manifest_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print(f"ALL4 cores: {summary['all4_cores']}")
    print(f"CPID mapped: {summary['cpid_mapped']}")


if __name__ == "__main__":
    main()
