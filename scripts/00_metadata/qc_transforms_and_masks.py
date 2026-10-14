#!/usr/bin/env python3
"""Quality-control registration transforms and analysis masks.

The script audits the canonical core manifest together with one or more
MAPK3-to-SMNDC1 transform tables. It reports:

- existence and size of manifest image files;
- existence of the expected Cellpose/protein masks derived from each channel;
- transform-table structure and per-core coverage;
- registration status and Dice distributions;
- existence of MAPK3 and SMNDC1 masks referenced by transform tables.

No image or mask files are modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from tma_if.naming import normalize_core as _normalize_core_shared
except Exception:
    _normalize_core_shared = None


MANIFEST_DATASETS = (
    "mapk3_exon",
    "mapk3_noexon",
    "smndc1_exon",
    "smndc1_noexon",
)
CHANNEL_MASK_SUFFIX = {
    "B": "_nuc_masks.tif",
    "G": "_prot_mask.tif",
    "R": "_cp_masks.tif",
}
MATRIX_COLUMNS = ("T00", "T01", "T02", "T10", "T11", "T12")


def normalize_core(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if _normalize_core_shared is not None:
        core, _ = _normalize_core_shared(str(value))
        return core

    import re

    match = re.search(r"(\d{1,2})\s*[_-]\s*(\d{1,2})", str(value))
    if not match:
        return None
    return f"{int(match.group(1))}_{int(match.group(2))}"


def normalize_condition(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().lower().replace("_", "").replace("-", "")
    if s in {"exon", "exon4", "e"}:
        return "Exon"
    if s in {"noexon", "noexon4", "noe", "noex"}:
        return "NoExon"
    return None


def load_yaml(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")
    return cfg


def expected_mask_path(image_path: Path, suffix: str) -> Path:
    name = image_path.name
    lower = name.lower()
    if lower.endswith(".tiff"):
        stem = name[:-5]
    elif lower.endswith(".tif"):
        stem = name[:-4]
    else:
        stem = image_path.stem
    return image_path.with_name(stem + suffix)


def file_record(
    *,
    core_norm: Optional[str],
    dataset: str,
    channel: str,
    asset_type: str,
    path: Optional[Path],
) -> dict[str, Any]:
    exists = bool(path and path.is_file())
    return {
        "core_norm": core_norm,
        "dataset": dataset,
        "channel": channel,
        "asset_type": asset_type,
        "path": str(path) if path else None,
        "exists": exists,
        "size_bytes": int(path.stat().st_size) if exists else np.nan,
    }


def audit_manifest_assets(manifest: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        core = normalize_core(row.get("core_norm"))
        for dataset in MANIFEST_DATASETS:
            for channel, mask_suffix in CHANNEL_MASK_SUFFIX.items():
                col = f"{dataset}_{channel}"
                if col not in manifest.columns:
                    continue

                value = row.get(col)
                image_path = None
                if value is not None and not pd.isna(value) and str(value).strip():
                    image_path = Path(str(value)).expanduser()

                records.append(
                    file_record(
                        core_norm=core,
                        dataset=dataset,
                        channel=channel,
                        asset_type="channel_image",
                        path=image_path,
                    )
                )

                mask_path = (
                    expected_mask_path(image_path, mask_suffix)
                    if image_path is not None
                    else None
                )
                records.append(
                    file_record(
                        core_norm=core,
                        dataset=dataset,
                        channel=channel,
                        asset_type={
                            "B": "nuclear_mask",
                            "G": "protein_mask",
                            "R": "cytoplasmic_mask",
                        }[channel],
                        path=mask_path,
                    )
                )

    columns = [
        "core_norm",
        "dataset",
        "channel",
        "asset_type",
        "path",
        "exists",
        "size_bytes",
    ]
    return pd.DataFrame(records, columns=columns)


def find_column_ci(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        col = lookup.get(candidate.lower())
        if col is not None:
            return col
    return None


def parse_transform_table(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(), {
            "source_csv": str(path),
            "n_rows": 0,
            "n_unique_cores": 0,
            "parse_status": "empty",
        }

    core_col = find_column_ci(df, ["core_norm", "core", "core_name", "corename"])
    if core_col is None:
        raise ValueError(f"No core identifier column found in transform table: {path}")

    condition_col = find_column_ci(df, ["condition", "cond"])
    status_col = find_column_ci(df, ["status", "registration_status"])
    dice_ds_col = find_column_ci(
        df,
        ["dice_tumor_ds", "dice_final", "dice", "dice_score"],
    )
    dice_full_col = find_column_ci(df, ["dice_tumor_fullres", "dice_fullres"])

    out = pd.DataFrame(index=df.index)
    out["source_csv"] = str(path)
    out["source_row"] = np.arange(len(df), dtype=int)
    out["core_norm"] = df[core_col].map(normalize_core)
    out["condition"] = (
        df[condition_col].map(normalize_condition)
        if condition_col is not None
        else pd.Series([None] * len(df), index=df.index, dtype="object")
    )
    out["status"] = (
        df[status_col].astype("string")
        if status_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["dice_tumor_ds"] = (
        pd.to_numeric(df[dice_ds_col], errors="coerce")
        if dice_ds_col is not None
        else np.nan
    )
    out["dice_tumor_fullres"] = (
        pd.to_numeric(df[dice_full_col], errors="coerce")
        if dice_full_col is not None
        else np.nan
    )

    source_matrix_lookup = {str(c).upper(): c for c in df.columns}
    for col in MATRIX_COLUMNS:
        source = source_matrix_lookup.get(col.upper())
        out[col] = pd.to_numeric(df[source], errors="coerce") if source else np.nan

    for canonical, candidates in {
        "m_cp_mask": ["m_cp_mask", "mapk3_cp_mask", "moving_mask"],
        "s_cp_mask": ["s_cp_mask", "smndc1_cp_mask", "reference_mask"],
    }.items():
        source = find_column_ci(df, candidates)
        out[canonical] = df[source].astype("string") if source else pd.NA

    out = out[out["core_norm"].notna()].copy()

    matrix_complete = out[list(MATRIX_COLUMNS)].notna().all(axis=1)
    summary = {
        "source_csv": str(path),
        "n_rows": int(len(df)),
        "n_rows_with_valid_core": int(len(out)),
        "n_unique_cores": int(out["core_norm"].nunique()),
        "has_condition": bool(condition_col is not None),
        "has_status": bool(status_col is not None),
        "has_dice_downsample": bool(pd.to_numeric(out["dice_tumor_ds"], errors="coerce").notna().any()),
        "has_dice_fullres": bool(pd.to_numeric(out["dice_tumor_fullres"], errors="coerce").notna().any()),
        "n_complete_matrices": int(matrix_complete.sum()),
        "parse_status": "ok",
    }

    return out.reset_index(drop=True), summary


def audit_transform_mask_paths(transforms: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if transforms.empty:
        return pd.DataFrame(
            columns=["source_csv", "condition", "core_norm", "mask_role", "path", "exists", "size_bytes"]
        )

    for _, row in transforms.iterrows():
        for col, role in (("m_cp_mask", "MAPK3 cytoplasmic mask"), ("s_cp_mask", "SMNDC1 cytoplasmic mask")):
            value = row.get(col)
            path = None
            if value is not None and not pd.isna(value) and str(value).strip():
                path = Path(str(value)).expanduser()
            exists = bool(path and path.is_file())
            records.append(
                {
                    "source_csv": row.get("source_csv"),
                    "condition": row.get("condition"),
                    "core_norm": row.get("core_norm"),
                    "mask_role": role,
                    "path": str(path) if path else None,
                    "exists": exists,
                    "size_bytes": int(path.stat().st_size) if exists else np.nan,
                }
            )
    return pd.DataFrame(records)


def coverage_table(
    transforms: pd.DataFrame,
    manifest_cores: set[str],
) -> pd.DataFrame:
    if transforms.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["source_csv"]
    if transforms["condition"].notna().any():
        group_cols.append("condition")

    for keys, sub in transforms.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keymap = dict(zip(group_cols, keys))
        cores = set(sub["core_norm"].dropna().astype(str))
        overlap = cores & manifest_cores

        status = sub["status"].astype("string")
        dice_ds = pd.to_numeric(sub["dice_tumor_ds"], errors="coerce")
        dice_full = pd.to_numeric(sub["dice_tumor_fullres"], errors="coerce")

        rows.append(
            {
                **keymap,
                "n_rows": int(len(sub)),
                "n_unique_cores": int(len(cores)),
                "manifest_cores_covered": int(len(overlap)),
                "manifest_coverage_fraction": float(len(overlap) / len(manifest_cores)) if manifest_cores else np.nan,
                "n_status_ok": int((status == "ok").sum()),
                "n_status_low_dice": int((status == "low_dice").sum()),
                "n_status_failed_or_error": int((~status.isin(["ok", "low_dice"]) & status.notna()).sum()),
                "dice_ds_median": float(dice_ds.median()) if dice_ds.notna().any() else np.nan,
                "dice_ds_q25": float(dice_ds.quantile(0.25)) if dice_ds.notna().any() else np.nan,
                "dice_ds_q75": float(dice_ds.quantile(0.75)) if dice_ds.notna().any() else np.nan,
                "dice_fullres_median": float(dice_full.median()) if dice_full.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_dice_distribution(
    transforms: pd.DataFrame,
    output: Path,
    metric: str,
    title: str,
) -> None:
    values = pd.to_numeric(transforms.get(metric), errors="coerce").dropna()
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.hist(values, bins=30, edgecolor="black", linewidth=0.5)
    ax.axvline(values.median(), linestyle="--", linewidth=1.4, label=f"Median = {values.median():.3f}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Number of registrations")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def discover_transform_csvs(config_path: Optional[Path]) -> list[Path]:
    if config_path is None:
        return []
    cfg = load_yaml(config_path)
    keywords = [str(x).lower() for x in cfg.get("transform_filename_keywords", [])]
    if not keywords:
        return []

    roots: list[Path] = []
    for key in ("base_dir", "previous_results_dir"):
        value = cfg.get(key)
        if value:
            root = Path(str(value)).expanduser().resolve()
            if root.is_dir():
                roots.append(root)

    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for path in root.rglob("*.csv"):
            if not any(k in path.name.lower() for k in keywords):
                continue
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                found.append(path.resolve())
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit registration transforms and expected masks for the TMA pipeline."
    )
    parser.add_argument("--manifest_csv", required=True, help="Canonical manifest_all4.csv.")
    parser.add_argument(
        "--transforms_csv",
        action="append",
        default=[],
        help=(
            "Transform CSV generated by register_mapk3_smndc1.py. "
            "May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional paths.yaml used only to discover historical transform CSVs "
            "when --transforms_csv is not supplied."
        ),
    )
    parser.add_argument("--outdir", required=True, help="Output QC directory.")
    parser.add_argument(
        "--dice_threshold",
        type=float,
        default=0.60,
        help="Reference threshold used only for QC counts (default: 0.60).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.dice_threshold <= 1.0:
        raise ValueError("--dice_threshold must be between 0 and 1.")

    manifest_path = Path(args.manifest_csv).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if "core_norm" not in manifest.columns:
        raise ValueError("Manifest must contain a 'core_norm' column.")
    manifest["core_norm"] = manifest["core_norm"].map(normalize_core)
    manifest_cores = set(manifest["core_norm"].dropna().astype(str))

    outdir = Path(args.outdir).expanduser().resolve()
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    asset_qc = audit_manifest_assets(manifest)
    asset_qc.to_csv(tables_dir / "manifest_asset_qc.csv", index=False)

    if asset_qc.empty:
        asset_summary = pd.DataFrame()
    else:
        asset_summary = (
            asset_qc.groupby(["dataset", "asset_type"], dropna=False)
            .agg(
                n_expected=("path", "size"),
                n_present=("exists", "sum"),
            )
            .reset_index()
        )
        asset_summary["present_fraction"] = (
            asset_summary["n_present"] / asset_summary["n_expected"].replace(0, np.nan)
        )
    asset_summary.to_csv(tables_dir / "manifest_asset_summary.csv", index=False)

    transform_paths = [Path(p).expanduser().resolve() for p in args.transforms_csv]
    if not transform_paths:
        config_path = Path(args.config).expanduser().resolve() if args.config else None
        transform_paths = discover_transform_csvs(config_path)

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in transform_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            unique_paths.append(path)

    parsed_tables: list[pd.DataFrame] = []
    transform_inventory: list[dict[str, Any]] = []
    transform_errors: list[dict[str, str]] = []

    for path in unique_paths:
        try:
            parsed, summary = parse_transform_table(path)
            transform_inventory.append(summary)
            if not parsed.empty:
                parsed_tables.append(parsed)
        except Exception as exc:
            transform_errors.append(
                {
                    "source_csv": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    inventory_df = pd.DataFrame(transform_inventory)
    inventory_df.to_csv(tables_dir / "transform_inventory.csv", index=False)
    pd.DataFrame(transform_errors).to_csv(tables_dir / "transform_parse_errors.csv", index=False)

    if parsed_tables:
        transforms = pd.concat(parsed_tables, ignore_index=True)
    else:
        transforms = pd.DataFrame(
            columns=[
                "source_csv",
                "source_row",
                "core_norm",
                "condition",
                "status",
                "dice_tumor_ds",
                "dice_tumor_fullres",
                *MATRIX_COLUMNS,
                "m_cp_mask",
                "s_cp_mask",
            ]
        )

    transforms.to_csv(tables_dir / "transforms_per_core.csv", index=False)

    coverage = coverage_table(transforms, manifest_cores)
    if not coverage.empty:
        coverage["n_dice_ds_ge_threshold"] = 0
        for idx, row in coverage.iterrows():
            sub = transforms[transforms["source_csv"] == row["source_csv"]]
            if "condition" in coverage.columns and pd.notna(row.get("condition")):
                sub = sub[sub["condition"] == row["condition"]]
            dice = pd.to_numeric(sub["dice_tumor_ds"], errors="coerce")
            coverage.at[idx, "n_dice_ds_ge_threshold"] = int((dice >= args.dice_threshold).sum())
    coverage.to_csv(tables_dir / "transform_coverage.csv", index=False)

    transform_mask_qc = audit_transform_mask_paths(transforms)
    transform_mask_qc.to_csv(tables_dir / "transform_mask_paths_qc.csv", index=False)

    plot_dice_distribution(
        transforms,
        figures_dir / "dice_tumor_downsample_distribution.png",
        "dice_tumor_ds",
        "MAPK3-to-SMNDC1 registration Dice",
    )
    plot_dice_distribution(
        transforms,
        figures_dir / "dice_tumor_fullres_distribution.png",
        "dice_tumor_fullres",
        "MAPK3-to-SMNDC1 registration Dice (full resolution)",
    )

    status_counts = {}
    if not transforms.empty and transforms["status"].notna().any():
        status_counts = {
            str(k): int(v)
            for k, v in transforms["status"].astype("string").value_counts(dropna=True).items()
        }

    summary = {
        "manifest_csv": str(manifest_path),
        "n_manifest_cores": int(len(manifest_cores)),
        "n_manifest_asset_records": int(len(asset_qc)),
        "n_missing_manifest_assets": int((~asset_qc["exists"]).sum()) if not asset_qc.empty else 0,
        "transform_csvs_requested_or_discovered": [str(p) for p in unique_paths],
        "n_transform_csvs_parsed": int(len(transform_inventory)),
        "n_transform_parse_errors": int(len(transform_errors)),
        "n_transform_rows": int(len(transforms)),
        "registration_status_counts": status_counts,
        "dice_threshold_for_qc_counts": float(args.dice_threshold),
        "n_dice_ds_ge_threshold": int(
            (pd.to_numeric(transforms["dice_tumor_ds"], errors="coerce") >= args.dice_threshold).sum()
        ) if not transforms.empty else 0,
        "n_missing_transform_mask_paths": int((~transform_mask_qc["exists"]).sum())
        if not transform_mask_qc.empty
        else 0,
    }
    (outdir / "qc_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"QC directory: {outdir}")
    print(f"Manifest cores: {summary['n_manifest_cores']}")
    print(f"Missing manifest assets: {summary['n_missing_manifest_assets']}")
    print(f"Transform rows: {summary['n_transform_rows']}")
    print(f"Transform parse errors: {summary['n_transform_parse_errors']}")


if __name__ == "__main__":
    main()
