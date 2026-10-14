#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import matplotlib.pyplot as plt
import pandas as pd
import yaml

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tma_if.excel_report import (  # noqa: E402
    build_core_to_cpid_map,
    find_excels,
    tumor_annotations_overview,
    xlsx_overview,
)
from tma_if.scan import (  # noqa: E402
    find_first_existing_dir,
    find_mask_like_assets,
    scan_group_channels,
)
from tma_if.transform_report import (  # noqa: E402
    find_transform_csvs,
    summarize_transform_csv,
)

GROUPS = (
    ("mapk3_exon", "MAPK3", "Exon4"),
    ("mapk3_noexon", "MAPK3", "NoExon4"),
    ("smndc1_exon", "SMNDC1", "Exon4"),
    ("smndc1_noexon", "SMNDC1", "NoExon4"),
)
CHANNELS = ("B", "G", "R")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def require_mapping(obj: Any, key: str) -> Dict[str, Any]:
    value = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(value, dict):
        raise ValueError(f"Configuration key '{key}' must be a mapping.")
    return value


def require_list(obj: Dict[str, Any], key: str) -> List[str]:
    value = obj.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Configuration key '{key}' must be a non-empty list.")
    return [str(x) for x in value]


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")
    if not cfg.get("base_dir"):
        raise ValueError("Configuration must define 'base_dir'.")

    groups = require_mapping(cfg, "groups")
    for protein in ("mapk3", "smndc1"):
        pcfg = require_mapping(groups, protein)
        require_list(pcfg, "exon_candidates")
        require_list(pcfg, "noexon_candidates")

    channels = require_mapping(cfg, "channels")
    for channel in CHANNELS:
        require_list(channels, channel)
    return cfg


def channel_core_set(channel_scan: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for channel in CHANNELS:
        out.update(channel_scan.get(channel, {}).get("cores", {}).keys())
    return out


def triple_core_set(channel_scan: Dict[str, Any]) -> Set[str]:
    sets = [set(channel_scan.get(ch, {}).get("cores", {}).keys()) for ch in CHANNELS]
    if any(not s for s in sets):
        return set()
    return set.intersection(*sets)


def safe_resolve(path_value: Optional[str], *, relative_to: Path) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = relative_to / p
    return p.resolve()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_barplot(path: Path, title: str, labels: Iterable[str], values: Iterable[int]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(list(labels), list(values))
    ax.set_title(title)
    ax.set_ylabel("Number of cores")
    ax.tick_params(axis="x", labelrotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_group_roots(base_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    groups = cfg["groups"]
    roots: Dict[str, Optional[Path]] = {}
    roots["mapk3_exon"] = find_first_existing_dir(base_dir, groups["mapk3"]["exon_candidates"])
    roots["mapk3_noexon"] = find_first_existing_dir(base_dir, groups["mapk3"]["noexon_candidates"])
    roots["smndc1_exon"] = find_first_existing_dir(base_dir, groups["smndc1"]["exon_candidates"])
    roots["smndc1_noexon"] = find_first_existing_dir(base_dir, groups["smndc1"]["noexon_candidates"])
    return roots


def scan_groups(
    roots: Dict[str, Optional[Path]],
    channels_cfg: Dict[str, List[str]],
    img_exts: List[str],
    mask_dir_keywords: List[str],
    mask_file_keywords: List[str],
) -> Dict[str, Dict[str, Any]]:
    scans: Dict[str, Dict[str, Any]] = {}
    for key, _, _ in GROUPS:
        root = roots[key]
        if root is None:
            scans[key] = {
                "root": None,
                "channels": {},
                "cores_any": set(),
                "cores_triple": set(),
                "mask_assets": [],
            }
            continue

        ch_scan = scan_group_channels(root, channels_cfg, img_exts)
        mask_assets = find_mask_like_assets(
            root,
            img_exts,
            mask_dir_keywords,
            mask_file_keywords,
        )
        scans[key] = {
            "root": root,
            "channels": ch_scan,
            "cores_any": channel_core_set(ch_scan),
            "cores_triple": triple_core_set(ch_scan),
            "mask_assets": mask_assets,
        }
    return scans


def build_group_inventory(scans: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, protein, condition in GROUPS:
        scan = scans[key]
        row: Dict[str, Any] = {
            "dataset": key,
            "protein": protein,
            "condition": condition,
            "root": str(scan["root"]) if scan["root"] else None,
            "n_cores_any_channel": len(scan["cores_any"]),
            "n_cores_complete_BGR": len(scan["cores_triple"]),
            "n_mask_like_assets": len(scan["mask_assets"]),
        }
        for channel in CHANNELS:
            info = scan.get("channels", {}).get(channel, {})
            row[f"channel_{channel}_dir"] = info.get("dir")
            row[f"channel_{channel}_files"] = int(info.get("n_files", 0))
            row[f"channel_{channel}_cores"] = len(info.get("cores", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def build_core_inventory(
    scans: Dict[str, Dict[str, Any]],
    core_to_cpid: Dict[str, str],
) -> pd.DataFrame:
    core_sets = {key: scans[key]["cores_any"] for key, _, _ in GROUPS}
    triple_sets = {key: scans[key]["cores_triple"] for key, _, _ in GROUPS}
    all_cores = sorted(set().union(*core_sets.values())) if core_sets else []

    rows: List[Dict[str, Any]] = []
    for core in all_cores:
        m_ex = core in core_sets["mapk3_exon"]
        m_no = core in core_sets["mapk3_noexon"]
        s_ex = core in core_sets["smndc1_exon"]
        s_no = core in core_sets["smndc1_noexon"]
        rows.append(
            {
                "core_norm": core,
                "CPID": core_to_cpid.get(core),
                "in_mapk3_exon": m_ex,
                "in_mapk3_noexon": m_no,
                "in_smndc1_exon": s_ex,
                "in_smndc1_noexon": s_no,
                "in_mapk3_pair": m_ex and m_no,
                "in_smndc1_pair": s_ex and s_no,
                "in_all_4": m_ex and m_no and s_ex and s_no,
                "mapk3_exon_has_BGR": core in triple_sets["mapk3_exon"],
                "mapk3_noexon_has_BGR": core in triple_sets["mapk3_noexon"],
                "smndc1_exon_has_BGR": core in triple_sets["smndc1_exon"],
                "smndc1_noexon_has_BGR": core in triple_sets["smndc1_noexon"],
            }
        )
    return pd.DataFrame(rows)


def excel_report_table(paths: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        report = xlsx_overview(path)
        if "error" in report:
            rows.append(
                {
                    "path": str(path),
                    "sheet": None,
                    "rows": None,
                    "columns": None,
                    "error": report["error"],
                }
            )
            continue
        for sheet in report.get("sheets", []):
            shape = report.get("sheet_shapes", {}).get(sheet, {})
            rows.append(
                {
                    "path": str(path),
                    "sheet": sheet,
                    "rows": shape.get("rows"),
                    "columns": shape.get("cols"),
                    "error": None,
                }
            )
    return pd.DataFrame(rows, columns=["path", "sheet", "rows", "columns", "error"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory a paired MAPK3/SMNDC1 TMA dataset and report dataset completeness."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to paths.yaml.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--timestamped",
        action="store_true",
        help="Create a timestamped diagnostic subdirectory inside --outdir.",
    )
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip diagnostic PNG figures.",
    )
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    cfg = load_config(cfg_path)
    config_dir = cfg_path.parent

    base_dir = safe_resolve(str(cfg["base_dir"]), relative_to=config_dir)
    assert base_dir is not None
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base_dir does not exist or is not a directory: {base_dir}")

    previous_results = safe_resolve(cfg.get("previous_results_dir"), relative_to=config_dir)
    if previous_results is not None and not previous_results.exists():
        previous_results = None

    outdir = args.outdir.expanduser().resolve()
    if args.timestamped:
        outdir = outdir / f"dataset_inventory_{now_tag()}"
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    img_exts = [str(x) for x in cfg.get("img_exts", [".tif", ".tiff"])]
    mask_dir_keywords = [str(x) for x in cfg.get("mask_dir_keywords", ["mask", "seg", "segmentation"])]
    mask_file_keywords = [str(x) for x in cfg.get("mask_file_keywords", ["mask", "seg"])]
    transform_keywords = [str(x) for x in cfg.get("transform_filename_keywords", ["transform", "registration"])]
    xlsx_candidates = [str(x) for x in cfg.get("xlsx_candidates", ["TMA_Array.xlsx", "Tumor_Annotations.xlsx"])]

    roots = resolve_group_roots(base_dir, cfg)
    scans = scan_groups(
        roots,
        cfg["channels"],
        img_exts,
        mask_dir_keywords,
        mask_file_keywords,
    )

    excels = find_excels(base_dir, xlsx_candidates)
    core_to_cpid: Dict[str, str] = {}
    tumor_annotations_report: Dict[str, Any] = {}
    for path in excels:
        if path.name == "TMA_Array.xlsx":
            core_to_cpid, _ = build_core_to_cpid_map(path)
        elif path.name == "Tumor_Annotations.xlsx":
            tumor_annotations_report = tumor_annotations_overview(path)

    group_inventory = build_group_inventory(scans)
    core_inventory = build_core_inventory(scans, core_to_cpid)
    group_inventory.to_csv(tables_dir / "group_inventory.csv", index=False)
    core_inventory.to_csv(tables_dir / "core_inventory.csv", index=False)

    excel_inventory = excel_report_table(excels)
    excel_inventory.to_csv(tables_dir / "excel_inventory.csv", index=False)
    write_json(outdir / "tumor_annotations_overview.json", tumor_annotations_report)

    transform_search_root = previous_results if previous_results is not None else base_dir
    transform_paths = find_transform_csvs(base_dir, transform_search_root, transform_keywords)
    transform_rows = [summarize_transform_csv(path) for path in transform_paths]
    transform_columns = [
        "path",
        "n_rows",
        "n_unique_cores",
        "has_dice",
        "dice_stats",
        "columns",
        "error",
    ]
    transform_inventory = pd.DataFrame(transform_rows)
    if transform_inventory.empty:
        transform_inventory = pd.DataFrame(columns=transform_columns)
    transform_inventory.to_csv(tables_dir / "transform_inventory.csv", index=False)

    core_sets = {key: scans[key]["cores_any"] for key, _, _ in GROUPS}
    m_pair = core_sets["mapk3_exon"] & core_sets["mapk3_noexon"]
    s_pair = core_sets["smndc1_exon"] & core_sets["smndc1_noexon"]
    all_four = set.intersection(*core_sets.values()) if core_sets else set()
    union_all = set.union(*core_sets.values()) if core_sets else set()

    counts = {
        "base_dir": str(base_dir),
        "previous_results_dir": str(previous_results) if previous_results else None,
        "resolved_group_roots": {
            key: str(path) if path is not None else None for key, path in roots.items()
        },
        "cores_by_group": {key: len(core_sets[key]) for key, _, _ in GROUPS},
        "complete_BGR_by_group": {
            key: len(scans[key]["cores_triple"]) for key, _, _ in GROUPS
        },
        "mask_like_assets_by_group": {
            key: len(scans[key]["mask_assets"]) for key, _, _ in GROUPS
        },
        "paired_core_counts": {
            "mapk3_exon_noexon": len(m_pair),
            "smndc1_exon_noexon": len(s_pair),
            "all_four_datasets": len(all_four),
            "union_all_datasets": len(union_all),
        },
        "n_cores_mapped_to_cpid": int(core_inventory["CPID"].notna().sum())
        if "CPID" in core_inventory.columns
        else 0,
        "n_transform_csvs_found": len(transform_paths),
        "n_expected_excel_files_found": len(excels),
    }
    write_json(outdir / "counts.json", counts)

    if not args.no_figures:
        save_barplot(
            figures_dir / "cores_by_dataset.png",
            "Cores detected by dataset",
            ["MAPK3 Exon4", "MAPK3 NoExon4", "SMNDC1 Exon4", "SMNDC1 NoExon4"],
            [len(core_sets[key]) for key, _, _ in GROUPS],
        )
        save_barplot(
            figures_dir / "paired_core_counts.png",
            "Paired core availability",
            ["MAPK3 pair", "SMNDC1 pair", "All four"],
            [len(m_pair), len(s_pair), len(all_four)],
        )

    summary_lines = [
        "TMA IF dataset inventory",
        "========================",
        f"Base directory: {base_dir}",
        f"Previous results directory: {previous_results if previous_results else 'not configured/found'}",
        "",
        "Resolved datasets",
    ]
    for key, protein, condition in GROUPS:
        summary_lines.append(f"- {protein} {condition}: {roots[key] if roots[key] else 'NOT FOUND'}")

    summary_lines.extend(
        [
            "",
            "Core counts",
            f"- MAPK3 Exon4: {len(core_sets['mapk3_exon'])}",
            f"- MAPK3 NoExon4: {len(core_sets['mapk3_noexon'])}",
            f"- SMNDC1 Exon4: {len(core_sets['smndc1_exon'])}",
            f"- SMNDC1 NoExon4: {len(core_sets['smndc1_noexon'])}",
            f"- MAPK3 paired: {len(m_pair)}",
            f"- SMNDC1 paired: {len(s_pair)}",
            f"- Present in all four datasets: {len(all_four)}",
            f"- Mapped to CPID: {counts['n_cores_mapped_to_cpid']}",
            "",
            "Metadata and registration",
            f"- Expected Excel files found: {len(excels)}",
            f"- Transform-like CSV files found: {len(transform_paths)}",
            "",
            "Outputs",
            f"- {tables_dir / 'group_inventory.csv'}",
            f"- {tables_dir / 'core_inventory.csv'}",
            f"- {tables_dir / 'excel_inventory.csv'}",
            f"- {tables_dir / 'transform_inventory.csv'}",
            f"- {outdir / 'counts.json'}",
            f"- {outdir / 'tumor_annotations_overview.json'}",
        ]
    )
    if not args.no_figures:
        summary_lines.append(f"- {figures_dir}")

    summary = "\n".join(summary_lines) + "\n"
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
