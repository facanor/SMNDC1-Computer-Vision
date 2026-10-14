#!/usr/bin/env python3
"""Analyze nuclear area relative to the cytoplasmic/cell compartment.

The script reproduces the manuscript morphology analysis for two populations:

Global
    SMNDC1-section cells reconstructed from the standard SMNDC1 nuclear table
    and the SMNDC1 cytoplasmic/cell masks.

Registered
    Spatially matched MAPK3-SMNDC1 observations from
    ``registered_mapk3_smndc1.csv``. Nuclear areas are aggregated per matched
    MAPK3 cell and compared with that cell's cytoplasmic area.

For each cell, two descriptive morphology metrics are calculated:

``nuclear_to_cytoplasm``
    Sum of nuclear areas assigned to the cell divided by cytoplasmic/cell area.

``nuclear_fraction_total``
    Sum of nuclear areas divided by nuclear area plus cytoplasmic/cell area.

Unit-level distributions are used for visualization only. Exon4 versus NoExon4
inference is performed on one median value per CPID using paired tests.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.stats import norm, ttest_rel, wilcoxon

EXON_COLOR = "tab:blue"
NOEXON_COLOR = "tab:orange"
CONDITIONS = ("Exon", "NoExon")
CONDITION_LABELS = {"Exon": "Exon4", "NoExon": "NoExon4"}

MANIFEST_COLUMNS = {
    "Exon": {
        "smndc1_B": "smndc1_exon_B",
        "smndc1_R": "smndc1_exon_R",
    },
    "NoExon": {
        "smndc1_B": "smndc1_noexon_B",
        "smndc1_R": "smndc1_noexon_R",
    },
}


def normalize_core(value: object) -> str:
    match = re.search(r"(\d+)\s*[_-]\s*(\d+)", str(value))
    if not match:
        raise ValueError(f"Could not parse core identifier: {value!r}")
    return f"{int(match.group(1))}_{int(match.group(2))}"


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"exon", "exon4", "e"}:
        return "Exon"
    if text in {"noexon", "noexon4", "noex", "n"}:
        return "NoExon"
    raise ValueError(f"Unrecognized condition label: {value!r}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def read_2d(path: Path) -> np.ndarray:
    array = np.asarray(tiff.imread(path))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D TIFF at {path}; found shape {array.shape}.")
    return array


def first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_mask_for_channel(channel_image: object, core: str, kind: str) -> Path | None:
    if pd.isna(channel_image) or not str(channel_image).strip():
        return None
    image_path = Path(str(channel_image))
    if not image_path.exists():
        return None

    directory = image_path.parent
    core_dash = core.replace("_", "-")
    if kind == "nucleus":
        suffixes = ("nuc_masks", "nucleus_masks", "nuclei_masks")
    elif kind == "cytoplasm":
        suffixes = ("cp_masks", "cyto_masks", "cytoplasm_masks")
    else:
        raise ValueError(f"Unsupported mask kind: {kind}")

    candidates: list[Path] = []
    for key in (core, core_dash):
        for suffix in suffixes:
            for extension in ("tif", "tiff"):
                candidates.extend(sorted(directory.glob(f"*{key}*{suffix}*.{extension}")))

    # Prefer masks sharing the channel image stem when several historical files exist.
    stem = image_path.stem.replace(".ome", "")
    preferred = [path for path in candidates if stem in path.stem]
    return first_existing(preferred or candidates)


def load_manifest(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"core_norm", "CPID"}
    for columns in MANIFEST_COLUMNS.values():
        required.update(columns.values())
    require_columns(frame, required, "manifest")

    frame = frame.copy()
    frame["core_norm"] = frame["core_norm"].map(normalize_core)
    if frame["core_norm"].duplicated().any():
        duplicated = sorted(
            frame.loc[frame["core_norm"].duplicated(keep=False), "core_norm"].unique()
        )
        raise ValueError(f"Manifest contains duplicate core_norm values: {duplicated[:10]}")
    return frame


def load_smndc1_global(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    aliases = {}
    if "core_norm" not in frame.columns and "core_key" in frame.columns:
        aliases["core_key"] = "core_norm"
    if "area_nuc_px" not in frame.columns and "area_nucleus_px" in frame.columns:
        aliases["area_nucleus_px"] = "area_nuc_px"
    if aliases:
        frame = frame.rename(columns=aliases)

    require_columns(
        frame,
        ["core_norm", "condition", "CPID", "nucleus_id", "area_nuc_px"],
        "SMNDC1 global table",
    )
    frame = frame.copy()
    frame["core_norm"] = frame["core_norm"].map(normalize_core)
    frame["condition"] = frame["condition"].map(normalize_condition)
    frame["nucleus_id"] = pd.to_numeric(frame["nucleus_id"], errors="coerce")
    frame["area_nuc_px"] = pd.to_numeric(frame["area_nuc_px"], errors="coerce")
    frame = frame[
        frame["nucleus_id"].notna()
        & np.isfinite(frame["area_nuc_px"])
        & (frame["area_nuc_px"] > 0)
    ].copy()
    frame["nucleus_id"] = frame["nucleus_id"].astype(np.int64)
    return frame


def load_registered(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    aliases = {}
    if "core_norm" not in frame.columns and "core_key" in frame.columns:
        aliases["core_key"] = "core_norm"
    if "area_nuc_px" not in frame.columns and "area_nucleus_px" in frame.columns:
        aliases["area_nucleus_px"] = "area_nuc_px"
    if "mapk3_cell_id" not in frame.columns and "cell_id_M" in frame.columns:
        aliases["cell_id_M"] = "mapk3_cell_id"
    if aliases:
        frame = frame.rename(columns=aliases)

    require_columns(
        frame,
        [
            "core_norm",
            "condition",
            "CPID",
            "mapk3_cell_id",
            "area_cyto_px",
            "area_nuc_px",
        ],
        "registered MAPK3-SMNDC1 table",
    )
    frame = frame.copy()
    frame["core_norm"] = frame["core_norm"].map(normalize_core)
    frame["condition"] = frame["condition"].map(normalize_condition)
    for column in ("mapk3_cell_id", "area_cyto_px", "area_nuc_px"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["mapk3_cell_id"].notna()
        & np.isfinite(frame["area_cyto_px"])
        & np.isfinite(frame["area_nuc_px"])
        & (frame["area_cyto_px"] > 0)
        & (frame["area_nuc_px"] > 0)
    ].copy()
    frame["mapk3_cell_id"] = frame["mapk3_cell_id"].astype(np.int64)
    return frame


def majority_nucleus_to_cell(nucleus_labels: np.ndarray, cell_labels: np.ndarray) -> pd.DataFrame:
    if nucleus_labels.shape != cell_labels.shape:
        raise ValueError(
            f"Nuclear and cytoplasmic masks have different shapes: "
            f"{nucleus_labels.shape} vs {cell_labels.shape}."
        )

    mask = nucleus_labels > 0
    if not np.any(mask):
        return pd.DataFrame(columns=["nucleus_id", "cell_id", "overlap_px"])

    nucleus_id = nucleus_labels[mask].astype(np.int64)
    cell_id = cell_labels[mask].astype(np.int64)
    valid = cell_id > 0
    if not np.any(valid):
        return pd.DataFrame(columns=["nucleus_id", "cell_id", "overlap_px"])

    pairs = np.stack([nucleus_id[valid], cell_id[valid]], axis=1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    table = pd.DataFrame(
        {
            "nucleus_id": unique_pairs[:, 0].astype(np.int64),
            "cell_id": unique_pairs[:, 1].astype(np.int64),
            "overlap_px": counts.astype(np.int64),
        }
    )
    table = table.sort_values(
        ["nucleus_id", "overlap_px", "cell_id"],
        ascending=[True, False, True],
    )
    return table.drop_duplicates("nucleus_id", keep="first").reset_index(drop=True)


def quantify_global(
    manifest: pd.DataFrame,
    smndc1_global: pd.DataFrame,
    min_cytoplasm_area: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    for condition in CONDITIONS:
        columns = MANIFEST_COLUMNS[condition]
        for _, manifest_row in manifest.iterrows():
            core = manifest_row["core_norm"]
            diagnostic: dict[str, object] = {
                "population": "Global",
                "condition": condition,
                "core_norm": core,
                "status": "not_run",
            }
            try:
                nuclei = smndc1_global[
                    (smndc1_global["condition"] == condition)
                    & (smndc1_global["core_norm"] == core)
                ].copy()
                if nuclei.empty:
                    raise ValueError("No standard SMNDC1 nuclei found for this core/condition.")

                nucleus_mask_path = find_mask_for_channel(
                    manifest_row[columns["smndc1_B"]], core, "nucleus"
                )
                cytoplasm_mask_path = find_mask_for_channel(
                    manifest_row[columns["smndc1_R"]], core, "cytoplasm"
                )
                if nucleus_mask_path is None:
                    raise FileNotFoundError("SMNDC1 nuclear label mask not found.")
                if cytoplasm_mask_path is None:
                    raise FileNotFoundError("SMNDC1 cytoplasmic/cell label mask not found.")

                nucleus_labels = read_2d(nucleus_mask_path).astype(np.int64)
                cell_labels = read_2d(cytoplasm_mask_path).astype(np.int64)
                mapping = majority_nucleus_to_cell(nucleus_labels, cell_labels)
                if mapping.empty:
                    raise ValueError("No nuclei overlapped a positive cytoplasmic/cell label.")

                max_cell = int(cell_labels.max())
                cell_area = np.bincount(
                    cell_labels.ravel().astype(np.int64), minlength=max_cell + 1
                ).astype(np.int64)

                matched = nuclei.merge(mapping, on="nucleus_id", how="inner", validate="one_to_one")
                if matched.empty:
                    raise ValueError("No standard nuclei could be assigned to cells.")

                grouped = (
                    matched.groupby("cell_id", as_index=False)
                    .agg(
                        area_nuc_px_sum=("area_nuc_px", "sum"),
                        n_nuclei=("nucleus_id", "nunique"),
                    )
                )
                grouped["area_cyto_px"] = grouped["cell_id"].map(
                    lambda cell_id: int(cell_area[int(cell_id)])
                    if 0 <= int(cell_id) < len(cell_area)
                    else 0
                )
                grouped = grouped[
                    (grouped["area_cyto_px"] >= min_cytoplasm_area)
                    & (grouped["area_nuc_px_sum"] > 0)
                ].copy()
                if grouped.empty:
                    raise ValueError("No cells remained after area filtering.")

                grouped["population"] = "Global"
                grouped["condition"] = condition
                grouped["core_norm"] = core
                grouped["CPID"] = str(manifest_row["CPID"])
                grouped["nuclear_to_cytoplasm"] = (
                    grouped["area_nuc_px_sum"] / grouped["area_cyto_px"]
                )
                grouped["nuclear_fraction_total"] = grouped["area_nuc_px_sum"] / (
                    grouped["area_nuc_px_sum"] + grouped["area_cyto_px"]
                )

                rows.append(
                    grouped[
                        [
                            "population",
                            "condition",
                            "core_norm",
                            "CPID",
                            "cell_id",
                            "n_nuclei",
                            "area_cyto_px",
                            "area_nuc_px_sum",
                            "nuclear_to_cytoplasm",
                            "nuclear_fraction_total",
                        ]
                    ]
                )
                diagnostic.update(
                    {
                        "status": "ok",
                        "n_input_nuclei": int(len(nuclei)),
                        "n_assigned_nuclei": int(len(matched)),
                        "n_cells": int(len(grouped)),
                        "nucleus_mask": str(nucleus_mask_path),
                        "cytoplasm_mask": str(cytoplasm_mask_path),
                    }
                )
            except Exception as exc:
                diagnostic.update({"status": "error", "error": str(exc)})
            diagnostics.append(diagnostic)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return result, pd.DataFrame(diagnostics)


def quantify_registered(registered: pd.DataFrame, min_cytoplasm_area: int) -> pd.DataFrame:
    work = registered[registered["area_cyto_px"] >= min_cytoplasm_area].copy()
    if work.empty:
        return pd.DataFrame()

    # The same MAPK3 cell may contain more than one matched SMNDC1 nucleus.
    # Nuclear area is therefore summed, while cytoplasmic area is constant per cell.
    result = (
        work.groupby(
            ["CPID", "condition", "core_norm", "mapk3_cell_id"],
            as_index=False,
            dropna=False,
        )
        .agg(
            area_cyto_px=("area_cyto_px", "median"),
            area_nuc_px_sum=("area_nuc_px", "sum"),
            n_nuclei=("area_nuc_px", "count"),
        )
    )
    result = result[
        np.isfinite(result["area_cyto_px"])
        & np.isfinite(result["area_nuc_px_sum"])
        & (result["area_cyto_px"] > 0)
        & (result["area_nuc_px_sum"] > 0)
    ].copy()
    result["population"] = "Registered"
    result["cell_id"] = result["mapk3_cell_id"].astype(np.int64)
    result["nuclear_to_cytoplasm"] = result["area_nuc_px_sum"] / result["area_cyto_px"]
    result["nuclear_fraction_total"] = result["area_nuc_px_sum"] / (
        result["area_nuc_px_sum"] + result["area_cyto_px"]
    )
    return result[
        [
            "population",
            "condition",
            "core_norm",
            "CPID",
            "cell_id",
            "n_nuclei",
            "area_cyto_px",
            "area_nuc_px_sum",
            "nuclear_to_cytoplasm",
            "nuclear_fraction_total",
        ]
    ].copy()


def gaussian_intersections(mu1: float, sd1: float, mu2: float, sd2: float) -> list[float]:
    if not all(np.isfinite(value) for value in (mu1, sd1, mu2, sd2)):
        return []
    if sd1 <= 0 or sd2 <= 0:
        return []

    a = 1.0 / (2.0 * sd2**2) - 1.0 / (2.0 * sd1**2)
    b = mu1 / sd1**2 - mu2 / sd2**2
    c = mu2**2 / (2.0 * sd2**2) - mu1**2 / (2.0 * sd1**2) + math.log(sd2 / sd1)

    roots: list[float] = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            roots = [-c / b]
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0:
            root = math.sqrt(discriminant)
            roots = [(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)]
    return sorted(float(value) for value in roots if np.isfinite(value))


def finite_values(frame: pd.DataFrame, metric: str, condition: str) -> np.ndarray:
    values = pd.to_numeric(
        frame.loc[frame["condition"] == condition, metric], errors="coerce"
    ).to_numpy(dtype=float)
    return values[np.isfinite(values) & (values >= 0)]


def draw_gaussian(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
) -> dict[str, object]:
    exon = finite_values(frame, metric, "Exon")
    noexon = finite_values(frame, metric, "NoExon")
    if len(exon) < 2 or len(noexon) < 2:
        ax.set_title(f"{title} (insufficient data)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized density (0–1)")
        return {
            "n_exon": int(len(exon)),
            "n_noexon": int(len(noexon)),
            "intersection": np.nan,
        }

    mu_ex = float(np.mean(exon))
    sd_ex = float(np.std(exon, ddof=1))
    mu_no = float(np.mean(noexon))
    sd_no = float(np.std(noexon, ddof=1))
    sd_ex = max(sd_ex, 1e-12)
    sd_no = max(sd_no, 1e-12)

    maximum = float(np.nanmax(np.concatenate([exon, noexon])))
    x_max = max(maximum * 1.05, mu_ex + 4.0 * sd_ex, mu_no + 4.0 * sd_no, 1e-6)
    x = np.linspace(0.0, x_max, 512)
    y_ex = norm.pdf(x, mu_ex, sd_ex)
    y_no = norm.pdf(x, mu_no, sd_no)
    ymax = max(float(np.max(y_ex)), float(np.max(y_no)), 1e-12)
    y_ex /= ymax
    y_no /= ymax

    ax.plot(x, y_ex, lw=2.4, color=EXON_COLOR, label=f"Exon4 (n={len(exon)})")
    ax.plot(x, y_no, lw=2.4, color=NOEXON_COLOR, label=f"NoExon4 (n={len(noexon)})")

    intersections = [value for value in gaussian_intersections(mu_ex, sd_ex, mu_no, sd_no) if 0 <= value <= x_max]
    intersection = intersections[0] if intersections else np.nan
    if np.isfinite(intersection):
        ax.axvline(intersection, color="black", ls="--", lw=0.8, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized density (0–1)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, ls="--", lw=0.7)

    return {
        "n_exon": int(len(exon)),
        "n_noexon": int(len(noexon)),
        "mean_exon": mu_ex,
        "sd_exon": sd_ex,
        "mean_noexon": mu_no,
        "sd_noexon": sd_no,
        "intersection": float(intersection) if np.isfinite(intersection) else np.nan,
    }


def make_figures(global_frame: pd.DataFrame, registered_frame: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    metrics = [
        (
            "nuclear_to_cytoplasm",
            "Nuclear-to-cytoplasmic area ratio",
            r"$A_{nuc}/A_{cyto}$",
        ),
        (
            "nuclear_fraction_total",
            "Nuclear fraction of total area",
            r"$A_{nuc}/(A_{nuc}+A_{cyto})$",
        ),
    ]
    populations = [("Global", global_frame), ("Registered", registered_frame)]
    rows: list[dict[str, object]] = []

    for population, frame in populations:
        for metric, title_metric, xlabel in metrics:
            fig, ax = plt.subplots(figsize=(8.8, 4.2))
            stats = draw_gaussian(
                ax,
                frame,
                metric,
                f"{population} — {title_metric}",
                xlabel,
            )
            ax.legend(frameon=True, framealpha=0.95)
            fig.tight_layout()
            fig.savefig(outdir / f"{population}_{metric}_gaussian.png", dpi=300)
            plt.close(fig)
            rows.append({"population": population, "metric": metric, **stats})

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))
    for row_index, (population, frame) in enumerate(populations):
        for column_index, (metric, title_metric, xlabel) in enumerate(metrics):
            draw_gaussian(
                axes[row_index, column_index],
                frame,
                metric,
                f"{population} — {title_metric}",
                xlabel,
            )
    axes[0, 0].legend(frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outdir / "nuclear_vs_cytoplasm_gaussian_panel.png", dpi=300)
    plt.close(fig)

    return pd.DataFrame(rows)


def paired_tests(frame: pd.DataFrame, population: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ("nuclear_to_cytoplasm", "nuclear_fraction_total")
    cpid_rows: list[pd.DataFrame] = []
    test_rows: list[dict[str, object]] = []

    for metric in metrics:
        summary = (
            frame.groupby(["CPID", "condition"], as_index=False)[metric]
            .median()
            .rename(columns={metric: "median_value"})
        )
        summary["population"] = population
        summary["metric"] = metric
        cpid_rows.append(summary)

        wide = summary.pivot(index="CPID", columns="condition", values="median_value")
        if not {"Exon", "NoExon"}.issubset(wide.columns):
            paired = pd.DataFrame(columns=["Exon", "NoExon"])
        else:
            paired = wide[["Exon", "NoExon"]].dropna()

        n = int(len(paired))
        if n >= 2:
            exon = paired["Exon"].to_numpy(dtype=float)
            noexon = paired["NoExon"].to_numpy(dtype=float)
            delta = exon - noexon
            if np.allclose(delta, 0.0, atol=1e-12):
                wilcoxon_p = 1.0
            else:
                try:
                    wilcoxon_p = float(wilcoxon(exon, noexon, alternative="two-sided").pvalue)
                except ValueError:
                    wilcoxon_p = np.nan
            ttest_p = float(ttest_rel(exon, noexon, nan_policy="omit").pvalue)
            median_delta = float(np.median(delta))
            mean_delta = float(np.mean(delta))
        else:
            wilcoxon_p = np.nan
            ttest_p = np.nan
            median_delta = np.nan
            mean_delta = np.nan

        test_rows.append(
            {
                "population": population,
                "metric": metric,
                "n_paired_cpid": n,
                "median_delta_exon_minus_noexon": median_delta,
                "mean_delta_exon_minus_noexon": mean_delta,
                "wilcoxon_p": wilcoxon_p,
                "paired_ttest_p": ttest_p,
            }
        )

    cpid_summary = pd.concat(cpid_rows, ignore_index=True) if cpid_rows else pd.DataFrame()
    return cpid_summary, pd.DataFrame(test_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze nuclear area relative to cytoplasmic/cell area in Global and Registered populations."
    )
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--smndc1_global_csv", required=True, help="Canonical smndc1_global.csv.")
    parser.add_argument(
        "--registered_csv",
        required=True,
        help="Canonical registered_mapk3_smndc1.csv.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--min_cytoplasm_area",
        type=int,
        default=60,
        help="Minimum cytoplasmic/cell mask area in pixels (default: 60).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_cytoplasm_area < 1:
        raise ValueError("--min_cytoplasm_area must be >= 1.")

    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    reports_dir = outdir / "reports"
    for directory in (tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest_csv)
    smndc1_global = load_smndc1_global(args.smndc1_global_csv)
    registered = load_registered(args.registered_csv)

    manifest_cores = set(manifest["core_norm"])
    smndc1_global = smndc1_global[smndc1_global["core_norm"].isin(manifest_cores)].copy()
    registered = registered[registered["core_norm"].isin(manifest_cores)].copy()

    global_frame, diagnostics = quantify_global(
        manifest,
        smndc1_global,
        min_cytoplasm_area=args.min_cytoplasm_area,
    )
    registered_frame = quantify_registered(
        registered,
        min_cytoplasm_area=args.min_cytoplasm_area,
    )

    if global_frame.empty:
        raise RuntimeError("No Global nuclear/cytoplasmic measurements were produced.")
    if registered_frame.empty:
        raise RuntimeError("No Registered nuclear/cytoplasmic measurements were produced.")

    global_frame.to_csv(tables_dir / "nuclear_cytoplasm_global.csv", index=False)
    registered_frame.to_csv(tables_dir / "nuclear_cytoplasm_registered.csv", index=False)
    diagnostics.to_csv(tables_dir / "nuclear_cytoplasm_qc.csv", index=False)

    global_cpid, global_tests = paired_tests(global_frame, "Global")
    registered_cpid, registered_tests = paired_tests(registered_frame, "Registered")
    cpid_summary = pd.concat([global_cpid, registered_cpid], ignore_index=True)
    tests = pd.concat([global_tests, registered_tests], ignore_index=True)
    cpid_summary.to_csv(tables_dir / "nuclear_cytoplasm_cpid_summary.csv", index=False)
    tests.to_csv(tables_dir / "nuclear_cytoplasm_paired_tests.csv", index=False)

    gaussian_summary = make_figures(global_frame, registered_frame, figures_dir)
    gaussian_summary.to_csv(tables_dir / "nuclear_cytoplasm_gaussian_parameters.csv", index=False)

    summary = {
        "inputs": {
            "manifest_csv": str(Path(args.manifest_csv).resolve()),
            "smndc1_global_csv": str(Path(args.smndc1_global_csv).resolve()),
            "registered_csv": str(Path(args.registered_csv).resolve()),
        },
        "min_cytoplasm_area": int(args.min_cytoplasm_area),
        "global": {
            "n_cells": int(len(global_frame)),
            "n_cores": int(global_frame["core_norm"].nunique()),
            "n_cpid": int(global_frame["CPID"].nunique()),
        },
        "registered": {
            "n_cells": int(len(registered_frame)),
            "n_cores": int(registered_frame["core_norm"].nunique()),
            "n_cpid": int(registered_frame["CPID"].nunique()),
        },
    }
    with open(reports_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[OK] {tables_dir / 'nuclear_cytoplasm_global.csv'}")
    print(f"[OK] {tables_dir / 'nuclear_cytoplasm_registered.csv'}")
    print(f"[OK] {tables_dir / 'nuclear_cytoplasm_paired_tests.csv'}")
    print(f"[OK] {figures_dir / 'nuclear_vs_cytoplasm_gaussian_panel.png'}")


if __name__ == "__main__":
    main()
