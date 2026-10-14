#!/usr/bin/env python3
"""Quantify MAPK3/SMNDC1 area fractions and generate distribution trends.

This script is the canonical ratio/trend stage of the clean TMA pipeline.
It does not segment nuclei and it does not estimate registration transforms.
Those operations are performed upstream by ``segment_cellpose.py``,
``register_mapk3_smndc1.py``, and ``quantify_smndc1_nuclear.py``.

Definitions
-----------
MAPK3 cytoplasmic ratio (``ratio_M_cyto``)
    MAPK3-positive pixels within a MAPK3 cell/cytoplasmic label divided by
    that label's cytoplasmic area in pixels.

SMNDC1 nuclear ratio (``ratio_S_nuc``)
    SMNDC1-positive pixels within a nucleus divided by that nucleus's area
    in pixels. The canonical upstream column is
    ``smndc1_positive_area_fraction``.

Analysis populations
--------------------
Global
    All eligible MAPK3 cells or standard SMNDC1 nuclei from the cores in the
    four-condition manifest.

Registered
    Matched MAPK3-SMNDC1 observations represented by
    ``nuclei_features_coincident.csv``. MAPK3 measurements are attached by
    ``condition + core_norm + mapk3_cell_id`` so that MAPK3 and SMNDC1 are
    evaluated on the same matched rows.

Gaussian intersections are reported only when they are mathematically
present within the observed ratio domain [0, 1]. No arbitrary cutoff is
substituted when an intersection is absent.
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
from scipy.stats import gaussian_kde, norm

EXON_COLOR = "tab:blue"
NOEXON_COLOR = "tab:orange"
CONDITIONS = ("Exon", "NoExon")
CONDITION_LABELS = {"Exon": "Exon4", "NoExon": "NoExon4"}
RATIO_DOMAIN = (0.0, 1.0)

MANIFEST_CHANNEL_COLUMNS = {
    "Exon": {
        "mapk3_G": "mapk3_exon_G",
        "mapk3_R": "mapk3_exon_R",
    },
    "NoExon": {
        "mapk3_G": "mapk3_noexon_G",
        "mapk3_R": "mapk3_noexon_R",
    },
}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def normalize_core(value: object) -> str:
    """Normalize a core identifier to ``row_column`` form."""
    match = re.search(r"(\d+)\s*[_-]\s*(\d+)", str(value))
    if not match:
        raise ValueError(f"Could not parse core identifier: {value!r}")
    return f"{int(match.group(1))}_{int(match.group(2))}"


def normalize_condition(value: object) -> str:
    """Normalize condition labels while rejecting ambiguous values."""
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


def read_mask_2d(path: Path) -> np.ndarray:
    """Read a 2-D label/binary mask without silently discarding dimensions."""
    array = tiff.imread(path)
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask at {path}; found shape {array.shape}.")
    return array


def finite_ratio(values: Sequence[float]) -> np.ndarray:
    """Return finite ratios in the biologically defined [0, 1] domain."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return values[(values >= RATIO_DOMAIN[0]) & (values <= RATIO_DOMAIN[1])]


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def mask_candidates(channel_image: Path, core: str, kind: str) -> list[Path]:
    """Return candidate masks adjacent to a channel image.

    Historical datasets used several suffix conventions. For MAPK3 protein
    masks, both explicit binary ``prot_mask`` files and historical G-channel
    ``cp_masks`` files are supported. Any nonzero pixel is subsequently
    treated as MAPK3-positive.
    """
    directory = channel_image.parent
    core_dash = core.replace("_", "-")
    keys = (core, core_dash)

    if kind == "cytoplasm":
        suffixes = ("cp_masks", "cyto_masks", "cytoplasm_masks")
    elif kind == "protein":
        suffixes = (
            "prot_mask",
            "prot_masks",
            "protein_mask",
            "protein_masks",
            "cp_masks",  # historical MAPK3 G-channel segmentation
        )
    else:
        raise ValueError(f"Unsupported mask kind: {kind}")

    candidates: list[Path] = []
    for key in keys:
        for suffix in suffixes:
            for extension in ("tif", "tiff"):
                candidates.extend(sorted(directory.glob(f"*{key}*{suffix}*.{extension}")))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def find_mask(channel_image: object, core: str, kind: str) -> Path | None:
    if pd.isna(channel_image) or str(channel_image).strip() == "":
        return None
    image_path = Path(str(channel_image))
    if not image_path.exists():
        return None
    return first_existing(mask_candidates(image_path, core, kind))


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------


def load_manifest(path: str) -> pd.DataFrame:
    manifest = pd.read_csv(path, low_memory=False)
    required = {"core_norm", "CPID"}
    for columns in MANIFEST_CHANNEL_COLUMNS.values():
        required.update(columns.values())
    require_columns(manifest, required, "manifest")

    manifest = manifest.copy()
    manifest["core_norm"] = manifest["core_norm"].map(normalize_core)
    if manifest["core_norm"].duplicated().any():
        duplicates = sorted(
            manifest.loc[manifest["core_norm"].duplicated(keep=False), "core_norm"].unique()
        )
        raise ValueError(f"Manifest contains duplicate core_norm values: {duplicates[:10]}")
    return manifest


def load_smndc1_table(path: str, source_name: str) -> pd.DataFrame:
    """Load canonical SMNDC1 nuclear data and standardize historical aliases."""
    frame = pd.read_csv(path, low_memory=False)

    if "condition" not in frame.columns:
        if "group" in frame.columns:
            frame = frame.rename(columns={"group": "condition"})
        elif "cohort" in frame.columns:
            frame = frame.rename(columns={"cohort": "condition"})
        else:
            raise ValueError(f"{source_name} is missing condition/group/cohort.")

    if "core_norm" not in frame.columns:
        if "core_key" in frame.columns:
            frame = frame.rename(columns={"core_key": "core_norm"})
        elif "core" in frame.columns:
            frame = frame.rename(columns={"core": "core_norm"})
        else:
            raise ValueError(f"{source_name} is missing core_norm/core_key/core.")

    frame = frame.copy()
    frame["core_norm"] = frame["core_norm"].map(normalize_core)
    frame["condition"] = frame["condition"].map(normalize_condition)

    if "smndc1_positive_area_fraction" in frame.columns:
        frame["ratio_S_nuc"] = pd.to_numeric(
            frame["smndc1_positive_area_fraction"], errors="coerce"
        )
    elif "ratio_S_nuc" in frame.columns:
        frame["ratio_S_nuc"] = pd.to_numeric(frame["ratio_S_nuc"], errors="coerce")
    elif {"protein_area_in_nucleus_px", "area_nucleus_px"}.issubset(frame.columns):
        positive_area = pd.to_numeric(frame["protein_area_in_nucleus_px"], errors="coerce")
        nuclear_area = pd.to_numeric(frame["area_nucleus_px"], errors="coerce")
        frame["ratio_S_nuc"] = positive_area / nuclear_area.where(nuclear_area > 0)
    else:
        raise ValueError(
            f"{source_name} does not contain an SMNDC1 positive-area fraction. "
            "Expected smndc1_positive_area_fraction, ratio_S_nuc, or the pair "
            "protein_area_in_nucleus_px + area_nucleus_px. Intensity-based "
            "protein_rate_in_nucleus is intentionally not used as an "
            "area-fraction substitute."
        )

    if "area_nuc_px" not in frame.columns and "area_nucleus_px" in frame.columns:
        frame["area_nuc_px"] = pd.to_numeric(frame["area_nucleus_px"], errors="coerce")
    elif "area_nuc_px" in frame.columns:
        frame["area_nuc_px"] = pd.to_numeric(frame["area_nuc_px"], errors="coerce")

    frame = frame[np.isfinite(frame["ratio_S_nuc"])].copy()
    frame = frame[frame["ratio_S_nuc"].between(0.0, 1.0, inclusive="both")].copy()
    return frame


# -----------------------------------------------------------------------------
# MAPK3 quantification
# -----------------------------------------------------------------------------


def quantify_mapk3_core(
    cytoplasm_mask_path: Path,
    protein_mask_path: Path,
    min_cell_area: int,
) -> pd.DataFrame:
    cytoplasm = read_mask_2d(cytoplasm_mask_path).astype(np.int64, copy=False)
    protein = read_mask_2d(protein_mask_path)
    if cytoplasm.shape != protein.shape:
        raise ValueError(
            f"Mask shape mismatch: cytoplasm {cytoplasm.shape}, protein {protein.shape}."
        )

    protein_positive = protein > 0
    positive_labels = cytoplasm > 0
    if not np.any(positive_labels):
        return pd.DataFrame(
            columns=[
                "cell_id",
                "area_cyto_px",
                "area_protein_in_cell_px",
                "mapk3_positive_area_px",
                "ratio_M_cyto",
            ]
        )

    max_label = int(cytoplasm.max())
    cell_ids = np.arange(1, max_label + 1, dtype=np.int64)

    cytoplasm_area = np.bincount(
        cytoplasm[positive_labels], minlength=max_label + 1
    ).astype(np.int64)
    protein_area = np.bincount(
        cytoplasm[positive_labels & protein_positive], minlength=max_label + 1
    ).astype(np.int64)

    area = cytoplasm_area[1:]
    positive = protein_area[1:]
    keep = area >= min_cell_area

    result = pd.DataFrame(
        {
            "cell_id": cell_ids[keep],
            "area_cyto_px": area[keep],
            "area_protein_in_cell_px": positive[keep],
        }
    )
    result["mapk3_positive_area_px"] = result["area_protein_in_cell_px"]
    result["ratio_M_cyto"] = (
        result["area_protein_in_cell_px"] / result["area_cyto_px"]
    )
    return result


def build_mapk3_global(
    manifest: pd.DataFrame,
    min_cell_area: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    for _, manifest_row in manifest.iterrows():
        core = normalize_core(manifest_row["core_norm"])
        for condition in CONDITIONS:
            columns = MANIFEST_CHANNEL_COLUMNS[condition]
            diagnostic: dict[str, object] = {
                "condition": condition,
                "core_norm": core,
                "status": "not_run",
            }
            try:
                cytoplasm_mask = find_mask(
                    manifest_row[columns["mapk3_R"]], core, "cytoplasm"
                )
                protein_mask = find_mask(
                    manifest_row[columns["mapk3_G"]], core, "protein"
                )
                if cytoplasm_mask is None:
                    raise FileNotFoundError("MAPK3 cytoplasmic mask not found.")
                if protein_mask is None:
                    raise FileNotFoundError("MAPK3 protein-positive mask not found.")

                core_frame = quantify_mapk3_core(
                    cytoplasm_mask,
                    protein_mask,
                    min_cell_area=min_cell_area,
                )
                core_frame.insert(0, "condition", condition)
                core_frame.insert(0, "core_norm", core)
                core_frame.insert(2, "CPID", manifest_row.get("CPID", np.nan))
                for clinical in ("OS_days", "OS_censor", "OS_event"):
                    if clinical in manifest.columns:
                        core_frame[clinical] = manifest_row.get(clinical, np.nan)

                rows.append(core_frame)
                diagnostic.update(
                    {
                        "status": "ok",
                        "n_cells": int(len(core_frame)),
                        "cytoplasm_mask": str(cytoplasm_mask),
                        "protein_mask": str(protein_mask),
                    }
                )
            except Exception as exc:
                diagnostic["status"] = f"error: {type(exc).__name__}: {exc}"
            diagnostics.append(diagnostic)

    global_frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return global_frame, pd.DataFrame(diagnostics)


def build_registered_pairs(
    smndc1_registered: pd.DataFrame,
    mapk3_global: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    require_columns(
        smndc1_registered,
        ["condition", "core_norm", "mapk3_cell_id", "ratio_S_nuc"],
        "SMNDC1 registered table",
    )
    require_columns(
        mapk3_global,
        ["condition", "core_norm", "cell_id", "ratio_M_cyto"],
        "MAPK3 global table",
    )

    registered = smndc1_registered.copy()
    registered["mapk3_cell_id"] = pd.to_numeric(
        registered["mapk3_cell_id"], errors="coerce"
    ).astype("Int64")

    mapk3_lookup = mapk3_global.copy().rename(columns={"cell_id": "mapk3_cell_id"})
    mapk3_lookup["mapk3_cell_id"] = pd.to_numeric(
        mapk3_lookup["mapk3_cell_id"], errors="coerce"
    ).astype("Int64")

    mapk3_columns = [
        "condition",
        "core_norm",
        "mapk3_cell_id",
        "ratio_M_cyto",
        "area_cyto_px",
        "area_protein_in_cell_px",
        "mapk3_positive_area_px",
    ]
    mapk3_lookup = mapk3_lookup[mapk3_columns].drop_duplicates(
        ["condition", "core_norm", "mapk3_cell_id"], keep="first"
    )

    before = len(registered)
    merged = registered.merge(
        mapk3_lookup,
        on=["condition", "core_norm", "mapk3_cell_id"],
        how="left",
        validate="many_to_one",
    )
    missing_mapk3 = int(merged["ratio_M_cyto"].isna().sum())
    merged = merged[
        np.isfinite(pd.to_numeric(merged["ratio_M_cyto"], errors="coerce"))
        & np.isfinite(pd.to_numeric(merged["ratio_S_nuc"], errors="coerce"))
    ].copy()

    stats = {
        "registered_input_rows": int(before),
        "registered_rows_missing_mapk3_ratio": missing_mapk3,
        "registered_complete_pairs": int(len(merged)),
        "registered_unique_mapk3_cells": int(
            merged[["condition", "core_norm", "mapk3_cell_id"]].drop_duplicates().shape[0]
        ),
    }
    return merged, stats


# -----------------------------------------------------------------------------
# Gaussian/KDE summaries
# -----------------------------------------------------------------------------


def gaussian_parameters(values: Sequence[float]) -> tuple[float, float]:
    values = finite_ratio(values)
    if values.size < 2:
        return np.nan, np.nan
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return mean, np.nan
    return mean, sd


def gaussian_intersections(
    mean_exon: float,
    sd_exon: float,
    mean_noexon: float,
    sd_noexon: float,
) -> tuple[list[float], list[float]]:
    """Solve equality of two normal PDFs and split roots by [0, 1]."""
    if not all(np.isfinite(x) for x in (mean_exon, sd_exon, mean_noexon, sd_noexon)):
        return [], []
    if sd_exon <= 0 or sd_noexon <= 0:
        return [], []

    a = 1.0 / (2.0 * sd_noexon**2) - 1.0 / (2.0 * sd_exon**2)
    b = mean_exon / (sd_exon**2) - mean_noexon / (sd_noexon**2)
    c = (
        mean_noexon**2 / (2.0 * sd_noexon**2)
        - mean_exon**2 / (2.0 * sd_exon**2)
        + math.log(sd_noexon / sd_exon)
    )

    roots: list[float] = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            roots = [-c / b]
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0:
            sqrt_discriminant = math.sqrt(discriminant)
            roots = [
                (-b - sqrt_discriminant) / (2.0 * a),
                (-b + sqrt_discriminant) / (2.0 * a),
            ]

    roots = sorted({float(root) for root in roots if np.isfinite(root)})
    inside = [root for root in roots if 0.0 <= root <= 1.0]
    outside = [root for root in roots if not (0.0 <= root <= 1.0)]
    return inside, outside


def dataset_values(frame: pd.DataFrame, ratio_column: str) -> tuple[np.ndarray, np.ndarray]:
    exon = finite_ratio(frame.loc[frame["condition"] == "Exon", ratio_column])
    noexon = finite_ratio(frame.loc[frame["condition"] == "NoExon", ratio_column])
    return exon, noexon


def summarize_distribution(
    frame: pd.DataFrame,
    protein: str,
    population: str,
    ratio_column: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    parameters: dict[str, tuple[float, float]] = {}

    for condition in CONDITIONS:
        values = finite_ratio(frame.loc[frame["condition"] == condition, ratio_column])
        mean, sd = gaussian_parameters(values)
        parameters[condition] = (mean, sd)
        rows.append(
            {
                "protein": protein,
                "population": population,
                "condition": CONDITION_LABELS[condition],
                "ratio_column": ratio_column,
                "n": int(values.size),
                "mean": mean,
                "sd": sd,
                "median": float(np.median(values)) if values.size else np.nan,
                "q25": float(np.quantile(values, 0.25)) if values.size else np.nan,
                "q75": float(np.quantile(values, 0.75)) if values.size else np.nan,
            }
        )

    inside, outside = gaussian_intersections(
        parameters["Exon"][0],
        parameters["Exon"][1],
        parameters["NoExon"][0],
        parameters["NoExon"][1],
    )
    intersection_summary = {
        "protein": protein,
        "population": population,
        "ratio_column": ratio_column,
        "n_intersections_in_0_1": len(inside),
        "first_intersection": inside[0] if inside else np.nan,
        "last_intersection": inside[-1] if inside else np.nan,
        "intersections_in_0_1": json.dumps(inside),
        "intersections_outside_0_1": json.dumps(outside),
    }
    return rows, intersection_summary


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_gaussian(
    frame: pd.DataFrame,
    ratio_column: str,
    title: str,
    xlabel: str,
    output: Path,
    normalize_y: bool,
    intersections: Sequence[float],
) -> None:
    exon, noexon = dataset_values(frame, ratio_column)
    mean_exon, sd_exon = gaussian_parameters(exon)
    mean_noexon, sd_noexon = gaussian_parameters(noexon)
    if not all(np.isfinite(x) for x in (mean_exon, sd_exon, mean_noexon, sd_noexon)):
        raise ValueError(f"Insufficient variation to fit Gaussian curves for {title}.")

    x = np.linspace(0.0, 1.0, 600)
    y_exon = norm.pdf(x, mean_exon, sd_exon)
    y_noexon = norm.pdf(x, mean_noexon, sd_noexon)
    if normalize_y:
        maximum = max(float(y_exon.max()), float(y_noexon.max()))
        if maximum > 0:
            y_exon = y_exon / maximum
            y_noexon = y_noexon / maximum

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.plot(x, y_exon, color=EXON_COLOR, linewidth=2.6, label=f"Exon4 (n={len(exon):,})")
    ax.plot(x, y_noexon, color=NOEXON_COLOR, linewidth=2.6, label=f"NoExon4 (n={len(noexon):,})")
    for cutoff in intersections:
        ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.0, alpha=0.65)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Gaussian density" if normalize_y else "Gaussian density")
    if normalize_y:
        ax.set_ylim(0.0, 1.03)
    ax.set_title(title)
    ax.grid(alpha=0.20, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_kde(
    frame: pd.DataFrame,
    ratio_column: str,
    title: str,
    xlabel: str,
    output: Path,
    normalize_y: bool,
    intersections: Sequence[float],
    bandwidth: str | float,
) -> None:
    exon, noexon = dataset_values(frame, ratio_column)
    if exon.size < 3 or noexon.size < 3:
        raise ValueError(f"Too few observations for KDE: {title}")
    if np.unique(exon).size < 2 or np.unique(noexon).size < 2:
        raise ValueError(f"Insufficient variation for KDE: {title}")

    x = np.linspace(0.0, 1.0, 600)
    kde_exon = gaussian_kde(exon, bw_method=bandwidth)
    kde_noexon = gaussian_kde(noexon, bw_method=bandwidth)
    y_exon = kde_exon(x)
    y_noexon = kde_noexon(x)
    if normalize_y:
        maximum = max(float(y_exon.max()), float(y_noexon.max()))
        if maximum > 0:
            y_exon = y_exon / maximum
            y_noexon = y_noexon / maximum

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.plot(x, y_exon, color=EXON_COLOR, linewidth=2.6, label=f"Exon4 (n={len(exon):,})")
    ax.plot(x, y_noexon, color=NOEXON_COLOR, linewidth=2.6, label=f"NoExon4 (n={len(noexon):,})")
    for cutoff in intersections:
        ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.0, alpha=0.65)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized KDE density" if normalize_y else "KDE density")
    if normalize_y:
        ax.set_ylim(0.0, 1.03)
    ax.set_title(title)
    ax.grid(alpha=0.20, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_two_panel_gaussian(
    mapk3: pd.DataFrame,
    smndc1: pd.DataFrame,
    population: str,
    output: Path,
    normalize_y: bool,
    mapk3_intersections: Sequence[float],
    smndc1_intersections: Sequence[float],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.0), constrained_layout=True)

    specifications = [
        (
            axes[0],
            mapk3,
            "ratio_M_cyto",
            "MAPK3 / cytoplasmic area ratio",
            mapk3_intersections,
            "MAPK3",
        ),
        (
            axes[1],
            smndc1,
            "ratio_S_nuc",
            "SMNDC1 / nuclear area ratio",
            smndc1_intersections,
            "SMNDC1",
        ),
    ]

    for ax, frame, ratio_column, xlabel, intersections, protein in specifications:
        exon, noexon = dataset_values(frame, ratio_column)
        mean_exon, sd_exon = gaussian_parameters(exon)
        mean_noexon, sd_noexon = gaussian_parameters(noexon)
        if not all(np.isfinite(x) for x in (mean_exon, sd_exon, mean_noexon, sd_noexon)):
            ax.text(0.5, 0.5, "Gaussian fit unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue

        x = np.linspace(0.0, 1.0, 600)
        y_exon = norm.pdf(x, mean_exon, sd_exon)
        y_noexon = norm.pdf(x, mean_noexon, sd_noexon)
        if normalize_y:
            maximum = max(float(y_exon.max()), float(y_noexon.max()))
            if maximum > 0:
                y_exon = y_exon / maximum
                y_noexon = y_noexon / maximum

        ax.plot(x, y_exon, color=EXON_COLOR, linewidth=2.5, label=f"Exon4 (n={len(exon):,})")
        ax.plot(x, y_noexon, color=NOEXON_COLOR, linewidth=2.5, label=f"NoExon4 (n={len(noexon):,})")
        for cutoff in intersections:
            ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized Gaussian density" if normalize_y else "Gaussian density")
        if normalize_y:
            ax.set_ylim(0.0, 1.03)
        ax.set_title(f"{protein} — {population}")
        ax.grid(alpha=0.20, linestyle="--", linewidth=0.7)
        ax.legend(frameon=False)

    fig.savefig(output, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify MAPK3 cytoplasmic area fractions, standardize SMNDC1 "
            "nuclear area fractions, construct matched registered pairs, and "
            "generate Exon4/NoExon4 Gaussian trend curves."
        )
    )
    parser.add_argument("--manifest_csv", required=True, help="Four-condition core manifest.")
    parser.add_argument(
        "--smndc1_standard_csv",
        required=True,
        help="nuclei_features_standard.csv from quantify_smndc1_nuclear.py.",
    )
    parser.add_argument(
        "--smndc1_registered_csv",
        required=True,
        help="nuclei_features_coincident.csv from quantify_smndc1_nuclear.py.",
    )
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--min_cell_area",
        type=int,
        default=60,
        help="Minimum MAPK3 cytoplasmic label area in pixels.",
    )
    parser.add_argument(
        "--normalize_y",
        action="store_true",
        help="Normalize each Gaussian/KDE panel to a maximum density of 1.",
    )
    parser.add_argument(
        "--include_kde",
        action="store_true",
        help="Also generate KDE versions of the four individual trend plots.",
    )
    parser.add_argument(
        "--kde_bw",
        default="scott",
        help="scipy gaussian_kde bandwidth: scott, silverman, or a positive float.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.min_cell_area <= 0:
        parser.error("--min_cell_area must be > 0.")

    bandwidth: str | float
    try:
        bandwidth = float(args.kde_bw)
        if bandwidth <= 0:
            parser.error("Numeric --kde_bw must be > 0.")
    except ValueError:
        bandwidth = args.kde_bw
        if bandwidth not in {"scott", "silverman"}:
            parser.error("--kde_bw must be scott, silverman, or a positive float.")

    outdir = Path(args.outdir).resolve()
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    reports_dir = outdir / "reports"
    for directory in (tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest_csv)
    standard_s = load_smndc1_table(args.smndc1_standard_csv, "SMNDC1 standard table")
    registered_s = load_smndc1_table(args.smndc1_registered_csv, "SMNDC1 registered table")

    manifest_cores = set(manifest["core_norm"])
    standard_s = standard_s[standard_s["core_norm"].isin(manifest_cores)].copy()
    registered_s = registered_s[registered_s["core_norm"].isin(manifest_cores)].copy()

    mapk3_global, mapk3_diagnostics = build_mapk3_global(
        manifest,
        min_cell_area=args.min_cell_area,
    )
    if mapk3_global.empty:
        raise RuntimeError("No MAPK3 cells were quantified. Check manifest paths and masks.")

    registered_pairs, registered_stats = build_registered_pairs(
        registered_s,
        mapk3_global,
    )
    if registered_pairs.empty:
        raise RuntimeError(
            "No complete registered MAPK3-SMNDC1 pairs were produced. "
            "Check mapk3_cell_id and mask availability."
        )

    # Canonical aliases expected by downstream analysis scripts.
    standard_s = standard_s.copy()
    registered_s = registered_s.copy()
    if "area_nuc_px" not in standard_s.columns:
        standard_s["area_nuc_px"] = np.nan
    if "area_nuc_px" not in registered_pairs.columns:
        registered_pairs["area_nuc_px"] = np.nan

    mapk3_global.to_csv(tables_dir / "mapk3_global.csv", index=False)
    standard_s.to_csv(tables_dir / "smndc1_global.csv", index=False)
    registered_pairs.to_csv(tables_dir / "registered_mapk3_smndc1.csv", index=False)
    mapk3_diagnostics.to_csv(tables_dir / "mapk3_quantification_qc.csv", index=False)

    datasets = [
        ("MAPK3", "Global", mapk3_global, "ratio_M_cyto", "MAPK3 / cytoplasmic area ratio"),
        ("SMNDC1", "Global", standard_s, "ratio_S_nuc", "SMNDC1 / nuclear area ratio"),
        ("MAPK3", "Registered", registered_pairs, "ratio_M_cyto", "MAPK3 / cytoplasmic area ratio"),
        ("SMNDC1", "Registered", registered_pairs, "ratio_S_nuc", "SMNDC1 / nuclear area ratio"),
    ]

    parameter_rows: list[dict[str, object]] = []
    intersection_rows: list[dict[str, object]] = []
    intersections_by_dataset: dict[tuple[str, str], list[float]] = {}

    for protein, population, frame, ratio_column, xlabel in datasets:
        rows, intersection = summarize_distribution(
            frame,
            protein,
            population,
            ratio_column,
        )
        parameter_rows.extend(rows)
        intersection_rows.append(intersection)
        roots = json.loads(str(intersection["intersections_in_0_1"]))
        intersections_by_dataset[(protein, population)] = roots

        plot_gaussian(
            frame,
            ratio_column,
            title=f"{protein} — {population}",
            xlabel=xlabel,
            output=figures_dir / f"{protein}_{population}_gaussian.png",
            normalize_y=args.normalize_y,
            intersections=roots,
        )

        if args.include_kde:
            plot_kde(
                frame,
                ratio_column,
                title=f"{protein} — {population} (KDE)",
                xlabel=xlabel,
                output=figures_dir / f"{protein}_{population}_kde.png",
                normalize_y=args.normalize_y,
                intersections=roots,
                bandwidth=bandwidth,
            )

    parameter_table = pd.DataFrame(parameter_rows)
    intersection_table = pd.DataFrame(intersection_rows)
    parameter_table.to_csv(tables_dir / "distribution_parameters.csv", index=False)
    intersection_table.to_csv(tables_dir / "gaussian_intersections.csv", index=False)

    plot_two_panel_gaussian(
        mapk3_global,
        standard_s,
        "Global",
        figures_dir / "Global_MAPK3_SMNDC1_gaussian.png",
        args.normalize_y,
        intersections_by_dataset[("MAPK3", "Global")],
        intersections_by_dataset[("SMNDC1", "Global")],
    )
    plot_two_panel_gaussian(
        registered_pairs,
        registered_pairs,
        "Registered",
        figures_dir / "Registered_MAPK3_SMNDC1_gaussian.png",
        args.normalize_y,
        intersections_by_dataset[("MAPK3", "Registered")],
        intersections_by_dataset[("SMNDC1", "Registered")],
    )

    summary = {
        "definitions": {
            "ratio_M_cyto": (
                "MAPK3-positive area within a MAPK3 cell/cytoplasmic label "
                "divided by cytoplasmic area in pixels."
            ),
            "ratio_S_nuc": (
                "SMNDC1-positive area within a nucleus divided by nuclear area "
                "in pixels; derived from smndc1_positive_area_fraction."
            ),
            "global": (
                "All eligible MAPK3 cells or standard SMNDC1 nuclei from cores "
                "included in the four-condition manifest."
            ),
            "registered": (
                "Matched MAPK3-SMNDC1 rows from the coincident SMNDC1 table, "
                "with MAPK3 measurements joined by condition, core, and MAPK3 cell ID."
            ),
        },
        "counts": {
            "manifest_cores": int(manifest["core_norm"].nunique()),
            "mapk3_global_rows": int(len(mapk3_global)),
            "smndc1_global_rows": int(len(standard_s)),
            "registered_pair_rows": int(len(registered_pairs)),
            **registered_stats,
        },
        "inputs": {
            "manifest_csv": str(Path(args.manifest_csv).resolve()),
            "smndc1_standard_csv": str(Path(args.smndc1_standard_csv).resolve()),
            "smndc1_registered_csv": str(Path(args.smndc1_registered_csv).resolve()),
        },
        "settings": {
            "min_cell_area": args.min_cell_area,
            "normalize_y": bool(args.normalize_y),
            "include_kde": bool(args.include_kde),
            "kde_bw": args.kde_bw,
        },
        "note": (
            "Gaussian intersections are reported only when roots fall inside [0, 1]. "
            "No fallback cutoff is inserted when no intersection exists."
        ),
    }
    with open(reports_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"[OK] MAPK3 Global:       {len(mapk3_global):,} cells")
    print(f"[OK] SMNDC1 Global:      {len(standard_s):,} nuclei")
    print(f"[OK] Registered pairs:   {len(registered_pairs):,} rows")
    print(f"[OK] Tables:             {tables_dir}")
    print(f"[OK] Figures:            {figures_dir}")
    print(f"[OK] Report:             {reports_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
