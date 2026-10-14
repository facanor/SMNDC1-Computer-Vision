#!/usr/bin/env python3
"""Plot MAPK3-SMNDC1 relationships in the registered TMA population.

The canonical input is ``registered_mapk3_smndc1.csv`` produced by
``quantify_ratios_and_trends.py``. Each row represents a registered SMNDC1
nucleus matched to a MAPK3 cell and contains both ``ratio_M_cyto`` and
``ratio_S_nuc``.

The script reproduces the historical A/B/C MAPK3-density classification in a
self-contained way. For the registered MAPK3 ratios, Gaussian distributions
are fitted separately to Exon4 and NoExon4. A registered unit is assigned to:

A
    Exon4 Gaussian density is greater than NoExon4 density.
B
    NoExon4 Gaussian density is greater than Exon4 density.
C
    The two densities are approximately equal within ``--tol_rel_equal``.

For the historical four-category figures, B and C are collapsed into ``BC``.
The classification is descriptive and is not used as an independent
inferential endpoint.

Cell/nucleus-level plots are descriptive. Correlation p-values are reported at
patient (CPID) level using per-patient medians to avoid treating individual
cells/nuclei as independent patients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

EXON_COLOR = "tab:blue"
NOEXON_COLOR = "tab:orange"
CONDITIONS = ("Exon", "NoExon")
CONDITION_LABELS = {"Exon": "Exon4", "NoExon": "NoExon4"}


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"exon", "exon4", "e"}:
        return "Exon"
    if text in {"noexon", "noexon4", "noex", "n"}:
        return "NoExon"
    raise ValueError(f"Unrecognized condition label: {value!r}")


def finite01(series: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values[(values >= 0.0) & (values <= 1.0)]


def gaussian_parameters(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return mean, np.nan
    return mean, sd


def classify_mapk3_density_groups(
    frame: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    exon = finite01(frame.loc[frame["condition"] == "Exon", "ratio_M_cyto"])
    noexon = finite01(frame.loc[frame["condition"] == "NoExon", "ratio_M_cyto"])
    mu_ex, sd_ex = gaussian_parameters(exon)
    mu_no, sd_no = gaussian_parameters(noexon)

    params = {
        "mu_exon": mu_ex,
        "sd_exon": sd_ex,
        "mu_noexon": mu_no,
        "sd_noexon": sd_no,
        "tol_rel_equal": float(tolerance),
    }

    out = frame.copy()
    if not all(np.isfinite(v) and v > 0 for v in (sd_ex, sd_no)):
        out["group_ABC"] = pd.NA
        out["group2"] = pd.NA
        return out, params

    x = pd.to_numeric(out["ratio_M_cyto"], errors="coerce").to_numpy(dtype=float)
    p_ex = norm.pdf(x, loc=mu_ex, scale=sd_ex)
    p_no = norm.pdf(x, loc=mu_no, scale=sd_no)
    max_pdf = np.maximum(np.maximum(p_ex, p_no), 1e-12)
    near_equal = np.abs(p_ex - p_no) <= tolerance * max_pdf

    groups = np.where(near_equal, "C", np.where(p_ex > p_no, "A", "B"))
    groups[~np.isfinite(x)] = ""
    out["group_ABC"] = pd.Series(groups, index=out.index).replace("", pd.NA)
    out["group2"] = out["group_ABC"].map({"A": "A", "B": "BC", "C": "BC"})
    return out, params


def aggregate_per_mapk3_cell(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "core_norm", "CPID", "mapk3_cell_id"]
    require_columns(frame, keys + ["ratio_M_cyto", "ratio_S_nuc"], "registered table")

    work = frame.copy()
    work["ratio_M_cyto"] = pd.to_numeric(work["ratio_M_cyto"], errors="coerce")
    work["ratio_S_nuc"] = pd.to_numeric(work["ratio_S_nuc"], errors="coerce")

    agg_dict: dict[str, str] = {
        "ratio_M_cyto": "first",
        "ratio_S_nuc": "mean",
    }
    for col in ("group_ABC", "group2"):
        if col in work.columns:
            agg_dict[col] = "first"
    for col in ("OS_days", "OS_censor", "OS_event"):
        if col in work.columns:
            agg_dict[col] = "first"

    cell = work.groupby(keys, dropna=False, as_index=False).agg(agg_dict)

    if {"smndc1_positive_area_px", "area_nucleus_px"}.issubset(work.columns):
        areas = work.copy()
        areas["smndc1_positive_area_px"] = pd.to_numeric(
            areas["smndc1_positive_area_px"], errors="coerce"
        )
        areas["area_nucleus_px"] = pd.to_numeric(areas["area_nucleus_px"], errors="coerce")
        summed = (
            areas.groupby(keys, dropna=False, as_index=False)[
                ["smndc1_positive_area_px", "area_nucleus_px"]
            ]
            .sum(min_count=1)
        )
        summed["ratio_S_nuc_cell"] = (
            summed["smndc1_positive_area_px"] / summed["area_nucleus_px"]
        )
        cell = cell.merge(summed[keys + ["ratio_S_nuc_cell"]], on=keys, how="left")
    else:
        cell["ratio_S_nuc_cell"] = cell["ratio_S_nuc"]

    cell["ratio_S_nuc_cell"] = pd.to_numeric(cell["ratio_S_nuc_cell"], errors="coerce")
    cell = cell[
        np.isfinite(cell["ratio_M_cyto"])
        & np.isfinite(cell["ratio_S_nuc_cell"])
        & cell["ratio_M_cyto"].between(0, 1)
        & cell["ratio_S_nuc_cell"].between(0, 1)
    ].copy()
    return cell


def spearman_safe(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, np.nan, int(len(pair))
    rho, p = spearmanr(pair["x"], pair["y"])
    return float(rho), float(p), int(len(pair))


def make_relationship_summary(cell: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for condition in CONDITIONS:
        sub = cell[cell["condition"] == condition].copy()
        rho_units, _, n_units = spearman_safe(sub["ratio_M_cyto"], sub["ratio_S_nuc_cell"])
        rows.append(
            {
                "analysis_level": "registered_MAPK3_cells_descriptive",
                "condition": CONDITION_LABELS[condition],
                "n": n_units,
                "spearman_rho": rho_units,
                "p_value": np.nan,
                "note": "Descriptive cell-level correlation; no inferential p-value reported.",
            }
        )

        patient = (
            sub.groupby("CPID", dropna=True, as_index=False)[["ratio_M_cyto", "ratio_S_nuc_cell"]]
            .median()
        )
        rho_patient, p_patient, n_patient = spearman_safe(
            patient["ratio_M_cyto"], patient["ratio_S_nuc_cell"]
        )
        rows.append(
            {
                "analysis_level": "CPID_medians",
                "condition": CONDITION_LABELS[condition],
                "n": n_patient,
                "spearman_rho": rho_patient,
                "p_value": p_patient,
                "note": "Inferential Spearman correlation using one median pair per CPID.",
            }
        )

    return pd.DataFrame(rows)


def quantile_trend(
    frame: pd.DataFrame,
    qbins: int,
    min_bin_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    categories = [
        ("A", "Exon"),
        ("A", "NoExon"),
        ("BC", "Exon"),
        ("BC", "NoExon"),
    ]

    for group2, condition in categories:
        sub = frame[(frame["group2"] == group2) & (frame["condition"] == condition)].copy()
        if len(sub) < max(min_bin_n, 3):
            continue
        try:
            sub["qbin"] = pd.qcut(sub["ratio_M_cyto"], q=qbins, duplicates="drop")
        except ValueError:
            continue
        grouped = sub.groupby("qbin", observed=True)
        for bin_index, (_, bin_frame) in enumerate(grouped, start=1):
            if len(bin_frame) < min_bin_n:
                continue
            rows.append(
                {
                    "group2": group2,
                    "condition": condition,
                    "bin": bin_index,
                    "n": int(len(bin_frame)),
                    "mapk3_median": float(bin_frame["ratio_M_cyto"].median()),
                    "smndc1_q25": float(bin_frame["ratio_S_nuc_cell"].quantile(0.25)),
                    "smndc1_median": float(bin_frame["ratio_S_nuc_cell"].median()),
                    "smndc1_q75": float(bin_frame["ratio_S_nuc_cell"].quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def plot_group_boxplots(frame: pd.DataFrame, outpath: Path, seed: int, max_points: int) -> None:
    categories = [
        ("A", "Exon", "A-Exon4", EXON_COLOR),
        ("A", "NoExon", "A-NoExon4", NOEXON_COLOR),
        ("BC", "Exon", "BC-Exon4", EXON_COLOR),
        ("BC", "NoExon", "BC-NoExon4", NOEXON_COLOR),
    ]
    data = [
        finite01(frame.loc[(frame["group2"] == group) & (frame["condition"] == cond), "ratio_S_nuc_cell"])
        for group, cond, _, _ in categories
    ]
    if any(len(values) == 0 for values in data):
        return

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    positions = np.arange(1, 5)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, (_, _, _, color) in zip(bp["boxes"], categories):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
        patch.set_edgecolor("black")

    for pos, values, (_, _, _, color) in zip(positions, data, categories):
        shown = values
        if len(shown) > max_points:
            shown = rng.choice(shown, size=max_points, replace=False)
        x = pos + rng.uniform(-0.14, 0.14, size=len(shown))
        ax.scatter(x, shown, s=8, alpha=0.16, color=color, edgecolors="none", rasterized=True)

    ax.set_xticks(positions, [label for _, _, label, _ in categories])
    ax.set_ylim(0, 1)
    ax.set_ylabel("SMNDC1 nuclear area fraction")
    ax.set_title("Registered SMNDC1 by MAPK3 Gaussian-density group")
    ax.grid(axis="y", alpha=0.22)

    for pos, values in zip(positions, data):
        ax.text(pos, 0.985, f"n={len(values):,}", ha="center", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_trends(
    cell: pd.DataFrame,
    trend: pd.DataFrame,
    outpath: Path,
    zoom_quantile: float,
    max_scatter_points: int,
    seed: int,
) -> tuple[Path, Path]:
    colors = {"Exon": EXON_COLOR, "NoExon": NOEXON_COLOR}
    linestyles = {"A": "-", "BC": "--"}
    markers = {"A": "o", "BC": "^"}
    rng = np.random.default_rng(seed)

    def _draw(frame: pd.DataFrame, trend_frame: pd.DataFrame, path: Path, xmax: float | None) -> None:
        fig, ax = plt.subplots(figsize=(9.2, 6.4))
        raw = frame
        if len(raw) > max_scatter_points:
            raw = raw.iloc[rng.choice(len(raw), size=max_scatter_points, replace=False)]
        ax.scatter(
            raw["ratio_M_cyto"],
            raw["ratio_S_nuc_cell"],
            s=7,
            alpha=0.045,
            color="0.2",
            edgecolors="none",
            rasterized=True,
        )

        for group2 in ("A", "BC"):
            for condition in CONDITIONS:
                sub = trend_frame[
                    (trend_frame["group2"] == group2)
                    & (trend_frame["condition"] == condition)
                ].sort_values("mapk3_median")
                if sub.empty:
                    continue
                label = f"{group2} — {CONDITION_LABELS[condition]}"
                ax.plot(
                    sub["mapk3_median"],
                    sub["smndc1_median"],
                    color=colors[condition],
                    linestyle=linestyles[group2],
                    marker=markers[group2],
                    linewidth=2.0,
                    markersize=5,
                    label=label,
                )
                ax.fill_between(
                    sub["mapk3_median"].to_numpy(dtype=float),
                    sub["smndc1_q25"].to_numpy(dtype=float),
                    sub["smndc1_q75"].to_numpy(dtype=float),
                    color=colors[condition],
                    alpha=0.09,
                )

        ax.set_xlabel("MAPK3 cytoplasmic area fraction")
        ax.set_ylabel("SMNDC1 nuclear area fraction")
        ax.set_ylim(0, 1)
        if xmax is not None:
            ax.set_xlim(0, xmax)
            ax.set_title(f"Registered MAPK3–SMNDC1 relationship (x ≤ p{int(zoom_quantile * 100)})")
        else:
            ax.set_xlim(0, 1)
            ax.set_title("Registered MAPK3–SMNDC1 relationship")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    full_path = outpath
    _draw(cell, trend, full_path, None)

    xmax = float(cell["ratio_M_cyto"].quantile(zoom_quantile))
    zoom_path = outpath.with_name(f"{outpath.stem}_zoom_p{int(zoom_quantile * 100)}{outpath.suffix}")
    zoom_cell = cell[cell["ratio_M_cyto"] <= xmax].copy()
    zoom_trend = trend[trend["mapk3_median"] <= xmax].copy()
    _draw(zoom_cell, zoom_trend, zoom_path, xmax)
    return full_path, zoom_path


def plot_patient_scatter(cell: pd.DataFrame, summary: pd.DataFrame, outpath: Path) -> None:
    patient = (
        cell.groupby(["condition", "CPID"], dropna=True, as_index=False)[
            ["ratio_M_cyto", "ratio_S_nuc_cell"]
        ]
        .median()
    )
    if patient.empty:
        return

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for condition, color in (("Exon", EXON_COLOR), ("NoExon", NOEXON_COLOR)):
        sub = patient[patient["condition"] == condition]
        if sub.empty:
            continue
        row = summary[
            (summary["analysis_level"] == "CPID_medians")
            & (summary["condition"] == CONDITION_LABELS[condition])
        ]
        if row.empty:
            label = CONDITION_LABELS[condition]
        else:
            rho = row.iloc[0]["spearman_rho"]
            p = row.iloc[0]["p_value"]
            label = f"{CONDITION_LABELS[condition]} (ρ={rho:.3f}, p={p:.3g})" if np.isfinite(rho) and np.isfinite(p) else CONDITION_LABELS[condition]
        ax.scatter(
            sub["ratio_M_cyto"],
            sub["ratio_S_nuc_cell"],
            s=34,
            alpha=0.75,
            color=color,
            edgecolors="none",
            label=label,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Patient median MAPK3 cytoplasmic area fraction")
    ax.set_ylabel("Patient median SMNDC1 nuclear area fraction")
    ax.set_title("Patient-level registered MAPK3–SMNDC1 relationship")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate registered MAPK3-SMNDC1 relationship figures and summaries."
    )
    parser.add_argument("--registered_csv", required=True, help="registered_mapk3_smndc1.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--tol_rel_equal",
        type=float,
        default=0.05,
        help="Relative Gaussian-density tolerance used to assign historical group C.",
    )
    parser.add_argument("--trend_qbins", type=int, default=10)
    parser.add_argument("--trend_min_bin_n", type=int, default=30)
    parser.add_argument("--trend_zoom_q", type=float, default=0.95)
    parser.add_argument("--max_points_per_box", type=int, default=2500)
    parser.add_argument("--max_scatter_points", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.tol_rel_equal < 1):
        raise ValueError("--tol_rel_equal must be in [0, 1).")
    if args.trend_qbins < 2:
        raise ValueError("--trend_qbins must be >= 2.")
    if args.trend_min_bin_n < 1:
        raise ValueError("--trend_min_bin_n must be >= 1.")
    if not (0 < args.trend_zoom_q <= 1):
        raise ValueError("--trend_zoom_q must be in (0, 1].")

    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    registered = pd.read_csv(args.registered_csv, low_memory=False)
    require_columns(
        registered,
        ["condition", "core_norm", "CPID", "mapk3_cell_id", "ratio_M_cyto", "ratio_S_nuc"],
        "registered CSV",
    )
    registered = registered.copy()
    registered["condition"] = registered["condition"].map(normalize_condition)
    registered["ratio_M_cyto"] = pd.to_numeric(registered["ratio_M_cyto"], errors="coerce")
    registered["ratio_S_nuc"] = pd.to_numeric(registered["ratio_S_nuc"], errors="coerce")
    registered = registered[
        registered["ratio_M_cyto"].between(0, 1)
        & registered["ratio_S_nuc"].between(0, 1)
    ].copy()
    if registered.empty:
        raise ValueError("No valid registered MAPK3-SMNDC1 rows remain after ratio filtering.")

    classified, gaussian_params = classify_mapk3_density_groups(
        registered,
        tolerance=args.tol_rel_equal,
    )
    classified.to_csv(tables_dir / "registered_mapk3_smndc1_density_groups.csv", index=False)

    with open(tables_dir / "mapk3_registered_gaussian_parameters.json", "w", encoding="utf-8") as handle:
        json.dump(gaussian_params, handle, indent=2)

    cell = aggregate_per_mapk3_cell(classified)
    cell.to_csv(tables_dir / "registered_mapk3_cell_level.csv", index=False)

    summary = make_relationship_summary(cell)
    summary.to_csv(tables_dir / "mapk3_smndc1_relationship_summary.csv", index=False)

    if cell["group2"].notna().any():
        trend = quantile_trend(cell, args.trend_qbins, args.trend_min_bin_n)
        trend.to_csv(tables_dir / "mapk3_smndc1_quantile_trends.csv", index=False)

        plot_group_boxplots(
            cell,
            figures_dir / "smndc1_by_mapk3_density_group.png",
            seed=args.seed,
            max_points=args.max_points_per_box,
        )
        if not trend.empty:
            plot_trends(
                cell,
                trend,
                figures_dir / "mapk3_vs_smndc1_quantile_trends.png",
                zoom_quantile=args.trend_zoom_q,
                max_scatter_points=args.max_scatter_points,
                seed=args.seed,
            )
    else:
        pd.DataFrame().to_csv(tables_dir / "mapk3_smndc1_quantile_trends.csv", index=False)

    plot_patient_scatter(
        cell,
        summary,
        figures_dir / "mapk3_vs_smndc1_cpid_medians.png",
    )

    run_summary = {
        "registered_input_rows": int(len(registered)),
        "registered_mapk3_cells": int(len(cell)),
        "unique_cpid": int(cell["CPID"].dropna().nunique()),
        "gaussian_parameters": gaussian_params,
        "outputs": {
            "classified_registered": str(tables_dir / "registered_mapk3_smndc1_density_groups.csv"),
            "cell_level": str(tables_dir / "registered_mapk3_cell_level.csv"),
            "relationship_summary": str(tables_dir / "mapk3_smndc1_relationship_summary.csv"),
            "quantile_trends": str(tables_dir / "mapk3_smndc1_quantile_trends.csv"),
        },
    }
    with open(tables_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print(f"[OK] Registered rows: {len(registered):,}")
    print(f"[OK] Registered MAPK3 cells: {len(cell):,}")
    print(f"[OK] Output directory: {outdir}")


if __name__ == "__main__":
    main()
