#!/usr/bin/env python3
"""Plot Exon4/NoExon4 Gaussian distributions and cutoff-side counts.

The script consumes the canonical ratio tables produced by
``quantify_ratios_and_trends.py`` and uses the Gaussian intersections
reported by that stage. No arbitrary cutoff is substituted when an
intersection is absent.

Four datasets are plotted:
    MAPK3 Global
    MAPK3 Registered
    SMNDC1 Global
    SMNDC1 Registered

For each dataset, four publication-ready figures are generated:
    Gaussian curves with count/percentage annotation
    Gaussian curves without annotation
    Gaussian curves + count histogram with annotation
    Gaussian curves + count histogram without annotation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXON_COLOR = "tab:blue"
NOEXON_COLOR = "tab:orange"
RATIO_DOMAIN = (0.0, 1.0)

DATASETS = (
    ("MAPK3", "Global", "ratio_M_cyto", "MAPK3 / cytoplasmic area ratio"),
    ("MAPK3", "Registered", "ratio_M_cyto", "MAPK3 / cytoplasmic area ratio"),
    ("SMNDC1", "Global", "ratio_S_nuc", "SMNDC1 / nuclear area ratio"),
    ("SMNDC1", "Registered", "ratio_S_nuc", "SMNDC1 / nuclear area ratio"),
)


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


def load_ratio_table(path: str, ratio_column: str, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    require_columns(frame, ["condition", ratio_column], source)

    frame = frame.copy()
    frame["condition"] = frame["condition"].map(normalize_condition)
    frame[ratio_column] = pd.to_numeric(frame[ratio_column], errors="coerce")
    frame = frame[np.isfinite(frame[ratio_column])].copy()
    frame = frame[
        frame[ratio_column].between(RATIO_DOMAIN[0], RATIO_DOMAIN[1], inclusive="both")
    ].copy()
    if frame.empty:
        raise ValueError(f"{source} contains no valid {ratio_column} values in [0, 1].")
    return frame


def gaussian_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        raise ValueError("Gaussian parameters must have finite mean and positive SD.")
    z = (x - mean) / sd
    return np.exp(-0.5 * z * z) / (sd * np.sqrt(2.0 * np.pi))


def fit_gaussian(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite observations are required per condition.")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Gaussian SD is zero or non-finite for one condition.")
    return mean, sd


def load_cutoffs(path: str, intersection_choice: str) -> dict[tuple[str, str], float]:
    table = pd.read_csv(path, low_memory=False)
    require_columns(table, ["protein", "population"], "Gaussian intersections table")

    cutoff_column = f"{intersection_choice}_intersection"
    if cutoff_column not in table.columns:
        raise ValueError(
            f"Gaussian intersections table is missing {cutoff_column}. "
            "Use gaussian_intersections.csv produced by quantify_ratios_and_trends.py."
        )

    cutoffs: dict[tuple[str, str], float] = {}
    for _, row in table.iterrows():
        protein = str(row["protein"]).strip().upper()
        population_raw = str(row["population"]).strip().lower()
        population = "Global" if population_raw == "global" else (
            "Registered" if population_raw == "registered" else str(row["population"]).strip()
        )
        cutoff = pd.to_numeric(row[cutoff_column], errors="coerce")
        if np.isfinite(cutoff) and RATIO_DOMAIN[0] <= float(cutoff) <= RATIO_DOMAIN[1]:
            cutoffs[(protein, population)] = float(cutoff)
    return cutoffs


def side_counts(values: np.ndarray, cutoff: float) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        raise ValueError("Cannot calculate cutoff-side counts for an empty condition.")
    left = int(np.sum(values < cutoff))
    right = int(n - left)
    return {
        "n": n,
        "left_n": left,
        "right_n": right,
        "left_pct": 100.0 * left / n,
        "right_pct": 100.0 * right / n,
    }


def annotation_location(protein: str) -> tuple[float, float, str, str]:
    if protein == "MAPK3":
        return 0.02, 0.02, "left", "bottom"
    return 0.02, 0.98, "left", "top"


def add_count_annotation(
    ax: plt.Axes,
    exon_counts: dict[str, float | int],
    noexon_counts: dict[str, float | int],
    protein: str,
) -> None:
    text = (
        f"Exon4:   left {exon_counts['left_pct']:.1f}% (n={exon_counts['left_n']}) | "
        f"right {exon_counts['right_pct']:.1f}% (n={exon_counts['right_n']})\n"
        f"NoExon4: left {noexon_counts['left_pct']:.1f}% (n={noexon_counts['left_n']}) | "
        f"right {noexon_counts['right_pct']:.1f}% (n={noexon_counts['right_n']})"
    )
    x, y, ha, va = annotation_location(protein)
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 5},
    )


def draw_gaussians(
    ax: plt.Axes,
    exon_values: np.ndarray,
    noexon_values: np.ndarray,
    cutoff: float,
    normalize_y: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    mean_e, sd_e = fit_gaussian(exon_values)
    mean_n, sd_n = fit_gaussian(noexon_values)

    x = np.linspace(RATIO_DOMAIN[0], RATIO_DOMAIN[1], 800)
    y_e = gaussian_pdf(x, mean_e, sd_e)
    y_n = gaussian_pdf(x, mean_n, sd_n)

    if normalize_y:
        maximum = max(float(np.max(y_e)), float(np.max(y_n)))
        y_e = y_e / maximum
        y_n = y_n / maximum
        ax.set_ylabel("Normalized Gaussian density")
        ax.set_ylim(0.0, 1.05)
    else:
        ax.set_ylabel("Gaussian density")

    ax.plot(x, y_e, color=EXON_COLOR, linewidth=2.7, label="Exon4")
    ax.plot(x, y_n, color=NOEXON_COLOR, linewidth=2.7, label="NoExon4")
    ax.axvline(cutoff, color="black", linestyle=":", linewidth=1.1, alpha=0.65)
    ax.set_xlim(RATIO_DOMAIN)
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return (mean_e, sd_e), (mean_n, sd_n)


def save_curves_only(
    frame: pd.DataFrame,
    ratio_column: str,
    cutoff: float,
    protein: str,
    population: str,
    xlabel: str,
    output: Path,
    normalize_y: bool,
    with_text: bool,
    dpi: int,
    show_title: bool,
) -> None:
    exon = frame.loc[frame["condition"] == "Exon", ratio_column].to_numpy(float)
    noexon = frame.loc[frame["condition"] == "NoExon", ratio_column].to_numpy(float)
    exon_counts = side_counts(exon, cutoff)
    noexon_counts = side_counts(noexon, cutoff)

    fig, ax = plt.subplots(figsize=(10.6, 5.8), constrained_layout=True)
    draw_gaussians(ax, exon, noexon, cutoff, normalize_y)
    ax.set_xlabel(xlabel)
    if show_title:
        ax.set_title(f"{protein} — {population}")
    if with_text:
        add_count_annotation(ax, exon_counts, noexon_counts, protein)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_curves_histogram(
    frame: pd.DataFrame,
    ratio_column: str,
    cutoff: float,
    protein: str,
    population: str,
    xlabel: str,
    output: Path,
    normalize_y: bool,
    with_text: bool,
    bins: int,
    dpi: int,
    show_title: bool,
) -> None:
    exon = frame.loc[frame["condition"] == "Exon", ratio_column].to_numpy(float)
    noexon = frame.loc[frame["condition"] == "NoExon", ratio_column].to_numpy(float)
    exon_counts = side_counts(exon, cutoff)
    noexon_counts = side_counts(noexon, cutoff)

    fig = plt.figure(figsize=(10.6, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 2])
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0], sharex=ax_top)

    draw_gaussians(ax_top, exon, noexon, cutoff, normalize_y)
    if show_title:
        ax_top.set_title(f"{protein} — {population}")
    if with_text:
        add_count_annotation(ax_top, exon_counts, noexon_counts, protein)

    edges = np.linspace(RATIO_DOMAIN[0], RATIO_DOMAIN[1], bins + 1)
    ax_bottom.hist(
        exon,
        bins=edges,
        alpha=0.28,
        color=EXON_COLOR,
        histtype="stepfilled",
        edgecolor=EXON_COLOR,
        linewidth=1.0,
        label=f"Exon4 (n={len(exon):,})",
    )
    ax_bottom.hist(
        noexon,
        bins=edges,
        alpha=0.28,
        color=NOEXON_COLOR,
        histtype="stepfilled",
        edgecolor=NOEXON_COLOR,
        linewidth=1.0,
        label=f"NoExon4 (n={len(noexon):,})",
    )
    ax_bottom.axvline(cutoff, color="black", linestyle=":", linewidth=1.1, alpha=0.65)
    ax_bottom.set_xlim(RATIO_DOMAIN)
    ax_bottom.set_ylabel("Count")
    ax_bottom.set_xlabel(xlabel)
    ax_bottom.grid(alpha=0.22, linestyle="--", linewidth=0.7)
    ax_bottom.legend(frameon=False, fontsize=9, loc="upper right")

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def override_cutoff(args: argparse.Namespace, protein: str, population: str) -> float | None:
    name = f"cutoff_{protein.lower()}_{population.lower()}"
    value = getattr(args, name)
    if value is None:
        return None
    value = float(value)
    if not RATIO_DOMAIN[0] <= value <= RATIO_DOMAIN[1]:
        raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Gaussian density and count-histogram figures for MAPK3 and "
            "SMNDC1 Global/Registered area-ratio distributions."
        )
    )
    parser.add_argument("--mapk3_global_csv", required=True)
    parser.add_argument("--smndc1_global_csv", required=True)
    parser.add_argument("--registered_csv", required=True)
    parser.add_argument(
        "--intersections_csv",
        required=True,
        help="gaussian_intersections.csv from quantify_ratios_and_trends.py.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--normalize_y",
        action="store_true",
        help="Normalize each pair of Gaussian curves to a maximum density of 1.",
    )
    parser.add_argument(
        "--intersection",
        choices=("first", "last"),
        default="first",
        help="Which reported Gaussian intersection to use when two exist.",
    )
    parser.add_argument("--show_titles", action="store_true")

    parser.add_argument("--cutoff_mapk3_global", type=float)
    parser.add_argument("--cutoff_mapk3_registered", type=float)
    parser.add_argument("--cutoff_smndc1_global", type=float)
    parser.add_argument("--cutoff_smndc1_registered", type=float)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.bins < 5:
        parser.error("--bins must be >= 5.")
    if args.dpi <= 0:
        parser.error("--dpi must be > 0.")

    outdir = Path(args.outdir).resolve()
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    mapk3_global = load_ratio_table(
        args.mapk3_global_csv, "ratio_M_cyto", "MAPK3 Global table"
    )
    smndc1_global = load_ratio_table(
        args.smndc1_global_csv, "ratio_S_nuc", "SMNDC1 Global table"
    )
    registered_mapk3 = load_ratio_table(
        args.registered_csv, "ratio_M_cyto", "Registered MAPK3-SMNDC1 table"
    )
    registered_smndc1 = load_ratio_table(
        args.registered_csv, "ratio_S_nuc", "Registered MAPK3-SMNDC1 table"
    )

    frames = {
        ("MAPK3", "Global"): mapk3_global,
        ("MAPK3", "Registered"): registered_mapk3,
        ("SMNDC1", "Global"): smndc1_global,
        ("SMNDC1", "Registered"): registered_smndc1,
    }

    cutoffs = load_cutoffs(args.intersections_csv, args.intersection)
    summary_rows: list[dict[str, object]] = []

    for protein, population, ratio_column, xlabel in DATASETS:
        frame = frames[(protein, population)]
        cutoff = override_cutoff(args, protein, population)
        cutoff_source = "explicit_override"
        if cutoff is None:
            cutoff = cutoffs.get((protein, population))
            cutoff_source = f"{args.intersection}_gaussian_intersection"
        if cutoff is None:
            raise RuntimeError(
                f"No valid Gaussian intersection is available for {protein} {population}. "
                f"Provide --cutoff_{protein.lower()}_{population.lower()} explicitly if a "
                "scientifically justified cutoff is intended."
            )

        exon = frame.loc[frame["condition"] == "Exon", ratio_column].to_numpy(float)
        noexon = frame.loc[frame["condition"] == "NoExon", ratio_column].to_numpy(float)
        exon_counts = side_counts(exon, cutoff)
        noexon_counts = side_counts(noexon, cutoff)
        mean_e, sd_e = fit_gaussian(exon)
        mean_n, sd_n = fit_gaussian(noexon)

        prefix = f"{protein}_{population}"
        save_curves_only(
            frame,
            ratio_column,
            cutoff,
            protein,
            population,
            xlabel,
            figures_dir / f"{prefix}_curves_TEXT.png",
            args.normalize_y,
            True,
            args.dpi,
            args.show_titles,
        )
        save_curves_only(
            frame,
            ratio_column,
            cutoff,
            protein,
            population,
            xlabel,
            figures_dir / f"{prefix}_curves_NOTEXT.png",
            args.normalize_y,
            False,
            args.dpi,
            args.show_titles,
        )
        save_curves_histogram(
            frame,
            ratio_column,
            cutoff,
            protein,
            population,
            xlabel,
            figures_dir / f"{prefix}_curvesHist_TEXT.png",
            args.normalize_y,
            True,
            args.bins,
            args.dpi,
            args.show_titles,
        )
        save_curves_histogram(
            frame,
            ratio_column,
            cutoff,
            protein,
            population,
            xlabel,
            figures_dir / f"{prefix}_curvesHist_NOTEXT.png",
            args.normalize_y,
            False,
            args.bins,
            args.dpi,
            args.show_titles,
        )

        for condition, counts, mean, sd in (
            ("Exon", exon_counts, mean_e, sd_e),
            ("NoExon", noexon_counts, mean_n, sd_n),
        ):
            summary_rows.append(
                {
                    "protein": protein,
                    "population": population,
                    "condition": condition,
                    "ratio_column": ratio_column,
                    "cutoff": float(cutoff),
                    "cutoff_source": cutoff_source,
                    "gaussian_mean": mean,
                    "gaussian_sd": sd,
                    "n": counts["n"],
                    "left_n": counts["left_n"],
                    "left_pct": counts["left_pct"],
                    "right_n": counts["right_n"],
                    "right_pct": counts["right_pct"],
                }
            )

        print(f"[OK] {protein} {population}: cutoff={cutoff:.6f}")

    pd.DataFrame(summary_rows).to_csv(
        tables_dir / "density_cutoff_counts_summary.csv", index=False
    )
    print(f"[OK] Figures: {figures_dir}")
    print(f"[OK] Summary: {tables_dir / 'density_cutoff_counts_summary.csv'}")


if __name__ == "__main__":
    main()
