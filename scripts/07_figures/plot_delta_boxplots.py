#!/usr/bin/env python3
"""Create publication-style MAPK3 and SMNDC1 boxplots stratified by paired patient delta."""

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
from scipy.stats import ttest_rel

EXON_COLOR = "tab:blue"
NOEXON_COLOR = "tab:orange"
GROUP_POS = "Exon4 > NoExon4"
GROUP_NEG = "Exon4 ≤ NoExon4"
CONDITIONS = ("Exon4", "NoExon4")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def normalize_condition(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"exon", "exon4", "e"}:
        return "Exon4"
    if text in {"noexon", "noexon4", "noex", "n"}:
        return "NoExon4"
    return None


def normalize_cpid(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        numeric = float(text)
        if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
    except ValueError:
        pass
    return text


def clean_ratio_table(
    frame: pd.DataFrame,
    ratio_column: str,
    source: str,
) -> pd.DataFrame:
    require_columns(frame, ["CPID", "condition", ratio_column], source)
    out = frame.copy()
    out["CPID"] = out["CPID"].map(normalize_cpid)
    out["condition"] = out["condition"].map(normalize_condition)
    out[ratio_column] = pd.to_numeric(out[ratio_column], errors="coerce")
    out = out[
        out["CPID"].notna()
        & out["condition"].isin(CONDITIONS)
        & np.isfinite(out[ratio_column])
        & out[ratio_column].between(0.0, 1.0, inclusive="both")
    ].copy()
    return out


def load_analysis_tables(
    mapk3_global_csv: Path,
    smndc1_global_csv: Path,
    registered_csv: Path,
) -> dict[tuple[str, str], tuple[pd.DataFrame, str]]:
    mapk3_global = clean_ratio_table(
        pd.read_csv(mapk3_global_csv, low_memory=False),
        "ratio_M_cyto",
        "MAPK3 Global",
    )
    smndc1_global = clean_ratio_table(
        pd.read_csv(smndc1_global_csv, low_memory=False),
        "ratio_S_nuc",
        "SMNDC1 Global",
    )
    registered = pd.read_csv(registered_csv, low_memory=False)
    registered_m = clean_ratio_table(registered, "ratio_M_cyto", "MAPK3 Registered")
    registered_s = clean_ratio_table(registered, "ratio_S_nuc", "SMNDC1 Registered")

    require_columns(
        registered_m,
        ["core_norm", "mapk3_cell_id"],
        "MAPK3 Registered",
    )
    registered_m["mapk3_cell_id"] = pd.to_numeric(
        registered_m["mapk3_cell_id"], errors="coerce"
    ).astype("Int64")
    registered_m = registered_m.dropna(subset=["mapk3_cell_id"]).drop_duplicates(
        ["condition", "core_norm", "mapk3_cell_id"], keep="first"
    )

    return {
        ("MAPK3", "Global"): (mapk3_global, "ratio_M_cyto"),
        ("MAPK3", "Registered"): (registered_m, "ratio_M_cyto"),
        ("SMNDC1", "Global"): (smndc1_global, "ratio_S_nuc"),
        ("SMNDC1", "Registered"): (registered_s, "ratio_S_nuc"),
    }


def patient_paired_table(
    units: pd.DataFrame,
    ratio_column: str,
    summary: str,
) -> pd.DataFrame:
    grouped = units.groupby(["CPID", "condition"], observed=True)[ratio_column]
    if summary == "mean":
        values = grouped.mean()
    elif summary == "median":
        values = grouped.median()
    else:
        raise ValueError(f"Unsupported patient summary: {summary}")

    wide = values.unstack("condition")
    if not set(CONDITIONS).issubset(wide.columns):
        return pd.DataFrame(
            columns=["CPID", "Exon4", "NoExon4", "delta_exon_minus_noexon", "dominance_group"]
        )

    wide = wide.dropna(subset=list(CONDITIONS)).copy()
    wide["delta_exon_minus_noexon"] = wide["Exon4"] - wide["NoExon4"]
    wide["dominance_group"] = np.where(
        wide["delta_exon_minus_noexon"] > 0,
        GROUP_POS,
        GROUP_NEG,
    )
    return wide.reset_index()


def paired_test(patient: pd.DataFrame) -> dict[str, float | int]:
    if len(patient) < 2:
        return {
            "n_paired_patients": int(len(patient)),
            "paired_t_statistic": np.nan,
            "paired_t_p_value": np.nan,
            "mean_exon4": np.nan,
            "mean_noexon4": np.nan,
            "mean_delta": np.nan,
            "sd_delta": np.nan,
            "cohen_dz": np.nan,
        }

    exon = patient["Exon4"].to_numpy(dtype=float)
    noex = patient["NoExon4"].to_numpy(dtype=float)
    result = ttest_rel(exon, noex, nan_policy="omit")
    delta = exon - noex
    sd_delta = float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan
    cohen_dz = float(np.mean(delta) / sd_delta) if np.isfinite(sd_delta) and sd_delta > 0 else np.nan
    return {
        "n_paired_patients": int(len(patient)),
        "paired_t_statistic": float(result.statistic),
        "paired_t_p_value": float(result.pvalue),
        "mean_exon4": float(np.mean(exon)),
        "mean_noexon4": float(np.mean(noex)),
        "mean_delta": float(np.mean(delta)),
        "sd_delta": sd_delta,
        "cohen_dz": cohen_dz,
    }


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def styled_boxplot(
    ax: plt.Axes,
    values_pos: np.ndarray,
    values_neg: np.ndarray,
    ylabel: str,
    title: str,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    arrays = [
        np.asarray(values_pos, dtype=float),
        np.asarray(values_neg, dtype=float),
    ]
    arrays = [x[np.isfinite(x)] for x in arrays]

    box = ax.boxplot(
        arrays,
        tick_labels=[GROUP_POS, GROUP_NEG],
        widths=0.62,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 5.5,
        },
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"linewidth": 1.4},
        whiskerprops={"linewidth": 1.3},
        capprops={"linewidth": 1.3},
        flierprops={"marker": "o", "markersize": 2.0, "alpha": 0.12},
    )

    for patch, color in zip(box["boxes"], (EXON_COLOR, NOEXON_COLOR)):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.24)

    for position, (values, color) in enumerate(
        zip(arrays, (EXON_COLOR, NOEXON_COLOR)), start=1
    ):
        if values.size == 0:
            continue
        if values.size > max_points:
            indices = rng.choice(values.size, size=max_points, replace=False)
            display = values[indices]
        else:
            display = values
        x = position + rng.normal(0.0, 0.045, size=display.size)
        ax.scatter(x, display, s=9, alpha=0.18, linewidths=0, color=color)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.8)
    ax.tick_params(axis="x", labelrotation=10)


def plot_patient_delta(
    ax: plt.Axes,
    patient: pd.DataFrame,
    protein: str,
    population: str,
    stats: dict[str, float | int],
    rng: np.random.Generator,
) -> None:
    positive = patient.loc[
        patient["dominance_group"] == GROUP_POS, "delta_exon_minus_noexon"
    ].to_numpy(dtype=float)
    nonpositive = patient.loc[
        patient["dominance_group"] == GROUP_NEG, "delta_exon_minus_noexon"
    ].to_numpy(dtype=float)

    styled_boxplot(
        ax,
        positive,
        nonpositive,
        ylabel="Δ ratio = Exon4 − NoExon4",
        title=f"{protein} — {population}",
        rng=rng,
        max_points=2500,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    p = float(stats["paired_t_p_value"])
    ax.text(
        0.02,
        0.98,
        f"Paired t-test Exon4 vs NoExon4\np = {format_p(p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    ax.text(
        0.98,
        0.98,
        f"patients: n={len(positive)} | n={len(nonpositive)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )


def plot_unit_ratio(
    ax: plt.Axes,
    units: pd.DataFrame,
    ratio_column: str,
    patient: pd.DataFrame,
    protein: str,
    population: str,
    rng: np.random.Generator,
    max_points: int,
) -> None:
    groups = patient[["CPID", "dominance_group"]].drop_duplicates("CPID")
    data = units.merge(groups, on="CPID", how="inner", validate="many_to_one")

    positive = data.loc[data["dominance_group"] == GROUP_POS, ratio_column].to_numpy(dtype=float)
    nonpositive = data.loc[data["dominance_group"] == GROUP_NEG, ratio_column].to_numpy(dtype=float)

    styled_boxplot(
        ax,
        positive,
        nonpositive,
        ylabel="Protein-positive area fraction",
        title=f"{protein} — {population}",
        rng=rng,
        max_points=max_points,
    )
    ax.set_ylim(-0.025, 1.025)

    n_pat_pos = int((patient["dominance_group"] == GROUP_POS).sum())
    n_pat_neg = int((patient["dominance_group"] == GROUP_NEG).sum())
    ax.text(
        0.98,
        0.98,
        f"units: n={len(positive):,} | n={len(nonpositive):,}\n"
        f"patients: n={n_pat_pos} | n={n_pat_neg}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )


def save_individual_figure(
    plot_function,
    out_path: Path,
    *args,
    **kwargs,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    plot_function(ax, *args, **kwargs)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MAPK3/SMNDC1 Global and Registered boxplots stratified by "
            "the sign of the paired patient Exon4-NoExon4 difference."
        )
    )
    parser.add_argument("--mapk3_global_csv", required=True, type=Path)
    parser.add_argument("--smndc1_global_csv", required=True, type=Path)
    parser.add_argument("--registered_csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--patient_summary",
        choices=("mean", "median"),
        default="mean",
        help="Statistic used to obtain one Exon4 and one NoExon4 ratio per CPID.",
    )
    parser.add_argument(
        "--max_unit_points_per_group",
        type=int,
        default=6000,
        help="Maximum number of jittered unit-level points drawn per box.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_unit_points_per_group < 1:
        raise ValueError("--max_unit_points_per_group must be at least 1.")

    outdir = args.outdir
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_analysis_tables(
        args.mapk3_global_csv,
        args.smndc1_global_csv,
        args.registered_csv,
    )
    order = [
        ("MAPK3", "Global"),
        ("MAPK3", "Registered"),
        ("SMNDC1", "Global"),
        ("SMNDC1", "Registered"),
    ]

    patient_tables: dict[tuple[str, str], pd.DataFrame] = {}
    stats_rows: list[dict[str, object]] = []
    patient_rows: list[pd.DataFrame] = []

    for protein, population in order:
        units, ratio_column = datasets[(protein, population)]
        patient = patient_paired_table(units, ratio_column, args.patient_summary)
        if patient.empty:
            raise ValueError(
                f"No paired Exon4/NoExon4 patients were available for {protein} {population}."
            )
        patient_tables[(protein, population)] = patient

        stats = paired_test(patient)
        stats_rows.append(
            {
                "protein": protein,
                "population": population,
                "patient_summary": args.patient_summary,
                **stats,
                "n_exon_dominant": int((patient["dominance_group"] == GROUP_POS).sum()),
                "n_noexon_dominant_or_equal": int((patient["dominance_group"] == GROUP_NEG).sum()),
            }
        )
        annotated = patient.copy()
        annotated.insert(0, "population", population)
        annotated.insert(0, "protein", protein)
        patient_rows.append(annotated)

    patient_all = pd.concat(patient_rows, ignore_index=True)
    stats_table = pd.DataFrame(stats_rows)
    patient_all.to_csv(tables_dir / "patient_paired_ratios_and_delta.csv", index=False)
    stats_table.to_csv(tables_dir / "paired_exon_noexon_tests.csv", index=False)

    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    for ax, (protein, population), stats_row in zip(
        axes.ravel(), order, stats_rows
    ):
        plot_patient_delta(
            ax,
            patient_tables[(protein, population)],
            protein,
            population,
            stats_row,
            rng,
        )
    fig.suptitle("Patient-level Exon4 − NoExon4 ratio differences", fontsize=14)
    fig.savefig(figures_dir / "patient_delta_boxplots_4panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for protein, population in order:
        row = next(
            item for item in stats_rows
            if item["protein"] == protein and item["population"] == population
        )
        save_individual_figure(
            plot_patient_delta,
            figures_dir / f"patient_delta_{protein.lower()}_{population.lower()}.png",
            patient_tables[(protein, population)],
            protein,
            population,
            row,
            np.random.default_rng(args.seed),
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    for ax, (protein, population) in zip(axes.ravel(), order):
        units, ratio_column = datasets[(protein, population)]
        plot_unit_ratio(
            ax,
            units,
            ratio_column,
            patient_tables[(protein, population)],
            protein,
            population,
            rng,
            args.max_unit_points_per_group,
        )
    fig.suptitle("Protein ratios grouped by patient Exon4/NoExon4 dominance", fontsize=14)
    fig.savefig(
        figures_dir / "unit_ratio_by_patient_delta_sign_4panel.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    for protein, population in order:
        units, ratio_column = datasets[(protein, population)]
        save_individual_figure(
            plot_unit_ratio,
            figures_dir / f"unit_ratio_{protein.lower()}_{population.lower()}.png",
            units,
            ratio_column,
            patient_tables[(protein, population)],
            protein,
            population,
            np.random.default_rng(args.seed),
            args.max_unit_points_per_group,
        )

    run_summary = {
        "patient_summary": args.patient_summary,
        "group_definition": {
            GROUP_POS: "patient-level Exon4 ratio greater than NoExon4 ratio",
            GROUP_NEG: "patient-level NoExon4 ratio greater than or equal to Exon4 ratio",
        },
        "inference": (
            "Paired Student t-tests compare Exon4 and NoExon4 patient summaries. "
            "No inferential test is performed between groups defined by the sign of the same delta."
        ),
        "outputs": {
            "patient_table": str(tables_dir / "patient_paired_ratios_and_delta.csv"),
            "paired_tests": str(tables_dir / "paired_exon_noexon_tests.csv"),
            "patient_figure": str(figures_dir / "patient_delta_boxplots_4panel.png"),
            "unit_figure": str(figures_dir / "unit_ratio_by_patient_delta_sign_4panel.png"),
        },
    }
    with (outdir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
