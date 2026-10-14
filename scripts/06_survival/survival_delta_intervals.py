#!/usr/bin/env python3
"""Patient-level survival analysis of paired Exon4-NoExon4 differences across size intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from statsmodels.nonparametric.smoothers_lowess import lowess

EXON_COLOR = "tab:blue"
NOEX_COLOR = "tab:orange"


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def normalize_condition(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"exon", "exon4", "e"}:
        return "Exon4"
    if text in {"noexon", "noexon4", "noe", "n"}:
        return "NoExon4"
    return None


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def fit_trend(x: np.ndarray, y: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 5:
        raise ValueError("At least five valid observations per condition are required for LOWESS.")
    fitted = lowess(y, x, frac=frac, it=0, return_sorted=True)
    order = np.argsort(fitted[:, 0])
    return fitted[order, 0], fitted[order, 1]


def find_trend_intersections(
    exon_x: np.ndarray,
    exon_y: np.ndarray,
    noex_x: np.ndarray,
    noex_y: np.ndarray,
    overlap_quantiles: tuple[float, float],
    grid_size: int = 2000,
) -> list[float]:
    qlo, qhi = overlap_quantiles
    exon_lo, exon_hi = np.quantile(exon_x, [qlo, qhi])
    noex_lo, noex_hi = np.quantile(noex_x, [qlo, qhi])
    xmin = max(float(exon_lo), float(noex_lo))
    xmax = min(float(exon_hi), float(noex_hi))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return []

    grid = np.linspace(xmin, xmax, grid_size)
    exon_interp = np.interp(grid, exon_x, exon_y)
    noex_interp = np.interp(grid, noex_x, noex_y)
    difference = exon_interp - noex_interp

    roots: list[float] = []
    exact = np.where(np.isclose(difference, 0.0, atol=1e-12))[0]
    for index in exact:
        roots.append(float(grid[index]))

    for index in np.where(difference[:-1] * difference[1:] < 0)[0]:
        x1, x2 = grid[index], grid[index + 1]
        y1, y2 = difference[index], difference[index + 1]
        root = x1 - y1 * (x2 - x1) / (y2 - y1)
        roots.append(float(root))

    if not roots:
        return []
    roots = sorted(roots)
    deduplicated = [roots[0]]
    for root in roots[1:]:
        if abs(root - deduplicated[-1]) > 1e-6:
            deduplicated.append(root)
    return deduplicated


def save_trend_figure(
    units: pd.DataFrame,
    exon_x: np.ndarray,
    exon_y: np.ndarray,
    noex_x: np.ndarray,
    noex_y: np.ndarray,
    x0: float,
    out_path: Path,
    protein: str,
    population: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    sample = units
    if len(sample) > 25000:
        sample = sample.sample(25000, random_state=7)

    for condition, color in (("Exon4", EXON_COLOR), ("NoExon4", NOEX_COLOR)):
        subset = sample[sample["condition"] == condition]
        ax.scatter(
            subset["unit_area_px"],
            subset["protein_positive_area_px"],
            s=5,
            alpha=0.08,
            linewidths=0,
            color=color,
        )

    ax.plot(exon_x, exon_y, color=EXON_COLOR, lw=2.2, label="Exon4 LOWESS")
    ax.plot(noex_x, noex_y, color=NOEX_COLOR, lw=2.2, label="NoExon4 LOWESS")
    ax.axvline(x0, color="black", ls="--", lw=1.2, label=f"Intersection = {x0:.1f} px")
    ax.set_xlabel("Unit area (px)")
    ax.set_ylabel("Protein-positive area (px)")
    ax.set_title(f"{protein} {population}: size-dependent protein trend")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summarize_patient_metric(units: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = units.groupby(["CPID", "condition"], observed=True)

    if metric == "mean_positive_area":
        summary = grouped["protein_positive_area_px"].mean().rename("metric")
    elif metric == "median_positive_area":
        summary = grouped["protein_positive_area_px"].median().rename("metric")
    elif metric == "sum_positive_area":
        summary = grouped["protein_positive_area_px"].sum().rename("metric")
    elif metric == "mean_ratio":
        summary = grouped["protein_ratio"].mean().rename("metric")
    elif metric == "median_ratio":
        summary = grouped["protein_ratio"].median().rename("metric")
    elif metric == "area_fraction":
        totals = grouped[["protein_positive_area_px", "unit_area_px"]].sum()
        summary = (
            totals["protein_positive_area_px"] / totals["unit_area_px"].replace(0, np.nan)
        ).rename("metric")
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return summary.reset_index().dropna(subset=["metric"])


def make_patient_table(units: pd.DataFrame, metric: str) -> pd.DataFrame:
    patient_metric = summarize_patient_metric(units, metric)
    wide = patient_metric.pivot(index="CPID", columns="condition", values="metric")
    if "Exon4" not in wide.columns or "NoExon4" not in wide.columns:
        return pd.DataFrame()
    wide = wide.dropna(subset=["Exon4", "NoExon4"]).copy()
    if wide.empty:
        return pd.DataFrame()

    survival = (
        units[["CPID", "OS_months", "event"]]
        .drop_duplicates()
        .dropna(subset=["CPID", "OS_months", "event"])
    )
    conflicts = survival.groupby("CPID")[["OS_months", "event"]].nunique(dropna=False).max(axis=1)
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.tolist()[:10]
        raise ValueError(f"Conflicting survival values were found for CPID(s): {bad}")
    survival = survival.drop_duplicates("CPID", keep="first").set_index("CPID")

    wide["delta_exon_minus_noexon"] = wide["Exon4"] - wide["NoExon4"]
    wide["ratio_exon_over_noexon"] = wide["Exon4"] / wide["NoExon4"].replace(0, np.nan)
    wide = wide.join(survival, how="inner")
    wide = wide[
        np.isfinite(wide["OS_months"])
        & (wide["OS_months"] >= 0)
        & wide["event"].isin([0, 1])
    ].copy()
    return wide.reset_index()


def km_logrank(
    patient_table: pd.DataFrame,
    group: pd.Series,
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
    out_path: Path,
) -> tuple[float, int, int]:
    data = patient_table.copy()
    data["KM_group"] = group.to_numpy()
    levels = list(labels)
    first = data[data["KM_group"] == levels[0]]
    second = data[data["KM_group"] == levels[1]]
    if first.empty or second.empty:
        return np.nan, len(first), len(second)

    result = logrank_test(
        first["OS_months"],
        second["OS_months"],
        event_observed_A=first["event"],
        event_observed_B=second["event"],
    )
    p_value = float(result.p_value)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for level in levels:
        subset = data[data["KM_group"] == level]
        fitter = KaplanMeierFitter()
        fitter.fit(
            subset["OS_months"].astype(float),
            subset["event"].astype(int),
            label=f"{labels[level]} (n={len(subset)})",
        )
        fitter.plot_survival_function(ax=ax, ci_show=True, lw=2.2, color=colors[level])

    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"{title}\nlog-rank p = {p_text(p_value)}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return p_value, len(first), len(second)


def fit_continuous_cox(
    patient_table: pd.DataFrame,
    out_csv: Path,
    penalizer: float,
) -> dict[str, float]:
    data = patient_table[["OS_months", "event", "delta_exon_minus_noexon"]].dropna().copy()
    if len(data) < 5 or data["event"].sum() < 2:
        return {"cox_hr_per_sd": np.nan, "cox_ci_lower": np.nan, "cox_ci_upper": np.nan, "cox_p": np.nan}

    sd = float(data["delta_exon_minus_noexon"].std(ddof=0))
    mean = float(data["delta_exon_minus_noexon"].mean())
    if not np.isfinite(sd) or sd <= 1e-12:
        return {"cox_hr_per_sd": np.nan, "cox_ci_lower": np.nan, "cox_ci_upper": np.nan, "cox_p": np.nan}

    data["delta_per_sd"] = (data["delta_exon_minus_noexon"] - mean) / sd
    fit_data = data[["OS_months", "event", "delta_per_sd"]]

    fitter = CoxPHFitter(penalizer=penalizer)
    fitter.fit(fit_data, duration_col="OS_months", event_col="event", robust=True)
    summary = fitter.summary.copy()
    summary.to_csv(out_csv)

    row = summary.loc["delta_per_sd"]
    return {
        "cox_hr_per_sd": float(np.exp(row["coef"])),
        "cox_ci_lower": float(np.exp(row["coef lower 95%"])),
        "cox_ci_upper": float(np.exp(row["coef upper 95%"])),
        "cox_p": float(row["p"]),
    }


def save_delta_distribution(patient_table: pd.DataFrame, out_path: Path, title: str) -> None:
    values = patient_table["delta_exon_minus_noexon"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return
    median = float(np.median(values))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.hist(values, bins=min(30, max(8, int(np.sqrt(values.size) * 2))))
    ax.axvline(0.0, color="black", lw=1.1)
    ax.axvline(median, color="black", ls="--", lw=1.1, label=f"Median Δ = {median:.3g}")
    ax.set_xlabel("Δ = Exon4 − NoExon4")
    ax.set_ylabel("Patients")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def analyze_interval(
    name: str,
    units: pd.DataFrame,
    metric: str,
    outdir: Path,
    cox_penalizer: float,
) -> dict[str, object]:
    interval_dir = outdir / name
    interval_dir.mkdir(parents=True, exist_ok=True)

    patient_table = make_patient_table(units, metric)
    if patient_table.empty:
        return {
            "interval": name,
            "n_units": int(len(units)),
            "n_patients": 0,
            "logrank_delta_sign_p": np.nan,
            "logrank_delta_median_p": np.nan,
            "cox_hr_per_sd": np.nan,
            "cox_ci_lower": np.nan,
            "cox_ci_upper": np.nan,
            "cox_p": np.nan,
        }

    patient_table.to_csv(interval_dir / "patient_metrics.csv", index=False)
    delta = patient_table["delta_exon_minus_noexon"]

    sign_group = pd.Series(
        np.where(delta > 0, "Delta_positive", "Delta_nonpositive"),
        index=patient_table.index,
    )
    p_sign, n_positive, n_nonpositive = km_logrank(
        patient_table,
        sign_group,
        labels={
            "Delta_positive": "Δ > 0 (Exon4 > NoExon4)",
            "Delta_nonpositive": "Δ ≤ 0 (NoExon4 ≥ Exon4)",
        },
        colors={"Delta_positive": EXON_COLOR, "Delta_nonpositive": NOEX_COLOR},
        title=f"{name}: survival by sign of Δ",
        out_path=interval_dir / "km_delta_sign.png",
    )

    median_delta = float(np.nanmedian(delta))
    median_group = pd.Series(
        np.where(delta >= median_delta, "High_delta", "Low_delta"),
        index=patient_table.index,
    )
    p_median, n_high, n_low = km_logrank(
        patient_table,
        median_group,
        labels={"High_delta": "Δ ≥ median", "Low_delta": "Δ < median"},
        colors={"High_delta": "tab:red", "Low_delta": "tab:green"},
        title=f"{name}: survival by median Δ",
        out_path=interval_dir / "km_delta_median.png",
    )

    cox = fit_continuous_cox(
        patient_table,
        out_csv=interval_dir / "cox_delta_continuous.csv",
        penalizer=cox_penalizer,
    )
    save_delta_distribution(
        patient_table,
        interval_dir / "delta_distribution.png",
        title=f"{name}: patient-level Δ distribution",
    )

    return {
        "interval": name,
        "n_units": int(len(units)),
        "n_patients": int(len(patient_table)),
        "n_delta_positive": int(n_positive),
        "n_delta_nonpositive": int(n_nonpositive),
        "n_high_delta": int(n_high),
        "n_low_delta": int(n_low),
        "median_delta": median_delta,
        "logrank_delta_sign_p": p_sign,
        "logrank_delta_median_p": p_median,
        **cox,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Perform patient-level Kaplan-Meier and Cox analyses of paired Exon4-NoExon4 "
            "differences within unit-size intervals."
        )
    )
    parser.add_argument(
        "--units_csv",
        required=True,
        help="One harmonized survival_units_*.csv file produced by prepare_survival_inputs.py.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--metric",
        choices=[
            "mean_positive_area",
            "median_positive_area",
            "sum_positive_area",
            "mean_ratio",
            "median_ratio",
            "area_fraction",
        ],
        default="mean_positive_area",
        help="Patient-condition metric used to calculate Δ = Exon4 - NoExon4.",
    )
    parser.add_argument("--lowess_frac", type=float, default=0.25)
    parser.add_argument("--overlap_q_low", type=float, default=0.01)
    parser.add_argument("--overlap_q_high", type=float, default=0.99)
    parser.add_argument(
        "--x0_override",
        type=float,
        default=None,
        help="Explicit unit-area cutoff. Required when the LOWESS curves have no true intersection.",
    )
    parser.add_argument(
        "--intersection_index",
        type=int,
        default=0,
        help="Zero-based intersection to use when more than one LOWESS intersection is present.",
    )
    parser.add_argument(
        "--similar_x_max",
        type=float,
        default=500.0,
        help="Upper unit-area boundary for the manuscript-defined SIMILAR interval.",
    )
    parser.add_argument("--cox_penalizer", type=float, default=0.2)
    args = parser.parse_args()

    if not 0 < args.lowess_frac <= 1:
        raise ValueError("--lowess_frac must be in (0, 1].")
    if not 0 <= args.overlap_q_low < args.overlap_q_high <= 1:
        raise ValueError("Overlap quantiles must satisfy 0 <= low < high <= 1.")
    if args.similar_x_max < 0:
        raise ValueError("--similar_x_max must be non-negative.")
    if args.cox_penalizer < 0:
        raise ValueError("--cox_penalizer must be non-negative.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    units = pd.read_csv(args.units_csv)
    require_columns(
        units,
        [
            "CPID",
            "condition",
            "unit_area_px",
            "protein_positive_area_px",
            "protein_ratio",
            "OS_months",
            "event",
        ],
        args.units_csv,
    )

    units = units.copy()
    units["condition"] = units["condition"].map(normalize_condition)
    for column in ["unit_area_px", "protein_positive_area_px", "protein_ratio", "OS_months", "event"]:
        units[column] = pd.to_numeric(units[column], errors="coerce")
    units = units[
        units["CPID"].notna()
        & units["condition"].isin(["Exon4", "NoExon4"])
        & np.isfinite(units["unit_area_px"])
        & (units["unit_area_px"] > 0)
        & np.isfinite(units["protein_positive_area_px"])
        & (units["protein_positive_area_px"] >= 0)
        & np.isfinite(units["protein_ratio"])
        & units["protein_ratio"].between(0, 1, inclusive="both")
        & np.isfinite(units["OS_months"])
        & (units["OS_months"] >= 0)
        & units["event"].isin([0, 1])
    ].copy()
    if units.empty:
        raise ValueError("No valid units remain after input validation.")

    protein = str(units["protein"].dropna().iloc[0]) if "protein" in units.columns and units["protein"].notna().any() else "Protein"
    population = str(units["population"].dropna().iloc[0]) if "population" in units.columns and units["population"].notna().any() else "Population"

    exon = units[units["condition"] == "Exon4"]
    noex = units[units["condition"] == "NoExon4"]
    exon_x, exon_y = fit_trend(
        exon["unit_area_px"].to_numpy(dtype=float),
        exon["protein_positive_area_px"].to_numpy(dtype=float),
        args.lowess_frac,
    )
    noex_x, noex_y = fit_trend(
        noex["unit_area_px"].to_numpy(dtype=float),
        noex["protein_positive_area_px"].to_numpy(dtype=float),
        args.lowess_frac,
    )

    roots = find_trend_intersections(
        exon_x,
        exon_y,
        noex_x,
        noex_y,
        overlap_quantiles=(args.overlap_q_low, args.overlap_q_high),
    )

    if args.x0_override is not None:
        x0 = float(args.x0_override)
        cutoff_method = "override"
    else:
        if not roots:
            raise ValueError(
                "The Exon4 and NoExon4 LOWESS curves have no true intersection within the "
                "specified overlap range. Provide an explicit --x0_override; no arbitrary "
                "fallback cutoff is used."
            )
        if args.intersection_index < 0 or args.intersection_index >= len(roots):
            raise ValueError(
                f"--intersection_index={args.intersection_index} is invalid; "
                f"{len(roots)} intersection(s) were found: {roots}"
            )
        x0 = float(roots[args.intersection_index])
        cutoff_method = "lowess_intersection"

    if not np.isfinite(x0) or x0 <= 0:
        raise ValueError("The selected x0 cutoff must be a positive finite number.")

    save_trend_figure(
        units,
        exon_x,
        exon_y,
        noex_x,
        noex_y,
        x0,
        outdir / "trend_intersection.png",
        protein=protein,
        population=population,
    )

    area = units["unit_area_px"]
    intervals: dict[str, pd.Series] = {
        "ALL": pd.Series(True, index=units.index),
        "LEFT_small": area < x0,
        "RIGHT_large": area >= x0,
        "SIMILAR_le500": area <= args.similar_x_max,
    }
    if x0 > args.similar_x_max:
        intervals["DIFFERENT_500_to_x0"] = (area > args.similar_x_max) & (area < x0)

    results: list[dict[str, object]] = []
    for name, mask in intervals.items():
        subset = units.loc[mask].copy()
        results.append(
            analyze_interval(
                name,
                subset,
                metric=args.metric,
                outdir=outdir / "intervals",
                cox_penalizer=args.cox_penalizer,
            )
        )

    summary = pd.DataFrame(results)
    summary.insert(0, "protein", protein)
    summary.insert(1, "population", population)
    summary.insert(2, "metric", args.metric)
    summary.insert(3, "x0_px", x0)
    summary.insert(4, "x0_method", cutoff_method)
    summary.to_csv(outdir / "survival_delta_intervals_summary.csv", index=False)

    cutoff_table = pd.DataFrame(
        {
            "protein": [protein],
            "population": [population],
            "cutoff_method": [cutoff_method],
            "selected_x0_px": [x0],
            "intersection_index": [args.intersection_index if cutoff_method == "lowess_intersection" else np.nan],
            "n_lowess_intersections": [len(roots)],
            "lowess_intersections_px": [";".join(f"{value:.8g}" for value in roots)],
            "similar_x_max_px": [args.similar_x_max],
            "lowess_frac": [args.lowess_frac],
            "overlap_q_low": [args.overlap_q_low],
            "overlap_q_high": [args.overlap_q_high],
        }
    )
    cutoff_table.to_csv(outdir / "interval_cutoff.csv", index=False)

    metadata = {
        "input": str(Path(args.units_csv).resolve()),
        "protein": protein,
        "population": population,
        "metric": args.metric,
        "delta_definition": "Exon4 - NoExon4",
        "ratio_definition": "Exon4 / NoExon4",
        "cutoff_method": cutoff_method,
        "selected_x0_px": x0,
        "all_lowess_intersections_px": roots,
        "similar_x_max_px": args.similar_x_max,
        "n_valid_units": int(len(units)),
        "n_valid_patients": int(units["CPID"].nunique()),
    }
    with open(outdir / "analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[OK] Survival analysis complete: {outdir}")
    print(f"  Protein: {protein}")
    print(f"  Population: {population}")
    print(f"  Metric: {args.metric}")
    print(f"  x0: {x0:.3f} px ({cutoff_method})")


if __name__ == "__main__":
    main()
