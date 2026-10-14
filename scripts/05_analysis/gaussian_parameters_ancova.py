#!/usr/bin/env python3
"""Patient-level distribution summaries and covariate-adjusted comparisons.

The script summarizes MAPK3 and SMNDC1 ratio distributions at the CPID level
for Global and Registered populations. Exon4 and NoExon4 are paired within
patient. Global and Registered results are compared only among CPIDs present in
both populations. ANCOVA-style models use cluster-robust standard errors with
CPID as the clustering unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_rel, wilcoxon

CONDITIONS = ("Exon", "NoExon")
POPULATIONS = ("Global", "Registered")
PROTEINS = ("MAPK3", "SMNDC1")


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"exon", "exon4", "e"}:
        return "Exon"
    if text in {"noexon", "noexon4", "noex", "n"}:
        return "NoExon"
    raise ValueError(f"Unrecognized condition label: {value!r}")


def normalize_cpid(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
        if np.isfinite(number) and abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
    except ValueError:
        pass
    return text


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def load_table(path: str, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    require_columns(frame, ["CPID", "condition", "core_norm"], source)
    frame = frame.copy()
    frame["CPID"] = frame["CPID"].map(normalize_cpid)
    frame = frame.dropna(subset=["CPID"]).copy()
    frame["condition"] = frame["condition"].map(normalize_condition)
    frame["core_norm"] = frame["core_norm"].astype(str)
    return frame


def prepare_mapk3_global(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        ["CPID", "condition", "core_norm", "cell_id", "ratio_M_cyto", "area_cyto_px"],
        "MAPK3 Global",
    )
    out = frame.copy()
    out["unit_id"] = out["core_norm"].astype(str) + "::" + out["cell_id"].astype(str)
    out["ratio"] = numeric(out, "ratio_M_cyto")
    out["size_px"] = numeric(out, "area_cyto_px")
    return out[["CPID", "condition", "core_norm", "unit_id", "ratio", "size_px"]]


def prepare_smndc1_global(frame: pd.DataFrame) -> pd.DataFrame:
    nucleus_col = "nucleus_id" if "nucleus_id" in frame.columns else "cell_id" if "cell_id" in frame.columns else None
    if nucleus_col is None:
        raise ValueError("SMNDC1 Global requires nucleus_id or cell_id.")
    size_col = "area_nuc_px" if "area_nuc_px" in frame.columns else "area_nucleus_px" if "area_nucleus_px" in frame.columns else None
    if size_col is None:
        raise ValueError("SMNDC1 Global requires area_nuc_px or area_nucleus_px.")
    require_columns(frame, ["CPID", "condition", "core_norm", "ratio_S_nuc"], "SMNDC1 Global")
    out = frame.copy()
    out["unit_id"] = out["core_norm"].astype(str) + "::" + out[nucleus_col].astype(str)
    out["ratio"] = numeric(out, "ratio_S_nuc")
    out["size_px"] = numeric(out, size_col)
    return out[["CPID", "condition", "core_norm", "unit_id", "ratio", "size_px"]]


def prepare_mapk3_registered(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        ["CPID", "condition", "core_norm", "mapk3_cell_id", "ratio_M_cyto", "area_cyto_px"],
        "Registered MAPK3-SMNDC1",
    )
    out = frame.copy()
    out["unit_id"] = out["core_norm"].astype(str) + "::" + out["mapk3_cell_id"].astype(str)
    out["ratio"] = numeric(out, "ratio_M_cyto")
    out["size_px"] = numeric(out, "area_cyto_px")
    out = (
        out.groupby(["CPID", "condition", "core_norm", "unit_id"], as_index=False)
        .agg(ratio=("ratio", "mean"), size_px=("size_px", "mean"))
    )
    return out


def prepare_smndc1_registered(frame: pd.DataFrame) -> pd.DataFrame:
    nucleus_col = "nucleus_id" if "nucleus_id" in frame.columns else None
    if nucleus_col is None:
        raise ValueError("Registered SMNDC1 requires nucleus_id.")
    size_col = "area_nuc_px" if "area_nuc_px" in frame.columns else "area_nucleus_px" if "area_nucleus_px" in frame.columns else None
    if size_col is None:
        raise ValueError("Registered SMNDC1 requires area_nuc_px or area_nucleus_px.")
    require_columns(frame, ["CPID", "condition", "core_norm", "ratio_S_nuc"], "Registered MAPK3-SMNDC1")
    out = frame.copy()
    out["unit_id"] = out["core_norm"].astype(str) + "::" + out[nucleus_col].astype(str)
    out["ratio"] = numeric(out, "ratio_S_nuc")
    out["size_px"] = numeric(out, size_col)
    out = (
        out.groupby(["CPID", "condition", "core_norm", "unit_id"], as_index=False)
        .agg(ratio=("ratio", "mean"), size_px=("size_px", "mean"))
    )
    return out


def clean_units(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out[np.isfinite(out["ratio"]) & np.isfinite(out["size_px"])].copy()
    out = out[out["ratio"].between(0.0, 1.0, inclusive="both")].copy()
    out = out[out["size_px"] > 0].copy()
    out = out.drop_duplicates(["CPID", "condition", "unit_id"], keep="first")
    return out


def summarize_group(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mu": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
        }
    q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    return {
        "n": int(values.size),
        "mu": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def summarize_cpid(
    frame: pd.DataFrame,
    protein: str,
    population: str,
    min_units: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cpid, condition), group in frame.groupby(["CPID", "condition"], sort=True):
        ratio_stats = summarize_group(group["ratio"].to_numpy(float))
        size_stats = summarize_group(group["size_px"].to_numpy(float))
        if ratio_stats["n"] < min_units:
            continue
        rows.append(
            {
                "protein": protein,
                "population": population,
                "CPID": cpid,
                "condition": condition,
                "n_units": ratio_stats["n"],
                "mu": ratio_stats["mu"],
                "sd": ratio_stats["sd"],
                "median": ratio_stats["median"],
                "q25": ratio_stats["q25"],
                "q75": ratio_stats["q75"],
                "iqr": ratio_stats["iqr"],
                "size_mean_px": size_stats["mu"],
                "size_sd_px": size_stats["sd"],
                "size_median_px": size_stats["median"],
                "size_iqr_px": size_stats["iqr"],
                "n_cores": int(group["core_norm"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_deltas(params: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protein, population), group in params.groupby(["protein", "population"], sort=True):
        exon = group[group["condition"] == "Exon"].set_index("CPID")
        noexon = group[group["condition"] == "NoExon"].set_index("CPID")
        shared = sorted(set(exon.index).intersection(noexon.index))
        for cpid in shared:
            e = exon.loc[cpid]
            n = noexon.loc[cpid]
            rows.append(
                {
                    "protein": protein,
                    "population": population,
                    "CPID": cpid,
                    "n_units_Exon": int(e["n_units"]),
                    "n_units_NoExon": int(n["n_units"]),
                    "n_cores_Exon": int(e["n_cores"]),
                    "n_cores_NoExon": int(n["n_cores"]),
                    "mu_Exon": e["mu"],
                    "mu_NoExon": n["mu"],
                    "sd_Exon": e["sd"],
                    "sd_NoExon": n["sd"],
                    "median_Exon": e["median"],
                    "median_NoExon": n["median"],
                    "iqr_Exon": e["iqr"],
                    "iqr_NoExon": n["iqr"],
                    "size_median_Exon_px": e["size_median_px"],
                    "size_median_NoExon_px": n["size_median_px"],
                    "delta_mu": e["mu"] - n["mu"],
                    "delta_sd": e["sd"] - n["sd"],
                    "delta_median": e["median"] - n["median"],
                    "delta_iqr": e["iqr"] - n["iqr"],
                    "size_median_pool_px": 0.5 * (e["size_median_px"] + n["size_median_px"]),
                }
            )
    return pd.DataFrame(rows)


def paired_arrays(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    aa = pd.to_numeric(a, errors="coerce").to_numpy(float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(float)
    keep = np.isfinite(aa) & np.isfinite(bb)
    return aa[keep], bb[keep]


def paired_test(a: pd.Series, b: pd.Series, min_pairs: int) -> dict[str, float | int]:
    aa, bb = paired_arrays(a, b)
    n = int(len(aa))
    result: dict[str, float | int] = {
        "n_pairs": n,
        "mean_a": float(np.mean(aa)) if n else np.nan,
        "mean_b": float(np.mean(bb)) if n else np.nan,
        "mean_difference_a_minus_b": float(np.mean(aa - bb)) if n else np.nan,
        "median_difference_a_minus_b": float(np.median(aa - bb)) if n else np.nan,
        "paired_t_p": np.nan,
        "wilcoxon_p": np.nan,
        "cohen_dz": np.nan,
    }
    if n < min_pairs:
        return result

    difference = aa - bb
    diff_sd = float(np.std(difference, ddof=1)) if n > 1 else np.nan
    scale = max(float(np.nanmax(np.abs(np.concatenate([aa, bb])))), 1.0)
    tol = 1e-12 * scale
    if np.all(np.abs(difference) <= tol):
        result["paired_t_p"] = 1.0
        result["cohen_dz"] = 0.0
    elif np.isfinite(diff_sd) and diff_sd > tol:
        result["cohen_dz"] = float(np.mean(difference) / diff_sd)
        result["paired_t_p"] = float(ttest_rel(aa, bb).pvalue)

    try:
        result["wilcoxon_p"] = 1.0 if np.all(np.abs(difference) <= tol) else float(wilcoxon(aa, bb).pvalue)
    except ValueError:
        result["wilcoxon_p"] = np.nan
    return result


def within_population_tests(deltas: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = {
        "mu": ("mu_Exon", "mu_NoExon"),
        "median": ("median_Exon", "median_NoExon"),
        "sd": ("sd_Exon", "sd_NoExon"),
        "iqr": ("iqr_Exon", "iqr_NoExon"),
    }
    for (protein, population), group in deltas.groupby(["protein", "population"], sort=True):
        for metric, (exon_col, noexon_col) in metrics.items():
            stats = paired_test(group[exon_col], group[noexon_col], min_pairs)
            rows.append(
                {
                    "protein": protein,
                    "population": population,
                    "metric": metric,
                    "comparison": "Exon4 - NoExon4",
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def global_registered_tests(deltas: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("delta_mu", "delta_median", "delta_sd", "delta_iqr")
    for protein in PROTEINS:
        group = deltas[deltas["protein"] == protein]
        for metric in metrics:
            pivot = group.pivot_table(index="CPID", columns="population", values=metric, aggfunc="first")
            if not set(POPULATIONS).issubset(pivot.columns):
                continue
            stats = paired_test(pivot["Registered"], pivot["Global"], min_pairs)
            rows.append(
                {
                    "protein": protein,
                    "metric": metric,
                    "comparison": "Registered - Global",
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def cluster_robust_ancova(deltas: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("delta_mu", "delta_median", "delta_sd", "delta_iqr")

    for protein in PROTEINS:
        protein_data = deltas[deltas["protein"] == protein].copy()
        for metric in metrics:
            pivot = protein_data.pivot_table(index="CPID", columns="population", values=metric, aggfunc="first")
            if not set(POPULATIONS).issubset(pivot.columns):
                continue
            paired_cpids = pivot.dropna(subset=list(POPULATIONS)).index.astype(str)
            stack = protein_data[
                protein_data["CPID"].astype(str).isin(paired_cpids)
                & protein_data["population"].isin(POPULATIONS)
            ].copy()
            stack[metric] = pd.to_numeric(stack[metric], errors="coerce")
            stack["size_median_pool_px"] = pd.to_numeric(stack["size_median_pool_px"], errors="coerce")
            stack = stack.dropna(subset=[metric, "size_median_pool_px", "CPID"]).copy()
            n_cpids = int(stack["CPID"].nunique())

            base = {
                "protein": protein,
                "metric": metric,
                "n_CPIDs": n_cpids,
                "n_rows": int(len(stack)),
                "model": f"{metric} ~ population_registered + size_z + population_registered:size_z",
                "status": "not_fit",
                "beta_population_registered": np.nan,
                "se_population_registered": np.nan,
                "p_population_registered": np.nan,
                "beta_size_z": np.nan,
                "se_size_z": np.nan,
                "p_size_z": np.nan,
                "beta_interaction": np.nan,
                "se_interaction": np.nan,
                "p_interaction": np.nan,
                "r_squared": np.nan,
            }

            if n_cpids < max(min_pairs, 4) or len(stack) < 8:
                base["status"] = "insufficient_paired_CPIDs"
                rows.append(base)
                continue

            outcome_sd = float(stack[metric].std(ddof=0))
            if not np.isfinite(outcome_sd) or outcome_sd <= 1e-12:
                base["status"] = "zero_outcome_variance"
                rows.append(base)
                continue

            size_sd = float(stack["size_median_pool_px"].std(ddof=0))
            if not np.isfinite(size_sd) or size_sd <= 1e-12:
                base["status"] = "zero_size_variance"
                rows.append(base)
                continue

            stack["size_z"] = (
                stack["size_median_pool_px"] - stack["size_median_pool_px"].mean()
            ) / size_sd
            stack["population_registered"] = (stack["population"] == "Registered").astype(float)
            stack["interaction"] = stack["population_registered"] * stack["size_z"]

            X = sm.add_constant(
                stack[["population_registered", "size_z", "interaction"]],
                has_constant="add",
            )
            y = stack[metric].astype(float)

            try:
                fit = sm.OLS(y, X).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": stack["CPID"].astype(str)},
                )
                base.update(
                    {
                        "status": "ok",
                        "beta_population_registered": float(fit.params["population_registered"]),
                        "se_population_registered": float(fit.bse["population_registered"]),
                        "p_population_registered": float(fit.pvalues["population_registered"]),
                        "beta_size_z": float(fit.params["size_z"]),
                        "se_size_z": float(fit.bse["size_z"]),
                        "p_size_z": float(fit.pvalues["size_z"]),
                        "beta_interaction": float(fit.params["interaction"]),
                        "se_interaction": float(fit.bse["interaction"]),
                        "p_interaction": float(fit.pvalues["interaction"]),
                        "r_squared": float(fit.rsquared),
                    }
                )
            except Exception as exc:
                base["status"] = f"fit_error: {type(exc).__name__}: {exc}"
            rows.append(base)

    return pd.DataFrame(rows)


def write_report(
    path: Path,
    inputs: dict[str, str],
    params: pd.DataFrame,
    deltas: pd.DataFrame,
    within: pd.DataFrame,
    between: pd.DataFrame,
    ancova: pd.DataFrame,
    min_units: int,
    min_pairs: int,
) -> None:
    lines = [
        "PATIENT-LEVEL DISTRIBUTION PARAMETERS AND COVARIATE-ADJUSTED COMPARISONS",
        "",
        "Definitions",
        "-----------",
        "mu and sd are the empirical mean and sample standard deviation of the",
        "cell/nucleus-level protein ratio within each CPID and condition.",
        "median and IQR are robust summaries of the same within-patient distributions.",
        "Delta is always Exon4 - NoExon4.",
        "Global versus Registered comparisons use only CPIDs represented in both populations.",
        "ANCOVA-style OLS uses cluster-robust standard errors clustered by CPID.",
        "",
        f"Minimum units per CPID-condition: {min_units}",
        f"Minimum paired CPIDs for inferential tests: {min_pairs}",
        "",
        "Inputs",
        "------",
    ]
    lines.extend([f"{key}: {value}" for key, value in inputs.items()])
    lines.extend(
        [
            "",
            "Counts",
            "------",
            f"CPID-condition summary rows: {len(params)}",
            f"Paired CPID delta rows: {len(deltas)}",
            "",
            "Exon4 versus NoExon4 paired tests",
            "----------------------------------",
            within.to_string(index=False) if not within.empty else "No valid paired comparisons.",
            "",
            "Registered versus Global paired delta tests",
            "---------------------------------------------",
            between.to_string(index=False) if not between.empty else "No valid paired comparisons.",
            "",
            "Cluster-robust ANCOVA-style models",
            "----------------------------------",
            ancova.to_string(index=False) if not ancova.empty else "No valid models.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize patient-level MAPK3/SMNDC1 distributions and compare Exon4, NoExon4, Global, and Registered populations."
    )
    parser.add_argument("--mapk3_global_csv", required=True)
    parser.add_argument("--smndc1_global_csv", required=True)
    parser.add_argument("--registered_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min_units_per_condition", type=int, default=1)
    parser.add_argument("--min_pairs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_units_per_condition < 1:
        raise ValueError("--min_units_per_condition must be >= 1.")
    if args.min_pairs < 2:
        raise ValueError("--min_pairs must be >= 2.")

    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    reports_dir = outdir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    mapk3_global_raw = load_table(args.mapk3_global_csv, "MAPK3 Global")
    smndc1_global_raw = load_table(args.smndc1_global_csv, "SMNDC1 Global")
    registered_raw = load_table(args.registered_csv, "Registered MAPK3-SMNDC1")

    prepared = {
        ("MAPK3", "Global"): clean_units(prepare_mapk3_global(mapk3_global_raw)),
        ("SMNDC1", "Global"): clean_units(prepare_smndc1_global(smndc1_global_raw)),
        ("MAPK3", "Registered"): clean_units(prepare_mapk3_registered(registered_raw)),
        ("SMNDC1", "Registered"): clean_units(prepare_smndc1_registered(registered_raw)),
    }

    summaries = [
        summarize_cpid(frame, protein, population, args.min_units_per_condition)
        for (protein, population), frame in prepared.items()
    ]
    params = pd.concat(summaries, ignore_index=True)
    if params.empty:
        raise RuntimeError("No CPID-level summaries were generated from the supplied inputs.")

    deltas = build_deltas(params)
    if deltas.empty:
        raise RuntimeError("No CPIDs have both Exon4 and NoExon4 measurements.")

    within = within_population_tests(deltas, args.min_pairs)
    between = global_registered_tests(deltas, args.min_pairs)
    ancova = cluster_robust_ancova(deltas, args.min_pairs)

    params_path = tables_dir / "distribution_parameters_by_cpid.csv"
    deltas_path = tables_dir / "exon_noexon_deltas_by_cpid.csv"
    within_path = tables_dir / "paired_tests_exon_noexon.csv"
    between_path = tables_dir / "paired_tests_global_registered.csv"
    ancova_path = tables_dir / "ancova_cluster_robust.csv"
    report_path = reports_dir / "gaussian_parameters_ancova_report.txt"
    metadata_path = reports_dir / "analysis_metadata.json"

    params.to_csv(params_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    within.to_csv(within_path, index=False)
    between.to_csv(between_path, index=False)
    ancova.to_csv(ancova_path, index=False)

    inputs = {
        "mapk3_global_csv": str(Path(args.mapk3_global_csv).resolve()),
        "smndc1_global_csv": str(Path(args.smndc1_global_csv).resolve()),
        "registered_csv": str(Path(args.registered_csv).resolve()),
    }
    write_report(
        report_path,
        inputs,
        params,
        deltas,
        within,
        between,
        ancova,
        args.min_units_per_condition,
        args.min_pairs,
    )

    metadata = {
        "inputs": inputs,
        "definitions": {
            "delta": "Exon4 - NoExon4",
            "mapk3_ratio": "MAPK3-positive area within the cytoplasmic/cell mask divided by cytoplasmic area",
            "smndc1_ratio": "SMNDC1-positive area within the nuclear mask divided by nuclear area",
            "patient_aggregation": "All eligible unique units are pooled across cores within each CPID-condition before distribution summaries are calculated.",
            "ancova": "OLS on patient-level deltas with cluster-robust standard errors clustered by CPID.",
        },
        "min_units_per_condition": args.min_units_per_condition,
        "min_pairs": args.min_pairs,
        "rows": {
            "distribution_parameters_by_cpid": int(len(params)),
            "exon_noexon_deltas_by_cpid": int(len(deltas)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] {params_path}")
    print(f"[OK] {deltas_path}")
    print(f"[OK] {within_path}")
    print(f"[OK] {between_path}")
    print(f"[OK] {ancova_path}")
    print(f"[OK] {report_path}")


if __name__ == "__main__":
    main()
