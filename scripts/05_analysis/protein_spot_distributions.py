#!/usr/bin/env python3
"""Quantify connected protein-positive spots in MAPK3 cells and SMNDC1 nuclei.

The script operates on the canonical clean-pipeline inputs:

- ``manifest_all4.csv`` from ``build_core_manifest.py``;
- ``registered_mapk3_smndc1.csv`` from ``quantify_ratios_and_trends.py``.

Global MAPK3 measurements use all labeled MAPK3 cytoplasmic units. Global
SMNDC1 measurements use all labeled SMNDC1 nuclei. Registered measurements
use the MAPK3 cell and SMNDC1 nucleus identifiers already present in the
registered pair table; registration transforms are therefore not re-applied
at this stage.

For each unit, connected components of the binary protein-positive mask are
measured after restriction to that cell or nucleus. Exon4 versus NoExon4
inference is performed at the CPID level using paired Wilcoxon signed-rank
tests on patient-level summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import find_objects, gaussian_filter1d, label as cc_label
from scipy.stats import wilcoxon

EXON_COLOR = "tab:blue"
NOEXON_COLOR = "tab:orange"
CONDITIONS = ("Exon", "NoExon")
POPULATIONS = ("Global", "Registered")

CHANNEL_COLUMNS = {
    ("MAPK3", "Exon"): ("mapk3_exon_R", "mapk3_exon_G"),
    ("MAPK3", "NoExon"): ("mapk3_noexon_R", "mapk3_noexon_G"),
    ("SMNDC1", "Exon"): ("smndc1_exon_B", "smndc1_exon_G"),
    ("SMNDC1", "NoExon"): ("smndc1_noexon_B", "smndc1_noexon_G"),
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


def read_mask_2d(path: Path) -> np.ndarray:
    array = np.asarray(tiff.imread(path))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask at {path}; found shape {array.shape}.")
    return array


def mask_candidates(channel_image: Path, core: str, kind: str) -> list[Path]:
    directory = channel_image.parent
    core_dash = core.replace("_", "-")
    keys = (core, core_dash)

    if kind == "cytoplasm":
        suffixes = ("cp_masks", "cyto_masks", "cytoplasm_masks")
    elif kind == "nucleus":
        suffixes = ("nuc_masks", "nucleus_masks", "nuclear_masks")
    elif kind == "protein":
        suffixes = (
            "prot_mask",
            "prot_masks",
            "protein_mask",
            "protein_masks",
            "cp_masks",
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
            seen.add(resolved)
            unique.append(candidate)
    return unique


def find_mask(channel_image: object, core: str, kind: str) -> Path | None:
    if pd.isna(channel_image) or str(channel_image).strip() == "":
        return None
    image_path = Path(str(channel_image))
    if not image_path.exists():
        return None
    for candidate in mask_candidates(image_path, core, kind):
        if candidate.exists():
            return candidate
    return None


def load_manifest(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"core_norm", "CPID"}
    for label_column, protein_column in CHANNEL_COLUMNS.values():
        required.add(label_column)
        required.add(protein_column)
    require_columns(frame, required, "manifest")

    frame = frame.copy()
    frame["core_norm"] = frame["core_norm"].map(normalize_core)
    if frame["core_norm"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["core_norm"].duplicated(keep=False), "core_norm"].unique()
        )
        raise ValueError(f"Manifest contains duplicate core_norm values: {duplicates[:10]}")
    return frame


def load_registered_pairs(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    require_columns(
        frame,
        ["condition", "core_norm", "mapk3_cell_id", "nucleus_id"],
        "registered MAPK3-SMNDC1 table",
    )
    frame = frame.copy()
    frame["condition"] = frame["condition"].map(normalize_condition)
    frame["core_norm"] = frame["core_norm"].map(normalize_core)
    frame["mapk3_cell_id"] = pd.to_numeric(frame["mapk3_cell_id"], errors="coerce").astype("Int64")
    frame["nucleus_id"] = pd.to_numeric(frame["nucleus_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["mapk3_cell_id", "nucleus_id"]).copy()
    return frame


def quantify_spots(
    label_image: np.ndarray,
    protein_mask: np.ndarray,
    unit_ids: Sequence[int] | None,
    min_positive_area_px: int,
    min_spot_area_px: int,
) -> pd.DataFrame:
    labels = np.asarray(label_image, dtype=np.int64)
    protein_positive = np.asarray(protein_mask) > 0
    if labels.shape != protein_positive.shape:
        raise ValueError(
            f"Mask shape mismatch: labels {labels.shape}, protein {protein_positive.shape}."
        )

    max_label = int(labels.max()) if labels.size else 0
    columns = [
        "unit_id",
        "compartment_area_px",
        "protein_positive_area_px",
        "protein_positive_fraction",
        "n_spots",
        "largest_spot_area_px",
        "largest_spot_fraction_of_positive_area",
        "mean_spot_area_px",
        "is_protein_positive",
    ]
    if max_label <= 0:
        return pd.DataFrame(columns=columns)

    if unit_ids is None:
        ids = np.unique(labels)
        ids = ids[ids > 0].astype(np.int64)
    else:
        ids = pd.to_numeric(pd.Series(list(unit_ids)), errors="coerce").dropna().astype(np.int64).unique()
        ids = ids[(ids > 0) & (ids <= max_label)]
    if len(ids) == 0:
        return pd.DataFrame(columns=columns)

    objects = find_objects(labels)
    structure = np.ones((3, 3), dtype=np.uint8)
    rows: list[dict[str, object]] = []

    for unit_id in ids:
        bbox = objects[unit_id - 1] if unit_id - 1 < len(objects) else None
        if bbox is None:
            continue

        unit_region = labels[bbox] == unit_id
        compartment_area = int(unit_region.sum())
        if compartment_area <= 0:
            continue

        protein_region = protein_positive[bbox] & unit_region
        raw_positive_area = int(protein_region.sum())

        n_spots = 0
        largest_spot_area = 0
        retained_positive_area = 0
        mean_spot_area = 0.0

        if raw_positive_area > 0:
            connected, n_connected = cc_label(protein_region, structure=structure)
            if n_connected > 0:
                sizes = np.bincount(connected.ravel())[1:].astype(np.int64)
                sizes = sizes[sizes >= min_spot_area_px]
                if sizes.size:
                    n_spots = int(sizes.size)
                    retained_positive_area = int(sizes.sum())
                    largest_spot_area = int(sizes.max())
                    mean_spot_area = float(sizes.mean())

        is_positive = int(retained_positive_area >= min_positive_area_px)
        if not is_positive:
            n_spots = 0
            retained_positive_area = 0
            largest_spot_area = 0
            mean_spot_area = 0.0

        positive_fraction = float(retained_positive_area / compartment_area)
        largest_fraction = (
            float(largest_spot_area / retained_positive_area)
            if retained_positive_area > 0
            else 0.0
        )

        rows.append(
            {
                "unit_id": int(unit_id),
                "compartment_area_px": compartment_area,
                "protein_positive_area_px": retained_positive_area,
                "protein_positive_fraction": positive_fraction,
                "n_spots": n_spots,
                "largest_spot_area_px": largest_spot_area,
                "largest_spot_fraction_of_positive_area": largest_fraction,
                "mean_spot_area_px": mean_spot_area,
                "is_protein_positive": is_positive,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def registered_unit_ids(
    registered: pd.DataFrame,
    protein: str,
    condition: str,
    core: str,
) -> np.ndarray:
    subset = registered[
        (registered["condition"] == condition) & (registered["core_norm"] == core)
    ]
    column = "mapk3_cell_id" if protein == "MAPK3" else "nucleus_id"
    return subset[column].dropna().astype(np.int64).unique()


def quantify_all_populations(
    manifest: pd.DataFrame,
    registered: pd.DataFrame,
    min_positive_area_px: int,
    min_spot_area_px: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        core = normalize_core(row["core_norm"])
        cpid = row["CPID"]

        for protein in ("MAPK3", "SMNDC1"):
            label_kind = "cytoplasm" if protein == "MAPK3" else "nucleus"
            unit_name = "cell" if protein == "MAPK3" else "nucleus"

            for condition in CONDITIONS:
                label_column, protein_column = CHANNEL_COLUMNS[(protein, condition)]
                diagnostic: dict[str, object] = {
                    "protein": protein,
                    "condition": condition,
                    "core_norm": core,
                    "CPID": cpid,
                    "status": "not_run",
                }

                try:
                    label_path = find_mask(row[label_column], core, label_kind)
                    protein_path = find_mask(row[protein_column], core, "protein")
                    if label_path is None:
                        raise FileNotFoundError(f"{protein} {label_kind} label mask not found.")
                    if protein_path is None:
                        raise FileNotFoundError(f"{protein} protein-positive mask not found.")

                    label_image = read_mask_2d(label_path)
                    protein_mask = read_mask_2d(protein_path)

                    global_frame = quantify_spots(
                        label_image,
                        protein_mask,
                        unit_ids=None,
                        min_positive_area_px=min_positive_area_px,
                        min_spot_area_px=min_spot_area_px,
                    )
                    if not global_frame.empty:
                        global_frame.insert(0, "population", "Global")
                        global_frame.insert(0, "protein", protein)
                        global_frame.insert(2, "condition", condition)
                        global_frame.insert(3, "core_norm", core)
                        global_frame.insert(4, "CPID", cpid)
                        global_frame.insert(6, "unit_type", unit_name)
                        result_frames.append(global_frame)

                    ids_registered = registered_unit_ids(registered, protein, condition, core)
                    registered_frame = quantify_spots(
                        label_image,
                        protein_mask,
                        unit_ids=ids_registered,
                        min_positive_area_px=min_positive_area_px,
                        min_spot_area_px=min_spot_area_px,
                    )
                    if not registered_frame.empty:
                        registered_frame.insert(0, "population", "Registered")
                        registered_frame.insert(0, "protein", protein)
                        registered_frame.insert(2, "condition", condition)
                        registered_frame.insert(3, "core_norm", core)
                        registered_frame.insert(4, "CPID", cpid)
                        registered_frame.insert(6, "unit_type", unit_name)
                        result_frames.append(registered_frame)

                    diagnostic.update(
                        {
                            "status": "ok",
                            "label_mask": str(label_path),
                            "protein_mask": str(protein_path),
                            "n_global_units": int(len(global_frame)),
                            "n_registered_ids_requested": int(len(ids_registered)),
                            "n_registered_units_quantified": int(len(registered_frame)),
                        }
                    )
                except Exception as exc:
                    diagnostic["status"] = f"error: {type(exc).__name__}: {exc}"
                qc_rows.append(diagnostic)

    features = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    return features, pd.DataFrame(qc_rows)


def paired_wilcoxon(exon: np.ndarray, noexon: np.ndarray) -> float:
    exon = np.asarray(exon, dtype=float)
    noexon = np.asarray(noexon, dtype=float)
    valid = np.isfinite(exon) & np.isfinite(noexon)
    exon = exon[valid]
    noexon = noexon[valid]
    if exon.size < 2:
        return np.nan
    if np.allclose(exon, noexon, atol=1e-12, rtol=0.0):
        return 1.0
    try:
        return float(wilcoxon(exon, noexon, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def patient_level_summaries(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (protein, population, cpid, condition), group in features.groupby(
        ["protein", "population", "CPID", "condition"], dropna=False
    ):
        positive = group["is_protein_positive"] == 1
        spot_counts_positive = group.loc[positive, "n_spots"].astype(float)
        rows.append(
            {
                "protein": protein,
                "population": population,
                "CPID": cpid,
                "condition": condition,
                "n_units": int(len(group)),
                "fraction_protein_positive": float(group["is_protein_positive"].mean()),
                "median_spots_all_units": float(group["n_spots"].median()),
                "median_spots_positive_units": (
                    float(spot_counts_positive.median()) if not spot_counts_positive.empty else np.nan
                ),
                "median_positive_fraction": float(group["protein_positive_fraction"].median()),
                "median_largest_spot_fraction": (
                    float(group.loc[positive, "largest_spot_fraction_of_positive_area"].median())
                    if positive.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_tests(patient_summary: pd.DataFrame) -> pd.DataFrame:
    if patient_summary.empty:
        return pd.DataFrame()

    metrics = (
        "fraction_protein_positive",
        "median_spots_all_units",
        "median_spots_positive_units",
        "median_positive_fraction",
        "median_largest_spot_fraction",
    )
    rows: list[dict[str, object]] = []

    for protein in ("MAPK3", "SMNDC1"):
        for population in POPULATIONS:
            subset = patient_summary[
                (patient_summary["protein"] == protein)
                & (patient_summary["population"] == population)
            ]
            if subset.empty:
                continue

            for metric in metrics:
                wide = subset.pivot_table(
                    index="CPID", columns="condition", values=metric, aggfunc="first"
                )
                if not set(CONDITIONS).issubset(wide.columns):
                    continue
                wide = wide[list(CONDITIONS)].dropna()
                if wide.empty:
                    continue

                exon = wide["Exon"].to_numpy(dtype=float)
                noexon = wide["NoExon"].to_numpy(dtype=float)
                rows.append(
                    {
                        "protein": protein,
                        "population": population,
                        "metric": metric,
                        "n_paired_CPID": int(len(wide)),
                        "median_Exon": float(np.median(exon)),
                        "median_NoExon": float(np.median(noexon)),
                        "median_delta_Exon_minus_NoExon": float(np.median(exon - noexon)),
                        "wilcoxon_p": paired_wilcoxon(exon, noexon),
                    }
                )
    return pd.DataFrame(rows)


def histogram_spot_counts(
    exon_values: Sequence[int],
    noexon_values: Sequence[int],
    output: Path,
    title: str,
    xlabel: str,
    kmax: int,
    smooth_sigma: float,
) -> None:
    exon = pd.to_numeric(pd.Series(exon_values), errors="coerce").dropna().astype(int).to_numpy()
    noexon = pd.to_numeric(pd.Series(noexon_values), errors="coerce").dropna().astype(int).to_numpy()
    if exon.size == 0 or noexon.size == 0:
        return

    exon = np.minimum(exon, kmax)
    noexon = np.minimum(noexon, kmax)
    bins = np.arange(-0.5, kmax + 1.5, 1.0)
    counts_exon, _ = np.histogram(exon, bins=bins)
    counts_noexon, _ = np.histogram(noexon, bins=bins)
    x = np.arange(kmax + 1)

    smooth_exon = gaussian_filter1d(counts_exon.astype(float), sigma=smooth_sigma, mode="nearest")
    smooth_noexon = gaussian_filter1d(counts_noexon.astype(float), sigma=smooth_sigma, mode="nearest")

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    width = 0.38
    ax.bar(
        x - width / 2,
        counts_exon,
        width=width,
        alpha=0.55,
        label=f"Exon4 (n={len(exon):,})",
        color=EXON_COLOR,
    )
    ax.bar(
        x + width / 2,
        counts_noexon,
        width=width,
        alpha=0.55,
        label=f"NoExon4 (n={len(noexon):,})",
        color=NOEXON_COLOR,
    )
    ax.plot(x, smooth_exon, linewidth=2.2, color=EXON_COLOR)
    ax.plot(x, smooth_noexon, linewidth=2.2, color=NOEXON_COLOR)

    labels = [str(i) for i in range(kmax)] + [f"{kmax}+"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    features: pd.DataFrame,
    figures_dir: Path,
    kmax: int,
    smooth_sigma: float,
) -> list[str]:
    generated: list[str] = []
    for protein, unit_name in (("MAPK3", "cell"), ("SMNDC1", "nucleus")):
        for population in POPULATIONS:
            subset = features[
                (features["protein"] == protein) & (features["population"] == population)
            ]
            if subset.empty:
                continue

            for positive_only, suffix in ((False, "all"), (True, "positive")):
                plotted = subset[subset["is_protein_positive"] == 1] if positive_only else subset
                exon = plotted.loc[plotted["condition"] == "Exon", "n_spots"]
                noexon = plotted.loc[plotted["condition"] == "NoExon", "n_spots"]
                if exon.empty or noexon.empty:
                    continue

                output = figures_dir / f"{protein}_{population}_spots_{suffix}.png"
                title_suffix = "protein-positive units" if positive_only else "all units"
                histogram_spot_counts(
                    exon,
                    noexon,
                    output,
                    title=f"{protein} {population}: spots per {unit_name} ({title_suffix})",
                    xlabel=f"Protein-positive spots per {unit_name}",
                    kmax=kmax,
                    smooth_sigma=smooth_sigma,
                )
                generated.append(str(output))
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify MAPK3 and SMNDC1 protein-positive spot distributions."
    )
    parser.add_argument("--manifest_csv", required=True, help="manifest_all4.csv")
    parser.add_argument(
        "--registered_csv",
        required=True,
        help="registered_mapk3_smndc1.csv from quantify_ratios_and_trends.py",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--min_positive_area_px",
        type=int,
        default=1,
        help="Minimum retained protein-positive area required to call a unit positive.",
    )
    parser.add_argument(
        "--min_spot_area_px",
        type=int,
        default=1,
        help="Minimum connected-component area retained as a protein-positive spot.",
    )
    parser.add_argument(
        "--kmax_hist",
        type=int,
        default=20,
        help="Pool spot counts greater than or equal to this value into the final histogram bin.",
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for the histogram count curves.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.min_positive_area_px < 1:
        parser.error("--min_positive_area_px must be >= 1.")
    if args.min_spot_area_px < 1:
        parser.error("--min_spot_area_px must be >= 1.")
    if args.kmax_hist < 1:
        parser.error("--kmax_hist must be >= 1.")
    if args.smooth_sigma <= 0:
        parser.error("--smooth_sigma must be > 0.")

    outdir = Path(args.outdir).resolve()
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    reports_dir = outdir / "reports"
    for directory in (tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest_csv)
    registered = load_registered_pairs(args.registered_csv)
    manifest_cores = set(manifest["core_norm"])
    registered = registered[registered["core_norm"].isin(manifest_cores)].copy()

    features, qc = quantify_all_populations(
        manifest,
        registered,
        min_positive_area_px=args.min_positive_area_px,
        min_spot_area_px=args.min_spot_area_px,
    )
    if features.empty:
        raise RuntimeError("No protein spot features were quantified. Check manifest paths and masks.")

    patient_summary = patient_level_summaries(features)
    tests = paired_tests(patient_summary)
    generated_figures = generate_figures(
        features,
        figures_dir,
        kmax=args.kmax_hist,
        smooth_sigma=args.smooth_sigma,
    )

    features.to_csv(tables_dir / "protein_spot_features.csv", index=False)
    patient_summary.to_csv(tables_dir / "protein_spot_summary_by_cpid.csv", index=False)
    tests.to_csv(tables_dir / "protein_spot_paired_tests.csv", index=False)
    qc.to_csv(tables_dir / "protein_spot_processing_qc.csv", index=False)

    report = {
        "manifest_csv": str(Path(args.manifest_csv).resolve()),
        "registered_csv": str(Path(args.registered_csv).resolve()),
        "min_positive_area_px": args.min_positive_area_px,
        "min_spot_area_px": args.min_spot_area_px,
        "kmax_hist": args.kmax_hist,
        "smooth_sigma": args.smooth_sigma,
        "n_feature_rows": int(len(features)),
        "n_patient_summary_rows": int(len(patient_summary)),
        "n_tests": int(len(tests)),
        "n_qc_records": int(len(qc)),
        "figures": generated_figures,
    }
    with open(reports_dir / "protein_spot_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[OK] features: {tables_dir / 'protein_spot_features.csv'}")
    print(f"[OK] patient summaries: {tables_dir / 'protein_spot_summary_by_cpid.csv'}")
    print(f"[OK] paired tests: {tables_dir / 'protein_spot_paired_tests.csv'}")
    print(f"[OK] figures: {figures_dir}")


if __name__ == "__main__":
    main()
