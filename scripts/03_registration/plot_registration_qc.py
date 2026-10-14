#!/usr/bin/env python3
"""Generate quality-control tables and figures for MAPK3-SMNDC1 registration.

The transform table is expected to come from ``register_mapk3_smndc1.py``.
MAPK3 is the moving section and SMNDC1 is the reference section. All overlap
metrics in this script are therefore evaluated in SMNDC1 space using the same
OpenCV warp convention as the registration workflow.

The script reports, for every valid transform:

- cytoplasmic/tumor-mask Dice and IoU after warping MAPK3 to SMNDC1;
- nuclear-mask Dice and IoU when both nuclear label masks are available;
- the number of SMNDC1 nuclei inside the SMNDC1 cytoplasmic mask;
- the number and fraction of those nuclei inside the MAPK3-SMNDC1 overlap;
- registration metadata copied from the transform table.

QC figures are generated for a configurable subset of cores, while the output
summary table includes all transforms that can be evaluated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk


THIS = Path(__file__).resolve()
PROJ_ROOT = THIS.parents[2]
SRC_DIR = PROJ_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from tma_if.image_io import imread_gray
from tma_if.mask_finder import find_mask_for_image
from tma_if.naming import normalize_core


CONDITION_COLUMNS = {
    "Exon": {
        "smndc1_B": "smndc1_exon_B",
        "smndc1_R": "smndc1_exon_R",
        "mapk3_B": "mapk3_exon_B",
        "mapk3_R": "mapk3_exon_R",
    },
    "NoExon": {
        "smndc1_B": "smndc1_noexon_B",
        "smndc1_R": "smndc1_noexon_R",
        "mapk3_B": "mapk3_noexon_B",
        "mapk3_R": "mapk3_noexon_R",
    },
}


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if text.startswith("noexon"):
        return "NoExon"
    if text.startswith("exon"):
        return "Exon"
    raise ValueError(f"Unrecognized condition label: {value!r}")


def normalize_core_string(value: object) -> str:
    core, _ = normalize_core(str(value))
    if core is None:
        raise ValueError(f"Could not normalize core identifier: {value!r}")
    return core


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def read_2d(path: str | Path) -> np.ndarray:
    array = np.asarray(imread_gray(str(path)))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {array.shape}: {path}")
    return array


def to_binary_mask(array: np.ndarray) -> np.ndarray:
    return np.asarray(array) > 0


def to_label_mask(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.size == 0:
        return np.zeros(array.shape, dtype=np.int32)
    maximum = float(np.nanmax(array))
    if maximum > 1.0:
        return array.astype(np.int32, copy=False)
    return label(array > 0).astype(np.int32)


def dilate_binary(mask: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius_px <= 0:
        return mask
    return binary_dilation(mask, disk(int(radius_px)))


def dice_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes differ: {a.shape} vs {b.shape}")

    intersection = int(np.logical_and(a, b).sum())
    size_a = int(a.sum())
    size_b = int(b.sum())
    union = int(np.logical_or(a, b).sum())

    dice = 2.0 * intersection / (size_a + size_b) if (size_a + size_b) > 0 else np.nan
    iou = intersection / union if union > 0 else np.nan
    return float(dice), float(iou)


def transform_from_row(row: pd.Series) -> np.ndarray:
    matrix = np.array(
        [
            [row["T00"], row["T01"], row["T02"]],
            [row["T10"], row["T11"], row["T12"]],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Transform contains non-finite values.")
    return matrix


def warp_mapk3_to_smndc1(
    source: np.ndarray,
    transform: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp a MAPK3 mask into SMNDC1 space using the registration convention."""
    height, width = output_shape
    warped = cv2.warpAffine(
        np.asarray(source),
        np.asarray(transform, dtype=np.float32),
        (int(width), int(height)),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def locate_mask(image_path: object, prefer: str) -> tuple[Path | None, str]:
    if image_path is None or pd.isna(image_path):
        return None, "image_path_missing"
    chosen, _, reason = find_mask_for_image(str(image_path), prefer=prefer)
    return (Path(chosen) if chosen else None), reason


def resolve_mask(
    transform_row: pd.Series,
    direct_column: str,
    manifest_image_path: object,
    prefer: str,
) -> tuple[Path | None, str]:
    direct = transform_row.get(direct_column, None)
    if direct is not None and not pd.isna(direct):
        direct_path = Path(str(direct))
        if direct_path.exists():
            return direct_path, f"transform_table:{direct_column}"
    return locate_mask(manifest_image_path, prefer=prefer)


def nucleus_ids_inside_region(nucleus_labels: np.ndarray, region_mask: np.ndarray) -> list[int]:
    region = np.asarray(region_mask, dtype=bool)
    if nucleus_labels.shape != region.shape:
        raise ValueError(
            f"Nuclear label shape {nucleus_labels.shape} does not match region {region.shape}."
        )
    ids = np.unique(nucleus_labels[region])
    return [int(value) for value in ids if int(value) > 0]


def select_plot_rows(
    summary: pd.DataFrame,
    condition: str,
    n_cores: int,
    mode: str,
    seed: int,
    minimum_dice: float,
) -> pd.DataFrame:
    subset = summary[
        (summary["condition"] == condition)
        & summary["dice_cytoplasm"].notna()
        & (summary["dice_cytoplasm"] >= minimum_dice)
        & (summary["qc_status"] == "ok")
    ].copy()

    if subset.empty or n_cores <= 0:
        return subset.iloc[0:0]

    n_cores = min(int(n_cores), len(subset))
    subset = subset.sort_values("dice_cytoplasm").reset_index(drop=True)

    if mode == "lowest":
        return subset.head(n_cores)
    if mode == "highest":
        return subset.tail(n_cores)
    if mode == "random":
        return subset.sample(n=n_cores, random_state=seed).sort_values("dice_cytoplasm")

    if n_cores == 1:
        positions = [len(subset) // 2]
    else:
        positions = np.linspace(0, len(subset) - 1, n_cores)
        positions = np.unique(np.rint(positions).astype(int))
    return subset.iloc[positions].copy()


def safe_contour(ax: plt.Axes, mask: np.ndarray, **kwargs: object) -> None:
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], **kwargs)


def make_qc_figure(
    row: pd.Series,
    figure_path: Path,
    downsample: int,
) -> None:
    ds = max(1, int(downsample))

    smndc1_cyto = np.load(row["_cache_smndc1_cyto"], allow_pickle=False).astype(bool)
    mapk3_cyto_warped = np.load(row["_cache_mapk3_cyto_warped"], allow_pickle=False).astype(bool)
    smndc1_nuclei = np.load(row["_cache_smndc1_nuclei"], allow_pickle=False).astype(np.int32)

    mapk3_nuclei_warped = None
    cache_nuc_m = row.get("_cache_mapk3_nuclei_warped", None)
    if isinstance(cache_nuc_m, str) and cache_nuc_m:
        path = Path(cache_nuc_m)
        if path.exists():
            mapk3_nuclei_warped = np.load(path, allow_pickle=False).astype(bool)

    overlap = np.logical_and(smndc1_cyto, mapk3_cyto_warped)

    def sample(array: np.ndarray) -> np.ndarray:
        return array[::ds, ::ds]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(sample(smndc1_cyto), cmap="gray", interpolation="nearest")
    safe_contour(axes[0], sample(mapk3_cyto_warped), linewidths=0.8)
    axes[0].imshow(sample(overlap), alpha=0.30, interpolation="nearest")
    axes[0].set_title(
        f"Cytoplasmic overlap\nDice={row['dice_cytoplasm']:.3f}, IoU={row['iou_cytoplasm']:.3f}"
    )
    axes[0].axis("off")

    smndc1_nuc_binary = smndc1_nuclei > 0
    axes[1].imshow(sample(smndc1_nuc_binary), cmap="gray", interpolation="nearest")
    if mapk3_nuclei_warped is not None:
        safe_contour(axes[1], sample(mapk3_nuclei_warped), linewidths=0.8)
        if np.isfinite(row.get("dice_nuclei", np.nan)):
            axes[1].set_title(
                f"Nuclear overlap\nDice={row['dice_nuclei']:.3f}, IoU={row['iou_nuclei']:.3f}"
            )
        else:
            axes[1].set_title("Nuclear overlap")
    else:
        axes[1].set_title("SMNDC1 nuclei\nMAPK3 nuclear mask unavailable")
    axes[1].axis("off")

    axes[2].imshow(sample(smndc1_nuc_binary), cmap="gray", interpolation="nearest")
    standard_ids = set(int(x) for x in str(row["standard_nucleus_ids"]).split(";") if x)
    coincident_ids = set(int(x) for x in str(row["coincident_nucleus_ids"]).split(";") if x)

    standard_only_x: list[float] = []
    standard_only_y: list[float] = []
    coincident_x: list[float] = []
    coincident_y: list[float] = []

    for region in regionprops(smndc1_nuclei):
        if region.label not in standard_ids:
            continue
        y, x = region.centroid
        if region.label in coincident_ids:
            coincident_x.append(x / ds)
            coincident_y.append(y / ds)
        else:
            standard_only_x.append(x / ds)
            standard_only_y.append(y / ds)

    if standard_only_x:
        axes[2].scatter(
            standard_only_x,
            standard_only_y,
            s=6,
            alpha=0.55,
            label="standard only",
        )
    if coincident_x:
        axes[2].scatter(
            coincident_x,
            coincident_y,
            s=8,
            alpha=0.85,
            label="coincident",
        )
    if standard_only_x or coincident_x:
        axes[2].legend(fontsize=8, frameon=True)
    axes[2].set_title(
        "SMNDC1 nuclei\n"
        f"standard={row['n_standard_nuclei']}  coincident={row['n_coincident_nuclei']}"
    )
    axes[2].axis("off")

    fig.suptitle(
        f"{row['condition']} | core {row['core_norm']} | registration status={row['registration_status']}",
        fontsize=12,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MAPK3-SMNDC1 registration using precomputed cell and nuclear masks."
    )
    parser.add_argument("--manifest_csv", required=True, help="Core manifest produced by build_core_manifest.py.")
    parser.add_argument(
        "--transforms_csv",
        required=True,
        help="Transform table produced by register_mapk3_smndc1.py register.",
    )
    parser.add_argument("--outdir", required=True, help="Output directory for QC tables and figures.")
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["ok", "low_dice"],
        help="Registration statuses to evaluate (default: ok low_dice).",
    )
    parser.add_argument(
        "--figure_dice_threshold",
        type=float,
        default=0.60,
        help="Minimum recomputed cytoplasmic Dice for a core to be eligible for plotting.",
    )
    parser.add_argument(
        "--n_cores_per_condition",
        type=int,
        default=12,
        help="Number of QC figures to generate per condition (default: 12).",
    )
    parser.add_argument(
        "--selection",
        choices=["spread", "random", "lowest", "highest"],
        default="spread",
        help="How cores are selected for QC figures (default: spread across Dice values).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed used when --selection random.")
    parser.add_argument(
        "--cyto_dilate_px",
        type=int,
        default=0,
        help="Optional binary dilation radius applied to both cytoplasmic masks before QC.",
    )
    parser.add_argument(
        "--plot_downsample",
        type=int,
        default=2,
        help="Integer stride used only when rendering QC figures (default: 2).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.figure_dice_threshold <= 1.0):
        raise ValueError("--figure_dice_threshold must be between 0 and 1.")
    if args.n_cores_per_condition < 0:
        raise ValueError("--n_cores_per_condition must be >= 0.")
    if args.cyto_dilate_px < 0:
        raise ValueError("--cyto_dilate_px must be >= 0.")
    if args.plot_downsample < 1:
        raise ValueError("--plot_downsample must be >= 1.")

    output_dir = Path(args.outdir).resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    cache_dir = output_dir / ".qc_cache"
    for directory in (figures_dir, tables_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(Path(args.manifest_csv).resolve(), low_memory=False)
    transforms = pd.read_csv(Path(args.transforms_csv).resolve(), low_memory=False)

    require_columns(manifest, ["core_norm"], args.manifest_csv)
    required_manifest_columns = set()
    for columns in CONDITION_COLUMNS.values():
        required_manifest_columns.update(columns.values())
    require_columns(manifest, sorted(required_manifest_columns), args.manifest_csv)

    # Accept legacy lowercase transform column names, but standardize internally.
    lowercase_map = {str(column).lower(): column for column in transforms.columns}
    rename: dict[str, str] = {}
    for canonical in ["T00", "T01", "T02", "T10", "T11", "T12"]:
        if canonical not in transforms.columns and canonical.lower() in lowercase_map:
            rename[lowercase_map[canonical.lower()]] = canonical
    if rename:
        transforms = transforms.rename(columns=rename)

    require_columns(
        transforms,
        ["condition", "core", "T00", "T01", "T02", "T10", "T11", "T12"],
        args.transforms_csv,
    )

    transforms["condition_norm"] = transforms["condition"].map(normalize_condition)
    transforms["core_norm"] = transforms["core"].map(normalize_core_string)
    manifest["core_norm"] = manifest["core_norm"].map(normalize_core_string)

    if "status" not in transforms.columns:
        transforms["status"] = "ok"
    transforms = transforms[transforms["status"].isin(args.statuses)].copy()
    if transforms.empty:
        raise RuntimeError("No transform rows remain after filtering by --statuses.")

    manifest_lookup = manifest.drop_duplicates("core_norm", keep="first").set_index("core_norm")
    summary_rows: list[dict[str, object]] = []

    for transform_index, transform_row in transforms.iterrows():
        condition = str(transform_row["condition_norm"])
        core = str(transform_row["core_norm"])
        registration_status = str(transform_row.get("status", "ok"))

        base: dict[str, object] = {
            "condition": condition,
            "core_norm": core,
            "registration_status": registration_status,
            "dice_registration_ds": pd.to_numeric(
                pd.Series([transform_row.get("dice_tumor_ds", np.nan)]), errors="coerce"
            ).iloc[0],
            "dice_registration_fullres": pd.to_numeric(
                pd.Series([transform_row.get("dice_tumor_fullres", np.nan)]), errors="coerce"
            ).iloc[0],
            "ecc_cc": pd.to_numeric(
                pd.Series([transform_row.get("ecc_cc", np.nan)]), errors="coerce"
            ).iloc[0],
            "motion": transform_row.get("motion", ""),
            "downsample_used": transform_row.get("downsample_used", np.nan),
            "qc_status": "ok",
            "qc_message": "",
        }

        if condition not in CONDITION_COLUMNS:
            base.update(qc_status="error", qc_message=f"unsupported_condition:{condition}")
            summary_rows.append(base)
            continue
        if core not in manifest_lookup.index:
            base.update(qc_status="error", qc_message="core_not_found_in_manifest")
            summary_rows.append(base)
            continue

        manifest_row = manifest_lookup.loc[core]
        columns = CONDITION_COLUMNS[condition]

        try:
            transform = transform_from_row(transform_row)

            s_cyto_path, s_cyto_source = resolve_mask(
                transform_row,
                "s_cp_mask",
                manifest_row[columns["smndc1_R"]],
                prefer="label",
            )
            m_cyto_path, m_cyto_source = resolve_mask(
                transform_row,
                "m_cp_mask",
                manifest_row[columns["mapk3_R"]],
                prefer="label",
            )
            s_nuc_path, s_nuc_source = locate_mask(
                manifest_row[columns["smndc1_B"]], prefer="label"
            )
            m_nuc_path, m_nuc_source = locate_mask(
                manifest_row[columns["mapk3_B"]], prefer="label"
            )

            if s_cyto_path is None:
                raise FileNotFoundError("SMNDC1 cytoplasmic mask was not found.")
            if m_cyto_path is None:
                raise FileNotFoundError("MAPK3 cytoplasmic mask was not found.")
            if s_nuc_path is None:
                raise FileNotFoundError("SMNDC1 nuclear mask was not found.")

            smndc1_cyto = dilate_binary(to_binary_mask(read_2d(s_cyto_path)), args.cyto_dilate_px)
            mapk3_cyto = dilate_binary(to_binary_mask(read_2d(m_cyto_path)), args.cyto_dilate_px)
            smndc1_nuclei = to_label_mask(read_2d(s_nuc_path))

            if not np.any(smndc1_cyto):
                raise ValueError("SMNDC1 cytoplasmic mask is empty.")
            if not np.any(mapk3_cyto):
                raise ValueError("MAPK3 cytoplasmic mask is empty.")
            if not np.any(smndc1_nuclei > 0):
                raise ValueError("SMNDC1 nuclear label mask is empty.")

            mapk3_cyto_warped = warp_mapk3_to_smndc1(
                mapk3_cyto.astype(np.uint8), transform, smndc1_cyto.shape
            ) > 0
            dice_cyto, iou_cyto = dice_iou(smndc1_cyto, mapk3_cyto_warped)
            overlap = np.logical_and(smndc1_cyto, mapk3_cyto_warped)

            standard_ids = nucleus_ids_inside_region(smndc1_nuclei, smndc1_cyto)
            coincident_ids = nucleus_ids_inside_region(smndc1_nuclei, overlap)

            mapk3_nuclei_warped: np.ndarray | None = None
            dice_nuclei = np.nan
            iou_nuclei = np.nan
            if m_nuc_path is not None:
                mapk3_nuclei_labels = to_label_mask(read_2d(m_nuc_path))
                if np.any(mapk3_nuclei_labels > 0):
                    mapk3_nuclei_warped = warp_mapk3_to_smndc1(
                        (mapk3_nuclei_labels > 0).astype(np.uint8),
                        transform,
                        smndc1_nuclei.shape,
                    ) > 0
                    dice_nuclei, iou_nuclei = dice_iou(
                        smndc1_nuclei > 0, mapk3_nuclei_warped
                    )

            cache_stem = f"{condition}_{core}_{transform_index}"
            cache_s_cyto = cache_dir / f"{cache_stem}_s_cyto.npy"
            cache_m_cyto_warped = cache_dir / f"{cache_stem}_m_cyto_warped.npy"
            cache_s_nuc = cache_dir / f"{cache_stem}_s_nuclei.npy"
            np.save(cache_s_cyto, smndc1_cyto.astype(np.uint8), allow_pickle=False)
            np.save(cache_m_cyto_warped, mapk3_cyto_warped.astype(np.uint8), allow_pickle=False)
            np.save(cache_s_nuc, smndc1_nuclei.astype(np.int32), allow_pickle=False)

            cache_m_nuc = ""
            if mapk3_nuclei_warped is not None:
                path = cache_dir / f"{cache_stem}_m_nuclei_warped.npy"
                np.save(path, mapk3_nuclei_warped.astype(np.uint8), allow_pickle=False)
                cache_m_nuc = str(path)

            base.update(
                {
                    "smndc1_cytoplasm_mask": str(s_cyto_path),
                    "mapk3_cytoplasm_mask": str(m_cyto_path),
                    "smndc1_nuclear_mask": str(s_nuc_path),
                    "mapk3_nuclear_mask": str(m_nuc_path) if m_nuc_path else "",
                    "smndc1_cytoplasm_mask_source": s_cyto_source,
                    "mapk3_cytoplasm_mask_source": m_cyto_source,
                    "smndc1_nuclear_mask_source": s_nuc_source,
                    "mapk3_nuclear_mask_source": m_nuc_source,
                    "dice_cytoplasm": dice_cyto,
                    "iou_cytoplasm": iou_cyto,
                    "dice_nuclei": dice_nuclei,
                    "iou_nuclei": iou_nuclei,
                    "n_standard_nuclei": len(standard_ids),
                    "n_coincident_nuclei": len(coincident_ids),
                    "fraction_coincident_nuclei": (
                        len(coincident_ids) / len(standard_ids) if standard_ids else np.nan
                    ),
                    "standard_nucleus_ids": ";".join(map(str, standard_ids)),
                    "coincident_nucleus_ids": ";".join(map(str, coincident_ids)),
                    "_cache_smndc1_cyto": str(cache_s_cyto),
                    "_cache_mapk3_cyto_warped": str(cache_m_cyto_warped),
                    "_cache_smndc1_nuclei": str(cache_s_nuc),
                    "_cache_mapk3_nuclei_warped": cache_m_nuc,
                }
            )
        except Exception as exc:
            base.update(
                qc_status="error",
                qc_message=f"{type(exc).__name__}: {exc}",
            )

        summary_rows.append(base)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("No registration rows were evaluated.")

    # Save the public QC table without temporary cache-path columns or full ID lists.
    public_columns = [
        column
        for column in summary.columns
        if not column.startswith("_cache_")
        and column not in {"standard_nucleus_ids", "coincident_nucleus_ids"}
    ]
    summary_path = tables_dir / "registration_qc_summary.csv"
    summary[public_columns].to_csv(summary_path, index=False)

    errors = summary[summary["qc_status"] != "ok"][public_columns].copy()
    errors_path = tables_dir / "registration_qc_errors.csv"
    errors.to_csv(errors_path, index=False)

    selected_frames: list[pd.DataFrame] = []
    for condition in ("Exon", "NoExon"):
        selected = select_plot_rows(
            summary,
            condition=condition,
            n_cores=args.n_cores_per_condition,
            mode=args.selection,
            seed=args.seed,
            minimum_dice=args.figure_dice_threshold,
        )
        selected_frames.append(selected)

        for _, row in selected.iterrows():
            filename = f"{condition}_{row['core_norm']}_registration_qc.png"
            make_qc_figure(
                row,
                figures_dir / filename,
                downsample=args.plot_downsample,
            )

    selected_all = (
        pd.concat(selected_frames, ignore_index=True)
        if any(not frame.empty for frame in selected_frames)
        else pd.DataFrame()
    )
    selected_path = tables_dir / "registration_qc_plotted_cores.csv"
    if selected_all.empty:
        pd.DataFrame(
            columns=["condition", "core_norm", "dice_cytoplasm", "iou_cytoplasm"]
        ).to_csv(selected_path, index=False)
    else:
        selected_all[public_columns].to_csv(selected_path, index=False)

    # Temporary arrays are useful only during figure generation.
    for path in cache_dir.glob("*.npy"):
        path.unlink(missing_ok=True)
    try:
        cache_dir.rmdir()
    except OSError:
        pass

    n_ok = int((summary["qc_status"] == "ok").sum())
    n_error = int((summary["qc_status"] != "ok").sum())
    n_plotted = int(len(selected_all))

    print(f"[OK] QC rows: {len(summary)}")
    print(f"[OK] Successfully evaluated: {n_ok}")
    print(f"[OK] QC errors: {n_error}")
    print(f"[OK] QC figures generated: {n_plotted}")
    print(f"[OK] Summary table: {summary_path}")
    print(f"[OK] Error table: {errors_path}")
    print(f"[OK] Plotted-core table: {selected_path}")
    print(f"[OK] Figures: {figures_dir}")


if __name__ == "__main__":
    main()
