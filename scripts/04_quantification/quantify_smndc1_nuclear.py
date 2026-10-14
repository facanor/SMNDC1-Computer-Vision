#!/usr/bin/env python3
"""Quantify nuclear SMNDC1 signal in standard and registered TMA populations.

This script quantifies SMNDC1 within Cellpose-derived nuclear masks and keeps
area-based and intensity-based measurements explicitly separate.

Standard population
-------------------
All SMNDC1 nuclei whose centroid lies inside the corresponding SMNDC1
cytoplasmic/tumor mask.

Registered (coincident) population
----------------------------------
The subset of standard SMNDC1 nuclei that also lies inside the registered
MAPK3 cell/tumor region. Registration transforms are expected to come from
``register_mapk3_smndc1.py`` and map MAPK3 coordinates into SMNDC1 space.

Per-nucleus SMNDC1 measurements
-------------------------------
``smndc1_mean_intensity``
    Mean SMNDC1-channel intensity inside the nucleus.

``smndc1_positive_area_px``
    Number of protein-positive pixels inside the nucleus, using the binary
    protein mask produced during preprocessing/segmentation.

``smndc1_positive_area_fraction``
    SMNDC1-positive nuclear area divided by total nuclear area. This is the
    area-fraction metric corresponding to the historical ``ratio_S_nuc``.

The intensity metric and the positive-area fraction are not interchangeable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.morphology import binary_erosion, disk
from tifffile import imread


THIS = Path(__file__).resolve()
PROJ_ROOT = THIS.parents[2]
SRC_DIR = PROJ_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

try:
    from tma_if.naming import normalize_core as shared_normalize_core
except Exception:
    shared_normalize_core = None


CONDITION_COLUMNS = {
    "Exon": {
        "mapk3_B": "mapk3_exon_B",
        "mapk3_R": "mapk3_exon_R",
        "smndc1_B": "smndc1_exon_B",
        "smndc1_G": "smndc1_exon_G",
        "smndc1_R": "smndc1_exon_R",
    },
    "NoExon": {
        "mapk3_B": "mapk3_noexon_B",
        "mapk3_R": "mapk3_noexon_R",
        "smndc1_B": "smndc1_noexon_B",
        "smndc1_G": "smndc1_noexon_G",
        "smndc1_R": "smndc1_noexon_R",
    },
}

TRANSFORM_COLUMNS = {
    "condition",
    "core",
    "status",
    "T00",
    "T01",
    "T02",
    "T10",
    "T11",
    "T12",
}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def normalize_core(value: object) -> str:
    """Normalize a core identifier to ``row_column`` form."""
    if shared_normalize_core is not None:
        normalized = shared_normalize_core(str(value))
        if isinstance(normalized, tuple):
            normalized = normalized[0]
        if normalized:
            return str(normalized)

    text = Path(str(value)).name
    match = re.search(r"(\d+)[_-](\d+)", text)
    if match:
        return f"{int(match.group(1))}_{int(match.group(2))}"
    return text.replace("-", "_").strip()


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if text.startswith("noexon"):
        return "NoExon"
    if text.startswith("exon"):
        return "Exon"
    raise ValueError(f"Unrecognized condition label: {value!r}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def read_2d(path: str | Path) -> np.ndarray:
    array = np.asarray(imread(str(path)))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {array.shape}: {path}")
    return array


def to_label_mask(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.size == 0:
        return np.zeros(array.shape, dtype=np.int32)
    if not np.issubdtype(array.dtype, np.integer):
        array = np.rint(array)
    labels = array.astype(np.int32, copy=False)
    labels[labels < 0] = 0
    return labels


def pad_or_crop_to_shape(array: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Top-left aligned crop/pad matching the registration workflow."""
    target_h, target_w = shape_hw
    source_h, source_w = array.shape[:2]
    cropped = array[: min(source_h, target_h), : min(source_w, target_w)]
    output = np.zeros((target_h, target_w), dtype=array.dtype)
    output[: cropped.shape[0], : cropped.shape[1]] = cropped
    return output


def find_mask_for_channel(image_path: str | Path, kind: str) -> Path | None:
    """Find a segmentation/protein mask adjacent to a channel image."""
    path = Path(image_path)
    if not path.exists():
        return None

    stem = path.stem
    if kind == "nucleus":
        suffixes = ["_nuc_masks.tif", "_nucleus_masks.tif", "_nuclei_masks.tif"]
    elif kind == "cytoplasm":
        suffixes = ["_cp_masks.tif", "_cyto_masks.tif", "_cytoplasm_masks.tif"]
    elif kind == "protein":
        suffixes = ["_prot_mask.tif", "_prot_masks.tif", "_protein_mask.tif", "_protein_masks.tif"]
    else:
        raise ValueError(f"Unknown mask kind: {kind}")

    for suffix in suffixes:
        candidate = path.parent / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    for suffix in suffixes:
        pattern = f"{stem}*{suffix.replace('.tif', '')}*.tif*"
        hits = sorted(path.parent.glob(pattern))
        if hits:
            return hits[0]
    return None


def transform_from_row(row: pd.Series) -> np.ndarray:
    transform = np.array(
        [
            [row["T00"], row["T01"], row["T02"]],
            [row["T10"], row["T11"], row["T12"]],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(transform).all():
        raise ValueError("Transform contains non-finite values.")
    return transform


def warp_labels_mapk3_to_smndc1(
    labels: np.ndarray,
    transform: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp MAPK3 labels into SMNDC1 space using the registration convention."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(output_shape, dtype=np.int32)

    max_label = int(np.nanmax(labels))
    if max_label <= np.iinfo(np.uint16).max:
        work = labels.astype(np.uint16, copy=False)
    else:
        work = labels.astype(np.float32, copy=False)

    height, width = output_shape
    warped = cv2.warpAffine(
        work,
        np.asarray(transform, dtype=np.float32),
        (int(width), int(height)),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.rint(warped).astype(np.int32)




# -----------------------------------------------------------------------------
# Nuclear measurements
# -----------------------------------------------------------------------------


def nucleus_geometry(labels: np.ndarray) -> pd.DataFrame:
    props = regionprops_table(
        labels,
        properties=["label", "area", "perimeter", "eccentricity", "centroid"],
    )
    if len(props.get("label", [])) == 0:
        return pd.DataFrame(
            columns=[
                "nucleus_id",
                "centroid_y",
                "centroid_x",
                "area_nucleus_px",
                "perim_nucleus_px",
                "eccentricity",
            ]
        )

    return pd.DataFrame(
        {
            "nucleus_id": props["label"].astype(np.int64),
            "centroid_y": props["centroid-0"].astype(float),
            "centroid_x": props["centroid-1"].astype(float),
            "area_nucleus_px": props["area"].astype(float),
            "perim_nucleus_px": props["perimeter"].astype(float),
            "eccentricity": props["eccentricity"].astype(float),
        }
    )


def nucleus_signal_measurements(
    labels: np.ndarray,
    protein_image: np.ndarray,
    protein_positive_mask: np.ndarray,
) -> pd.DataFrame:
    labels_flat = labels.ravel().astype(np.int64)
    intensity_flat = np.asarray(protein_image, dtype=np.float64).ravel()
    positive_flat = np.asarray(protein_positive_mask, dtype=np.float64).ravel()

    max_label = int(labels_flat.max()) if labels_flat.size else 0
    if max_label <= 0:
        return pd.DataFrame(
            columns=[
                "nucleus_id",
                "smndc1_intensity_sum",
                "smndc1_mean_intensity",
                "smndc1_positive_area_px",
                "smndc1_positive_area_fraction",
            ]
        )

    area = np.bincount(labels_flat, minlength=max_label + 1).astype(np.float64)
    intensity_sum = np.bincount(
        labels_flat,
        weights=intensity_flat,
        minlength=max_label + 1,
    ).astype(np.float64)
    positive_area = np.bincount(
        labels_flat,
        weights=positive_flat,
        minlength=max_label + 1,
    ).astype(np.float64)

    ids = np.arange(1, max_label + 1, dtype=np.int64)
    nucleus_area = np.maximum(area[1:], 1.0)

    return pd.DataFrame(
        {
            "nucleus_id": ids,
            "smndc1_intensity_sum": intensity_sum[1:],
            "smndc1_mean_intensity": intensity_sum[1:] / nucleus_area,
            "smndc1_positive_area_px": positive_area[1:],
            "smndc1_positive_area_fraction": positive_area[1:] / nucleus_area,
        }
    )


def match_nuclei_by_iou(
    smndc1_labels: np.ndarray,
    mapk3_labels_warped: np.ndarray,
    iou_threshold: float,
) -> pd.DataFrame:
    """Greedy one-to-one nucleus matching using intersection-over-union."""
    smndc1 = np.asarray(smndc1_labels, dtype=np.int64)
    mapk3 = np.asarray(mapk3_labels_warped, dtype=np.int64)
    overlap = (smndc1 > 0) & (mapk3 > 0)
    if not np.any(overlap):
        return pd.DataFrame(columns=["nucleus_id", "mapk3_nucleus_id", "nucleus_iou"])

    pair_values = np.stack([smndc1[overlap], mapk3[overlap]], axis=1)
    unique_pairs, intersections = np.unique(pair_values, axis=0, return_counts=True)

    area_s = np.bincount(smndc1.ravel())
    area_m = np.bincount(mapk3.ravel())
    s_ids = unique_pairs[:, 0].astype(np.int64)
    m_ids = unique_pairs[:, 1].astype(np.int64)
    intersections = intersections.astype(np.float64)
    unions = area_s[s_ids] + area_m[m_ids] - intersections
    iou = intersections / np.maximum(unions, 1.0)

    candidates = pd.DataFrame(
        {
            "nucleus_id": s_ids,
            "mapk3_nucleus_id": m_ids,
            "nucleus_iou": iou,
        }
    )
    candidates = candidates[candidates["nucleus_iou"] >= iou_threshold].copy()
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values("nucleus_iou", ascending=False)
    used_s: set[int] = set()
    used_m: set[int] = set()
    selected: list[pd.Series] = []
    for _, candidate in candidates.iterrows():
        sid = int(candidate["nucleus_id"])
        mid = int(candidate["mapk3_nucleus_id"])
        if sid in used_s or mid in used_m:
            continue
        used_s.add(sid)
        used_m.add(mid)
        selected.append(candidate)

    return pd.DataFrame(selected).reset_index(drop=True)


def core_summary(frame: pd.DataFrame, metadata: dict[str, object], mode: str) -> dict[str, object]:
    row = dict(metadata)
    row["mode"] = mode
    row["n_nuclei"] = int(len(frame))

    if frame.empty:
        row.update(
            {
                "smndc1_intensity_sum_total": 0.0,
                "smndc1_positive_area_total_px": 0.0,
                "smndc1_mean_intensity_mean": np.nan,
                "smndc1_mean_intensity_median": np.nan,
                "smndc1_positive_area_fraction_mean": np.nan,
                "smndc1_positive_area_fraction_median": np.nan,
                "eccentricity_mean": np.nan,
                "eccentricity_median": np.nan,
                "nuclear_area_mean_px": np.nan,
                "nuclear_area_median_px": np.nan,
            }
        )
        return row

    row.update(
        {
            "smndc1_intensity_sum_total": float(frame["smndc1_intensity_sum"].sum()),
            "smndc1_positive_area_total_px": float(frame["smndc1_positive_area_px"].sum()),
            "smndc1_mean_intensity_mean": float(frame["smndc1_mean_intensity"].mean()),
            "smndc1_mean_intensity_median": float(frame["smndc1_mean_intensity"].median()),
            "smndc1_positive_area_fraction_mean": float(frame["smndc1_positive_area_fraction"].mean()),
            "smndc1_positive_area_fraction_median": float(frame["smndc1_positive_area_fraction"].median()),
            "eccentricity_mean": float(frame["eccentricity"].mean()),
            "eccentricity_median": float(frame["eccentricity"].median()),
            "nuclear_area_mean_px": float(frame["area_nucleus_px"].mean()),
            "nuclear_area_median_px": float(frame["area_nucleus_px"].median()),
        }
    )
    return row


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"core_norm"}
    for columns in CONDITION_COLUMNS.values():
        required.update(columns.values())
    require_columns(manifest, required, "manifest")

    manifest = manifest.copy()
    manifest["core_norm"] = manifest["core_norm"].map(normalize_core)
    if manifest["core_norm"].duplicated().any():
        duplicates = sorted(manifest.loc[manifest["core_norm"].duplicated(), "core_norm"].unique())
        raise ValueError(f"Manifest contains duplicate core_norm values: {duplicates[:10]}")
    return manifest


def load_transforms(path: str | Path, include_low_dice: bool) -> pd.DataFrame:
    transforms = pd.read_csv(path)
    require_columns(transforms, TRANSFORM_COLUMNS, "transform table")

    transforms = transforms.copy()
    transforms["condition"] = transforms["condition"].map(normalize_condition)
    transforms["core_norm"] = transforms["core"].map(normalize_core)

    allowed_status = {"ok", "low_dice"} if include_low_dice else {"ok"}
    transforms = transforms[transforms["status"].isin(allowed_status)].copy()

    if transforms.duplicated(["condition", "core_norm"]).any():
        duplicates = transforms.loc[
            transforms.duplicated(["condition", "core_norm"], keep=False),
            ["condition", "core_norm"],
        ]
        raise ValueError(
            "Transform table contains multiple accepted transforms for the same condition/core: "
            + duplicates.drop_duplicates().head(10).to_string(index=False)
        )

    return transforms.set_index(["condition", "core_norm"], drop=False)


def registration_dice(row: pd.Series | None) -> float:
    if row is None:
        return np.nan
    for column in ("dice_tumor_fullres", "dice_tumor_ds", "dice_final"):
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    return np.nan


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------


def process_standard_core(
    manifest_row: pd.Series,
    condition: str,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, np.ndarray | Path]]:
    columns = CONDITION_COLUMNS[condition]
    path_b = Path(str(manifest_row[columns["smndc1_B"]]))
    path_g = Path(str(manifest_row[columns["smndc1_G"]]))
    path_r = Path(str(manifest_row[columns["smndc1_R"]]))

    for path in (path_b, path_g, path_r):
        if not path.exists():
            raise FileNotFoundError(f"Missing SMNDC1 channel image: {path}")

    nucleus_mask_path = find_mask_for_channel(path_b, "nucleus")
    cytoplasm_mask_path = find_mask_for_channel(path_r, "cytoplasm")
    protein_mask_path = find_mask_for_channel(path_g, "protein")

    missing = []
    if nucleus_mask_path is None:
        missing.append("SMNDC1 nuclear mask")
    if cytoplasm_mask_path is None:
        missing.append("SMNDC1 cytoplasmic mask")
    if protein_mask_path is None:
        missing.append("SMNDC1 protein-positive mask")
    if missing:
        raise FileNotFoundError("; ".join(missing))

    smndc1_labels = to_label_mask(read_2d(nucleus_mask_path))
    smndc1_cytoplasm = read_2d(cytoplasm_mask_path) > 0
    smndc1_protein_mask = read_2d(protein_mask_path) > 0
    smndc1_g = read_2d(path_g).astype(np.float32)

    shape = smndc1_g.shape
    for name, array in (
        ("nuclear mask", smndc1_labels),
        ("cytoplasmic mask", smndc1_cytoplasm),
        ("protein-positive mask", smndc1_protein_mask),
    ):
        if array.shape != shape:
            raise ValueError(
                f"SMNDC1 {name} shape {array.shape} does not match protein-channel shape {shape}."
            )

    geometry = nucleus_geometry(smndc1_labels)
    signal = nucleus_signal_measurements(smndc1_labels, smndc1_g, smndc1_protein_mask)
    features = geometry.merge(signal, on="nucleus_id", how="left")

    if features.empty:
        standard = features.copy()
    else:
        rr = np.clip(np.rint(features["centroid_y"]).astype(int), 0, shape[0] - 1)
        cc = np.clip(np.rint(features["centroid_x"]).astype(int), 0, shape[1] - 1)
        keep = smndc1_cytoplasm[rr, cc]
        standard = features.loc[keep].copy()

    metadata = {
        "core_norm": normalize_core(manifest_row["core_norm"]),
        "condition": condition,
        "CPID": manifest_row.get("CPID", np.nan),
        "OS_days": manifest_row.get("OS_days", np.nan),
        "OS_censor": manifest_row.get("OS_censor", np.nan),
        "OS_event": manifest_row.get("OS_event", np.nan),
    }
    for key, value in metadata.items():
        standard[key] = value
    standard["mode"] = "standard"

    # Put identifiers first for readable CSVs.
    identifier_columns = [
        "core_norm",
        "condition",
        "CPID",
        "OS_days",
        "OS_censor",
        "OS_event",
        "mode",
    ]
    remaining = [column for column in standard.columns if column not in identifier_columns]
    standard = standard[identifier_columns + remaining]

    assets: dict[str, np.ndarray | Path] = {
        "smndc1_labels": smndc1_labels,
        "smndc1_cytoplasm": smndc1_cytoplasm,
        "smndc1_protein_mask": smndc1_protein_mask,
        "smndc1_g": smndc1_g,
        "nucleus_mask_path": nucleus_mask_path,
        "cytoplasm_mask_path": cytoplasm_mask_path,
        "protein_mask_path": protein_mask_path,
    }
    return standard, metadata, assets


def process_coincident_core(
    standard: pd.DataFrame,
    manifest_row: pd.Series,
    condition: str,
    transform_row: pd.Series,
    assets: dict[str, np.ndarray | Path],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = CONDITION_COLUMNS[condition]
    mapk3_r = Path(str(manifest_row[columns["mapk3_R"]]))
    mapk3_b = Path(str(manifest_row[columns["mapk3_B"]]))

    mapk3_cp_path = find_mask_for_channel(mapk3_r, "cytoplasm")
    if mapk3_cp_path is None:
        raise FileNotFoundError(f"MAPK3 cytoplasmic mask not found for {mapk3_r}")

    mapk3_cells = to_label_mask(read_2d(mapk3_cp_path))
    smndc1_labels = np.asarray(assets["smndc1_labels"], dtype=np.int32)
    smndc1_cytoplasm = np.asarray(assets["smndc1_cytoplasm"], dtype=bool)
    output_shape = smndc1_labels.shape

    transform = transform_from_row(transform_row)
    mapk3_cells = pad_or_crop_to_shape(mapk3_cells, output_shape)
    mapk3_cells_warped = warp_labels_mapk3_to_smndc1(mapk3_cells, transform, output_shape)

    overlap_region = mapk3_cells_warped > 0
    if args.erode_mapk3_overlap_px > 0:
        overlap_region = binary_erosion(overlap_region, disk(args.erode_mapk3_overlap_px))

    if standard.empty:
        coincident = standard.copy()
        coincident["mapk3_cell_id"] = pd.Series(dtype="Int64")
        coincident["mapk3_nucleus_id"] = pd.Series(dtype="Int64")
        coincident["nucleus_iou"] = pd.Series(dtype=float)
        coincident["distance_to_mapk3_cell_centroid_px"] = pd.Series(dtype=float)
        return coincident, {
            "mapk3_cp_mask": str(mapk3_cp_path),
            "mapk3_nucleus_mask": "",
            "matching_method": "none",
        }

    rr = np.clip(np.rint(standard["centroid_y"]).astype(int), 0, output_shape[0] - 1)
    cc = np.clip(np.rint(standard["centroid_x"]).astype(int), 0, output_shape[1] - 1)
    mapk3_cell_ids = mapk3_cells_warped[rr, cc].astype(np.int64)
    keep = smndc1_cytoplasm[rr, cc] & overlap_region[rr, cc] & (mapk3_cell_ids > 0)

    coincident = standard.loc[keep].copy()
    coincident["mapk3_cell_id"] = mapk3_cell_ids[keep]
    coincident["mode"] = "coincident"

    mapk3_nucleus_path: Path | None = None
    matching_method = "cell_centroid"

    if args.match_mapk3_nuclei:
        mapk3_nucleus_path = find_mask_for_channel(mapk3_b, "nucleus")
        if mapk3_nucleus_path is not None:
            mapk3_nuclei = to_label_mask(read_2d(mapk3_nucleus_path))
            mapk3_nuclei = pad_or_crop_to_shape(mapk3_nuclei, output_shape)
            mapk3_nuclei_warped = warp_labels_mapk3_to_smndc1(
                mapk3_nuclei,
                transform,
                output_shape,
            )
            matches = match_nuclei_by_iou(
                smndc1_labels,
                mapk3_nuclei_warped,
                args.nucleus_iou_threshold,
            )
            if not matches.empty:
                coincident = coincident.merge(matches, on="nucleus_id", how="inner")
                coincident["distance_to_mapk3_cell_centroid_px"] = np.nan
                matching_method = "nucleus_iou"

    if matching_method != "nucleus_iou":
        if coincident.empty:
            coincident["mapk3_nucleus_id"] = pd.Series(dtype="Int64")
            coincident["nucleus_iou"] = pd.Series(dtype=float)
            coincident["distance_to_mapk3_cell_centroid_px"] = pd.Series(dtype=float)
        else:
            # Compute MAPK3 cell centroids after warping. This avoids applying
            # a second coordinate transform and guarantees that centroid
            # distances use exactly the same registered label image used for
            # overlap selection.
            props = regionprops_table(mapk3_cells_warped, properties=["label", "centroid"])
            mapk3_centroids = pd.DataFrame(
                {
                    "mapk3_cell_id": props["label"].astype(np.int64),
                    "registered_centroid_y": props["centroid-0"].astype(float),
                    "registered_centroid_x": props["centroid-1"].astype(float),
                }
            )
            coincident = coincident.merge(
                mapk3_centroids[
                    ["mapk3_cell_id", "registered_centroid_y", "registered_centroid_x"]
                ],
                on="mapk3_cell_id",
                how="left",
            )
            dy = coincident["centroid_y"] - coincident["registered_centroid_y"]
            dx = coincident["centroid_x"] - coincident["registered_centroid_x"]
            coincident["distance_to_mapk3_cell_centroid_px"] = np.sqrt(dy * dy + dx * dx)
            coincident = coincident[
                coincident["distance_to_mapk3_cell_centroid_px"]
                <= args.cell_centroid_max_distance_px
            ].copy()

            if args.one_nucleus_per_mapk3_cell and not coincident.empty:
                coincident = coincident.sort_values(
                    ["mapk3_cell_id", "distance_to_mapk3_cell_centroid_px"],
                    ascending=[True, True],
                )
                coincident = coincident.groupby("mapk3_cell_id", as_index=False).head(1).copy()

            coincident["mapk3_nucleus_id"] = pd.NA
            coincident["nucleus_iou"] = np.nan
            coincident = coincident.drop(
                columns=["registered_centroid_y", "registered_centroid_x"],
                errors="ignore",
            )

    return coincident, {
        "mapk3_cp_mask": str(mapk3_cp_path),
        "mapk3_nucleus_mask": str(mapk3_nucleus_path) if mapk3_nucleus_path else "",
        "matching_method": matching_method,
    }


def save_debug_overlay(
    outdir: Path,
    condition: str,
    core: str,
    standard: pd.DataFrame,
    coincident: pd.DataFrame,
    assets: dict[str, np.ndarray | Path],
    transform_dice_value: float,
) -> None:
    debug_dir = outdir / "debug_overlays" / condition
    debug_dir.mkdir(parents=True, exist_ok=True)

    background = np.asarray(assets["smndc1_g"], dtype=np.float32)
    finite = background[np.isfinite(background)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        background = np.clip((background - lo) / max(hi - lo, 1e-8), 0, 1)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(background, cmap="gray")
    cytoplasm = np.asarray(assets["smndc1_cytoplasm"], dtype=bool)
    ax.contour(cytoplasm.astype(float), levels=[0.5], linewidths=0.8)

    if not standard.empty:
        ax.scatter(
            standard["centroid_x"],
            standard["centroid_y"],
            s=5,
            alpha=0.25,
            label="standard",
        )
    if not coincident.empty:
        ax.scatter(
            coincident["centroid_x"],
            coincident["centroid_y"],
            s=10,
            alpha=0.85,
            label="registered",
        )

    dice_text = f"{transform_dice_value:.3f}" if np.isfinite(transform_dice_value) else "NA"
    ax.set_title(f"{condition} {core} | registration Dice={dice_text}")
    if not standard.empty or not coincident.empty:
        ax.legend(loc="upper right")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(debug_dir / f"{core}_smndc1_quantification_qc.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify nuclear SMNDC1 area fractions and fluorescence intensity "
            "for standard and MAPK3-registered TMA populations."
        )
    )
    parser.add_argument("--manifest_csv", required=True, help="Core/channel manifest produced by build_core_manifest.py.")
    parser.add_argument("--transforms_csv", required=True, help="MAPK3-to-SMNDC1 transform table from register_mapk3_smndc1.py.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--include_low_dice",
        action="store_true",
        help="Allow transforms labeled low_dice in addition to status=ok.",
    )
    parser.add_argument(
        "--min_registration_dice",
        type=float,
        default=0.0,
        help="Optional minimum accepted registration Dice for the registered subset.",
    )
    parser.add_argument(
        "--erode_mapk3_overlap_px",
        type=int,
        default=2,
        help="Erode the registered MAPK3 overlap mask before selecting coincident nuclei.",
    )
    parser.add_argument(
        "--cell_centroid_max_distance_px",
        type=float,
        default=60.0,
        help="Maximum SMNDC1-nucleus to transformed MAPK3-cell-centroid distance when nucleus IoU matching is unavailable.",
    )
    parser.add_argument(
        "--one_nucleus_per_mapk3_cell",
        action="store_true",
        help="In centroid-matching mode, retain only the closest SMNDC1 nucleus per MAPK3 cell.",
    )
    parser.add_argument(
        "--match_mapk3_nuclei",
        action="store_true",
        help="When MAPK3 nuclear masks exist, require one-to-one nucleus matching by IoU.",
    )
    parser.add_argument(
        "--nucleus_iou_threshold",
        type=float,
        default=0.15,
        help="Minimum IoU for optional MAPK3-SMNDC1 nucleus matching.",
    )
    parser.add_argument("--debug_visuals", action="store_true", help="Generate per-core QC overlays.")
    parser.add_argument("--max_cores", type=int, default=0, help="Limit the number of cores for testing (0 = all).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not (0.0 <= args.min_registration_dice <= 1.0):
        parser.error("--min_registration_dice must be between 0 and 1.")
    if not (0.0 <= args.nucleus_iou_threshold <= 1.0):
        parser.error("--nucleus_iou_threshold must be between 0 and 1.")
    if args.erode_mapk3_overlap_px < 0:
        parser.error("--erode_mapk3_overlap_px must be >= 0.")
    if args.cell_centroid_max_distance_px <= 0:
        parser.error("--cell_centroid_max_distance_px must be > 0.")
    if args.max_cores < 0:
        parser.error("--max_cores must be >= 0.")

    outdir = Path(args.outdir).resolve()
    tables_dir = outdir / "tables"
    reports_dir = outdir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest_csv)
    transforms = load_transforms(args.transforms_csv, args.include_low_dice)

    all_standard: list[pd.DataFrame] = []
    all_coincident: list[pd.DataFrame] = []
    standard_summaries: list[dict[str, object]] = []
    coincident_summaries: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    manifest_rows = list(manifest.iterrows())
    if args.max_cores > 0:
        manifest_rows = manifest_rows[: args.max_cores]

    for condition in ("Exon", "NoExon"):
        for _, manifest_row in manifest_rows:
            core = normalize_core(manifest_row["core_norm"])
            diagnostic: dict[str, object] = {
                "core_norm": core,
                "condition": condition,
                "standard_status": "not_run",
                "registered_status": "not_run",
            }

            try:
                standard, metadata, assets = process_standard_core(manifest_row, condition)
                diagnostic["standard_status"] = "ok"
                diagnostic["nuclei_standard"] = int(len(standard))
                diagnostic["smndc1_nucleus_mask"] = str(assets["nucleus_mask_path"])
                diagnostic["smndc1_cytoplasm_mask"] = str(assets["cytoplasm_mask_path"])
                diagnostic["smndc1_protein_mask"] = str(assets["protein_mask_path"])

                all_standard.append(standard)

                transform_row: pd.Series | None = None
                key = (condition, core)
                if key in transforms.index:
                    selected = transforms.loc[key]
                    if isinstance(selected, pd.DataFrame):
                        raise ValueError(f"Multiple transforms found for {condition} {core}.")
                    transform_row = selected

                dice_value = registration_dice(transform_row)
                diagnostic["registration_dice"] = dice_value
                metadata_registered = dict(metadata)
                metadata_registered["transform_dice"] = dice_value
                metadata_standard = dict(metadata)
                metadata_standard["transform_dice"] = dice_value

                standard["transform_dice"] = dice_value
                standard_summaries.append(core_summary(standard, metadata_standard, "standard"))

                if transform_row is None:
                    diagnostic["registered_status"] = "missing_or_rejected_transform"
                    diagnostics.append(diagnostic)
                    continue

                if np.isfinite(dice_value) and dice_value < args.min_registration_dice:
                    diagnostic["registered_status"] = "below_min_registration_dice"
                    diagnostics.append(diagnostic)
                    continue

                coincident, matching_info = process_coincident_core(
                    standard,
                    manifest_row,
                    condition,
                    transform_row,
                    assets,
                    args,
                )
                coincident["transform_dice"] = dice_value
                all_coincident.append(coincident)
                coincident_summaries.append(
                    core_summary(coincident, metadata_registered, "coincident")
                )

                diagnostic["registered_status"] = "ok"
                diagnostic["nuclei_coincident"] = int(len(coincident))
                diagnostic["fraction_coincident"] = (
                    float(len(coincident) / len(standard)) if len(standard) > 0 else np.nan
                )
                diagnostic.update(matching_info)

                if args.debug_visuals:
                    save_debug_overlay(
                        outdir,
                        condition,
                        core,
                        standard,
                        coincident,
                        assets,
                        dice_value,
                    )

            except Exception as exc:
                if diagnostic["standard_status"] == "not_run":
                    diagnostic["standard_status"] = f"error: {type(exc).__name__}: {exc}"
                else:
                    diagnostic["registered_status"] = f"error: {type(exc).__name__}: {exc}"

            diagnostics.append(diagnostic)

    standard_df = pd.concat(all_standard, ignore_index=True) if all_standard else pd.DataFrame()
    coincident_df = pd.concat(all_coincident, ignore_index=True) if all_coincident else pd.DataFrame()
    standard_core_df = pd.DataFrame(standard_summaries)
    coincident_core_df = pd.DataFrame(coincident_summaries)
    diagnostic_df = pd.DataFrame(diagnostics)

    standard_df.to_csv(tables_dir / "nuclei_features_standard.csv", index=False)
    coincident_df.to_csv(tables_dir / "nuclei_features_coincident.csv", index=False)
    standard_core_df.to_csv(tables_dir / "core_summary_standard.csv", index=False)
    coincident_core_df.to_csv(tables_dir / "core_summary_coincident.csv", index=False)
    diagnostic_df.to_csv(tables_dir / "diagnostics_per_core.csv", index=False)

    summary = {
        "manifest_csv": str(Path(args.manifest_csv).resolve()),
        "transforms_csv": str(Path(args.transforms_csv).resolve()),
        "n_standard_nuclei": int(len(standard_df)),
        "n_coincident_nuclei": int(len(coincident_df)),
        "n_standard_core_condition_records": int(len(standard_core_df)),
        "n_coincident_core_condition_records": int(len(coincident_core_df)),
        "standard_status_counts": diagnostic_df.get("standard_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "registered_status_counts": diagnostic_df.get("registered_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "definitions": {
            "smndc1_mean_intensity": "SMNDC1-channel intensity sum within a nucleus divided by nuclear area in pixels.",
            "smndc1_positive_area_fraction": "SMNDC1-positive pixels within a nucleus divided by total nuclear area in pixels.",
            "standard": "SMNDC1 nuclei with centroids inside the SMNDC1 cytoplasmic/tumor mask.",
            "coincident": "Standard SMNDC1 nuclei retained after spatial overlap/matching with registered MAPK3 masks.",
        },
    }
    with open(reports_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"[OK] Standard nuclei:   {len(standard_df)}")
    print(f"[OK] Coincident nuclei: {len(coincident_df)}")
    print(f"[OK] Tables: {tables_dir}")
    print(f"[OK] Report: {reports_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
