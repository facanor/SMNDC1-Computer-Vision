#!/usr/bin/env python3
"""Register MAPK3 and SMNDC1 TMA sections and identify matched MAPK3 cells.

The workflow has two stages:

1. ``register``
   Registers MAPK3 (moving) to SMNDC1 (reference) using the pan-cytokeratin /
   cytoplasmic cell masks. Registration uses multistart ECC over a rotation
   sweep, optional phase-correlation translation initialization, optional
   coarse-to-fine refinement, and an optional affine fallback.

2. ``presence``
   Applies the accepted MAPK3-to-SMNDC1 transform to MAPK3 cell labels and
   measures the fraction of every target MAPK3 cell that falls inside the
   SMNDC1 tissue/cell mask. Cells meeting the requested overlap threshold are
   retained as spatially present in the SMNDC1 section.

The transform matrix stored in the output table maps MAPK3 coordinates into
SMNDC1 coordinates. OpenCV warping is therefore performed with
``WARP_INVERSE_MAP`` to preserve the convention used during ECC estimation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import morphology
from tifffile import imread
from tqdm import tqdm

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    phase_cross_correlation = None


CP_MASK_SUFFIX = "_cp_masks.tif"
CYTOPLASM_DIR = "CHAN_R_citoplasmaIF"


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_core(value: str | os.PathLike[str]) -> str:
    """Normalize a core identifier or filename to the ``row_column`` form."""
    basename = os.path.basename(str(value))
    match = re.search(r"^(.+?)\.ome", basename, flags=re.IGNORECASE)
    if match:
        core = match.group(1)
    else:
        core = os.path.splitext(basename)[0]
        core = re.sub(
            r"_(?:B_nucleos8|G_proteina8|R_citoplasma8)(?:_(?:nuc|cp|prot)_?masks?)?$",
            "",
            core,
            flags=re.IGNORECASE,
        )
    return core.replace("-", "_").strip()


def normalize_condition(value: object) -> str:
    """Normalize Exon4/NoExon4-like labels to ``Exon`` or ``NoExon``."""
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if text.startswith("noexon"):
        return "NoExon"
    if text.startswith("exon"):
        return "Exon"
    raise ValueError(f"Unrecognized condition label: {value!r}")


def list_cp_masks(channel_dir: str | os.PathLike[str]) -> list[str]:
    path = Path(channel_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Cytoplasmic-channel directory not found: {path}")
    return sorted(str(p) for p in path.iterdir() if p.is_file() and p.name.endswith(CP_MASK_SUFFIX))


def read_label_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Read a 2-D label image without silently selecting an arbitrary plane."""
    arr = np.asarray(imread(path))
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D label image, got shape {arr.shape}: {path}")
    return arr


def pad_or_crop_to_shape(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Top-left aligned crop/pad used by the original registration workflow."""
    target_h, target_w = shape_hw
    source_h, source_w = image.shape[:2]
    cropped = image[: min(source_h, target_h), : min(source_w, target_w)]
    out = np.zeros((target_h, target_w), dtype=cropped.dtype)
    out[: cropped.shape[0], : cropped.shape[1]] = cropped
    return out


def downsample_stride(image: np.ndarray, factor: int) -> np.ndarray:
    return image if factor <= 1 else image[::factor, ::factor]


def cleanup_binary(binary: np.ndarray, close_radius: int = 2, min_size: int = 128) -> np.ndarray:
    mask = binary.astype(bool)
    if close_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(int(close_radius)))
    if min_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_size))
    return mask.astype(np.uint8)


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    area_a = int(a_bool.sum())
    area_b = int(b_bool.sum())
    if area_a + area_b == 0:
        return 1.0
    intersection = int(np.logical_and(a_bool, b_bool).sum())
    return float(2.0 * intersection / (area_a + area_b))


# -----------------------------------------------------------------------------
# Affine transform conversion between image scales
# -----------------------------------------------------------------------------


def warp_2x3_to_3x3(warp: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2, :3] = np.asarray(warp, dtype=np.float32)
    return matrix


def warp_3x3_to_2x3(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:2, :3], dtype=np.float32)


def scale_matrix(factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Scale factor must be > 0.")
    return np.array(
        [[1.0 / factor, 0.0, 0.0], [0.0, 1.0 / factor, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def convert_warp_scale(warp: np.ndarray, k_from: int, k_to: int) -> np.ndarray:
    """Convert a warp from ``full/k_from`` coordinates to ``full/k_to``."""
    matrix = warp_2x3_to_3x3(warp)
    s_from = scale_matrix(k_from)
    s_to = scale_matrix(k_to)
    full = np.linalg.inv(s_from) @ matrix @ s_from
    target = s_to @ full @ np.linalg.inv(s_to)
    return warp_3x3_to_2x3(target)


def convert_warp_to_full(warp: np.ndarray, factor: int) -> np.ndarray:
    matrix = warp_2x3_to_3x3(warp)
    scale = scale_matrix(factor)
    full = np.linalg.inv(scale) @ matrix @ scale
    return warp_3x3_to_2x3(full)


# -----------------------------------------------------------------------------
# ECC registration helpers
# -----------------------------------------------------------------------------


def binary_to_ecc_image(binary: np.ndarray, mode: str = "dt") -> np.ndarray:
    mask = (binary > 0).astype(np.uint8)
    if mode == "binary":
        return mask.astype(np.float32)

    distance = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 3)
    maximum = float(distance.max()) if distance.size else 0.0
    if maximum > 0:
        distance = distance / maximum
    return 0.25 * mask.astype(np.float32) + 0.75 * distance.astype(np.float32)


def ecc_motion_flag(motion: str) -> int:
    if motion == "affine":
        return cv2.MOTION_AFFINE
    if motion == "euclidean":
        return cv2.MOTION_EUCLIDEAN
    raise ValueError(f"Unsupported ECC motion model: {motion}")


def ecc_register(
    template: np.ndarray,
    moving: np.ndarray,
    warp_init: np.ndarray | None = None,
    iterations: int = 200,
    epsilon: float = 1e-5,
    motion: str = "euclidean",
) -> tuple[np.ndarray, float]:
    warp = (
        np.asarray(warp_init, dtype=np.float32).copy()
        if warp_init is not None
        else np.eye(2, 3, dtype=np.float32)
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(iterations),
        float(epsilon),
    )
    correlation, warp = cv2.findTransformECC(
        np.asarray(template, dtype=np.float32),
        np.asarray(moving, dtype=np.float32),
        warp,
        ecc_motion_flag(motion),
        criteria,
        None,
        5,
    )
    return np.asarray(warp, dtype=np.float32), float(correlation)


def warp_affine_nearest(image: np.ndarray, warp: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    return cv2.warpAffine(
        image,
        np.asarray(warp, dtype=np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_affine_linear(image: np.ndarray, warp: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    return cv2.warpAffine(
        image,
        np.asarray(warp, dtype=np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    if phase_cross_correlation is None:
        raise RuntimeError(
            "--use_phasecorr_init requires skimage.registration.phase_cross_correlation."
        )
    shift, _, _ = phase_cross_correlation(
        reference.astype(np.float32),
        moving.astype(np.float32),
        upsample_factor=1,
    )
    dy, dx = shift
    return float(dy), float(dx)


def parse_angles(specification: str) -> list[float]:
    """Parse either ``start:stop:step`` or a comma-separated angle list."""
    specification = specification.strip()
    if not specification:
        raise ValueError("Angle specification cannot be empty.")

    if ":" in specification:
        parts = specification.split(":")
        if len(parts) != 3:
            raise ValueError("Angle range must have the form start:stop:step.")
        start, stop, step = map(float, parts)
        if step == 0:
            raise ValueError("Angle step cannot be zero.")
        if (stop - start) * step < 0:
            raise ValueError("Angle step has the wrong sign for the requested range.")
        count = int(np.floor((stop - start) / step)) + 1
        angles = [start + i * step for i in range(max(count, 0))]
    else:
        angles = [float(item) for item in specification.split(",") if item.strip()]

    if not angles:
        raise ValueError("No valid registration angles were provided.")
    return angles


def rotation_initialization(center_xy: tuple[float, float], angle_deg: float) -> np.ndarray:
    return cv2.getRotationMatrix2D(center_xy, angle_deg, 1.0).astype(np.float32)


def build_condition_pairs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    return [
        ("Exon", args.m_exon_root, args.s_exon_root),
        ("NoExon", args.m_noex_root, args.s_noex_root),
    ]


def mask_maps(mapk3_root: str, smndc1_root: str) -> tuple[dict[str, str], dict[str, str]]:
    mapk3_dir = Path(mapk3_root) / CYTOPLASM_DIR
    smndc1_dir = Path(smndc1_root) / CYTOPLASM_DIR
    mapk3_masks = list_cp_masks(mapk3_dir)
    smndc1_masks = list_cp_masks(smndc1_dir)
    return (
        {normalize_core(path): path for path in mapk3_masks},
        {normalize_core(path): path for path in smndc1_masks},
    )


def evaluate_downsample_dice(moving_binary: np.ndarray, reference_binary: np.ndarray, warp: np.ndarray) -> float:
    warped = warp_affine_nearest(
        (moving_binary * 255).astype(np.uint8),
        warp,
        reference_binary.shape,
    ) > 0
    return dice_coefficient(warped, reference_binary.astype(bool))


# -----------------------------------------------------------------------------
# Registration stage
# -----------------------------------------------------------------------------


def cmd_register(args: argparse.Namespace) -> None:
    output_tables = Path(args.outdir) / "tables"
    ensure_dir(output_tables)

    angles = parse_angles(args.angles_deg)
    if args.try_flip_180:
        angles = angles + [angle + 180.0 for angle in angles]
    angles = list(dict.fromkeys(round(angle, 10) for angle in angles))

    if args.downsample <= 0 or args.downsample_fine <= 0:
        raise ValueError("Downsample factors must be positive integers.")
    if not (0.0 <= args.min_dice_keep <= 1.0):
        raise ValueError("--min_dice_keep must be between 0 and 1.")
    if not (0.0 <= args.affine_fallback_lt <= 1.0):
        raise ValueError("--affine_fallback_lt must be between 0 and 1.")
    if args.use_phasecorr_init and phase_cross_correlation is None:
        raise RuntimeError(
            "--use_phasecorr_init was requested, but phase_cross_correlation is unavailable."
        )

    rows: list[dict[str, object]] = []

    for condition, mapk3_root, smndc1_root in build_condition_pairs(args):
        mapk3_map, smndc1_map = mask_maps(mapk3_root, smndc1_root)
        cores = sorted(set(mapk3_map).intersection(smndc1_map))
        if args.max_cores > 0:
            cores = cores[: args.max_cores]

        print(
            f"[register] {condition}: {len(cores)} common cores; "
            f"motion={args.motion}; angles={len(angles)}; "
            f"coarse_to_fine={args.coarse_to_fine}"
        )

        for core in tqdm(cores, desc=f"register {condition}", ncols=100):
            mapk3_path = mapk3_map[core]
            smndc1_path = smndc1_map[core]

            base_row: dict[str, object] = {
                "condition": condition,
                "core": core,
                "m_cp_mask": mapk3_path,
                "s_cp_mask": smndc1_path,
            }

            try:
                mapk3_labels = read_label_image(mapk3_path)
                smndc1_labels = read_label_image(smndc1_path)

                mapk3_binary = (mapk3_labels > 0).astype(np.uint8)
                smndc1_binary = (smndc1_labels > 0).astype(np.uint8)

                if not np.any(mapk3_binary):
                    raise ValueError("MAPK3 cytoplasmic mask contains no labeled pixels.")
                if not np.any(smndc1_binary):
                    raise ValueError("SMNDC1 cytoplasmic mask contains no labeled pixels.")

                reference_shape = smndc1_binary.shape
                original_mapk3_shape = mapk3_binary.shape
                mapk3_binary = pad_or_crop_to_shape(mapk3_binary, reference_shape)

                coarse_factor = int(args.downsample)
                fine_factor = int(args.downsample_fine) if args.coarse_to_fine else coarse_factor

                mapk3_coarse = cleanup_binary(
                    downsample_stride(mapk3_binary, coarse_factor),
                    close_radius=args.close_r,
                    min_size=args.min_size,
                )
                smndc1_coarse = cleanup_binary(
                    downsample_stride(smndc1_binary, coarse_factor),
                    close_radius=args.close_r,
                    min_size=args.min_size,
                )

                if not np.any(mapk3_coarse) or not np.any(smndc1_coarse):
                    raise ValueError(
                        "Foreground disappeared after downsampling/cleanup. "
                        "Review --downsample, --close_r, and --min_size."
                    )

                mapk3_ecc = binary_to_ecc_image(mapk3_coarse, mode=args.ecc_image)
                smndc1_ecc = binary_to_ecc_image(smndc1_coarse, mode=args.ecc_image)
                height, width = smndc1_coarse.shape
                center = (width / 2.0, height / 2.0)

                best: dict[str, object] = {
                    "dice": -1.0,
                    "cc": np.nan,
                    "warp": None,
                    "motion": args.motion,
                    "used_k": coarse_factor,
                    "initial_angle_deg": np.nan,
                }

                for angle in angles:
                    warp0 = rotation_initialization(center, angle)
                    if args.use_phasecorr_init:
                        rotated = warp_affine_linear(
                            (mapk3_ecc * 255).astype(np.uint8),
                            warp0,
                            smndc1_coarse.shape,
                        ).astype(np.float32) / 255.0
                        dy, dx = phase_correlation_shift(smndc1_ecc, rotated)
                        warp0[0, 2] += dx
                        warp0[1, 2] += dy

                    try:
                        candidate_warp, correlation = ecc_register(
                            smndc1_ecc,
                            mapk3_ecc,
                            warp_init=warp0,
                            iterations=args.ecc_iters,
                            epsilon=args.ecc_eps,
                            motion=args.motion,
                        )
                        candidate_dice = evaluate_downsample_dice(
                            mapk3_coarse,
                            smndc1_coarse,
                            candidate_warp,
                        )
                        if candidate_dice > float(best["dice"]):
                            best.update(
                                {
                                    "dice": candidate_dice,
                                    "cc": correlation,
                                    "warp": candidate_warp,
                                    "motion": args.motion,
                                    "used_k": coarse_factor,
                                    "initial_angle_deg": angle,
                                }
                            )
                    except cv2.error:
                        continue

                if best["warp"] is None:
                    rows.append(
                        {
                            **base_row,
                            "m_shape_h": int(original_mapk3_shape[0]),
                            "m_shape_w": int(original_mapk3_shape[1]),
                            "s_shape_h": int(reference_shape[0]),
                            "s_shape_w": int(reference_shape[1]),
                            "motion": args.motion,
                            "downsample_used": coarse_factor,
                            "dice_tumor_ds": np.nan,
                            "dice_tumor_fullres": np.nan,
                            "ecc_cc": np.nan,
                            "initial_angle_deg": np.nan,
                            "T00": np.nan,
                            "T01": np.nan,
                            "T02": np.nan,
                            "T10": np.nan,
                            "T11": np.nan,
                            "T12": np.nan,
                            "status": "registration_failed",
                        }
                    )
                    continue

                if args.coarse_to_fine and fine_factor != coarse_factor:
                    mapk3_fine = cleanup_binary(
                        downsample_stride(mapk3_binary, fine_factor),
                        close_radius=args.close_r,
                        min_size=args.min_size,
                    )
                    smndc1_fine = cleanup_binary(
                        downsample_stride(smndc1_binary, fine_factor),
                        close_radius=args.close_r,
                        min_size=args.min_size,
                    )
                    if np.any(mapk3_fine) and np.any(smndc1_fine):
                        mapk3_fine_ecc = binary_to_ecc_image(mapk3_fine, mode=args.ecc_image)
                        smndc1_fine_ecc = binary_to_ecc_image(smndc1_fine, mode=args.ecc_image)
                        fine_initialization = convert_warp_scale(
                            np.asarray(best["warp"]),
                            k_from=coarse_factor,
                            k_to=fine_factor,
                        )
                        try:
                            fine_warp, fine_correlation = ecc_register(
                                smndc1_fine_ecc,
                                mapk3_fine_ecc,
                                warp_init=fine_initialization,
                                iterations=args.ecc_iters_fine,
                                epsilon=args.ecc_eps,
                                motion=str(best["motion"]),
                            )
                            fine_dice = evaluate_downsample_dice(
                                mapk3_fine,
                                smndc1_fine,
                                fine_warp,
                            )
                            # The historical workflow uses the fine solution when
                            # ECC converges. Keep that behavior for reproducibility.
                            best.update(
                                {
                                    "warp": fine_warp,
                                    "cc": fine_correlation,
                                    "dice": fine_dice,
                                    "used_k": fine_factor,
                                }
                            )
                        except cv2.error:
                            pass

                if (
                    args.affine_fallback
                    and float(best["dice"]) < args.affine_fallback_lt
                    and args.motion != "affine"
                ):
                    fallback_factor = int(best["used_k"])
                    mapk3_fallback = cleanup_binary(
                        downsample_stride(mapk3_binary, fallback_factor),
                        close_radius=args.close_r,
                        min_size=args.min_size,
                    )
                    smndc1_fallback = cleanup_binary(
                        downsample_stride(smndc1_binary, fallback_factor),
                        close_radius=args.close_r,
                        min_size=args.min_size,
                    )
                    mapk3_fallback_ecc = binary_to_ecc_image(
                        mapk3_fallback, mode=args.ecc_image
                    )
                    smndc1_fallback_ecc = binary_to_ecc_image(
                        smndc1_fallback, mode=args.ecc_image
                    )
                    fallback_height, fallback_width = smndc1_fallback.shape
                    fallback_center = (fallback_width / 2.0, fallback_height / 2.0)
                    best_affine: dict[str, object] = {
                        "dice": -1.0,
                        "cc": np.nan,
                        "warp": None,
                        "motion": "affine",
                        "used_k": fallback_factor,
                        "initial_angle_deg": np.nan,
                    }

                    for angle in angles:
                        warp0 = rotation_initialization(fallback_center, angle)
                        if args.use_phasecorr_init:
                            rotated = warp_affine_linear(
                                (mapk3_fallback_ecc * 255).astype(np.uint8),
                                warp0,
                                smndc1_fallback.shape,
                            ).astype(np.float32) / 255.0
                            dy, dx = phase_correlation_shift(smndc1_fallback_ecc, rotated)
                            warp0[0, 2] += dx
                            warp0[1, 2] += dy
                        try:
                            affine_warp, correlation = ecc_register(
                                smndc1_fallback_ecc,
                                mapk3_fallback_ecc,
                                warp_init=warp0,
                                iterations=args.ecc_iters_affine,
                                epsilon=args.ecc_eps,
                                motion="affine",
                            )
                            affine_dice = evaluate_downsample_dice(
                                mapk3_fallback,
                                smndc1_fallback,
                                affine_warp,
                            )
                            if affine_dice > float(best_affine["dice"]):
                                best_affine.update(
                                    {
                                        "dice": affine_dice,
                                        "cc": correlation,
                                        "warp": affine_warp,
                                        "initial_angle_deg": angle,
                                    }
                                )
                        except cv2.error:
                            continue

                    if (
                        best_affine["warp"] is not None
                        and float(best_affine["dice"]) > float(best["dice"])
                    ):
                        best = best_affine

                warp_full = convert_warp_to_full(
                    np.asarray(best["warp"]),
                    int(best["used_k"]),
                )

                # A full-resolution overlap metric is recorded for QC while the
                # historical downsample Dice remains the acceptance metric.
                mapk3_full_aligned = warp_affine_nearest(
                    (mapk3_binary * 255).astype(np.uint8),
                    warp_full,
                    reference_shape,
                ) > 0
                fullres_dice = dice_coefficient(
                    mapk3_full_aligned,
                    smndc1_binary.astype(bool),
                )

                registration_status = (
                    "ok" if float(best["dice"]) >= args.min_dice_keep else "low_dice"
                )

                rows.append(
                    {
                        **base_row,
                        "m_shape_h": int(original_mapk3_shape[0]),
                        "m_shape_w": int(original_mapk3_shape[1]),
                        "s_shape_h": int(reference_shape[0]),
                        "s_shape_w": int(reference_shape[1]),
                        "motion": str(best["motion"]),
                        "downsample_used": int(best["used_k"]),
                        "dice_tumor_ds": float(best["dice"]),
                        "dice_tumor_fullres": float(fullres_dice),
                        "ecc_cc": float(best["cc"]) if np.isfinite(best["cc"]) else np.nan,
                        "initial_angle_deg": float(best["initial_angle_deg"])
                        if np.isfinite(best["initial_angle_deg"])
                        else np.nan,
                        "T00": float(warp_full[0, 0]),
                        "T01": float(warp_full[0, 1]),
                        "T02": float(warp_full[0, 2]),
                        "T10": float(warp_full[1, 0]),
                        "T11": float(warp_full[1, 1]),
                        "T12": float(warp_full[1, 2]),
                        "status": registration_status,
                    }
                )

            except Exception as exc:
                rows.append(
                    {
                        **base_row,
                        "dice_tumor_ds": np.nan,
                        "dice_tumor_fullres": np.nan,
                        "status": f"error: {type(exc).__name__}: {exc}",
                    }
                )

    timestamp = now_tag()
    output_csv = output_tables / f"transforms_panck_M_to_S_exhaustive_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    parameters = vars(args).copy()
    parameters.pop("func", None)
    parameters.pop("command", None)
    with open(output_tables / f"register_panck_params_exhaustive_{timestamp}.json", "w", encoding="utf-8") as handle:
        json.dump(parameters, handle, indent=2, default=str)

    print(f"[OK] Transform table: {output_csv}")


# -----------------------------------------------------------------------------
# Cell-presence stage
# -----------------------------------------------------------------------------


def validate_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def resolve_cp_mask(root: str, core: str, fallback: object) -> str | None:
    directory = Path(root) / CYTOPLASM_DIR
    candidates = [
        directory / f"{core}.ome_R_citoplasma8_cp_masks.tif",
        directory / f"{core}_R_citoplasma8_cp_masks.tif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if isinstance(fallback, str) and fallback and os.path.exists(fallback):
        return fallback
    return None


def cmd_presence(args: argparse.Namespace) -> None:
    output_tables = Path(args.outdir) / "tables"
    ensure_dir(output_tables)

    if not (0.0 <= args.presence_threshold <= 1.0):
        raise ValueError("--presence_threshold must be between 0 and 1.")

    transforms = pd.read_csv(args.transforms_csv)
    transform_required = {
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
    validate_columns(transforms, transform_required, args.transforms_csv)

    allowed_status = {"ok", "low_dice"} if args.include_low_dice else {"ok"}
    transforms = transforms[transforms["status"].isin(allowed_status)].copy()
    if transforms.empty:
        raise RuntimeError("No valid registration transforms remain after status filtering.")

    groups = pd.read_csv(args.groups_csv).copy()
    group_required = {
        "condition",
        "core_key",
        "cell_id",
        "group_ABC",
        "ratio_prot_cyto",
    }
    validate_columns(groups, group_required, args.groups_csv)

    groups["condition"] = groups["condition"].map(normalize_condition)
    groups["core"] = groups["core_key"].map(normalize_core)
    groups["group2"] = np.where(groups["group_ABC"].astype(str) == "A", "A", "BC")
    groups["cell_id"] = pd.to_numeric(groups["cell_id"], errors="coerce")
    groups["ratio_prot_cyto"] = pd.to_numeric(groups["ratio_prot_cyto"], errors="coerce")
    groups = groups.dropna(subset=["cell_id", "ratio_prot_cyto"]).copy()
    groups["cell_id"] = groups["cell_id"].astype(int)
    groups = groups[groups["cell_id"] > 0]
    groups = groups.drop_duplicates(["condition", "core", "cell_id"], keep="first")

    cells_by_core = {
        key: subset.sort_values("cell_id").reset_index(drop=True)
        for key, subset in groups.groupby(["condition", "core"], sort=False)
    }

    smndc1_roots = {"Exon": args.s_exon_root, "NoExon": args.s_noex_root}
    mapk3_roots = {"Exon": args.m_exon_root, "NoExon": args.m_noex_root}

    cell_rows: list[dict[str, object]] = []
    core_rows: list[dict[str, object]] = []

    for _, transform_row in tqdm(
        transforms.iterrows(), total=len(transforms), desc="presence", ncols=100
    ):
        try:
            condition = normalize_condition(transform_row["condition"])
            core = normalize_core(transform_row["core"])
        except ValueError as exc:
            core_rows.append(
                {
                    "condition": transform_row.get("condition", ""),
                    "core": transform_row.get("core", ""),
                    "status": f"error: {exc}",
                }
            )
            continue

        key = (condition, core)
        if key not in cells_by_core:
            core_rows.append(
                {
                    "condition": condition,
                    "core": core,
                    "status": "no_target_cells_in_groups_table",
                }
            )
            continue

        smndc1_cp = resolve_cp_mask(
            smndc1_roots[condition],
            core,
            transform_row.get("s_cp_mask"),
        )
        mapk3_cp = resolve_cp_mask(
            mapk3_roots[condition],
            core,
            transform_row.get("m_cp_mask"),
        )

        if smndc1_cp is None or mapk3_cp is None:
            core_rows.append(
                {
                    "condition": condition,
                    "core": core,
                    "status": "missing_masks",
                    "m_cp_mask": mapk3_cp,
                    "s_cp_mask": smndc1_cp,
                }
            )
            continue

        try:
            smndc1_labels = read_label_image(smndc1_cp)
            mapk3_labels = read_label_image(mapk3_cp)
        except Exception as exc:
            core_rows.append(
                {
                    "condition": condition,
                    "core": core,
                    "status": f"mask_read_error: {type(exc).__name__}: {exc}",
                }
            )
            continue

        smndc1_binary = (smndc1_labels > 0).astype(np.uint8)
        reference_shape = smndc1_binary.shape
        mapk3_labels = pad_or_crop_to_shape(mapk3_labels, reference_shape).astype(
            np.int32, copy=False
        )

        try:
            warp = np.array(
                [
                    [transform_row["T00"], transform_row["T01"], transform_row["T02"]],
                    [transform_row["T10"], transform_row["T11"], transform_row["T12"]],
                ],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            core_rows.append(
                {
                    "condition": condition,
                    "core": core,
                    "status": f"invalid_transform: {exc}",
                }
            )
            continue

        if not np.isfinite(warp).all():
            core_rows.append(
                {
                    "condition": condition,
                    "core": core,
                    "status": "invalid_transform_nonfinite",
                }
            )
            continue

        mapk3_in_smndc1 = warp_affine_nearest(mapk3_labels, warp, reference_shape)
        flat_labels = mapk3_in_smndc1.reshape(-1)
        max_id = int(flat_labels.max()) if flat_labels.size else 0
        if max_id <= 0:
            core_rows.append(
                {"condition": condition, "core": core, "status": "empty_after_warp"}
            )
            continue

        area_by_id = np.bincount(flat_labels, minlength=max_id + 1).astype(np.int64)
        overlap_by_id = np.bincount(
            flat_labels[smndc1_binary.reshape(-1).astype(bool)],
            minlength=max_id + 1,
        ).astype(np.int64)

        target_cells = cells_by_core[key].copy()
        target_cells = target_cells[target_cells["cell_id"] <= max_id].copy()
        if target_cells.empty:
            core_rows.append(
                {"condition": condition, "core": core, "status": "no_ids_in_warped_image"}
            )
            continue

        ids = target_cells["cell_id"].to_numpy(dtype=int)
        area_values = area_by_id[ids]
        overlap_values = overlap_by_id[ids]
        with np.errstate(divide="ignore", invalid="ignore"):
            presence_values = np.where(
                area_values > 0,
                overlap_values / area_values,
                0.0,
            )
        present_flags = presence_values >= float(args.presence_threshold)
        registration_dice = pd.to_numeric(
            pd.Series([transform_row.get("dice_tumor_ds", np.nan)]), errors="coerce"
        ).iloc[0]
        registration_dice_full = pd.to_numeric(
            pd.Series([transform_row.get("dice_tumor_fullres", np.nan)]), errors="coerce"
        ).iloc[0]

        for i, target_row in enumerate(target_cells.itertuples(index=False)):
            cell_rows.append(
                {
                    "condition": condition,
                    "core": core,
                    "cell_id_M": int(target_row.cell_id),
                    "group_ABC": target_row.group_ABC,
                    "group2": target_row.group2,
                    "ratio_M_cell": float(target_row.ratio_prot_cyto),
                    "presence": float(presence_values[i]),
                    "present": bool(present_flags[i]),
                    "area_px": int(area_values[i]),
                    "overlap_px": int(overlap_values[i]),
                    "dice_tumor_ds": float(registration_dice)
                    if pd.notna(registration_dice)
                    else np.nan,
                    "dice_tumor_fullres": float(registration_dice_full)
                    if pd.notna(registration_dice_full)
                    else np.nan,
                    "T00": float(transform_row["T00"]),
                    "T01": float(transform_row["T01"]),
                    "T02": float(transform_row["T02"]),
                    "T10": float(transform_row["T10"]),
                    "T11": float(transform_row["T11"]),
                    "T12": float(transform_row["T12"]),
                }
            )

        core_rows.append(
            {
                "condition": condition,
                "core": core,
                "n_cells_targets": int(len(target_cells)),
                "n_cells_present": int(present_flags.sum()),
                "frac_present": float(present_flags.mean()),
                "presence_threshold": float(args.presence_threshold),
                "dice_tumor_ds": float(registration_dice)
                if pd.notna(registration_dice)
                else np.nan,
                "dice_tumor_fullres": float(registration_dice_full)
                if pd.notna(registration_dice_full)
                else np.nan,
                "status": "ok",
            }
        )

    timestamp = now_tag()
    cells_output = output_tables / f"cells_present_M_in_S_{timestamp}.csv"
    cores_output = output_tables / f"core_presence_summary_{timestamp}.csv"
    pd.DataFrame(cell_rows).to_csv(cells_output, index=False)
    pd.DataFrame(core_rows).to_csv(cores_output, index=False)

    parameters = vars(args).copy()
    parameters.pop("func", None)
    parameters.pop("command", None)
    with open(output_tables / f"presence_cells_params_{timestamp}.json", "w", encoding="utf-8") as handle:
        json.dump(parameters, handle, indent=2, default=str)

    print(f"[OK] Cell-presence table: {cells_output}")
    print(f"[OK] Core summary: {cores_output}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def add_registration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--m_exon_root", required=True, help="MAPK3 Exon4 dataset root.")
    parser.add_argument("--m_noex_root", required=True, help="MAPK3 NoExon4 dataset root.")
    parser.add_argument("--s_exon_root", required=True, help="SMNDC1 Exon4 dataset root.")
    parser.add_argument("--s_noex_root", required=True, help="SMNDC1 NoExon4 dataset root.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--motion",
        choices=["euclidean", "affine"],
        default="euclidean",
        help="Primary ECC motion model (default: euclidean).",
    )
    parser.add_argument(
        "--ecc_image",
        choices=["dt", "binary"],
        default="dt",
        help="Image representation used by ECC (default: distance-transform blend).",
    )
    parser.add_argument(
        "--angles_deg",
        default="-25:25:2.5",
        help='Rotation sweep as "start:stop:step" or a comma-separated list.',
    )
    parser.add_argument(
        "--try_flip_180",
        action="store_true",
        help="Also test each initialization angle plus 180 degrees.",
    )
    parser.add_argument(
        "--use_phasecorr_init",
        action="store_true",
        help="Initialize translation by phase correlation after each rotation initialization.",
    )
    parser.add_argument("--downsample", type=int, default=8, help="Coarse downsample factor.")
    parser.add_argument(
        "--coarse_to_fine",
        action="store_true",
        help="Refine the best coarse solution at a smaller downsample factor.",
    )
    parser.add_argument(
        "--downsample_fine",
        type=int,
        default=4,
        help="Fine-stage downsample factor (default: 4).",
    )
    parser.add_argument("--ecc_iters", type=int, default=220)
    parser.add_argument("--ecc_iters_fine", type=int, default=160)
    parser.add_argument("--ecc_iters_affine", type=int, default=220)
    parser.add_argument("--ecc_eps", type=float, default=1e-5)
    parser.add_argument("--close_r", type=int, default=2, help="Binary-closing radius.")
    parser.add_argument(
        "--min_size",
        type=int,
        default=128,
        help="Minimum foreground-object size after downsampling.",
    )
    parser.add_argument(
        "--min_dice_keep",
        type=float,
        default=0.35,
        help="Minimum downsample Dice required for status=ok.",
    )
    parser.add_argument(
        "--affine_fallback",
        action="store_true",
        help="Try affine ECC when the primary registration remains below the fallback threshold.",
    )
    parser.add_argument(
        "--affine_fallback_lt",
        type=float,
        default=0.55,
        help="Trigger affine fallback when the current Dice is below this value.",
    )
    parser.add_argument(
        "--max_cores",
        type=int,
        default=0,
        help="Limit cores per condition for testing (0 = all).",
    )


def add_presence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--transforms_csv", required=True, help="Transform table generated by register.")
    parser.add_argument(
        "--groups_csv",
        required=True,
        help=(
            "MAPK3 cell table containing condition, core_key, cell_id, group_ABC, "
            "and ratio_prot_cyto."
        ),
    )
    parser.add_argument("--m_exon_root", required=True, help="MAPK3 Exon4 dataset root.")
    parser.add_argument("--m_noex_root", required=True, help="MAPK3 NoExon4 dataset root.")
    parser.add_argument("--s_exon_root", required=True, help="SMNDC1 Exon4 dataset root.")
    parser.add_argument("--s_noex_root", required=True, help="SMNDC1 NoExon4 dataset root.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--presence_threshold",
        type=float,
        default=0.50,
        help="Minimum fraction of warped MAPK3 cell area overlapping the SMNDC1 mask.",
    )
    parser.add_argument(
        "--include_low_dice",
        action="store_true",
        help="Include transforms labeled low_dice in addition to status=ok.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register MAPK3 to SMNDC1 TMA sections using cytoplasmic masks and "
            "identify MAPK3 cells spatially present in the SMNDC1 section."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser(
        "register",
        aliases=["register_panck"],
        help="Register MAPK3 cytoplasmic masks to SMNDC1 cytoplasmic masks.",
    )
    add_registration_arguments(register_parser)
    register_parser.set_defaults(func=cmd_register)

    presence_parser = subparsers.add_parser(
        "presence",
        aliases=["presence_cells"],
        help="Measure MAPK3 cell presence after applying accepted transforms.",
    )
    add_presence_arguments(presence_parser)
    presence_parser.set_defaults(func=cmd_presence)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
