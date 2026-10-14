#!/usr/bin/env python3
"""Split multichannel IF TIFF images into analysis channels.

Channel convention
------------------
C0 -> nuclei (B)
C1 -> protein (G)
C2 -> cytoplasm / pan-cytokeratin (R)

The script always writes robustly scaled 8-bit channel images used by the
analysis pipeline. With ``--keep16``, the original per-channel data are also
written as 16-bit TIFF files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff


def select_largest_series(tf: tiff.TiffFile):
    """Return the largest TIFF series containing spatial Y and X axes."""
    best_series = None
    best_area = -1

    for series in tf.series:
        axes = series.axes
        shape = series.shape
        if "Y" not in axes or "X" not in axes:
            continue

        area = int(shape[axes.index("Y")]) * int(shape[axes.index("X")])
        if area > best_area:
            best_series = series
            best_area = area

    if best_series is None:
        raise RuntimeError("No TIFF series with Y and X axes was found.")

    return best_series


def to_yxc(array: np.ndarray, axes: str) -> np.ndarray:
    """Convert an input TIFF array to Y x X x C layout.

    Non-spatial dimensions other than channels are collapsed using a maximum
    projection, matching the behavior of the original analysis script.
    """
    arr = np.asarray(array)
    current_axes = list(axes)

    if "C" in current_axes:
        c_index = current_axes.index("C")
        arr = np.moveaxis(arr, c_index, -1)
        current_axes.pop(c_index)
        current_axes.append("C")

    while arr.ndim > 3:
        arr = arr.max(axis=0)
        if current_axes:
            current_axes.pop(0)

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.ndim != 3:
        raise RuntimeError(f"Unable to convert input to Y x X x C layout; shape={arr.shape}.")

    return arr


def robust_u01(channel: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Robustly scale one image channel to the [0, 1] interval."""
    ch = np.asarray(channel, dtype=np.float32)
    low, high = np.percentile(ch, [p_low, p_high])

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(ch))
        high = float(np.nanmax(ch))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.zeros_like(ch, dtype=np.float32)

    scaled = (ch - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def as_uint16(channel: np.ndarray) -> np.ndarray:
    """Return channel data as uint16 without altering values when possible."""
    ch = np.asarray(channel)
    if ch.dtype == np.uint16:
        return ch

    if np.issubdtype(ch.dtype, np.integer):
        return np.clip(ch, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    ch = np.nan_to_num(ch, nan=0.0, posinf=np.iinfo(np.uint16).max, neginf=0.0)
    return np.clip(ch, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def process_one(
    path: Path,
    out_b: Path,
    out_g: Path,
    out_r: Path,
    p_low: float,
    p_high: float,
    keep16: bool,
) -> None:
    """Split one multichannel TIFF into nuclear, protein, and cytoplasmic channels."""
    with tiff.TiffFile(path) as tf:
        series = select_largest_series(tf)
        array = series.asarray()
        axes = series.axes

    array = to_yxc(array, axes)
    n_channels = array.shape[-1]
    if n_channels < 3:
        raise RuntimeError(
            f"{path.name}: at least 3 channels are required; found {n_channels}."
        )

    b_raw = array[..., 0]
    g_raw = array[..., 1]
    r_raw = array[..., 2]
    stem = path.stem

    # Standard analysis outputs: always write robustly scaled 8-bit images.
    b8 = (robust_u01(b_raw, p_low, p_high) * 255.0).round().astype(np.uint8)
    g8 = (robust_u01(g_raw, p_low, p_high) * 255.0).round().astype(np.uint8)
    r8 = (robust_u01(r_raw, p_low, p_high) * 255.0).round().astype(np.uint8)

    tiff.imwrite(out_b / f"{stem}_B_nucleos8.tif", b8)
    tiff.imwrite(out_g / f"{stem}_G_proteina8.tif", g8)
    tiff.imwrite(out_r / f"{stem}_R_citoplasma8.tif", r8)

    # Optional archival outputs: retain original per-channel information in uint16.
    if keep16:
        tiff.imwrite(out_b / f"{stem}_B_nucleos_16bit.tif", as_uint16(b_raw))
        tiff.imwrite(out_g / f"{stem}_G_proteina_16bit.tif", as_uint16(g_raw))
        tiff.imwrite(out_r / f"{stem}_R_citoplasma_16bit.tif", as_uint16(r_raw))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split multichannel IF TIFF images into C0=nuclei, C1=protein, "
            "and C2=cytoplasm/pan-cytokeratin channels."
        )
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument(
        "--p_low",
        type=float,
        default=0.5,
        help="Lower percentile used for robust 8-bit scaling (default: 0.5).",
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=99.5,
        help="Upper percentile used for robust 8-bit scaling (default: 99.5).",
    )
    parser.add_argument(
        "--keep16",
        action="store_true",
        help="Also save the original three channels as uint16 TIFF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.p_low < args.p_high <= 100.0:
        raise ValueError("Require 0 <= p_low < p_high <= 100.")

    input_dir = args.input_dir.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    out_b = out_root / "CHAN_B_nucleosIF"
    out_g = out_root / "CHAN_G_proteinaIF"
    out_r = out_root / "CHAN_R_citoplasmaIF"
    for directory in (out_b, out_g, out_r):
        directory.mkdir(parents=True, exist_ok=True)

    files = sorted({*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")})
    if not files:
        raise FileNotFoundError(f"No .tif or .tiff files were found in {input_dir}")

    failures = 0
    for index, path in enumerate(files, start=1):
        try:
            process_one(
                path=path,
                out_b=out_b,
                out_g=out_g,
                out_r=out_r,
                p_low=args.p_low,
                p_high=args.p_high,
                keep16=args.keep16,
            )
            print(f"[{index}/{len(files)}] OK: {path.name}")
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(files)}] ERROR: {path.name}: {exc}")

    if failures:
        raise SystemExit(f"Completed with {failures} failed file(s).")


if __name__ == "__main__":
    main()
