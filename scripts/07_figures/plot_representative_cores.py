#!/usr/bin/env python3
"""Select and render representative MAPK3 TMA cores from canonical quantification tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import tifffile

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
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


def normalize_core(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    text = text.replace(".ome.tiff", "").replace(".ome.tif", "")
    text = text.replace(".tiff", "").replace(".tif", "")
    text = text.replace("-", "_")
    return text


def load_mapk3_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    require_columns(
        frame,
        [
            "condition",
            "core_norm",
            "cell_id",
            "ratio_M_cyto",
            "area_cyto_px",
            "mapk3_positive_area_px",
        ],
        "MAPK3 Global table",
    )
    out = frame.copy()
    out["condition"] = out["condition"].map(normalize_condition)
    out["core_norm"] = out["core_norm"].map(normalize_core)
    for column in ("cell_id", "ratio_M_cyto", "area_cyto_px", "mapk3_positive_area_px"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[
        out["condition"].isin(CONDITIONS)
        & out["core_norm"].notna()
        & np.isfinite(out["cell_id"])
        & np.isfinite(out["ratio_M_cyto"])
        & np.isfinite(out["area_cyto_px"])
        & np.isfinite(out["mapk3_positive_area_px"])
    ].copy()
    out["cell_id"] = out["cell_id"].astype(np.int64)
    return out


def aggregate_cores(frame: pd.DataFrame, min_cells: int, rank_by: str) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_cells": ("cell_id", "size"),
        "mean_ratio": ("ratio_M_cyto", "mean"),
        "median_ratio": ("ratio_M_cyto", "median"),
        "mean_positive_area_px": ("mapk3_positive_area_px", "mean"),
        "sum_positive_area_px": ("mapk3_positive_area_px", "sum"),
        "mean_cytoplasmic_area_px": ("area_cyto_px", "mean"),
    }
    if "CPID" in frame.columns:
        aggregations["CPID"] = ("CPID", "first")

    summary = (
        frame.groupby(["condition", "core_norm"], observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    summary = summary[summary["n_cells"] >= int(min_cells)].copy()

    rank_column = {
        "mean_ratio": "mean_ratio",
        "median_ratio": "median_ratio",
        "mean_positive_area": "mean_positive_area_px",
        "sum_positive_area": "sum_positive_area_px",
    }[rank_by]
    summary["rank_value"] = summary[rank_column]
    return summary


def select_extremes(summary: pd.DataFrame, k: int) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        subset = summary[summary["condition"] == condition].copy()
        if subset.empty:
            continue
        high = subset.nlargest(k, "rank_value").assign(selection="high")
        low = subset.nsmallest(k, "rank_value").assign(selection="low")
        selections.extend([high, low])
    if not selections:
        return pd.DataFrame(columns=list(summary.columns) + ["selection"])
    selected = pd.concat(selections, ignore_index=True)
    return selected.drop_duplicates(["condition", "core_norm", "selection"], keep="first")


def image_candidates(root: Path, core: str) -> list[Path]:
    tokens = {core, core.replace("_", "-")}
    hits: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = path.name.lower()
        if "mask" in name:
            continue
        if any(token.lower() in name for token in tokens):
            hits.add(path)
    return sorted(hits)


def image_score(path: Path, condition: str) -> tuple[int, int, str]:
    text = str(path).lower().replace("_", "").replace("-", "")
    desired = "noexon" if condition == "NoExon4" else "exon"
    opposite = "exon" if condition == "NoExon4" else "noexon"

    score = 0
    if desired in text:
        score -= 30
    if opposite in text:
        score += 30
    name = path.name.lower()
    for keyword in ("merge", "merged", "rgb", "composite", "overlay", "multistain"):
        if keyword in name:
            score -= 5
    return score, len(str(path)), str(path)


def find_core_image(root: Path, core: str, condition: str) -> Path | None:
    candidates = image_candidates(root, core)
    if not candidates:
        return None
    return min(candidates, key=lambda path: image_score(path, condition))


def find_cytoplasm_mask(root: Path, core: str) -> Path | None:
    channel_root = root / "CHAN_R_citoplasmaIF"
    if not channel_root.exists():
        return None
    tokens = (core, core.replace("_", "-"))
    suffixes = (
        "R_citoplasma8_cp_masks.tif",
        "R_citoplasma8_cp_masks.tiff",
        "cp_masks.tif",
        "cp_masks.tiff",
    )
    candidates: list[Path] = []
    for path in channel_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if any(token in name for token in tokens) and any(name.endswith(suffix) for suffix in suffixes):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: (len(path.name), str(path)))
    return candidates[0]


def read_2d_label_mask(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D label mask, got shape {array.shape}: {path}")
    return array


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(path))
    return np.asarray(iio.imread(path))


def stretch_u8(array: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    lo = float(np.percentile(values[finite], low))
    hi = float(np.percentile(values[finite], high))
    if not np.isfinite(lo):
        lo = float(np.nanmin(values))
    if not np.isfinite(hi):
        hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def to_rgb(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Convert a grayscale, RGB, or 3-channel TMA image to display RGB.

    Channel-first 3-channel images are interpreted as DAPI, pan-cytokeratin,
    BaseScope/protein and displayed as blue, green, red, respectively.
    """
    array = np.squeeze(np.asarray(image))

    while array.ndim > 3:
        array = np.max(array, axis=0)

    if array.ndim == 2:
        gray = stretch_u8(array, low, high)
        return np.stack([gray, gray, gray], axis=-1)

    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.shape[-1] in (3, 4):
        rgb = array[..., :3]
        if rgb.dtype == np.uint8:
            return rgb
        return np.stack([stretch_u8(rgb[..., i], low, high) for i in range(3)], axis=-1)

    if array.shape[0] >= 3:
        dapi = array[0]
        panck = array[1]
        protein = array[2]
        return np.stack(
            [
                stretch_u8(protein, low, high),
                stretch_u8(panck, low, high),
                stretch_u8(dapi, low, high),
            ],
            axis=-1,
        )

    raise ValueError(f"Could not identify image channels from shape {array.shape}")


def make_thumbnail(rgb: np.ndarray, max_dimension: int) -> tuple[np.ndarray, int]:
    height, width = rgb.shape[:2]
    stride = max(1, int(np.ceil(max(height, width) / max_dimension)))
    return rgb[::stride, ::stride, :], stride


def safe_crop(
    rgb: np.ndarray,
    center_x: float,
    center_y: float,
    window: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = rgb.shape[:2]
    crop_width = min(int(window), width)
    crop_height = min(int(window), height)
    x0 = int(round(center_x - crop_width / 2))
    y0 = int(round(center_y - crop_height / 2))
    x0 = int(np.clip(x0, 0, max(0, width - crop_width)))
    y0 = int(np.clip(y0, 0, max(0, height - crop_height)))
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    return rgb[y0:y1, x0:x1], (x0, y0, x1, y1)


def hotspot_cell(core_units: pd.DataFrame, metric: str) -> pd.Series | None:
    metric_column = "ratio_M_cyto" if metric == "ratio" else "mapk3_positive_area_px"
    valid = core_units[np.isfinite(core_units[metric_column])].copy()
    if valid.empty:
        return None
    return valid.loc[valid[metric_column].idxmax()]


def label_centroid(mask: np.ndarray, cell_id: int) -> tuple[float, float] | None:
    rows, cols = np.nonzero(mask == int(cell_id))
    if rows.size == 0:
        return None
    return float(cols.mean()), float(rows.mean())


def render_core(
    image_path: Path,
    mask_path: Path | None,
    hotspot_id: int | None,
    zoom_window: int,
    thumbnail_max_dimension: int,
    percentile_low: float,
    percentile_high: float,
) -> tuple[plt.Figure, dict[str, object]]:
    rgb = to_rgb(read_image(image_path), percentile_low, percentile_high)
    thumbnail, stride = make_thumbnail(rgb, thumbnail_max_dimension)

    mask = None
    centroid = None
    if mask_path is not None and hotspot_id is not None:
        mask = read_2d_label_mask(mask_path)
        centroid = label_centroid(mask, hotspot_id)

    if centroid is None:
        center_x = rgb.shape[1] / 2.0
        center_y = rgb.shape[0] / 2.0
    else:
        source_x, source_y = centroid
        scale_x = rgb.shape[1] / mask.shape[1]
        scale_y = rgb.shape[0] / mask.shape[0]
        center_x = source_x * scale_x
        center_y = source_y * scale_y

    zoom, (x0, y0, x1, y1) = safe_crop(rgb, center_x, center_y, zoom_window)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=200)
    axes[0].imshow(thumbnail)
    axes[0].add_patch(
        Rectangle(
            (x0 / stride, y0 / stride),
            (x1 - x0) / stride,
            (y1 - y0) / stride,
            fill=False,
            linewidth=1.8,
            edgecolor="yellow",
        )
    )
    axes[0].set_title("Full TMA core")
    axes[0].axis("off")

    axes[1].imshow(zoom)
    axes[1].set_title("Representative hotspot")
    axes[1].axis("off")
    fig.tight_layout()

    metadata = {
        "image_height_px": int(rgb.shape[0]),
        "image_width_px": int(rgb.shape[1]),
        "hotspot_x_image_px": float(center_x),
        "hotspot_y_image_px": float(center_y),
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "hotspot_centroid_from_mask": centroid is not None,
    }
    return fig, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank MAPK3 TMA cores from mapk3_global.csv and render full-core and "
            "hotspot views from the original multistain images."
        )
    )
    parser.add_argument("--mapk3_global_csv", required=True, type=Path)
    parser.add_argument("--m_exon_root", required=True, type=Path)
    parser.add_argument("--m_noex_root", required=True, type=Path)
    parser.add_argument("--tma_cores_root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--k", type=int, default=3, help="Number of high and low cores per condition.")
    parser.add_argument("--min_cells_per_core", type=int, default=50)
    parser.add_argument(
        "--rank_by",
        choices=("mean_ratio", "median_ratio", "mean_positive_area", "sum_positive_area"),
        default="mean_ratio",
    )
    parser.add_argument("--hotspot_by", choices=("ratio", "positive_area"), default="ratio")
    parser.add_argument("--zoom_window", type=int, default=1024)
    parser.add_argument("--thumbnail_max_dimension", type=int, default=2000)
    parser.add_argument("--percentile_low", type=float, default=1.0)
    parser.add_argument("--percentile_high", type=float, default=99.0)
    parser.add_argument("--image_format", choices=("png", "pdf", "svg"), default="png")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.k < 1:
        raise ValueError("--k must be at least 1.")
    if args.min_cells_per_core < 1:
        raise ValueError("--min_cells_per_core must be at least 1.")
    if args.zoom_window < 1 or args.thumbnail_max_dimension < 1:
        raise ValueError("Image dimensions must be positive.")
    if not (0.0 <= args.percentile_low < args.percentile_high <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100.")

    figures_dir = args.outdir / "figures"
    tables_dir = args.outdir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    units = load_mapk3_table(args.mapk3_global_csv)
    summary = aggregate_cores(units, args.min_cells_per_core, args.rank_by)
    selected = select_extremes(summary, args.k)
    if selected.empty:
        raise RuntimeError("No cores met the selection criteria.")

    records: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        condition = str(selected_row["condition"])
        core = str(selected_row["core_norm"])
        selection = str(selected_row["selection"])

        source_image = find_core_image(args.tma_cores_root, core, condition)
        mapk3_root = args.m_exon_root if condition == "Exon4" else args.m_noex_root
        cytoplasm_mask = find_cytoplasm_mask(mapk3_root, core)

        core_units = units[(units["condition"] == condition) & (units["core_norm"] == core)]
        hotspot = hotspot_cell(core_units, args.hotspot_by)
        hotspot_id = int(hotspot["cell_id"]) if hotspot is not None else None
        hotspot_value = (
            float(hotspot["ratio_M_cyto"])
            if hotspot is not None and args.hotspot_by == "ratio"
            else float(hotspot["mapk3_positive_area_px"])
            if hotspot is not None
            else np.nan
        )

        record = selected_row.to_dict()
        record.update(
            {
                "source_image": str(source_image) if source_image else "",
                "cytoplasm_mask": str(cytoplasm_mask) if cytoplasm_mask else "",
                "hotspot_cell_id": hotspot_id,
                "hotspot_metric": args.hotspot_by,
                "hotspot_value": hotspot_value,
                "status": "not_rendered",
                "figure": "",
            }
        )

        if source_image is None:
            record["status"] = "source_image_not_found"
            records.append(record)
            continue

        try:
            fig, rendering = render_core(
                image_path=source_image,
                mask_path=cytoplasm_mask,
                hotspot_id=hotspot_id,
                zoom_window=args.zoom_window,
                thumbnail_max_dimension=args.thumbnail_max_dimension,
                percentile_low=args.percentile_low,
                percentile_high=args.percentile_high,
            )
            cpid = selected_row.get("CPID", None)
            cpid_text = "" if cpid is None or pd.isna(cpid) else f" | CPID {cpid}"
            fig.suptitle(
                f"MAPK3 {condition} | core {core}{cpid_text} | {selection} {args.rank_by}",
                fontsize=12,
            )
            output_name = f"MAPK3_{condition}_{selection}_{core}.{args.image_format}"
            output_path = figures_dir / output_name
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)

            record.update(rendering)
            record["status"] = "ok"
            record["figure"] = str(output_path)
        except Exception as exc:
            record["status"] = f"error: {type(exc).__name__}: {exc}"
        records.append(record)

    rendered = pd.DataFrame(records)
    summary.to_csv(tables_dir / "mapk3_core_summary.csv", index=False)
    rendered.to_csv(tables_dir / "mapk3_representative_cores.csv", index=False)

    n_ok = int((rendered["status"] == "ok").sum())
    print(f"[OK] Selected cores: {len(rendered)}")
    print(f"[OK] Rendered cores: {n_ok}")
    print(f"[OK] Tables: {tables_dir}")
    print(f"[OK] Figures: {figures_dir}")


if __name__ == "__main__":
    main()
