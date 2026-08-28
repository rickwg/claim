"""
Visualize the synthetic lesion generation pipeline.

Creates a 5-column figure showing the diffusion inpainting process:
(a) Healthy mammogram
(b) Inpainting region (rectangular mask overlay)
(c) Synthetic lesion (composited result)
(d) Pixel difference
(e) Ground truth mask (contour on composited)

Usage:
    uv run python -m visualizations.main --config config/visualize_pipeline.yaml
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from loguru import logger

from analyses._colors import (
    DISTORTION_COLOR,
    LESION_ARTIFACT,
    LESION_COLOR,
    artifact_color,
)

INSET_FRAME_COLOR = "#5a6472"

SUPPORTED_MODES = ("lesion", "combined", "boxed_samples", "overview", "regimes")
EXCLUDE_DIRS = {"masks", "ground_truth"}
EXCLUDE_FILES = {"config.yaml", "dataset_metadata.jsonl"}
COLUMN_TITLES = [
    "(a) Healthy",
    "(b) Inpainting region",
    "(c) Synthetic lesion",
    "(d) Pixel difference",
    "(e) Ground truth mask",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_matching_samples(
    preprocessed_dir: Path, synthetic_dir: Path
) -> list[tuple[str, str]]:
    prep_studies = {
        d.name for d in preprocessed_dir.iterdir() if d.is_dir()
    }
    synth_studies = {
        d.name
        for d in synthetic_dir.iterdir()
        if d.is_dir() and d.name not in EXCLUDE_DIRS
    }
    common_studies = sorted(prep_studies & synth_studies)

    samples = []
    for study_id in common_studies:
        prep_images = {f.name for f in (preprocessed_dir / study_id).glob("*.png")}
        synth_images = {f.name for f in (synthetic_dir / study_id).glob("*.png")}
        for image_name in sorted(prep_images & synth_images):
            samples.append((study_id, image_name))

    logger.info(
        f"Discovered {len(samples)} matching samples across {len(common_studies)} studies"
    )
    return samples


def load_sample(
    preprocessed_dir: Path, synthetic_dir: Path, study_id: str, image_name: str
) -> dict | None:
    paths = {
        "original": preprocessed_dir / study_id / image_name,
        "composited": synthetic_dir / study_id / image_name,
        "inpainting_mask": synthetic_dir / "masks" / study_id / image_name,
        "ground_truth": synthetic_dir / "ground_truth" / study_id / image_name,
    }

    images = {}
    for key, path in paths.items():
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Failed to load {key}: {path}")
            return None
        images[key] = img

    images["study_id"] = study_id
    images["image_name"] = image_name
    return images


def discover_combined_records(synthetic_dir: Path) -> list[dict]:
    metadata_path = synthetic_dir / "dataset_metadata.jsonl"
    if not metadata_path.exists():
        logger.error(f"No dataset_metadata.jsonl in {synthetic_dir}")
        return []

    records: list[dict] = []
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    combined = [r for r in records if r.get("variant") == "combined"]
    combined.sort(key=lambda r: (r["subject_id"], r["image_id"]))
    logger.info(
        f"Discovered {len(combined)} combined records in {metadata_path.name}"
    )
    return combined


def load_combined_sample(
    record: dict, preprocessed_dir: Path, synthetic_dir: Path
) -> dict | None:
    subject_id = record["subject_id"]
    image_id = record["image_id"]
    source_image_id = record["source_image_id"]

    paths = {
        "original": preprocessed_dir / subject_id / f"{source_image_id}.png",
        "combined": synthetic_dir / "combined" / subject_id / f"{image_id}.png",
        "ground_truth": (
            synthetic_dir / "ground_truth" / "combined" / subject_id / f"{image_id}.png"
        ),
        "distortion_ground_truth": (
            synthetic_dir
            / "distortion_ground_truth"
            / "combined"
            / subject_id
            / f"{image_id}.png"
        ),
    }

    images: dict = {}
    for key, path in paths.items():
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Failed to load {key}: {path}")
            return None
        images[key] = img

    images["lesion_bbox"] = (
        int(record["lesion_x1"]),
        int(record["lesion_y1"]),
        int(record["lesion_x2"]),
        int(record["lesion_y2"]),
    )
    images["distortion_center"] = (
        int(record["distortion_center_x"]),
        int(record["distortion_center_y"]),
    )
    images["distortion_radius"] = int(record["distortion_radius"])
    images["classification_target"] = record.get("classification_target", "lesion")
    images["subject_id"] = subject_id
    images["image_id"] = image_id
    return images


def extract_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def compute_zoom_extent(
    mask_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    zoom_padding: int = 40,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = mask_bbox
    h, w = image_shape
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_size = max(x2 - x1, y2 - y1) / 2 + zoom_padding
    return (
        max(0, cx - half_size),
        min(w, cx + half_size),
        min(h, cy + half_size),
        max(0, cy - half_size),
    )


def add_zoom_inset(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    inset_bounds: tuple[float, float, float, float] = (0.55, 0.55, 0.42, 0.42),
) -> plt.Axes:
    inset = ax.inset_axes(inset_bounds)
    inset.set_xlim(extent[0], extent[1])
    inset.set_ylim(extent[2], extent[3])
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor(INSET_FRAME_COLOR)
        spine.set_linewidth(1.5)
    _, connectors = ax.indicate_inset_zoom(
        inset, edgecolor=INSET_FRAME_COLOR, linewidth=1.0
    )
    for connector in connectors:
        connector.set_visible(False)

    ylim = ax.get_ylim()
    rect_top_axes = (extent[3] - ylim[0]) / (ylim[1] - ylim[0])
    rect_bottom_axes = (extent[2] - ylim[0]) / (ylim[1] - ylim[0])
    inset_bottom_axes = inset_bounds[1]
    inset_top_axes = inset_bounds[1] + inset_bounds[3]

    rect_tl = (extent[0], extent[3])
    rect_tr = (extent[1], extent[3])
    rect_br = (extent[1], extent[2])
    rect_bl = (extent[0], extent[2])
    inset_tl = (0.0, 1.0)
    inset_tr = (1.0, 1.0)
    inset_br = (1.0, 0.0)
    inset_bl = (0.0, 0.0)

    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if inset_top_axes > rect_top_axes:
        edges.append((rect_tl, inset_tl))
        edges.append((rect_tr, inset_tr))
    if inset_bottom_axes < rect_bottom_axes:
        edges.append((rect_br, inset_br))
        edges.append((rect_bl, inset_bl))

    # for rect_xy_data, inset_xy_axes in edges:
    #     conn = patches.ConnectionPatch(
    #         xyA=rect_xy_data, coordsA=ax.transData,
    #         xyB=inset_xy_axes, coordsB=inset.transAxes,
    #         edgecolor="yellow", linewidth=1.0,
    #     )
    #     ax.add_patch(conn)
    return inset


def plot_pipeline(samples: list[dict], output_path: Path, config: dict) -> None:
    panel_w, panel_h = config.get("figsize_per_panel", [4, 4])
    dpi = config.get("dpi", 300)
    mask_alpha = config.get("mask_overlay_alpha", 0.2)
    contour_color = config.get("contour_color", LESION_COLOR)
    contour_lw = config.get("contour_linewidth", 1.5)
    zoom_padding = config.get("zoom_padding", 40)

    num_rows = len(samples)
    num_cols = 5
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * panel_w, num_rows * panel_h)
    )
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        original = sample["original"]
        composited = sample["composited"]
        mask = sample["inpainting_mask"]
        gt = sample["ground_truth"]
        mask_bbox = extract_mask_bbox(mask)
        x1, y1, x2, y2 = mask_bbox
        zoom = compute_zoom_extent(mask_bbox, original.shape, zoom_padding)

        axes[row, 0].imshow(original, cmap="gray")

        axes[row, 1].imshow(original, cmap="gray")
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            facecolor=LESION_COLOR,
            alpha=mask_alpha,
            edgecolor=LESION_COLOR,
            linewidth=1.5,
        )
        axes[row, 1].add_patch(rect)
        inset = add_zoom_inset(axes[row, 1], zoom)
        inset.imshow(original, cmap="gray")
        inset_rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            facecolor=LESION_COLOR,
            alpha=mask_alpha,
            edgecolor=LESION_COLOR,
            linewidth=1.5,
        )
        inset.add_patch(inset_rect)

        axes[row, 2].imshow(composited, cmap="gray")
        inset = add_zoom_inset(axes[row, 2], zoom)
        inset.imshow(composited, cmap="gray")

        diff = cv.absdiff(original, composited)
        axes[row, 3].imshow(diff, cmap="magma")
        inset = add_zoom_inset(axes[row, 3], zoom)
        inset.imshow(diff, cmap="magma")

        axes[row, 4].imshow(composited, cmap="gray")
        contours, _ = cv.findContours(gt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for contour in contours:
            xs = contour[:, 0, 0]
            ys = contour[:, 0, 1]
            axes[row, 4].plot(
                np.append(xs, xs[0]),
                np.append(ys, ys[0]),
                color=contour_color,
                linewidth=contour_lw,
            )
        inset = add_zoom_inset(axes[row, 4], zoom)
        inset.imshow(composited, cmap="gray")
        for contour in contours:
            xs = contour[:, 0, 0]
            ys = contour[:, 0, 1]
            inset.plot(
                np.append(xs, xs[0]),
                np.append(ys, ys[0]),
                color=contour_color,
                linewidth=contour_lw,
            )

    for col, title in enumerate(COLUMN_TITLES):
        axes[0, col].set_title(title)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved pipeline figure: {output_path}")


COMBINED_PIPELINE_TITLES = [
    "(a) Healthy",
    "(b) Inpainting + distortion regions",
    "(c) Combined synthetic",
    "(d) Pixel difference",
    "(e) Ground truth masks",
]


def _make_region_patch(
    target: str,
    lesion_bbox: tuple[int, int, int, int],
    distortion_center: tuple[int, int],
    distortion_radius: int,
    *,
    facecolor,
    edgecolor,
    alpha: float,
    linewidth: float,
) -> patches.Patch:
    if target == LESION_ARTIFACT:
        x1, y1, x2, y2 = lesion_bbox
        return patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
    return patches.Circle(
        distortion_center,
        distortion_radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
    )


def _discriminator_bbox(
    target: str,
    lesion_bbox: tuple[int, int, int, int],
    distortion_center: tuple[int, int],
    distortion_radius: int,
) -> tuple[int, int, int, int]:
    if target == LESION_ARTIFACT:
        return lesion_bbox
    cx, cy = distortion_center
    r = distortion_radius
    return (cx - r, cy - r, cx + r, cy + r)


def plot_combined_pipeline(
    samples: list[dict], output_path: Path, config: dict
) -> None:
    panel_w, panel_h = config.get("figsize_per_panel", [4, 4])
    dpi = config.get("dpi", 300)
    mask_alpha = config.get("mask_overlay_alpha", 0.2)
    contour_color = config.get("contour_color", LESION_COLOR)
    distortion_color = config.get("distortion_color", DISTORTION_COLOR)
    contour_lw = config.get("contour_linewidth", 1.5)

    num_rows = len(samples)
    num_cols = 5
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * panel_w, num_rows * panel_h)
    )
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        original = sample["original"]
        combined = sample["combined"]
        gt = sample["ground_truth"]
        dist_gt = sample["distortion_ground_truth"]
        target = sample["classification_target"]
        x1, y1, x2, y2 = sample["lesion_bbox"]

        axes[row, 0].imshow(original, cmap="gray")

        axes[row, 1].imshow(original, cmap="gray")
        axes[row, 1].add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                facecolor=LESION_COLOR,
                alpha=mask_alpha,
                edgecolor=LESION_COLOR,
                linewidth=1.5,
            )
        )
        axes[row, 1].add_patch(
            patches.Circle(
                sample["distortion_center"],
                sample["distortion_radius"],
                facecolor=distortion_color,
                alpha=mask_alpha,
                edgecolor=distortion_color,
                linewidth=1.5,
            )
        )

        axes[row, 2].imshow(combined, cmap="gray")

        diff = cv.absdiff(original, combined)
        axes[row, 3].imshow(diff, cmap="magma")

        axes[row, 4].imshow(combined, cmap="gray")
        if target == LESION_ARTIFACT:
            lesion_contours, _ = cv.findContours(
                gt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
        else:
            lesion_contours = []
            axes[row, 4].add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    facecolor="none",
                    edgecolor=contour_color,
                    linewidth=contour_lw,
                )
            )
        for contour in lesion_contours:
            xs = contour[:, 0, 0]
            ys = contour[:, 0, 1]
            axes[row, 4].plot(
                np.append(xs, xs[0]),
                np.append(ys, ys[0]),
                color=contour_color,
                linewidth=contour_lw,
            )
        dist_contours, _ = cv.findContours(
            dist_gt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )
        for contour in dist_contours:
            xs = contour[:, 0, 0]
            ys = contour[:, 0, 1]
            axes[row, 4].plot(
                np.append(xs, xs[0]),
                np.append(ys, ys[0]),
                color=distortion_color,
                linewidth=contour_lw,
            )

    for col, title in enumerate(COMBINED_PIPELINE_TITLES):
        axes[0, col].set_title(title)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined pipeline figure: {output_path}")


COMPOSITING_TITLES = [
    "(a) Inpainting mask",
    "(b) Distance transform",
    "(c) Feathered blend mask",
    r"(d) original $\times$ (1 - blend)",
    r"(e) matched $\times$ blend",
]

GROUND_TRUTH_TITLES = [
    "(a) Healthy",
    "(b) Composited",
    "(c) Pixel difference",
    "(d) Ground truth",
]


def compute_crop(
    mask_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = mask_bbox
    h, w = image_shape
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2 + padding
    return (
        max(0, cx - half),
        max(0, cy - half),
        min(w, cx + half),
        min(h, cy + half),
    )


def plot_compositing_details(
    samples: list[dict], output_path: Path, config: dict
) -> None:
    panel_w, panel_h = config.get("figsize_per_panel", [4, 4])
    dpi = config.get("dpi", 300)
    taper_pixels = config.get("taper_pixels", 15)
    crop_padding = config.get("crop_padding", 30)

    num_rows = len(samples)
    num_cols = 5
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * panel_w, num_rows * panel_h)
    )
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        original = sample["original"]
        composited = sample["composited"]
        mask = sample["inpainting_mask"]

        mask_bbox = extract_mask_bbox(mask)
        cx1, cy1, cx2, cy2 = compute_crop(mask_bbox, original.shape, crop_padding)

        binary_crop = mask[cy1:cy2, cx1:cx2]
        axes[row, 0].imshow(binary_crop, cmap="gray", vmin=0, vmax=255)

        dist = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
        dist_crop = dist[cy1:cy2, cx1:cx2]
        axes[row, 1].imshow(dist_crop, cmap="magma")

        feathered = np.zeros_like(dist, dtype=np.float64)
        feathered[dist >= taper_pixels] = 1.0
        taper_zone = (dist > 0) & (dist < taper_pixels)
        feathered[taper_zone] = 0.5 * (
            1 - np.cos(np.pi * dist[taper_zone] / taper_pixels)
        )
        feathered_crop = feathered[cy1:cy2, cx1:cx2]
        axes[row, 2].imshow(feathered_crop, cmap="magma", vmin=0, vmax=1)

        original_f = original.astype(np.float64)
        composited_f = composited.astype(np.float64)
        original_contribution = original_f * (1 - feathered)
        matched_contribution = composited_f - original_contribution

        orig_contrib_crop = original_contribution[cy1:cy2, cx1:cx2]
        axes[row, 3].imshow(orig_contrib_crop, cmap="gray", vmin=0, vmax=255)

        matched_contrib_crop = matched_contribution[cy1:cy2, cx1:cx2]
        axes[row, 4].imshow(matched_contrib_crop, cmap="gray", vmin=0, vmax=255)

    for col, title in enumerate(COMPOSITING_TITLES):
        axes[0, col].set_title(title)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved compositing detail figure: {output_path}")


def plot_ground_truth_details(
    samples: list[dict], output_path: Path, config: dict
) -> None:
    panel_w, panel_h = config.get("figsize_per_panel", [4, 4])
    dpi = config.get("dpi", 300)
    crop_padding = config.get("crop_padding", 30)

    num_rows = len(samples)
    num_cols = 4
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * panel_w, num_rows * panel_h)
    )
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        original = sample["original"]
        composited = sample["composited"]

        mask_bbox = extract_mask_bbox(sample["inpainting_mask"])
        cx1, cy1, cx2, cy2 = compute_crop(mask_bbox, original.shape, crop_padding)

        original_crop = original[cy1:cy2, cx1:cx2]
        axes[row, 0].imshow(original_crop, cmap="gray")

        composited_crop = composited[cy1:cy2, cx1:cx2]
        axes[row, 1].imshow(composited_crop, cmap="gray")

        diff = cv.absdiff(original, composited)
        diff_crop = diff[cy1:cy2, cx1:cx2]
        axes[row, 2].imshow(diff_crop, cmap="magma")

        gt_crop = sample["ground_truth"][cy1:cy2, cx1:cx2]
        axes[row, 3].imshow(gt_crop, cmap="gray", vmin=0, vmax=255)

    for col, title in enumerate(GROUND_TRUTH_TITLES):
        axes[0, col].set_title(title)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ground truth detail figure: {output_path}")


ATTRIBUTION_TITLES = [
    "(a) Inpainting region",
    "(b) Synthetic lesion",
    "(c) Pixel difference",
    "(d) Attribution heatmap",
]


def generate_illustrative_attribution(
    gt_mask: np.ndarray, image: np.ndarray
) -> np.ndarray:
    attribution = gt_mask.astype(np.float64) / 255.0
    attribution = cv.GaussianBlur(attribution, (0, 0), sigmaX=25)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, attribution.shape)
    noise = cv.GaussianBlur(noise, (0, 0), sigmaX=20)
    noise = np.maximum(noise, 0)
    if noise.max() > 0:
        noise = noise / noise.max()
    attribution = attribution + noise * 0.25
    image_mask = (image > 5).astype(np.float64)
    image_mask = cv.GaussianBlur(image_mask, (0, 0), sigmaX=10)
    attribution = attribution * image_mask
    if attribution.max() > 0:
        attribution = attribution / attribution.max()
    return attribution


def overlay_attribution(
    attribution: np.ndarray,
    cmap_name: str = "magma",
    alpha_scale: float = 0.75,
) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(attribution)
    rgba[..., 3] = np.clip(attribution * alpha_scale, 0, 1)
    return rgba


def plot_attribution_illustration(
    samples: list[dict], output_path: Path, config: dict
) -> None:
    panel_w, panel_h = config.get("figsize_per_panel", [4, 4])
    dpi = config.get("dpi", 300)
    mask_alpha = config.get("mask_overlay_alpha", 0.3)
    zoom_padding = config.get("zoom_padding", 40)

    sample = samples[-1]
    original = sample["original"]
    composited = sample["composited"]
    mask = sample["inpainting_mask"]
    gt = sample["ground_truth"]
    mask_bbox = extract_mask_bbox(mask)
    x1, y1, x2, y2 = mask_bbox
    zoom = compute_zoom_extent(mask_bbox, original.shape, zoom_padding)

    num_cols = 4
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * panel_w, panel_h))

    axes[0].imshow(original, cmap="gray")
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        facecolor=LESION_COLOR,
        alpha=mask_alpha,
        edgecolor=LESION_COLOR,
        linewidth=1.5,
    )
    axes[0].add_patch(rect)
    inset = add_zoom_inset(axes[0], zoom)
    inset.imshow(original, cmap="gray")
    inset_rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        facecolor=LESION_COLOR,
        alpha=mask_alpha,
        edgecolor=LESION_COLOR,
        linewidth=1.5,
    )
    inset.add_patch(inset_rect)

    axes[1].imshow(composited, cmap="gray")
    inset = add_zoom_inset(axes[1], zoom)
    inset.imshow(composited, cmap="gray")

    diff = cv.absdiff(original, composited)
    axes[2].imshow(diff, cmap="magma")
    inset = add_zoom_inset(axes[2], zoom)
    inset.imshow(diff, cmap="magma")

    attribution = generate_illustrative_attribution(gt, composited)
    overlay = overlay_attribution(attribution)
    axes[3].imshow(composited, cmap="gray")
    axes[3].imshow(overlay)
    axes[3].add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            facecolor="none",
            edgecolor=LESION_COLOR,
            linewidth=1.5,
        )
    )
    inset = add_zoom_inset(axes[3], zoom)
    inset.imshow(composited, cmap="gray")
    inset.imshow(overlay)
    inset.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            facecolor="none",
            edgecolor=LESION_COLOR,
            linewidth=1.5,
        )
    )

    for col, title in enumerate(ATTRIBUTION_TITLES):
        axes[col].set_title(title)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved attribution illustration figure: {output_path}")


def plot_combined_attribution_illustration(
    samples: list[dict], output_path: Path, config: dict
) -> None:
    panel_w, panel_h = config.get("figsize_per_panel", [4, 4])
    dpi = config.get("dpi", 300)
    mask_alpha = config.get("mask_overlay_alpha", 0.3)
    zoom_padding = config.get("zoom_padding", 40)

    sample = samples[-1]
    original = sample["original"]
    combined = sample["combined"]
    gt = sample["ground_truth"]
    target = sample["classification_target"]
    lesion_bbox = sample["lesion_bbox"]
    distortion_center = sample["distortion_center"]
    distortion_radius = sample["distortion_radius"]

    discr_bbox = _discriminator_bbox(
        target, lesion_bbox, distortion_center, distortion_radius
    )
    zoom = compute_zoom_extent(discr_bbox, original.shape, zoom_padding)

    num_cols = 4
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * panel_w, panel_h))

    axes[0].imshow(original, cmap="gray")
    axes[0].add_patch(
        _make_region_patch(
            target, lesion_bbox, distortion_center, distortion_radius,
            facecolor=artifact_color(target), edgecolor=artifact_color(target),
            alpha=mask_alpha, linewidth=1.5,
        )
    )
    inset = add_zoom_inset(axes[0], zoom)
    inset.imshow(original, cmap="gray")
    inset.add_patch(
        _make_region_patch(
            target, lesion_bbox, distortion_center, distortion_radius,
            facecolor=artifact_color(target), edgecolor=artifact_color(target),
            alpha=mask_alpha, linewidth=1.5,
        )
    )

    axes[1].imshow(combined, cmap="gray")
    inset = add_zoom_inset(axes[1], zoom)
    inset.imshow(combined, cmap="gray")

    diff = cv.absdiff(original, combined)
    axes[2].imshow(diff, cmap="magma")
    inset = add_zoom_inset(axes[2], zoom)
    inset.imshow(diff, cmap="magma")

    attribution = generate_illustrative_attribution(gt, combined)
    overlay = overlay_attribution(attribution)
    axes[3].imshow(combined, cmap="gray")
    axes[3].imshow(overlay)
    axes[3].add_patch(
        _make_region_patch(
            target, lesion_bbox, distortion_center, distortion_radius,
            facecolor="none", edgecolor=artifact_color(target), alpha=1.0, linewidth=1.5,
        )
    )
    inset = add_zoom_inset(axes[3], zoom)
    inset.imshow(combined, cmap="gray")
    inset.imshow(overlay)
    inset.add_patch(
        _make_region_patch(
            target, lesion_bbox, distortion_center, distortion_radius,
            facecolor="none", edgecolor=artifact_color(target), alpha=1.0, linewidth=1.5,
        )
    )

    for col, title in enumerate(ATTRIBUTION_TITLES):
        axes[col].set_title(title)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined attribution illustration: {output_path}")


REGION_BOX_COLORS = {"lesion": LESION_COLOR, "distortion": DISTORTION_COLOR}

LESION_ABSENT_VARIANTS = {"distortion_only"}
DISTORTION_ABSENT_VARIANTS = {"lesion_only"}


def load_jsonl_records(metadata_path: Path) -> list[dict]:
    records: list[dict] = []
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_region_boxes(
    record: dict,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    variant = record.get("variant")
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []

    if "lesion_x1" in record and variant not in LESION_ABSENT_VARIANTS:
        boxes.append(
            (
                "lesion",
                (
                    int(record["lesion_x1"]),
                    int(record["lesion_y1"]),
                    int(record["lesion_x2"]),
                    int(record["lesion_y2"]),
                ),
            )
        )

    if variant not in DISTORTION_ABSENT_VARIANTS:
        if "distortion_center_x" in record:
            center_x = int(record["distortion_center_x"])
            center_y = int(record["distortion_center_y"])
            radius = int(record["distortion_radius"])
        elif "lesion_center_x" in record:
            center_x = int(record["lesion_center_x"])
            center_y = int(record["lesion_center_y"])
            radius = int(record["lesion_radius"])
        else:
            return boxes
        boxes.append(
            (
                "distortion",
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
            )
        )

    return boxes


def render_boxed_sample(
    image: np.ndarray,
    boxes: list[tuple[str, tuple[int, int, int, int]]],
    output_path: Path,
    config: dict,
) -> None:
    box_colors = {**REGION_BOX_COLORS, **(config.get("box_colors") or {})}
    box_linewidth = config.get("box_linewidth", 3.0)
    upscale = config.get("upscale", 2)
    render_dpi = 100

    height, width = image.shape
    fig = plt.figure(
        figsize=(width * upscale / render_dpi, height * upscale / render_dpi),
        dpi=render_dpi,
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(
        image, cmap="gray", vmin=0, vmax=255, extent=(0.0, width, height, 0.0)
    )
    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    ax.set_axis_off()

    for region, (x1, y1, x2, y2) in boxes:
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                facecolor="none",
                edgecolor=box_colors.get(region, artifact_color(region)),
                linewidth=box_linewidth,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=render_dpi)
    plt.close(fig)


def run_boxed_samples(config: dict, output_dir: Path) -> None:
    project_dir = Path(config.get("project_dir", "."))
    num_images = config.get("num_images", 10)
    seed = config.get("seed", 42)

    for dataset in config["datasets"]:
        name = dataset["name"]
        metadata_path = Path(dataset["metadata_path"])
        variant_filter = dataset.get("variant_filter")

        if not metadata_path.exists():
            logger.warning(f"[{name}] metadata not found: {metadata_path}")
            continue

        records = load_jsonl_records(metadata_path)
        if variant_filter is not None:
            records = [r for r in records if r.get("variant") == variant_filter]

        if not records:
            logger.warning(f"[{name}] no records to sample")
            continue

        rng = random.Random(f"{seed}:{name}")
        count = len(records) if num_images is None else min(num_images, len(records))
        selected = rng.sample(records, count)

        dataset_output_dir = output_dir / name
        saved = 0
        for record in selected:
            image_path = project_dir / record["image_path"]
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"[{name}] failed to load {image_path}")
                continue
            boxes = extract_region_boxes(record)
            output_path = dataset_output_dir / f"{record['image_id']}.png"
            render_boxed_sample(image, boxes, output_path, config)
            saved += 1

        logger.info(
            f"[{name}] saved {saved} boxed samples "
            f"(of {len(records)} available) to {dataset_output_dir}"
        )


def _reduce_attribution(attribution_path: Path) -> np.ndarray:
    attribution = np.abs(np.load(attribution_path)).astype(np.float64)
    if attribution.ndim == 3:
        attribution = attribution.sum(axis=0)
    return attribution


def _find_attribution_file(
    attribution_dir: Path, method: str, image_id: str
) -> Path | None:
    matches = sorted((attribution_dir / method).glob(f"nan-{image_id}-test-*.npy"))
    return matches[0] if matches else None


def _select_overview_record(metadata_path: Path, image_id: str | None) -> dict | None:
    for record in load_jsonl_records(metadata_path):
        if record.get("variant") != "combined":
            continue
        if image_id is None or record.get("image_id") == image_id:
            return record
    return None


def _normalized_mass(attribution: np.ndarray, mask: np.ndarray) -> float:
    total = attribution.sum()
    return float(attribution[mask > 0].sum() / total) if total > 0 else 0.0


def _mean_density(attribution: np.ndarray, mask: np.ndarray) -> float:
    pixels = int((mask > 0).sum())
    return float(attribution[mask > 0].sum() / pixels) if pixels > 0 else 0.0


def plot_overview(config: dict, output_path: Path) -> None:
    project_root = Path(config.get("project_dir", "."))
    metadata_path = Path(config["combined_metadata_path"])
    attribution_dir = Path(config["attribution_dir"])
    method = config.get("attribution_method", "integrated_gradient")
    image_id = config.get("image_id")
    discriminator_color = config.get("discriminator_color", LESION_COLOR)
    distractor_color = config.get("distractor_color", DISTORTION_COLOR)
    inset_color = config.get("inset_color", INSET_FRAME_COLOR)
    ink = config.get("ink_color", "#33404f")
    muted = config.get("muted_color", "#5a6472")
    dpi = config.get("dpi", 300)
    figure_width, figure_height = config.get("figsize", [7.6, 3.2])
    font_scale = config.get("font_scale", 1.1)

    record = _select_overview_record(metadata_path, image_id)
    if record is None:
        logger.error(f"No combined record for image_id={image_id} in {metadata_path}")
        return
    image_id = record["image_id"]

    def resolve(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else project_root / path

    source = cv.imread(str(resolve(record["source_image_path"])), cv.IMREAD_GRAYSCALE)
    combined = cv.imread(str(resolve(record["image_path"])), cv.IMREAD_GRAYSCALE)
    lesion_rect = cv.imread(
        str(resolve(record["lesion_ground_truth_rect_mask_path"])), cv.IMREAD_GRAYSCALE
    )
    distortion_rect = cv.imread(
        str(resolve(record["distortion_ground_truth_rect_mask_path"])),
        cv.IMREAD_GRAYSCALE,
    )
    attribution_file = _find_attribution_file(attribution_dir, method, image_id)
    if attribution_file is None:
        logger.error(f"No {method} attribution for {image_id} under {attribution_dir}")
        return
    attribution = _reduce_attribution(attribution_file)

    lx1, ly1, lx2, ly2 = (
        record["lesion_x1"], record["lesion_y1"], record["lesion_x2"], record["lesion_y2"]
    )
    dcx, dcy, dradius = (
        record["distortion_center_x"],
        record["distortion_center_y"],
        record["distortion_radius"],
    )

    breast = source > 10
    normalized = attribution / attribution.sum()
    mass_discriminator = _normalized_mass(attribution, lesion_rect)
    mass_distractor = _normalized_mass(attribution, distortion_rect)
    preference = mass_discriminator / (mass_discriminator + mass_distractor)
    surround = breast & (lesion_rect == 0) & (distortion_rect == 0)
    ri_discriminator = _mean_density(normalized, lesion_rect) / _mean_density(normalized, surround)
    ri_distractor = _mean_density(normalized, distortion_rect) / _mean_density(normalized, surround)

    saturation = np.percentile(attribution[breast], 99.8)
    heat = np.clip(attribution / saturation, 0, 1) ** 0.8

    def heat_overlay(intensity: np.ndarray) -> np.ndarray:
        rgba = plt.get_cmap("magma")(intensity)
        rgba[..., 3] = np.clip((intensity - 0.18) / 0.82, 0, 1) * 0.92
        return rgba

    ys, xs = np.where(breast)
    bx1, bx2 = max(int(xs.min()) - 8, 0), min(int(xs.max()) + 8, source.shape[1])
    by1, by2 = max(int(ys.min()) - 8, 0), min(int(ys.max()) + 8, source.shape[0])
    pad = 12
    zx1 = max(min(lx1, dcx - dradius) - pad, 0)
    zx2 = min(max(lx2, dcx + dradius) + pad, source.shape[1])
    zy1 = max(min(ly1, dcy - dradius) - pad, 0)
    zy2 = min(max(ly2, dcy + dradius) + pad, source.shape[0])

    fig = plt.figure(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("white")
    row_bottom, row_height = 0.10, 0.76
    crop_aspect = (bx2 - bx1) / (by2 - by1)
    image_width = crop_aspect * row_height * figure_height / figure_width
    classifier_width, score_width, gap = 0.13, 0.19, 0.014
    x_healthy = (1.0 - (3 * image_width + classifier_width + score_width + 4 * gap)) / 2.0
    x_combined = x_healthy + image_width + gap
    x_classifier = x_combined + image_width + gap
    x_attribute = x_classifier + classifier_width + gap
    x_score = x_attribute + image_width + gap

    def image_panel(position: list[float], panel_title: str):
        ax = fig.add_axes(position)
        ax.set_xlim(bx1, bx2)
        ax.set_ylim(by2, by1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(panel_title, fontsize=11, pad=5)
        return ax

    def discriminator_patch(linewidth: float = 2.0):
        return patches.Rectangle(
            (lx1, ly1), lx2 - lx1, ly2 - ly1,
            fill=False, edgecolor=discriminator_color, linewidth=linewidth,
        )

    def distractor_patch(linewidth: float = 2.0):
        return patches.Rectangle(
            (dcx - dradius, dcy - dradius), 2 * dradius, 2 * dradius,
            fill=False, edgecolor=distractor_color, linewidth=linewidth,
        )

    healthy_ax = image_panel([x_healthy, row_bottom, image_width, row_height], "Healthy")
    healthy_ax.imshow(source, cmap="gray", vmin=0, vmax=255)

    combined_ax = image_panel([x_combined, row_bottom, image_width, row_height], "Synthesize")
    combined_ax.imshow(combined, cmap="gray", vmin=0, vmax=255)
    combined_ax.add_patch(discriminator_patch())
    combined_ax.add_patch(distractor_patch())

    classifier_ax = fig.add_axes([x_classifier, row_bottom, classifier_width, row_height])
    classifier_ax.axis("off")
    classifier_ax.set_xlim(0, 1)
    classifier_ax.set_ylim(0, 1)
    classifier_ax.set_title("Train", fontsize=11, pad=5)
    for block_x, block_height in zip([0.06, 0.29, 0.52, 0.75], [0.70, 0.56, 0.42, 0.28]):
        classifier_ax.add_patch(
            patches.FancyBboxPatch(
                (block_x, 0.5 - block_height / 2), 0.17, block_height,
                boxstyle="round,pad=0.005", facecolor="#dfe6f2",
                edgecolor="#6b7d99", linewidth=1.0,
            )
        )
    classifier_ax.text(0.5, 0.94, "ConvNeXt-Tiny", ha="center", fontsize=8.5, color=ink)
    classifier_ax.text(
        0.5, 0.06, "healthy vs. unhealthy", ha="center", fontsize=7, style="italic", color=muted
    )

    attribution_ax = image_panel(
        [x_attribute, row_bottom, image_width, row_height], "Attribute"
    )
    attribution_ax.imshow(combined, cmap="gray", vmin=0, vmax=255)
    attribution_ax.imshow(heat_overlay(heat))
    attribution_ax.add_patch(discriminator_patch())
    attribution_ax.add_patch(distractor_patch())
    zoom_ax = attribution_ax.inset_axes([0.50, 0.02, 0.48, 0.30])
    zoom_ax.imshow(combined, cmap="gray", vmin=0, vmax=255)
    zoom_ax.imshow(heat_overlay(heat))
    zoom_ax.set_xlim(zx1, zx2)
    zoom_ax.set_ylim(zy2, zy1)
    zoom_ax.add_patch(discriminator_patch(1.6))
    zoom_ax.add_patch(distractor_patch(1.6))
    zoom_ax.set_xticks([])
    zoom_ax.set_yticks([])
    for spine in zoom_ax.spines.values():
        spine.set_color(inset_color)
        spine.set_linewidth(1.4)
    attribution_ax.indicate_inset_zoom(zoom_ax, edgecolor=inset_color, alpha=0.45)

    score_ax = fig.add_axes([x_score, row_bottom, score_width, row_height])
    score_ax.axis("off")
    score_ax.set_xlim(0, 1)
    score_ax.set_ylim(0, 1)
    score_ax.set_title("Score", fontsize=11, pad=5)

    def rect_crop(mask: np.ndarray) -> np.ndarray:
        mask_ys, mask_xs = np.where(mask > 0)
        crop_pad = 6
        return combined[
            max(int(mask_ys.min()) - crop_pad, 0): int(mask_ys.max()) + crop_pad,
            max(int(mask_xs.min()) - crop_pad, 0): int(mask_xs.max()) + crop_pad,
        ]

    discriminator_crop_ax = score_ax.inset_axes([0.02, 0.66, 0.47, 0.28])
    discriminator_crop_ax.imshow(rect_crop(lesion_rect), cmap="gray")
    discriminator_crop_ax.set_xticks([])
    discriminator_crop_ax.set_yticks([])
    for spine in discriminator_crop_ax.spines.values():
        spine.set_color(discriminator_color)
        spine.set_linewidth(2.4)
    distractor_crop_ax = score_ax.inset_axes([0.51, 0.66, 0.47, 0.28])
    distractor_crop_ax.imshow(rect_crop(distortion_rect), cmap="gray")
    distractor_crop_ax.set_xticks([])
    distractor_crop_ax.set_yticks([])
    for spine in distractor_crop_ax.spines.values():
        spine.set_color(distractor_color)
        spine.set_linewidth(2.4)

    score_ax.text(0.5, 0.46, "relative importance", ha="center", fontsize=8.5, color=ink)
    score_ax.text(0.46, 0.24, f"{ri_discriminator:.1f}×", ha="right", fontsize=14,
                  color=discriminator_color, weight="bold")
    score_ax.text(0.50, 0.255, "/", ha="center", fontsize=11, color=muted)
    score_ax.text(0.54, 0.24, f"{ri_distractor:.1f}×", ha="left", fontsize=14,
                  color=distractor_color, weight="bold")

    arrow_gaps = [
        (x_healthy + image_width, x_combined),
        (x_combined + image_width, x_classifier),
        (x_classifier + classifier_width, x_attribute),
        (x_attribute + image_width, x_score),
    ]
    arrow_y = row_bottom + row_height / 2.0
    for gap_start, gap_end in arrow_gaps:
        fig.add_artist(
            patches.FancyArrowPatch(
                (gap_start + 0.004, arrow_y),
                (gap_end - 0.004, arrow_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=12,
                color="#8a94a3",
                lw=2.0,
            )
        )

    for text_object in fig.findobj(lambda obj: hasattr(obj, "set_fontsize")):
        text_object.set_fontsize(text_object.get_fontsize() * font_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)
    logger.info(
        f"Saved overview figure: {output_path} "
        f"(preference={preference:.3f}, ri_disc={ri_discriminator:.2f}, ri_dist={ri_distractor:.2f})"
    )


REGIME_BOX_COLORS = {"lesion": LESION_COLOR, "distortion": DISTORTION_COLOR}


def _load_gray(path_value: str, project_root: Path) -> np.ndarray | None:
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return cv.imread(str(path), cv.IMREAD_GRAYSCALE)


def _regime_boxes(record: dict, kind: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    if kind in ("lesion", "combined"):
        boxes.append(
            ("lesion", (record["lesion_x1"], record["lesion_y1"], record["lesion_x2"], record["lesion_y2"]))
        )
    if kind == "distortion":
        cx, cy, r = record["lesion_center_x"], record["lesion_center_y"], record["lesion_radius"]
        boxes.append(("distortion", (cx - r, cy - r, cx + r, cy + r)))
    if kind == "combined":
        cx, cy, r = record["distortion_center_x"], record["distortion_center_y"], record["distortion_radius"]
        boxes.append(("distortion", (cx - r, cy - r, cx + r, cy + r)))
    return boxes


def _boxes_union(boxes: list[tuple[str, tuple[int, int, int, int]]]) -> tuple[int, int, int, int]:
    xs1 = min(b[1][0] for b in boxes)
    ys1 = min(b[1][1] for b in boxes)
    xs2 = max(b[1][2] for b in boxes)
    ys2 = max(b[1][3] for b in boxes)
    return xs1, ys1, xs2, ys2


def _select_regime_record(spec: dict, project_root: Path, scan_limit: int = 150) -> dict | None:
    records = [r for r in load_jsonl_records(Path(spec["metadata_path"])) if r.get("label") == 1]
    if spec["kind"] == "combined":
        records = [r for r in records if r.get("variant") == "combined"]
    if spec.get("image_id"):
        for r in records:
            if r.get("image_id") == spec["image_id"]:
                return r
    prefer_close = spec["kind"] == "combined"
    best, best_score = None, None
    for record in records[:scan_limit]:
        boxes = _regime_boxes(record, spec["kind"])
        source = _load_gray(record["source_image_path"], project_root)
        if source is None:
            continue
        height, width = source.shape
        margins = []
        in_tissue = True
        for _, (bx1, by1, bx2, by2) in boxes:
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            if not (0 <= cx < width and 0 <= cy < height and source[cy, cx] > 10):
                in_tissue = False
                break
            margins.append(min(bx1, by1, width - bx2, height - by2))
        if not in_tissue or min(margins) < 12:
            continue
        xs1, ys1, xs2, ys2 = _boxes_union(boxes)
        score = max(xs2 - xs1, ys2 - ys1) if prefer_close else -min(margins)
        if best_score is None or score < best_score:
            best_score, best = score, record
    return best


def _square_crop(
    boxes: list[tuple[str, tuple[int, int, int, int]]], factor: float, width: int, height: int
) -> tuple[float, float, float, float]:
    xs1, ys1, xs2, ys2 = _boxes_union(boxes)
    cx, cy = (xs1 + xs2) / 2.0, (ys1 + ys2) / 2.0
    half = max(xs2 - xs1, ys2 - ys1) * factor / 2.0
    x1, x2, y1, y2 = cx - half, cx + half, cy - half, cy + half
    if x1 < 0:
        x2 -= x1; x1 = 0.0
    if y1 < 0:
        y2 -= y1; y1 = 0.0
    if x2 > width:
        x1 -= x2 - width; x2 = float(width)
    if y2 > height:
        y1 -= y2 - height; y2 = float(height)
    return max(x1, 0.0), x2, max(y1, 0.0), y2


def plot_regimes(config: dict, output_path: Path) -> None:
    project_root = Path(config.get("project_dir", "."))
    regimes = config["regimes"]
    crop_factor = config.get("crop_factor", 2.6)
    colors = {
        "lesion": config.get("lesion_color", REGIME_BOX_COLORS["lesion"]),
        "distortion": config.get("distortion_color", REGIME_BOX_COLORS["distortion"]),
    }
    ink = config.get("ink_color", "#33404f")
    dpi = config.get("dpi", 300)
    row_labels = ["Source", "Manipulated"]

    panels = []
    for spec in regimes:
        record = _select_regime_record(spec, project_root)
        if record is None:
            logger.error(f"No well-placed record for regime {spec['label']}")
            return
        source = _load_gray(record["source_image_path"], project_root)
        manipulated = _load_gray(record["image_path"], project_root)
        if source is None or manipulated is None:
            logger.error(f"Failed to load images for regime {spec['label']}")
            return
        boxes = _regime_boxes(record, spec["kind"])
        height, width = source.shape
        panels.append(
            {
                "label": spec["label"],
                "images": [source, manipulated],
                "boxes": boxes,
                "crop": _square_crop(boxes, spec.get("crop_factor", crop_factor), width, height),
            }
        )

    n_cols, n_rows = len(panels), 2
    panel = config.get("panel_size", 1.35)
    label_width, header_height, col_gap, row_gap = 0.42, 0.30, 0.06, 0.06
    figure_width = label_width + n_cols * panel + (n_cols - 1) * col_gap
    figure_height = header_height + n_rows * panel + (n_rows - 1) * row_gap
    fig = plt.figure(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("white")

    def cell(col: int, row: int) -> list[float]:
        x = (label_width + col * (panel + col_gap)) / figure_width
        top = header_height + row * (panel + row_gap)
        return [x, 1.0 - (top + panel) / figure_height, panel / figure_width, panel / figure_height]

    for col, data in enumerate(panels):
        x1, x2, y1, y2 = data["crop"]
        for row in range(n_rows):
            ax = fig.add_axes(cell(col, row))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.imshow(data["images"][row], cmap="gray", vmin=0, vmax=255)
            ax.set_xlim(x1, x2); ax.set_ylim(y2, y1)
            if row == 1:
                for kind, (bx1, by1, bx2, by2) in data["boxes"]:
                    ax.add_patch(
                        patches.Rectangle(
                            (bx1, by1), bx2 - bx1, by2 - by1,
                            fill=False, edgecolor=colors[kind], linewidth=1.8,
                        )
                    )
        header_box = cell(col, 0)
        fig.text(
            header_box[0] + header_box[2] / 2.0, 1.0 - header_height * 0.42 / figure_height,
            data["label"], ha="center", va="center", fontsize=10.5, color=ink,
        )
    for row in range(n_rows):
        label_box = cell(0, row)
        fig.text(
            label_width * 0.42 / figure_width, label_box[1] + label_box[3] / 2.0,
            row_labels[row], ha="center", va="center", rotation=90, fontsize=9.5, color=ink,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved regimes figure: {output_path}  ({[p['label'] for p in panels]})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the synthetic lesion generation pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    mode = config.get("mode", "lesion")
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"mode must be one of {SUPPORTED_MODES}, got {mode!r}"
        )

    output_dir = Path(config["output_dir"])

    if mode == "boxed_samples":
        run_boxed_samples(config, output_dir)
        return

    if mode == "overview":
        plot_overview(config, output_dir / "figure1_overview.png")
        return

    if mode == "regimes":
        plot_regimes(config, output_dir / "figure2_regimes.png")
        return

    preprocessed_dir = Path(config["preprocessed_dir"])
    synthetic_dir = Path(config["synthetic_dir"])
    num_images = config.get("num_images")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode == "lesion":
        matching = discover_matching_samples(preprocessed_dir, synthetic_dir)
        if not matching:
            logger.error("No matching samples found")
            return

        if num_images is not None:
            matching = matching[:num_images]

        samples = []
        for study_id, image_name in matching:
            sample = load_sample(
                preprocessed_dir, synthetic_dir, study_id, image_name
            )
            if sample is not None:
                samples.append(sample)

        if not samples:
            logger.error("No samples could be loaded")
            return

        plot_pipeline(samples, output_dir / f"pipeline-{timestamp}.png", config)
        plot_compositing_details(
            samples, output_dir / f"compositing_details-{timestamp}.png", config
        )
        plot_ground_truth_details(
            samples, output_dir / f"ground_truth_details-{timestamp}.png", config
        )
        plot_attribution_illustration(
            samples,
            output_dir / f"attribution_illustration-{timestamp}.png",
            config,
        )
        return

    records = discover_combined_records(synthetic_dir)
    if not records:
        logger.error("No combined records found")
        return

    if num_images is not None:
        records = records[:num_images]

    samples = []
    for record in records:
        sample = load_combined_sample(record, preprocessed_dir, synthetic_dir)
        if sample is not None:
            samples.append(sample)

    if not samples:
        logger.error("No combined samples could be loaded")
        return

    plot_combined_pipeline(
        samples, output_dir / f"combined_pipeline-{timestamp}.png", config
    )
    plot_combined_attribution_illustration(
        samples,
        output_dir / f"combined_attribution_illustration-{timestamp}.png",
        config,
    )


if __name__ == "__main__":
    main()
