"""
Measure low-level property changes caused by masked distortions and generated lesions.

Computes per-image metrics comparing original vs manipulated mammograms:
  - Pixel value statistics within the manipulated area
  - Gradient-based metrics (Sobel, Laplacian, first-order differences) within the mask
  - First-order differences along the mask border (seamlessness)

Supports both sources:
  - Masked distortions (twirl/spherize) from generate_masked_distortion.py
  - Generated lesions (inpainting) from generate_lesions.py

The source type is auto-detected from the metadata CSV columns.

Usage:
    python exploration/anomaly_distortion_metrics.py --config config/anomaly_distortion_metrics.yaml
"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import yaml
from skimage.metrics import structural_similarity as ssim


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------


def _load_image_pair(
    original_dir: Path,
    generated_dir: Path,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load original image, manipulated image, and binary mask for one row.

    Returns (original, manipulated, mask) or None on failure.
    """
    filename = row["filename"]
    has_transformation = "transformation" in row.index and pd.notna(
        row.get("transformation")
    )

    if has_transformation:
        transformation = str(row["transformation"])
        manip_path = generated_dir / transformation / filename
        mask_path = generated_dir / transformation / "masks" / filename
    else:
        manip_path = generated_dir / filename
        mask_path = generated_dir / "masks" / filename

    orig_path = original_dir / filename

    if not orig_path.exists() or not manip_path.exists():
        return None

    original = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
    manipulated = cv.imread(str(manip_path), cv.IMREAD_GRAYSCALE)
    if original is None or manipulated is None:
        return None

    if mask_path.exists():
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        if mask is not None:
            _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        else:
            mask = _mask_from_bbox(original.shape, row)
    else:
        mask = _mask_from_bbox(original.shape, row)

    if mask is None:
        return None

    return original, manipulated, mask


def _mask_from_bbox(
    shape: tuple[int, int], row: pd.Series
) -> np.ndarray | None:
    """Reconstruct a rectangular mask from lesion bounding-box columns."""
    try:
        x1 = int(row["lesion_x1"])
        y1 = int(row["lesion_y1"])
        x2 = int(row["lesion_x2"])
        y2 = int(row["lesion_y2"])
    except (KeyError, ValueError):
        return None
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude (float64)."""
    gx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def _laplacian(image: np.ndarray) -> np.ndarray:
    """Compute Laplacian response (float64)."""
    return cv.Laplacian(image, cv.CV_64F, ksize=3)


def _first_order_diff(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute first-order finite differences (dx, dy) as float64."""
    dx = np.zeros_like(image, dtype=np.float64)
    dy = np.zeros_like(image, dtype=np.float64)
    dx[:, :-1] = np.diff(image.astype(np.float64), axis=1)
    dy[:-1, :] = np.diff(image.astype(np.float64), axis=0)
    return dx, dy


def _border_pixels(mask: np.ndarray, width: int = 1):
    """Return boolean masks for inner-border and outer-border pixels.

    inner_border: mask pixels that are adjacent to background.
    outer_border: background pixels that are adjacent to the mask.
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2 * width + 1, 2 * width + 1))
    dilated = cv.dilate(mask, kernel, iterations=1)
    eroded = cv.erode(mask, kernel, iterations=1)
    inner_border = (mask > 0) & (eroded == 0)
    outer_border = (dilated > 0) & (mask == 0)
    return inner_border, outer_border


def _compute_ssim(original: np.ndarray, manipulated: np.ndarray, mask: np.ndarray) -> float:
    """Compute SSIM within the masked region."""
    m = mask > 0
    if m.sum() == 0:
        return 0.0
    try:
        score = ssim(original, manipulated, data_range=255)
        return float(score)
    except Exception:
        return 0.0


def _compute_histogram_distance(original: np.ndarray, manipulated: np.ndarray, mask: np.ndarray) -> float:
    """Compute Bhattacharyya distance between histograms within the masked region."""
    m = mask > 0
    if m.sum() < 2:
        return 0.0
    
    orig_patch = original[m].astype(np.float32)
    manip_patch = manipulated[m].astype(np.float32)
    
    # Compute histograms
    hist_orig = cv.calcHist([orig_patch], [0], None, [256], [0, 256])
    hist_manip = cv.calcHist([manip_patch], [0], None, [256], [0, 256])
    
    # Normalize
    hist_orig = cv.normalize(hist_orig, hist_orig).flatten()
    hist_manip = cv.normalize(hist_manip, hist_manip).flatten()
    
    # Bhattacharyya distance: -ln(sum(sqrt(p*q)))
    bc = cv.compareHist(hist_orig, hist_manip, cv.HISTCMP_BHATTACHARYYA)
    return float(bc)


def _compute_local_contrast(original: np.ndarray, manipulated: np.ndarray, mask: np.ndarray, patch_size: int = 16) -> float:
    """Compute RMS contrast difference in local patches within the masked region."""
    m = mask > 0
    if m.sum() < patch_size * patch_size:
        return 0.0
    
    orig_f = original.astype(np.float64)
    manip_f = manipulated.astype(np.float64)
    
    h, w = original.shape
    contrast_diffs = []
    
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch_mask = m[y:y+patch_size, x:x+patch_size]
            if patch_mask.sum() < patch_size * patch_size // 2:
                continue
            
            orig_patch = orig_f[y:y+patch_size, x:x+patch_size]
            manip_patch = manip_f[y:y+patch_size, x:x+patch_size]
            
            orig_contrast = float(np.std(orig_patch))
            manip_contrast = float(np.std(manip_patch))
            
            if orig_contrast > 0:
                contrast_diffs.append(abs(manip_contrast - orig_contrast) / orig_contrast)
    
    return float(np.mean(contrast_diffs)) if contrast_diffs else 0.0


def compute_metrics(
    original: np.ndarray,
    manipulated: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compute all metrics for one image pair.

    Returns a flat dictionary of metric name → value.
    """
    m = mask > 0
    if m.sum() == 0:
        return {}

    orig_f = original.astype(np.float64)
    manip_f = manipulated.astype(np.float64)
    diff = manip_f - orig_f

    metrics: dict[str, float] = {}

    # ---- Pixel-value statistics within the mask ----
    metrics["pixel_mae"] = float(np.mean(np.abs(diff[m])))
    metrics["pixel_rmse"] = float(np.sqrt(np.mean(diff[m] ** 2)))
    metrics["pixel_max_diff"] = float(np.max(np.abs(diff[m])))
    metrics["pixel_mean_orig"] = float(np.mean(orig_f[m]))
    metrics["pixel_mean_manip"] = float(np.mean(manip_f[m]))
    metrics["pixel_std_orig"] = float(np.std(orig_f[m]))
    metrics["pixel_std_manip"] = float(np.std(manip_f[m]))

    # ---- Sobel gradient metrics within the mask ----
    sobel_orig = _sobel_magnitude(original)
    sobel_manip = _sobel_magnitude(manipulated)
    metrics["sobel_mean_orig"] = float(np.mean(sobel_orig[m]))
    metrics["sobel_mean_manip"] = float(np.mean(sobel_manip[m]))
    metrics["sobel_mae"] = float(np.mean(np.abs(sobel_manip[m] - sobel_orig[m])))

    # ---- Laplacian metrics within the mask ----
    lap_orig = _laplacian(original)
    lap_manip = _laplacian(manipulated)
    metrics["laplacian_abs_mean_orig"] = float(np.mean(np.abs(lap_orig[m])))
    metrics["laplacian_abs_mean_manip"] = float(np.mean(np.abs(lap_manip[m])))
    metrics["laplacian_mae"] = float(np.mean(np.abs(lap_manip[m] - lap_orig[m])))

    # ---- First-order difference metrics within the mask ----
    dx_orig, dy_orig = _first_order_diff(original)
    dx_manip, dy_manip = _first_order_diff(manipulated)
    metrics["grad_dx_mae"] = float(np.mean(np.abs(dx_manip[m] - dx_orig[m])))
    metrics["grad_dy_mae"] = float(np.mean(np.abs(dy_manip[m] - dy_orig[m])))
    grad_mag_orig = np.sqrt(dx_orig**2 + dy_orig**2)
    grad_mag_manip = np.sqrt(dx_manip**2 + dy_manip**2)
    metrics["grad_mag_mean_orig"] = float(np.mean(grad_mag_orig[m]))
    metrics["grad_mag_mean_manip"] = float(np.mean(grad_mag_manip[m]))

    # ---- Border metrics (first-order differences along the mask edge) ----
    inner_border, outer_border = _border_pixels(mask, width=1)

    if inner_border.any() and outer_border.any():
        # Mean intensity just inside vs just outside the mask (manipulated image)
        mean_inner_manip = float(np.mean(manip_f[inner_border]))
        mean_outer = float(np.mean(orig_f[outer_border]))
        mean_inner_orig = float(np.mean(orig_f[inner_border]))

        metrics["border_intensity_step_manip"] = abs(
            mean_inner_manip - mean_outer
        )
        metrics["border_intensity_step_orig"] = abs(
            mean_inner_orig - mean_outer
        )

        # Gradient magnitude at the border (measures edge visibility)
        metrics["border_grad_mag_manip"] = float(
            np.mean(sobel_manip[inner_border | outer_border])
        )
        metrics["border_grad_mag_orig"] = float(
            np.mean(sobel_orig[inner_border | outer_border])
        )

        # Difference in gradient at border vs original (added edge energy)
        border_region = inner_border | outer_border
        metrics["border_grad_mag_diff"] = float(
            np.mean(np.abs(sobel_manip[border_region] - sobel_orig[border_region]))
        )

        # Per-pixel intensity difference across the border for the manipulated image:
        # For each inner-border pixel, compare with nearest outer-border pixel.
        metrics["border_pixel_mae_manip"] = _cross_border_mae(
            manip_f, inner_border, outer_border
        )
        metrics["border_pixel_mae_orig"] = _cross_border_mae(
            orig_f, inner_border, outer_border
        )

        # Border diff: change in border visibility caused by the distortion
        metrics["border_intensity_step_diff"] = (
            metrics["border_intensity_step_manip"] - metrics["border_intensity_step_orig"]
        )
        metrics["border_pixel_mae_diff"] = (
            metrics["border_pixel_mae_manip"] - metrics["border_pixel_mae_orig"]
        )

    # ---- Integration quality metrics ----
    metrics["ssim"] = _compute_ssim(original, manipulated, mask)
    metrics["histogram_dist"] = _compute_histogram_distance(original, manipulated, mask)
    metrics["local_contrast_diff"] = _compute_local_contrast(original, manipulated, mask)

    return metrics


def _cross_border_mae(
    image: np.ndarray,
    inner_border: np.ndarray,
    outer_border: np.ndarray,
) -> float:
    """Mean absolute intensity difference between inner-border and nearest outer-border pixels."""
    inner_ys, inner_xs = np.where(inner_border)
    if len(inner_ys) == 0:
        return 0.0

    # For each inner-border pixel, look in a small neighborhood for the closest outer pixel
    inner_vals = image[inner_ys, inner_xs]

    # Find nearest outer-border pixel via dilation-based matching
    outer_vals_at_inner = np.full(len(inner_ys), np.nan)
    h, w = image.shape[:2]
    for i, (iy, ix) in enumerate(zip(inner_ys, inner_xs)):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = iy + dy, ix + dx
            if 0 <= ny < h and 0 <= nx < w and outer_border[ny, nx]:
                outer_vals_at_inner[i] = image[ny, nx]
                break

    valid = ~np.isnan(outer_vals_at_inner)
    if not valid.any():
        return 0.0

    return float(np.mean(np.abs(inner_vals[valid] - outer_vals_at_inner[valid])))


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_all(config: dict):
    """Compute metrics for all images and save results."""
    original_dir = Path(config["original_dir"])
    generated_dir = Path(config["generated_dir"])
    output_dir = Path(config["output_dir"])
    metadata_path = config.get("metadata_path")
    num_images = config.get("num_images")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    if metadata_path is None:
        metadata_path = generated_dir / "lesion_metadata.csv"
    else:
        metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return

    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata for {len(metadata)} images")

    if num_images is not None:
        metadata = metadata.head(num_images)

    has_transformation = "transformation" in metadata.columns
    source_type = "masked_distortion" if has_transformation else "generated_lesion"
    print(f"Detected source type: {source_type}")

    records = []
    skipped = 0

    for idx, row in metadata.iterrows():
        result = _load_image_pair(original_dir, generated_dir, row)
        if result is None:
            skipped += 1
            continue

        original, manipulated, mask = result
        metrics = compute_metrics(original, manipulated, mask)
        if not metrics:
            skipped += 1
            continue

        record = {"filename": row["filename"]}
        if has_transformation:
            record["transformation"] = row["transformation"]
        record.update(metrics)
        records.append(record)

    if not records:
        print("No images processed successfully.")
        return

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\nSaved per-image metrics to {output_dir / 'metrics.csv'}")

    # Print summary
    metric_cols = [c for c in results_df.columns if c not in ("filename", "transformation")]
    summary = results_df[metric_cols].describe().T[["mean", "std", "min", "max"]]
    summary.to_csv(output_dir / "metrics_summary.csv")

    print(f"\n{'Metric':<35s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 77)
    for name, s in summary.iterrows():
        print(f"{name:<35s} {s['mean']:10.3f} {s['std']:10.3f} {s['min']:10.3f} {s['max']:10.3f}")

    # Per-transformation summary if applicable
    if has_transformation:
        print("\n--- Per-transformation summary ---")
        for transform, group in results_df.groupby("transformation"):
            group_summary = group[metric_cols].mean()
            print(f"\n[{transform}]")
            for name, val in group_summary.items():
                print(f"  {name:<33s} {val:10.3f}")

    if skipped:
        print(f"\nSkipped {skipped} images (missing files or empty masks)")
    print(f"\nOutput saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure low-level property changes from masked distortions / generated lesions"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    print("Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)

    process_all(config)


if __name__ == "__main__":
    main()
