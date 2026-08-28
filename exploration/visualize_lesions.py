"""
Visualize generated mammograms with synthetic lesions.

Creates a 6-column plot for each image showing:
1. Original image
2. Generated image (raw from diffusion model)
3. Seamless cloned (Poisson blending)
4. Thresholded image with lesion bounding box
5. Difference: Original vs Generated
6. Difference: Original vs Seamless

Usage:
    python visualize_lesions.py --config config/visualize_lesions.yaml
"""

import argparse
import os
from pathlib import Path

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def seamless_clone(
    original: np.ndarray,
    generated: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 5,
) -> np.ndarray:
    """Apply Poisson seamless cloning to blend lesion region into original.

    Args:
        original: Original grayscale image
        generated: Generated image with lesion
        bbox: Lesion bounding box (x1, y1, x2, y2)
        padding: Extra padding around bbox for better blending

    Returns:
        Seamlessly blended image
    """
    x1, y1, x2, y2 = bbox
    h, w = original.shape[:2]

    # Add padding to bbox (clamped to image boundaries)
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)

    # Create mask for the lesion region (with padding)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1_pad:y2_pad, x1_pad:x2_pad] = 255

    # Center of the bounding box (required by seamlessClone)
    center = ((x1_pad + x2_pad) // 2, (y1_pad + y2_pad) // 2)

    # Convert to BGR (seamlessClone requires 3-channel images)
    original_bgr = cv.cvtColor(original, cv.COLOR_GRAY2BGR)
    generated_bgr = cv.cvtColor(generated, cv.COLOR_GRAY2BGR)

    # Apply Poisson seamless cloning
    result_bgr = cv.seamlessClone(
        generated_bgr, original_bgr, mask, center, cv.NORMAL_CLONE
    )

    # Convert back to grayscale
    result = cv.cvtColor(result_bgr, cv.COLOR_BGR2GRAY)

    return result


def plot_comparison(
    original: np.ndarray,
    generated: np.ndarray,
    lesion_bbox: tuple[int, int, int, int],
    output_path: str,
    title: str = "",
    threshold: int = 10,
    figsize: tuple[int, int] = (24, 4),
    dpi: int = 150,
    seamless_padding: int = 5,
):
    """Create a 6-column comparison plot for a single image.

    Args:
        original: Original grayscale image
        generated: Generated image with lesion
        lesion_bbox: Lesion bounding box (x1, y1, x2, y2)
        output_path: Path to save the plot
        title: Title for the figure
        threshold: Threshold value for binary mask
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
        seamless_padding: Padding around bbox for seamless cloning
    """
    fig, axes = plt.subplots(1, 6, figsize=figsize)

    x1, y1, x2, y2 = lesion_bbox
    width = x2 - x1
    height = y2 - y1

    # Apply seamless cloning
    seamless = seamless_clone(original, generated, lesion_bbox, padding=seamless_padding)

    # 1. Original image
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 2. Generated image (raw)
    axes[1].imshow(generated, cmap="gray")
    rect1 = patches.Rectangle(
        (x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none"
    )
    # axes[1].add_patch(rect1)
    axes[1].set_title("Generated (Raw)")
    axes[1].axis("off")

    # 3. Seamless cloned
    axes[2].imshow(seamless, cmap="gray")
    rect2 = patches.Rectangle(
        (x1, y1), width, height, linewidth=2, edgecolor="lime", facecolor="none"
    )
    # axes[2].add_patch(rect2)
    axes[2].set_title("Seamless Cloned")
    axes[2].axis("off")

    # 4. Thresholded with lesion bbox
    _, thresh = cv.threshold(original, threshold, 255, cv.THRESH_BINARY)
    axes[3].imshow(thresh, cmap="gray")
    rect3 = patches.Rectangle(
        (x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none"
    )
    # axes[3].add_patch(rect3)
    axes[3].set_title("Threshold + BBox")
    axes[3].axis("off")

    # 5. Difference: Original vs Generated
    diff_generated = cv.absdiff(original, generated)
    axes[4].imshow(diff_generated, cmap="hot")
    axes[4].set_title("Diff (Raw)")
    axes[4].axis("off")

    # 6. Difference: Original vs Seamless
    diff_seamless = cv.absdiff(original, seamless)
    axes[5].imshow(diff_seamless, cmap="hot")
    axes[5].set_title("Diff (Seamless)")
    axes[5].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def visualize_all(config: dict):
    """Generate visualization plots for all processed images.

    Args:
        config: Configuration dictionary
    """
    original_dir = Path(config["original_dir"])
    generated_dir = Path(config["generated_dir"])
    output_dir = Path(config["output_dir"])
    metadata_path = config.get("metadata_path")
    threshold = config.get("threshold", 10)
    figsize = tuple(config.get("figsize", [24, 4]))
    dpi = config.get("dpi", 150)
    num_images = config.get("num_images")
    seamless_padding = config.get("seamless_padding", 5)
    healthy_dir = Path(config.get("healthy_dir", output_dir / "healthy"))
    seamless_dir = Path(config.get("seamless_dir", output_dir / "seamless"))
    masks_dir = generated_dir / "masks"
    masked_dir = output_dir / "masked"

    output_dir.mkdir(parents=True, exist_ok=True)
    healthy_dir.mkdir(parents=True, exist_ok=True)
    seamless_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
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
        print(f"Processing first {num_images} images")

    for idx, row in metadata.iterrows():
        filename = row["filename"]
        lesion_bbox = (
            int(row["lesion_x1"]),
            int(row["lesion_y1"]),
            int(row["lesion_x2"]),
            int(row["lesion_y2"]),
        )

        # Load images
        original_path = original_dir / filename
        generated_path = generated_dir / filename

        if not original_path.exists():
            print(f"Original not found: {original_path}")
            continue

        if not generated_path.exists():
            print(f"Generated not found: {generated_path}")
            continue

        original = cv.imread(str(original_path), cv.IMREAD_GRAYSCALE)
        generated = cv.imread(str(generated_path), cv.IMREAD_GRAYSCALE)

        if original is None or generated is None:
            print(f"Failed to load images for {filename}")
            continue

        # Apply seamless cloning
        seamless = seamless_clone(original, generated, lesion_bbox, padding=seamless_padding)

        # Create output path (preserve directory structure)
        output_path = output_dir / Path(filename).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save healthy (original) image
        healthy_path = healthy_dir / filename
        healthy_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(healthy_path), original)

        # Save seamless image
        seamless_path = seamless_dir / filename
        seamless_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(seamless_path), seamless)

        # Save masked image (healthy image with mask applied)
        mask_path = masks_dir / filename
        if mask_path.exists():
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            if mask is not None:
                inverted_mask = cv.bitwise_not(mask)
                masked = cv.bitwise_and(original, inverted_mask)
                masked_path = masked_dir / filename
                masked_path.parent.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(masked_path), masked)
                print(f"Saved masked image: {masked_path}")
            else:
                print(f"Failed to load mask: {mask_path}")
        else:
            print(f"Mask not found: {mask_path}")

        # Generate plot
        # Note: plot_comparison will recompute seamless for visualization
        plot_comparison(
            original=original,
            generated=generated,
            lesion_bbox=lesion_bbox,
            output_path=str(output_path),
            title=filename,
            threshold=threshold,
            figsize=figsize,
            dpi=dpi,
            seamless_padding=seamless_padding,
        )

        print(f"Saved visualization: {output_path}")
        print(f"Saved healthy image: {healthy_path}")
        print(f"Saved seamless image: {seamless_path}")
    print(f"  Visualizations saved to: {output_dir}")
    print(f"  Healthy images saved to: {healthy_dir}")
    print(f"  Seamless images saved to: {seamless_dir}")
    print(f"  Masked images saved to: {masked_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated mammograms with synthetic lesions"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # Print configuration
    print("Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)

    visualize_all(config)


if __name__ == "__main__":
    main()
