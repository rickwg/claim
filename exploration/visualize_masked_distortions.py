"""
Visualize mammograms with masked distortions (twirl, spherize).

Creates a 3-column plot for each image showing:
1. Original image
2. Distorted image
3. Difference: Original vs Distorted

Usage:
    python exploration/visualize_masked_distortions.py --config config/visualize_masked_distortions.yaml
"""

import argparse
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def plot_comparison(
    original: np.ndarray,
    distorted: np.ndarray,
    output_path: str,
    title: str = "",
    figsize: tuple[int, int] = (12, 4),
    dpi: int = 150,
):
    """Create a 3-column comparison plot for a single image."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(distorted, cmap="gray")
    axes[1].set_title("Distorted")
    axes[1].axis("off")

    diff = cv.absdiff(original, distorted)
    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Difference")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def visualize_all(config: dict):
    """Generate visualization plots for all masked distortion images."""
    original_dir = Path(config["original_dir"])
    generated_dir = Path(config["generated_dir"])
    output_dir = Path(config["output_dir"])
    metadata_path = config.get("metadata_path")
    figsize = tuple(config.get("figsize", [12, 4]))
    dpi = config.get("dpi", 150)
    num_images = config.get("num_images")

    output_dir.mkdir(parents=True, exist_ok=True)

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
        transformation = str(row["transformation"])

        distorted_path = generated_dir / transformation / filename
        original_path = original_dir / filename

        if not original_path.exists():
            print(f"Original not found: {original_path}")
            continue

        if not distorted_path.exists():
            print(f"Distorted not found: {distorted_path}")
            continue

        original = cv.imread(str(original_path), cv.IMREAD_GRAYSCALE)
        distorted = cv.imread(str(distorted_path), cv.IMREAD_GRAYSCALE)

        if original is None or distorted is None:
            print(f"Failed to load images for {filename}")
            continue

        sub_path = Path(transformation) / filename

        plot_path = output_dir / sub_path.with_suffix(".png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(
            original=original,
            distorted=distorted,
            output_path=str(plot_path),
            title=f"[{transformation}] {filename}",
            figsize=figsize,
            dpi=dpi,
        )

        print(f"[{transformation}] {filename}")

    print(f"\nOutput saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mammograms with masked distortions"
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

    visualize_all(config)


if __name__ == "__main__":
    main()
