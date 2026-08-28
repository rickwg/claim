"""
Generate synthetic mammograms with lesions using geometric distortions.

This script takes preprocessed mammography images and adds synthetic lesions
by applying distortions within a circular region of the breast tissue.

Supported transformations (matching GIMP filters):
  - twirl:    Swirl/spiral distortion (GIMP Twirl, angle in degrees)
  - spherize: Spherical bulge or dent (GIMP Pinch, amount in %)

Usage:
    python exploration/generate_masked_distortion.py --config config/generate_masked_distortion.yaml
"""

import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import map_coordinates
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_healthy_image_ids(annotations_file: str) -> set[str]:
    """Load image IDs that have no findings (healthy) from annotations."""
    df = pd.read_csv(annotations_file)
    healthy_df = df[df["finding_categories"] == "['No Finding']"]
    return set(healthy_df["image_id"].unique())


def get_breast_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Extract bounding box of the breast region from a mammogram."""
    _, mask = cv.threshold(image, 10, 255, cv.THRESH_BINARY)
    nb_components, output, stats, _ = cv.connectedComponentsWithStats(
        mask, connectivity=4
    )
    if nb_components < 2:
        return None

    sizes = stats[1:, cv.CC_STAT_AREA]
    max_label = 1 + np.argmax(sizes)
    breast_mask = np.zeros(output.shape, dtype=np.uint8)
    breast_mask[output == max_label] = 255

    contours, _ = cv.findContours(breast_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if not contours:
        return None

    x, y, w, h = cv.boundingRect(contours[0])
    return x, y, x + w, y + h


def sample_lesion_center(
    breast_bbox: tuple[int, int, int, int],
    breast_mask: np.ndarray,
    radius: int,
    max_attempts: int = 100,
) -> tuple[int, int] | None:
    """Sample a random center for a lesion ensuring the full circle fits in breast tissue."""
    x1_b, y1_b, x2_b, y2_b = breast_bbox

    for _ in range(max_attempts):
        if x2_b - 2 * radius <= x1_b or y2_b - 2 * radius <= y1_b:
            return None

        cx = np.random.randint(x1_b + radius, x2_b - radius)
        cy = np.random.randint(y1_b + radius, y2_b - radius)

        # Check that the circular region is entirely within the breast
        y_coords, x_coords = np.ogrid[
            -radius : radius + 1, -radius : radius + 1
        ]
        circle = x_coords**2 + y_coords**2 <= radius**2

        region = breast_mask[
            cy - radius : cy + radius + 1, cx - radius : cx + radius + 1
        ]
        if (
            region.shape[0] == 2 * radius + 1
            and region.shape[1] == 2 * radius + 1
            and np.all(region[circle] > 0)
        ):
            return cx, cy

    return None


def apply_twirl(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    angle_degrees: float = -237.0,
) -> np.ndarray:
    """Apply twirl distortion to the image, matching GIMP's Twirl filter.

    Uses a linear decay from full rotation at center to zero at the radius
    boundary, matching GIMP's behavior (unlike skimage.swirl which uses a
    Gaussian decay that concentrates the effect too tightly).

    Args:
        image: Grayscale image (uint8)
        center: (x, y) center of the twirl
        radius: Radius of the twirl effect in pixels
        angle_degrees: Twirl angle in degrees (GIMP convention)

    Returns:
        Twirled image (uint8)
    """
    h, w = image.shape[:2]
    cx, cy = center

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dx = x_coords.astype(np.float64) - cx
    dy = y_coords.astype(np.float64) - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle_orig = np.arctan2(dy, dx)

    d_norm = dist / radius
    within = d_norm < 1.0

    # Squared-cosine falloff: strong at center, steep fade near the edge to
    # minimise interpolation artifacts at the mask border.
    falloff = np.where(d_norm < 1.0, (0.5 * (1.0 + np.cos(np.pi * d_norm))) ** 2, 0.0)
    rotation = np.zeros_like(dist)
    rotation[within] = np.radians(angle_degrees) * falloff[within]

    new_angle = angle_orig + rotation
    src_x = np.where(within, cx + dist * np.cos(new_angle), x_coords).astype(
        np.float64
    )
    src_y = np.where(within, cy + dist * np.sin(new_angle), y_coords).astype(
        np.float64
    )

    result = map_coordinates(image, [src_y, src_x], order=3, mode="reflect")
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_spherize(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    amount: float = -67.0,
) -> np.ndarray:
    """Apply spherize/pinch distortion matching GIMP's Pinch filter.

    Args:
        image: Grayscale image (uint8)
        center: (x, y) center of the distortion
        radius: Radius of the effect in pixels
        amount: Pinch amount in % (-100 to 100).
                Negative = bulge/pop-out, Positive = pinch/dent.

    Returns:
        Distorted image (uint8)
    """
    h, w = image.shape
    cx, cy = center

    y_coords, x_coords = np.mgrid[0:h, 0:w]

    dx = x_coords.astype(np.float64) - cx
    dy = y_coords.astype(np.float64) - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    d_norm = dist / radius
    within = d_norm < 1.0

    # Power factor: amount=-67 → k=0.33 (bulge), amount=67 → k=1.67 (pinch)
    k = max(1.0 + amount / 100.0, 0.01)

    # Inverse mapping: given output distance, find input distance
    distorted_d_norm = np.copy(d_norm)
    distorted_d_norm[within] = np.clip(d_norm[within] ** (1.0 / k), 0, 1)

    # Cosine falloff: interpolate between original and distorted coordinates
    falloff = np.where(d_norm < 1.0, 0.5 * (1.0 + np.cos(np.pi * d_norm)), 0.0)
    new_d_norm = d_norm + (distorted_d_norm - d_norm) * falloff

    new_dist = new_d_norm * radius
    src_x = np.where(within, cx + new_dist * np.cos(angle), x_coords).astype(
        np.float64
    )
    src_y = np.where(within, cy + new_dist * np.sin(angle), y_coords).astype(
        np.float64
    )

    result = map_coordinates(image, [src_y, src_x], order=3, mode="reflect")
    return np.clip(result, 0, 255).astype(np.uint8)


def create_blend_mask(
    shape: tuple[int, int], center: tuple[int, int], radius: int
) -> np.ndarray:
    """Create a circular blending mask with a thin cosine taper at the edge.

    The distortion functions handle the radial fade-out internally via
    displacement modulation. This mask only provides a thin taper at the
    boundary (outer 5%) to avoid hard cutoff artifacts.
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    inner_radius = radius * 0.95
    mask = np.zeros_like(dist, dtype=np.float64)

    mask[dist <= inner_radius] = 1.0
    taper = (dist > inner_radius) & (dist <= radius)
    mask[taper] = 0.5 * (
        1 + np.cos(np.pi * (dist[taper] - inner_radius) / (radius - inner_radius))
    )

    return mask


def generate_lesion(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    transformation: str = "twirl",
    twirl_angle: float = -237.0,
    spherize_amount: float = -67.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic lesion using the specified transformation with smooth blending.

    Args:
        image: Grayscale mammogram (uint8)
        center: (x, y) center of the lesion
        radius: Radius of the lesion
        transformation: One of "twirl" or "spherize"
        twirl_angle: Twirl angle in degrees (for twirl)
        spherize_amount: Pinch amount in % (for spherize)

    Returns:
        Tuple of (result image, binary mask of affected area)
    """
    if transformation == "twirl":
        distorted = apply_twirl(image, center, radius, twirl_angle)
    elif transformation == "spherize":
        distorted = apply_spherize(image, center, radius, spherize_amount)
    else:
        raise ValueError(f"Unknown transformation: {transformation}")

    blend_mask = create_blend_mask(image.shape, center, radius)

    result = (
        image.astype(np.float64) * (1 - blend_mask)
        + distorted.astype(np.float64) * blend_mask
    )
    result = np.clip(result, 0, 255).astype(np.uint8)

    binary_mask = np.zeros(image.shape, dtype=np.uint8)
    binary_mask[blend_mask > 0] = 255

    return result, binary_mask


def process_images(config: dict):
    """Process mammography images and generate synthetic lesions."""
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    annotations_file = config.get("annotations_file")
    transformations = config.get("transformations", ["twirl"])
    twirl_angle = config.get("twirl_angle", -237)
    spherize_amount = config.get("spherize_amount", -67)
    min_lesion_size = config.get("min_lesion_size", 20)
    max_lesion_size = config.get("max_lesion_size", 40)
    num_images = config.get("num_images")
    seed = config.get("seed")

    if seed is not None:
        np.random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    healthy_ids = None
    if annotations_file and os.path.exists(annotations_file):
        healthy_ids = load_healthy_image_ids(annotations_file)
        print(f"Loaded {len(healthy_ids)} healthy image IDs")

    image_files = list(input_path.rglob("*.png"))
    if healthy_ids is not None:
        original_count = len(image_files)
        image_files = [f for f in image_files if f.stem in healthy_ids]
        skipped = original_count - len(image_files)
        if skipped > 0:
            print(f"Filtered out {skipped} images with existing lesions")

    if num_images is not None:
        if num_images > len(image_files):
            raise ValueError(
                f"Requested num_images={num_images} exceeds available images ({len(image_files)})"
            )
        image_files = image_files[:num_images]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images with transformations: {transformations}")

    metadata_records = []

    for img_path in tqdm(image_files, desc="Generating distortions"):
        try:
            image_gray = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if image_gray is None:
                print(f"Failed to load {img_path}")
                continue

            breast_bbox = get_breast_bbox(image_gray)
            if breast_bbox is None:
                print(f"Could not detect breast region in {img_path}")
                continue

            _, breast_mask = cv.threshold(image_gray, 10, 255, cv.THRESH_BINARY)

            radius = np.random.randint(min_lesion_size // 2, max_lesion_size // 2 + 1)

            center = sample_lesion_center(breast_bbox, breast_mask, radius)
            if center is None:
                print(f"Could not find valid lesion location in {img_path}")
                continue

            relative_path = img_path.relative_to(input_path)
            x1 = center[0] - radius
            y1 = center[1] - radius
            x2 = center[0] + radius
            y2 = center[1] + radius

            for transformation in transformations:
                result, mask = generate_lesion(
                    image_gray,
                    center,
                    radius,
                    transformation=transformation,
                    twirl_angle=twirl_angle,
                    spherize_amount=spherize_amount,
                )

                output_file = output_path / transformation / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(output_file), result)

                mask_file = output_path / transformation / "masks" / relative_path
                mask_file.parent.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(mask_file), mask)

                metadata_records.append(
                    {
                        "filename": str(relative_path),
                        "transformation": transformation,
                        "lesion_center_x": center[0],
                        "lesion_center_y": center[1],
                        "lesion_radius": radius,
                        "lesion_x1": x1,
                        "lesion_y1": y1,
                        "lesion_x2": x2,
                        "lesion_y2": y2,
                        "twirl_angle": twirl_angle if transformation == "twirl" else None,
                        "spherize_amount": spherize_amount if transformation == "spherize" else None,
                        "breast_x1": breast_bbox[0],
                        "breast_y1": breast_bbox[1],
                        "breast_x2": breast_bbox[2],
                        "breast_y2": breast_bbox[3],
                    }
                )

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(output_path / "lesion_metadata.csv", index=False)
        print(f"Saved metadata to {output_path / 'lesion_metadata.csv'}")

    print(f"Processed {len(metadata_records)} images → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic lesions using geometric distortions"
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

    process_images(config)


if __name__ == "__main__":
    main()
