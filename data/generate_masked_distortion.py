"""
Generate synthetic mammograms with lesions using geometric distortions.

Takes preprocessed healthy mammography images and adds synthetic lesions
by applying distortions within a circular region of the breast tissue.

Supported transformations:
  - twirl:    Swirl/spiral distortion (angle in degrees)
  - spherize: Spherical bulge or dent (amount in %)
  - all:      Run both twirl and spherize in one invocation

Outputs (per transformation):
  - Distorted images in output_dir/{study_id}/{image_id}.png
  - Binary masks in output_dir/masks/{study_id}/{image_id}.png
  - dataset_metadata.jsonl with per-image generation parameters

Usage:
    python data/generate_masked_distortion.py --config config/generate_masked_distortion.yaml
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2 as cv
import numpy as np
import yaml
from loguru import logger
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from data._common import get_breast_bbox, load_healthy_image_ids
from utils import dump_as_jsonl_file

SUPPORTED_TRANSFORMATIONS = ("twirl", "spherize")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    """Apply twirl distortion with squared-cosine falloff."""
    h, w = image.shape[:2]
    cx, cy = center

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dx = x_coords.astype(np.float64) - cx
    dy = y_coords.astype(np.float64) - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle_orig = np.arctan2(dy, dx)

    d_norm = dist / radius
    within = d_norm < 1.0

    falloff = np.where(d_norm < 1.0, (0.5 * (1.0 + np.cos(np.pi * d_norm))) ** 2, 0.0)
    rotation = np.zeros_like(dist)
    rotation[within] = np.radians(angle_degrees) * falloff[within]

    new_angle = angle_orig + rotation
    src_x = np.where(within, cx + dist * np.cos(new_angle), x_coords).astype(np.float64)
    src_y = np.where(within, cy + dist * np.sin(new_angle), y_coords).astype(np.float64)

    result = map_coordinates(image, [src_y, src_x], order=3, mode="reflect")
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_spherize(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    amount: float = -67.0,
) -> np.ndarray:
    """Apply spherize/pinch distortion."""
    h, w = image.shape
    cx, cy = center

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dx = x_coords.astype(np.float64) - cx
    dy = y_coords.astype(np.float64) - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    d_norm = dist / radius
    within = d_norm < 1.0

    k = max(1.0 + amount / 100.0, 0.01)

    distorted_d_norm = np.copy(d_norm)
    distorted_d_norm[within] = np.clip(d_norm[within] ** (1.0 / k), 0, 1)

    falloff = np.where(d_norm < 1.0, 0.5 * (1.0 + np.cos(np.pi * d_norm)), 0.0)
    new_d_norm = d_norm + (distorted_d_norm - d_norm) * falloff

    new_dist = new_d_norm * radius
    src_x = np.where(within, cx + new_dist * np.cos(angle), x_coords).astype(np.float64)
    src_y = np.where(within, cy + new_dist * np.sin(angle), y_coords).astype(np.float64)

    result = map_coordinates(image, [src_y, src_x], order=3, mode="reflect")
    return np.clip(result, 0, 255).astype(np.uint8)


def create_blend_mask(
    shape: tuple[int, int], center: tuple[int, int], radius: int
) -> np.ndarray:
    """Create a circular blending mask with a thin cosine taper at the edge."""
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


def create_bounding_box_rect_mask(
    shape: tuple[int, int], center: tuple[int, int], radius: int
) -> np.ndarray:
    """Create a filled rectangle covering the distortion disk's bounding box."""
    h, w = shape
    cx, cy = center
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, w - 1)
    y2 = min(cy + radius, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    return mask


def generate_lesion(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    transformation: str = "twirl",
    twirl_angle: float = -237.0,
    spherize_amount: float = -67.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic lesion using the specified transformation with smooth blending."""
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


def _flatten(values: list) -> list:
    """Flatten nested lists into a single list of scalars."""
    result = []
    for v in values:
        if isinstance(v, list):
            result.extend(_flatten(v))
        else:
            result.append(v)
    return result


def _resolve_transformations(config: dict) -> list[str]:
    """Resolve one or more transformations from config.

    Supports:
      - transformation: twirl|spherize|all
      - transformations: [twirl, spherize]
    """
    configured_transformations = config.get("transformations")
    if configured_transformations is None:
        configured_transformations = [config.get("transformation", "twirl")]
    elif not isinstance(configured_transformations, list) or not configured_transformations:
        raise ValueError(
            "Config key 'transformations' must be a non-empty list when provided."
        )

    resolved = []
    for transformation in configured_transformations:
        if not isinstance(transformation, str):
            raise ValueError(
                f"Transformation values must be strings, got: {transformation}"
            )
        normalized = transformation.strip().lower()
        if normalized == "all":
            resolved.extend(SUPPORTED_TRANSFORMATIONS)
            continue
        if normalized not in SUPPORTED_TRANSFORMATIONS:
            raise ValueError(
                f"Unknown transformation '{transformation}'. "
                f"Supported values: {list(SUPPORTED_TRANSFORMATIONS)} or 'all'."
            )
        resolved.append(normalized)

    # Keep order but remove duplicates (e.g. transformations: [all, twirl]).
    return list(dict.fromkeys(resolved))


def _resolve_param_values(config: dict, transformation: str) -> list[tuple[str, float]]:
    """Resolve parameter values for the selected transformation.

    Supports both singular and plural config keys:
      - twirl_angles: [-70, -237]  or  twirl_angle: -70
      - spherize_amounts: [-17, -67]  or  spherize_amount: -67

    Returns list of (param_key, value) pairs, e.g.:
      [("twirl_angle", -70), ("twirl_angle", -237)]
    """
    if transformation == "twirl":
        values = config.get("twirl_angles")
        if values is None:
            values = [config.get("twirl_angle", -237)]
        return [("twirl_angle", v) for v in _flatten(values)]
    elif transformation == "spherize":
        values = config.get("spherize_amounts")
        if values is None:
            values = [config.get("spherize_amount", -67)]
        return [("spherize_amount", v) for v in _flatten(values)]
    else:
        raise ValueError(f"Unknown transformation: {transformation}")


def _make_output_filename(
    image_id: str, transformation: str, param_value: float, multi_param: bool,
) -> str:
    """Build output filename, adding a param suffix when multiple values are used."""
    if not multi_param:
        return f"{image_id}.png"
    # e.g. image_id_twirl_-70.png  or  image_id_spherize_-17.png
    return f"{image_id}_{transformation}_{param_value}.png"


def _resolve_output_dir(output_dir_template: str, transformation: str) -> str:
    """Resolve output directory, allowing {transformation} in the template."""
    if not isinstance(output_dir_template, str) or not output_dir_template.strip():
        raise ValueError("Config key 'output_dir' must be a non-empty string.")
    try:
        return output_dir_template.format(transformation=transformation)
    except KeyError as error:
        unknown_placeholder = error.args[0]
        raise ValueError(
            f"Unsupported placeholder '{{{unknown_placeholder}}}' in output_dir "
            f"template '{output_dir_template}'. Only '{{transformation}}' is supported."
        ) from error


def process_images(config: dict):
    """Process mammography images and generate synthetic lesions.

    Outputs images and masks in {study_id}/{filename}.png structure,
    plus a dataset_metadata.jsonl with per-image generation parameters.

    When multiple parameter values are specified (e.g. twirl_angles: [-70, -237]),
    each image is generated once per value with a param suffix in the filename
    to avoid collisions (e.g. {image_id}_twirl_-70.png).
    """
    transformations = _resolve_transformations(config)
    input_dir = config["input_dir"]
    output_dir_template = config["output_dir"]
    if len(transformations) > 1 and "{transformation}" not in output_dir_template:
        raise ValueError(
            "output_dir must include '{transformation}' when generating multiple "
            "transformations in one run."
        )
    annotations_file = config.get("annotations_file")
    min_lesion_size = config.get("min_lesion_size", 60)
    max_lesion_size = config.get("max_lesion_size", 90)
    num_images = config.get("num_images")
    seed = config.get("seed")

    if seed is not None:
        np.random.seed(seed)

    input_path = Path(input_dir)
    created = config.get("created", "")
    if created:
        created = f"{created}-{min_lesion_size}-{max_lesion_size}"
    healthy_ids = None
    if annotations_file and os.path.exists(annotations_file):
        healthy_ids = load_healthy_image_ids(annotations_file)
        logger.info(f"Loaded {len(healthy_ids)} healthy image IDs")

    image_files = sorted(input_path.rglob("*.png"))
    if healthy_ids is not None:
        original_count = len(image_files)
        image_files = [f for f in image_files if f.stem in healthy_ids]
        logger.info(f"Filtered to {len(image_files)} healthy images (from {original_count})")

    if num_images is not None:
        image_files = image_files[:num_images]

    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return

    for transformation in transformations:
        output_dir = _resolve_output_dir(
            output_dir_template=output_dir_template,
            transformation=transformation,
        )
        if created:
            output_dir = str(Path(output_dir) / created)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        mask_output_path = output_path / "masks"
        mask_output_path.mkdir(parents=True, exist_ok=True)
        ground_truth_rect_output_path = output_path / "ground_truth_rect"
        ground_truth_rect_output_path.mkdir(parents=True, exist_ok=True)

        param_values = _resolve_param_values(config=config, transformation=transformation)
        multi_param = len(param_values) > 1
        param_desc = ", ".join(f"{k}={v}" for k, v in param_values)
        logger.info(
            f"Processing {len(image_files)} images × {len(param_values)} param(s) "
            f"({param_desc}) with {transformation}"
        )

        resolved_config = dict(config)
        resolved_config["transformation"] = transformation
        resolved_config["output_dir"] = str(output_path)
        with open(output_path / "config.yaml", "w") as f:
            yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

        metadata_records = []
        for img_path in tqdm(image_files, desc=f"Generating {transformation}"):
            try:
                image_gray = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
                if image_gray is None:
                    continue

                breast_bbox = get_breast_bbox(image_gray)
                if breast_bbox is None:
                    continue

                _, breast_mask = cv.threshold(image_gray, 10, 255, cv.THRESH_BINARY)
                radius = np.random.randint(min_lesion_size // 2, max_lesion_size // 2 + 1)
                center = sample_lesion_center(breast_bbox, breast_mask, radius)
                if center is None:
                    continue

                relative_path = img_path.relative_to(input_path)
                study_id = str(relative_path.parent)
                image_id = relative_path.stem

                for param_key, param_value in param_values:
                    twirl_angle = param_value if param_key == "twirl_angle" else -237
                    spherize_amount = param_value if param_key == "spherize_amount" else -67

                    result, mask = generate_lesion(
                        image_gray, center, radius,
                        transformation=transformation,
                        twirl_angle=twirl_angle,
                        spherize_amount=spherize_amount,
                    )

                    out_filename = _make_output_filename(
                        image_id, transformation, param_value, multi_param,
                    )
                    output_file = output_path / study_id / out_filename
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    cv.imwrite(str(output_file), result)

                    mask_file = mask_output_path / study_id / out_filename
                    mask_file.parent.mkdir(parents=True, exist_ok=True)
                    cv.imwrite(str(mask_file), mask)

                    ground_truth_rect_mask = create_bounding_box_rect_mask(
                        image_gray.shape, center, radius
                    )
                    ground_truth_rect_file = ground_truth_rect_output_path / study_id / out_filename
                    ground_truth_rect_file.parent.mkdir(parents=True, exist_ok=True)
                    cv.imwrite(str(ground_truth_rect_file), ground_truth_rect_mask)

                    record = {
                        "image_path": str(output_file),
                        "mask_path": str(mask_file),
                        "ground_truth_rect_mask_path": str(ground_truth_rect_file),
                        "source_image_path": str(img_path),
                        "image_id": image_id,
                        "study_id": study_id,
                        "label": 1,
                        "transformation": transformation,
                        param_key: param_value,
                        "lesion_center_x": int(center[0]),
                        "lesion_center_y": int(center[1]),
                        "lesion_radius": int(radius),
                    }
                    metadata_records.append(record)

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        if metadata_records:
            jsonl_path = str(output_path / "dataset_metadata.jsonl")
            dump_as_jsonl_file(data=metadata_records, file_path=jsonl_path)
            logger.info(f"Saved {len(metadata_records)} records to {jsonl_path}")

        logger.info(f"Processed {len(metadata_records)} images → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic lesions using geometric distortions"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    process_images(config)


if __name__ == "__main__":
    main()
