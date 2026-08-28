import os
from pathlib import Path
from typing import Tuple, Optional

import cv2 as cv
import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw
from joblib import Parallel, delayed
from loguru import logger
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
from tqdm import tqdm

from utils import load_config_file, today_formatted, dump_as_yaml_file
from common import EnvironmentVariables

CONFIG = load_config_file(
    os.environ[EnvironmentVariables.PREPROCESS_CONFIG_FILE_PATH.value]
)


def get_image_laterality(input_path: str) -> str:
    """Extract laterality information from DICOM file.

    Args:
        input_path: Path to DICOM file

    Returns:
        'L' for left, 'R' for right, or 'UNKNOWN' if not found
    """
    try:
        ds = pydicom.dcmread(input_path)
        if hasattr(ds, "ImageLaterality"):
            return ds.ImageLaterality
        elif hasattr(ds, "Laterality"):
            return ds.Laterality
        else:
            logger.warning(f"No laterality info found in {input_path}")
            return "UNKNOWN"
    except Exception as e:
        logger.error(f"Error reading DICOM {input_path}: {e}")
        return "UNKNOWN"


def resize_and_center_crop(
    image: Image.Image, target_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """Resize image preserving aspect ratio and center crop to target size.

    Args:
        image: PIL Image to process
        target_size: Target dimensions (width, height)

    Returns:
        Processed PIL Image
    """
    target_width, target_height = target_size
    preprocess = Compose([
        Resize(min(target_width, target_height), interpolation=InterpolationMode.BILINEAR),
        CenterCrop((target_height, target_width)),
    ])
    return preprocess(image)


def normalize_and_convert(
    image: Image.Image, max_value: float = 3500.0, to_rgb: bool = False
) -> np.ndarray:
    """Normalize and convert image to uint8, optionally converting to RGB.

    Args:
        image: Input PIL Image
        max_value: Maximum value for normalization (maps to 255)
        to_rgb: If True, convert grayscale to 3-channel RGB

    Returns:
        Normalized numpy array (grayscale or RGB uint8)
    """
    a = np.array(image).astype(np.float32)
    a = (a / max_value) * 255.0
    a[a > 255] = 255
    a = a.astype(np.uint8)

    if to_rgb:
        a = np.stack((a,) * 3, axis=-1)

    return a


def process_single_image(
    study_id: str,
    image_id: str,
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    max_value: float = 3500.0,
    to_rgb: bool = False,
    laterality: Optional[str] = None,
) -> Tuple[str, bool]:
    """Process a single image with flipping, resizing, and normalization.

    Args:
        study_id: Study identifier
        image_id: Image identifier
        input_dir: Input directory containing DICOM and PNG files
        output_dir: Output directory for processed images
        target_size: Target size for resizing (width, height)
        max_value: Maximum value for normalization
        to_rgb: If True, convert to 3-channel RGB
        laterality: Image laterality ('L' or 'R'). If None, tries to read from DICOM.

    Returns:
        Tuple of (image_id, success_status)
    """
    png_path = os.path.join(input_dir, "png", study_id, f"{image_id}.png")
    image = Image.open(png_path)
    
    # If laterality not provided, try to get from DICOM
    if laterality is None:
        dicom_path = os.path.join(input_dir, "dicom", study_id, f"{image_id}.dicom")
        if os.path.exists(dicom_path):
            laterality = get_image_laterality(input_path=dicom_path)
        else:
            laterality = "UNKNOWN"
    
    # Flip right images to left
    if laterality == "R":
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    image = resize_and_center_crop(image=image, target_size=target_size)
    image_array = normalize_and_convert(image=image, max_value=max_value, to_rgb=to_rgb)

    output_path = os.path.join(output_dir, study_id, f"{image_id}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv.imwrite(output_path, image_array)

    return image_id, True


def collect_image_ids(input_dir: str) -> list[Tuple[str, str]]:
    png_dir = os.path.join(input_dir, "png")
    image_pairs = list()
    for study_id in os.listdir(png_dir):
        study_path = os.path.join(png_dir, study_id)
        if not os.path.isdir(study_path):
            continue
        for png_file in os.listdir(study_path):
            if png_file.endswith(".png"):
                image_id = png_file.replace(".png", "")
                image_pairs.append((study_id, image_id))

    return image_pairs


def build_manufacturer_map(input_dir: str) -> dict[str, str]:
    metadata_path = os.path.join(input_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    if metadata_df.empty:
        raise ValueError(f"Metadata file is empty: {metadata_path}")

    image_id_column = metadata_df.columns[0]
    manufacturer_column = "Manufacturer"
    if manufacturer_column not in metadata_df.columns:
        raise ValueError(
            f"Column '{manufacturer_column}' not found in metadata file: {metadata_path}"
        )

    metadata_df = metadata_df[[image_id_column, manufacturer_column]].dropna(
        subset=[image_id_column, manufacturer_column]
    )
    metadata_df[image_id_column] = metadata_df[image_id_column].astype(str)
    metadata_df[manufacturer_column] = metadata_df[manufacturer_column].astype(str)
    metadata_df = metadata_df.drop_duplicates(subset=[image_id_column], keep="first")

    return metadata_df.set_index(image_id_column)[manufacturer_column].to_dict()


def _check_source_max_value(
    study_id: str, image_id: str, input_dir: str, max_source_value: float
) -> Tuple[str, str, bool, bool]:
    png_path = os.path.join(input_dir, "png", study_id, f"{image_id}.png")
    if not os.path.exists(png_path):
        return study_id, image_id, False, False

    with Image.open(png_path) as image:
        image_max = float(np.asarray(image).max())

    return study_id, image_id, True, image_max <= max_source_value


def filter_image_pairs_by_source_max_value(
    image_pairs: list[Tuple[str, str]], input_dir: str, max_source_value: float,
    n_jobs: int = -1,
) -> tuple[list[Tuple[str, str]], int]:
    results = Parallel(n_jobs=n_jobs)(
        delayed(_check_source_max_value)(study_id, image_id, input_dir, max_source_value)
        for study_id, image_id in tqdm(image_pairs, desc="Filtering by source max value")
    )

    filtered_pairs = []
    missing_source_count = 0
    for study_id, image_id, found, passed in results:
        if not found:
            missing_source_count += 1
        elif passed:
            filtered_pairs.append((study_id, image_id))

    return filtered_pairs, missing_source_count


def generate_lesion_mask(
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    bbox_coords: Tuple[float, float, float, float],
    laterality: str,
) -> Image.Image:
    """Generate a binary mask for a lesion from bounding box coordinates.

    Args:
        original_size: Original image size (width, height)
        target_size: Target size after preprocessing (width, height)
        bbox_coords: Bounding box (xmin, ymin, xmax, ymax) in original coordinates
        laterality: Image laterality ('L' or 'R')

    Returns:
        PIL Image mask (binary, white lesion area on black background)
    """
    xmin, ymin, xmax, ymax = bbox_coords
    orig_width, orig_height = original_size
    target_width, target_height = target_size

    # Adjust coordinates if image was flipped (right laterality)
    if laterality == "R":
        xmin_new = orig_width - xmax
        xmax_new = orig_width - xmin
        xmin, xmax = xmin_new, xmax_new

    # Calculate scaling factor (matching resize_and_center_crop logic)
    scale = max(target_width / orig_width, target_height / orig_height)
    scaled_width = int(orig_width * scale)
    scaled_height = int(orig_height * scale)

    # Scale bounding box coordinates
    xmin_scaled = int(xmin * scale)
    ymin_scaled = int(ymin * scale)
    xmax_scaled = int(xmax * scale)
    ymax_scaled = int(ymax * scale)

    # Adjust for center crop
    left_crop = (scaled_width - target_width) // 2
    top_crop = (scaled_height - target_height) // 2

    # Adjust bbox coordinates relative to crop
    xmin_final = xmin_scaled - left_crop
    ymin_final = ymin_scaled - top_crop
    xmax_final = xmax_scaled - left_crop
    ymax_final = ymax_scaled - top_crop

    # Clip to target size boundaries
    xmin_final = max(0, min(xmin_final, target_width))
    ymin_final = max(0, min(ymin_final, target_height))
    xmax_final = max(0, min(xmax_final, target_width))
    ymax_final = max(0, min(ymax_final, target_height))

    # Create mask
    mask = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(mask)

    # Only draw if the bounding box is valid
    if xmax_final > xmin_final and ymax_final > ymin_final:
        draw.rectangle([xmin_final, ymin_final, xmax_final, ymax_final], fill=255)

    return mask


def process_single_image_with_mask(
    study_id: str,
    image_id: str,
    input_dir: str,
    output_dir: str,
    mask_output_dir: str,
    annotations_df: pd.DataFrame,
    target_size: Tuple[int, int] = (512, 512),
    max_value: float = 3500.0,
    to_rgb: bool = False,
    laterality: Optional[str] = None,
) -> Tuple[str, bool, int]:
    """Process a single image and generate masks for its lesions.

    For images with findings, generates masks from bounding box annotations.
    For healthy images (no findings), generates an all-zero mask.

    Args:
        study_id: Study identifier
        image_id: Image identifier
        input_dir: Input directory containing DICOM and PNG files
        output_dir: Output directory for processed images
        mask_output_dir: Output directory for lesion masks
        annotations_df: DataFrame with all annotations (including 'No Finding')
        target_size: Target size for resizing (width, height)
        max_value: Maximum value for normalization
        to_rgb: If True, convert to 3-channel RGB
        laterality: Image laterality ('L' or 'R'). If None, tries to read from DICOM.

    Returns:
        Tuple of (image_id, success_status, num_masks_generated)
    """
    png_path = os.path.join(input_dir, "png", study_id, f"{image_id}.png")
    if not os.path.exists(png_path):
        logger.debug(f"Skipping image {image_id} - PNG not found")
        return image_id, False, 0

    # Process the image (flip, resize, normalize, save)
    _, success = process_single_image(
        study_id, image_id, input_dir, output_dir, target_size, max_value, to_rgb, laterality
    )
    if not success:
        return image_id, False, 0

    # Find lesion annotations (excluding 'No Finding')
    image_annotations = annotations_df[
        (annotations_df["image_id"] == image_id)
        & (annotations_df["finding_categories"] != "['No Finding']")
    ]

    if len(image_annotations) == 0:
        # Healthy image — save all-zero mask
        mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        mask_path = os.path.join(mask_output_dir, study_id, f"{image_id}.png")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv.imwrite(mask_path, mask)
        return image_id, True, 1

    # Image has findings — generate lesion masks from bounding boxes
    image = Image.open(png_path)
    original_size = image.size

    if laterality is None:
        dicom_path = os.path.join(input_dir, "dicom", study_id, f"{image_id}.dicom")
        laterality = get_image_laterality(dicom_path) if os.path.exists(dicom_path) else "UNKNOWN"

    num_masks = 0
    for _, row in image_annotations.iterrows():
        bbox = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
        mask = generate_lesion_mask(original_size, target_size, bbox, laterality)

        mask_filename = f"{image_id}_{num_masks}.png" if num_masks > 0 else f"{image_id}.png"
        mask_path = os.path.join(mask_output_dir, study_id, mask_filename)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv.imwrite(mask_path, np.array(mask))
        num_masks += 1

    return image_id, True, num_masks


def main():
    input_dir = CONFIG.get("input_dir", "vindrmammo_data")
    output_dir = CONFIG.get("output_dir", "vindrmammo_data/processed")
    target_size = tuple(CONFIG.get("target_size", [512, 512]))
    max_value = CONFIG.get("max_value", 3500.0)
    n_jobs = CONFIG.get("n_jobs", -1)
    to_rgb = CONFIG.get("to_rgb", False)
    mask_output_dir = CONFIG.get("mask_output_dir", os.path.join(output_dir, "masks"))
    image_filter = CONFIG.get("image_filter", None)
    manufacturer_filter = CONFIG.get("manufacturer_filter", None)
    source_max_value_filter = CONFIG.get("source_max_value_filter", None)

    dump_as_yaml_file(
        data=CONFIG, file_path=os.path.join(output_dir, "preprocess_config.yaml")
    )

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mask output directory: {mask_output_dir}")
    logger.info(f"Target size: {target_size}")
    logger.info(f"Max value for normalization: {max_value}")
    logger.info(f"Convert to RGB: {to_rgb}")
    logger.info(f"Image filter: {image_filter}")
    logger.info(f"Manufacturer filter: {manufacturer_filter}")
    logger.info(f"Source max value filter: {source_max_value_filter}")
    logger.info(f"Number of parallel jobs: {n_jobs}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_output_dir).mkdir(parents=True, exist_ok=True)

    # Load all annotations
    annotations_path = os.path.join(input_dir, "finding_annotations.csv")
    if not os.path.exists(annotations_path):
        logger.error(f"Finding annotations not found at {annotations_path}")
        return

    annotations_df = pd.read_csv(annotations_path)
    logger.info(f"Loaded {len(annotations_df)} annotations")

    # Build laterality map from annotations
    laterality_map = (
        annotations_df.drop_duplicates(subset=["image_id"])
        .set_index("image_id")["laterality"]
        .to_dict()
    )

    # Collect all available images from disk
    image_pairs = collect_image_ids(input_dir=input_dir)
    logger.info(f"Found {len(image_pairs)} images on disk")

    if len(image_pairs) == 0:
        logger.error("No images found in the input directory!")
        return

    if manufacturer_filter:
        manufacturer_values = (
            [manufacturer_filter]
            if isinstance(manufacturer_filter, str)
            else manufacturer_filter
        )
        if not isinstance(manufacturer_values, list):
            raise ValueError("manufacturer_filter must be a string or a list of strings")

        allowed_manufacturers = {
            str(value).strip().upper()
            for value in manufacturer_values
            if str(value).strip()
        }
        if not allowed_manufacturers:
            raise ValueError("manufacturer_filter is set but empty after normalization")

        manufacturer_map = build_manufacturer_map(input_dir=input_dir)
        missing_manufacturer_count = 0
        filtered_image_pairs = []
        for study_id, image_id in image_pairs:
            manufacturer = manufacturer_map.get(image_id)
            if manufacturer is None:
                missing_manufacturer_count += 1
                continue
            if manufacturer.strip().upper() in allowed_manufacturers:
                filtered_image_pairs.append((study_id, image_id))

        image_pairs = filtered_image_pairs
        logger.info(
            f"Filtered to {len(image_pairs)} images matching manufacturers: "
            f"{sorted(allowed_manufacturers)}"
        )
        if missing_manufacturer_count > 0:
            logger.warning(
                f"Skipped {missing_manufacturer_count} images with missing manufacturer metadata"
            )

    if source_max_value_filter is not None:
        max_source_value = float(source_max_value_filter)
        if max_source_value <= 0:
            raise ValueError("source_max_value_filter must be > 0")

        image_pairs, missing_source_count = filter_image_pairs_by_source_max_value(
            image_pairs=image_pairs,
            input_dir=input_dir,
            max_source_value=max_source_value,
            n_jobs=n_jobs,
        )
        logger.info(
            f"Filtered to {len(image_pairs)} images with source max <= {max_source_value}"
        )
        if missing_source_count > 0:
            logger.warning(
                f"Skipped {missing_source_count} images with missing source PNG files"
            )

    # Optionally filter by finding categories
    if image_filter:
        matching_ids = set(
            annotations_df[
                annotations_df["finding_categories"].apply(
                    lambda x: any(f in x for f in image_filter)
                )
            ]["image_id"].unique()
        )
        image_pairs = [(sid, iid) for sid, iid in image_pairs if iid in matching_ids]
        logger.info(
            f"Filtered to {len(image_pairs)} images matching categories: {image_filter}"
        )

    if len(image_pairs) == 0:
        logger.error("No images left after applying filters!")
        return

    # Process all images and generate masks
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_image_with_mask)(
            study_id,
            image_id,
            input_dir,
            output_dir,
            mask_output_dir,
            annotations_df,
            target_size,
            max_value,
            to_rgb,
            laterality=laterality_map.get(image_id),
        )
        for study_id, image_id in tqdm(image_pairs, desc="Processing")
    )

    total_masks = sum(r[2] for r in results)
    successful = sum(1 for r in results if r[1])
    logger.info(f"Successfully processed {successful}/{len(results)} images")
    logger.info(f"Generated {total_masks} masks total")


if __name__ == "__main__":
    main()
