"""
Generate synthetic mammograms with lesions using diffusion inpainting.

Takes preprocessed healthy mammography images and adds synthetic lesions
using a fine-tuned Stable Diffusion inpainting model.

Outputs:
  - Composited images in output_dir/{study_id}/{image_id}.png
  - Rectangular inpainting masks in output_dir/masks/{study_id}/{image_id}.png
  - Ground-truth lesion contour masks in output_dir/ground_truth/{study_id}/{image_id}.png
  - dataset_metadata.jsonl with per-image generation parameters

Usage:
    python data/generate_lesions.py --config config/generate_lesions.yaml
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2 as cv
import numpy as np
import torch
import yaml
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from loguru import logger
from PIL import Image
from tqdm import tqdm

from data._common import get_breast_bbox, load_healthy_image_ids
from utils import dump_as_jsonl_file


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sample_lesion_bbox(
    breast_bbox: tuple[int, int, int, int],
    breast_mask: np.ndarray,
    min_size: int = 30,
    max_size: int = 80,
    max_attempts: int = 100,
) -> tuple[int, int, int, int] | None:
    """Sample a random bounding box for a lesion within the breast region."""
    x1_b, y1_b, x2_b, y2_b = breast_bbox

    for _ in range(max_attempts):
        width = np.random.randint(min_size, max_size)
        height = np.random.randint(min_size, max_size)

        if x2_b - width <= x1_b or y2_b - height <= y1_b:
            continue

        x1 = np.random.randint(x1_b, x2_b - width)
        y1 = np.random.randint(y1_b, y2_b - height)
        x2 = x1 + width
        y2 = y1 + height

        lesion_region = breast_mask[y1:y2, x1:x2]
        if lesion_region.shape[0] > 0 and lesion_region.shape[1] > 0:
            if np.all(lesion_region > 0):
                return x1, y1, x2, y2

    return None


def create_lesion_mask(
    image_size: tuple[int, int], lesion_bbox: tuple[int, int, int, int]
) -> Image.Image:
    """Create a binary mask for the lesion region."""
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    x1, y1, x2, y2 = lesion_bbox
    mask[y1:y2, x1:x2] = 255
    return Image.fromarray(mask)


def intensity_match(
    original: np.ndarray,
    generated: np.ndarray,
    binary_mask: np.ndarray,
    dilate_radius: int = 20,
) -> np.ndarray:
    """Normalize generated image intensity to match original in the mask neighborhood."""
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
    )
    dilated = cv.dilate(binary_mask, kernel, iterations=1)
    neighborhood = (dilated > 0) & (binary_mask == 0)

    if neighborhood.sum() < 10:
        neighborhood = binary_mask > 0

    orig_vals = original[neighborhood].astype(np.float64)
    gen_vals = generated[neighborhood].astype(np.float64)

    mu_orig, sigma_orig = orig_vals.mean(), orig_vals.std()
    mu_gen, sigma_gen = gen_vals.mean(), gen_vals.std()

    if sigma_gen < 1e-6:
        sigma_gen = 1.0

    matched = generated.astype(np.float64)
    matched = (matched - mu_gen) * (sigma_orig / sigma_gen) + mu_orig
    return np.clip(matched, 0, 255).astype(np.uint8)


def create_feathered_mask(
    binary_mask: np.ndarray,
    taper_pixels: int = 15,
) -> np.ndarray:
    """Create a soft blending mask from a binary mask using distance transform."""
    dist = cv.distanceTransform(binary_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    mask = np.zeros_like(dist, dtype=np.float64)
    mask[dist >= taper_pixels] = 1.0
    taper = (dist > 0) & (dist < taper_pixels)
    mask[taper] = 0.5 * (1 - np.cos(np.pi * dist[taper] / taper_pixels))

    return mask


def composite_lesion(
    original: np.ndarray,
    generated: np.ndarray,
    binary_mask: np.ndarray,
    taper_pixels: int = 15,
    dilate_radius: int = 20,
    intensity_matching: bool = True,
) -> np.ndarray:
    """Composite generated lesion onto original via optional intensity matching and feathered blending."""
    if intensity_matching:
        foreground = intensity_match(original, generated, binary_mask, dilate_radius)
    else:
        foreground = generated
    blend_mask = create_feathered_mask(binary_mask, taper_pixels)

    result = (
        original.astype(np.float64) * (1 - blend_mask)
        + foreground.astype(np.float64) * blend_mask
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_inscribed_ellipse(
    canvas: np.ndarray, bbox: tuple[int, int, int, int]
) -> None:
    """Draw a filled ellipse inscribed in an axis-aligned rectangle."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max((x2 - x1) // 2, 1), max((y2 - y1) // 2, 1)
    cv.ellipse(canvas, (cx, cy), (ax, ay), 0, 0, 360, 255, cv.FILLED)


def extract_ground_truth_mask(
    original: np.ndarray,
    composited: np.ndarray,
    diff_threshold: int = 5,
    morph_kernel_size: int = 3,
) -> np.ndarray:
    """Derive a tight lesion mask from actual pixel differences."""
    diff = cv.absdiff(original, composited)
    _, mask = cv.threshold(diff, diff_threshold, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    return mask


def extract_lesion_ground_truth(
    image_gray: np.ndarray,
    composited: np.ndarray,
    lesion_bbox: tuple[int, int, int, int],
    diff_threshold: int = 5,
    morph_kernel_size: int = 3,
    min_lesion_size: int = 30,
    min_gt_fraction: float = 0.6,
) -> np.ndarray:
    """Diff-extract → ellipse-fit → erode → pad-if-too-small lesion GT mask."""
    raw_mask = extract_ground_truth_mask(
        image_gray, composited,
        diff_threshold=diff_threshold,
        morph_kernel_size=morph_kernel_size,
    )

    h_img, w_img = image_gray.shape
    ellipse_canvas = np.zeros((h_img, w_img), dtype=np.uint8)
    ys, xs = np.nonzero(raw_mask)

    if xs.size >= 5:
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        try:
            (cx, cy), (maj, minr), angle = cv.fitEllipse(points)
            cv.ellipse(
                ellipse_canvas,
                (int(round(cx)), int(round(cy))),
                (max(int(round(maj / 2)), 1), max(int(round(minr / 2)), 1)),
                angle, 0, 360, 255, cv.FILLED,
            )
        except cv.error:
            draw_inscribed_ellipse(ellipse_canvas, lesion_bbox)
    else:
        draw_inscribed_ellipse(ellipse_canvas, lesion_bbox)

    erode_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    ground_truth = cv.erode(ellipse_canvas, erode_kernel, iterations=1)

    if not np.any(ground_truth):
        ground_truth = np.zeros((h_img, w_img), dtype=np.uint8)
        draw_inscribed_ellipse(ground_truth, lesion_bbox)

    min_gt_size = int(min_gt_fraction * min_lesion_size)
    gt_contours, _ = cv.findContours(
        ground_truth, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if gt_contours:
        _, _, gt_w, gt_h = cv.boundingRect(np.concatenate(gt_contours))
        pad = max(
            (min_gt_size - gt_w + 1) // 2,
            (min_gt_size - gt_h + 1) // 2,
            0,
        )
        if pad > 0:
            dilate_kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)
            )
            ground_truth = cv.dilate(ground_truth, dilate_kernel, iterations=1)

    return ground_truth


def load_pipeline(model_id: str, device: str = "cuda"):
    """Load the inpainting diffusion pipeline."""
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


def generate_lesion(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str = "a mammogram with a lesion",
    num_inference_steps: int = 40,
    guidance_scale: float = 4.0,
) -> np.ndarray:
    """Generate a lesion on the mammogram using inpainting."""
    with torch.autocast("cuda"), torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
        ).images[0]

    result_array = np.array(result)
    if len(result_array.shape) == 3:
        result_array = cv.cvtColor(result_array, cv.COLOR_RGB2GRAY)

    return result_array


def process_images(config: dict):
    """Process mammography images and generate synthetic lesions via inpainting.

    Outputs images and masks in {study_id}/{image_id}.png structure,
    plus a dataset_metadata.jsonl with per-image generation parameters.
    """
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    annotations_file = config.get("annotations_file")
    model_id = config.get("model_id", "Likalto4/vindr_lesion-inpainting")
    device = config.get("device", "cuda")
    prompt = config.get("prompt", "a mammogram with a lesion")
    num_inference_steps = config.get("num_inference_steps", 40)
    guidance_scale = config.get("guidance_scale", 4.0)
    min_lesion_size = config.get("min_lesion_size", 30)
    max_lesion_size = config.get("max_lesion_size", 80)
    compositing = config.get("compositing", True)
    taper_pixels = config.get("taper_pixels", 15)
    created = config.get("created", "")
    if created:
        created = f"{created}-{min_lesion_size}-{max_lesion_size}"
        if compositing:
            created = f"{created}-blending"
        output_dir = str(Path(output_dir) / created)
    num_images = config.get("num_images")
    seed = config.get("seed")
    intensity_match_dilate = config.get("intensity_match_dilate", 20)
    intensity_matching = config.get("intensity_matching", True)
    diff_threshold = config.get("diff_threshold", 5)
    morph_kernel_size = config.get("morph_kernel_size", 3)
    min_gt_fraction = config.get("min_gt_fraction", 0.9)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mask_output_path = output_path / "masks"
    mask_output_path.mkdir(parents=True, exist_ok=True)
    ground_truth_path = output_path / "ground_truth"
    ground_truth_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

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

    logger.info(f"Processing {len(image_files)} images")
    logger.info(f"Loading inpainting model: {model_id}")

    pipe = load_pipeline(model_id, device)

    metadata_records = []

    for img_path in tqdm(image_files, desc="Generating lesions"):
        try:
            image_gray = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if image_gray is None:
                continue

            breast_bbox = get_breast_bbox(image_gray)
            if breast_bbox is None:
                continue

            _, breast_mask = cv.threshold(image_gray, 10, 255, cv.THRESH_BINARY)

            lesion_bbox = sample_lesion_bbox(
                breast_bbox, breast_mask,
                min_size=min_lesion_size, max_size=max_lesion_size,
            )
            if lesion_bbox is None:
                continue

            mask = create_lesion_mask(
                (image_gray.shape[1], image_gray.shape[0]), lesion_bbox
            )

            image_rgb = Image.fromarray(image_gray).convert("RGB")

            raw_result = generate_lesion(
                pipe, image_rgb, mask,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            if compositing:
                mask_array = np.array(mask)
                result = composite_lesion(
                    image_gray, raw_result, mask_array,
                    taper_pixels=taper_pixels,
                    dilate_radius=intensity_match_dilate,
                    intensity_matching=intensity_matching,
                )
            else:
                result = raw_result

            ground_truth = extract_lesion_ground_truth(
                image_gray, result, lesion_bbox,
                diff_threshold=diff_threshold,
                morph_kernel_size=morph_kernel_size,
                min_lesion_size=min_lesion_size,
                min_gt_fraction=min_gt_fraction,
            )

            relative_path = img_path.relative_to(input_path)
            study_id = str(relative_path.parent)
            image_id = relative_path.stem

            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(output_file), result)

            mask_file = mask_output_path / relative_path
            mask_file.parent.mkdir(parents=True, exist_ok=True)
            mask.save(str(mask_file))

            gt_file = ground_truth_path / relative_path
            gt_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(gt_file), ground_truth)

            metadata_records.append({
                "image_path": str(output_file),
                "mask_path": str(mask_file),
                "ground_truth_mask_path": str(gt_file),
                "source_image_path": str(img_path),
                "image_id": image_id,
                "study_id": study_id,
                "label": 1,
                "transformation": "inpainting",
                "model_id": model_id,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "lesion_x1": int(lesion_bbox[0]),
                "lesion_y1": int(lesion_bbox[1]),
                "lesion_x2": int(lesion_bbox[2]),
                "lesion_y2": int(lesion_bbox[3]),
            })

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
        description="Generate synthetic mammograms with lesions using inpainting"
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
