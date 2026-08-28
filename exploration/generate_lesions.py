"""
Generate synthetic mammograms with lesions using the Likalto4/vindr_lesion-inpainting model.

This script takes preprocessed mammography images and adds synthetic lesions
using a fine-tuned Stable Diffusion inpainting model.

Usage:
    python generate_lesions.py --config config/generate_lesions.yaml
"""

import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_healthy_image_ids(annotations_file: str) -> set[str]:
    """Load the set of healthy image IDs from the annotations file.

    Args:
        annotations_file: Path to finding_annotations.csv

    Returns:
        Set of image_ids that have no findings (healthy)
    """
    df = pd.read_csv(annotations_file)
    healthy_df = df[df["finding_categories"] == "['No Finding']"]
    return set(healthy_df["image_id"].unique())


def get_breast_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Extract the bounding box of the breast region from a mammogram.

    Args:
        image: Grayscale mammogram as numpy array

    Returns:
        Tuple of (x1, y1, x2, y2) or None if no breast region found
    """
    # Threshold to get binary mask
    _, mask = cv.threshold(image, 10, 255, cv.THRESH_BINARY)

    # Find connected components and keep the largest one (breast)
    nb_components, output, stats, _ = cv.connectedComponentsWithStats(
        mask, connectivity=4
    )

    if nb_components < 2:
        return None

    # Find the largest component (excluding background at index 0)
    sizes = stats[1:, cv.CC_STAT_AREA]
    max_label = 1 + np.argmax(sizes)

    # Create mask of the largest component
    breast_mask = np.zeros(output.shape, dtype=np.uint8)
    breast_mask[output == max_label] = 255

    # Find contours and get bounding box
    contours, _ = cv.findContours(breast_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if not contours:
        return None

    x, y, w, h = cv.boundingRect(contours[0])
    return x, y, x + w, y + h


def sample_lesion_bbox(
    breast_bbox: tuple[int, int, int, int],
    breast_mask: np.ndarray,
    min_size: int = 30,
    max_size: int = 80,
    max_attempts: int = 100,
) -> tuple[int, int, int, int] | None:
    """Sample a random bounding box for a lesion within the breast region.

    Args:
        breast_bbox: Bounding box of the breast (x1, y1, x2, y2)
        breast_mask: Binary mask of the breast region
        min_size: Minimum lesion size in pixels
        max_size: Maximum lesion size in pixels
        max_attempts: Maximum attempts to find a valid location

    Returns:
        Tuple of (x1, y1, x2, y2) or None if no valid location found
    """
    x1_b, y1_b, x2_b, y2_b = breast_bbox

    for _ in range(max_attempts):
        # Sample lesion size
        width = np.random.randint(min_size, max_size)
        height = np.random.randint(min_size, max_size)

        # Sample top-left corner within breast bbox
        if x2_b - width <= x1_b or y2_b - height <= y1_b:
            continue

        x1 = np.random.randint(x1_b, x2_b - width)
        y1 = np.random.randint(y1_b, y2_b - height)
        x2 = x1 + width
        y2 = y1 + height

        # Check if the entire lesion bbox is within the breast mask
        lesion_region = breast_mask[y1:y2, x1:x2]
        if lesion_region.shape[0] > 0 and lesion_region.shape[1] > 0:
            if np.all(lesion_region > 0):
                return x1, y1, x2, y2

    return None


def create_lesion_mask(
    image_size: tuple[int, int], lesion_bbox: tuple[int, int, int, int]
) -> Image.Image:
    """Create a binary mask for the lesion region.

    Args:
        image_size: Size of the image (width, height)
        lesion_bbox: Bounding box of the lesion (x1, y1, x2, y2)

    Returns:
        PIL Image mask (white region where lesion will be inpainted)
    """
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    x1, y1, x2, y2 = lesion_bbox
    mask[y1:y2, x1:x2] = 255
    return Image.fromarray(mask)


def load_pipeline(model_id: str, device: str = "cuda"):
    """Load the inpainting diffusion pipeline.

    Args:
        model_id: HuggingFace model ID
        device: Device to run on ("cuda" or "cpu")

    Returns:
        Configured diffusion pipeline
    """
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Enable memory optimizations if available
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available

    return pipe


def generate_lesion(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str = "a mammogram with a lesion",
    num_inference_steps: int = 40,
    guidance_scale: float = 4.0,
) -> np.ndarray:
    """Generate a lesion on the mammogram using inpainting.

    Args:
        pipe: Diffusion pipeline
        image: Input mammogram as PIL Image (RGB)
        mask: Binary mask indicating where to inpaint
        prompt: Text prompt for generation
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale

    Returns:
        Generated image as grayscale numpy array
    """
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

    # Convert to grayscale
    result_array = np.array(result)
    if len(result_array.shape) == 3:
        result_array = cv.cvtColor(result_array, cv.COLOR_RGB2GRAY)

    return result_array


def process_images(config: dict):
    """Process mammography images and generate synthetic lesions.

    Args:
        config: Configuration dictionary with all parameters
    """
    # Extract config values
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
    num_images = config.get("num_images")
    seed = config.get("seed")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory for this run: {output_path}")

    # Save config to output directory for reproducibility
    config_output_path = output_path / "config.yaml"
    with open(config_output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved config to {config_output_path}")

    # Load healthy image IDs if annotations file provided
    healthy_ids = None
    if annotations_file:
        if not os.path.exists(annotations_file):
            print(f"Warning: Annotations file not found: {annotations_file}")
        else:
            healthy_ids = load_healthy_image_ids(annotations_file)
            print(f"Loaded {len(healthy_ids)} healthy image IDs from annotations")

    # Collect all PNG images (recursively)
    image_files = list(input_path.rglob("*.png"))

    # Filter to only healthy images if annotations provided
    if healthy_ids is not None:
        original_count = len(image_files)
        image_files = [f for f in image_files if f.stem in healthy_ids]
        skipped = original_count - len(image_files)
        if skipped > 0:
            print(f"Filtered out {skipped} images with existing lesions")

    if num_images is not None:
        image_files = image_files[:num_images]

    if not image_files:
        print(f"No PNG images found in {input_dir}")
        if healthy_ids is not None:
            print("(after filtering for healthy images)")
        return

    print(f"Found {len(image_files)} healthy images to process")
    print(f"Loading inpainting model: {model_id}")

    pipe = load_pipeline(model_id, device)

    # Metadata for generated lesions
    metadata_records = []

    for img_path in tqdm(image_files, desc="Generating lesions"):
        try:
            # Load image
            image_gray = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if image_gray is None:
                print(f"Failed to load {img_path}")
                continue

            # Get breast bounding box
            breast_bbox = get_breast_bbox(image_gray)
            if breast_bbox is None:
                print(f"Could not detect breast region in {img_path}")
                continue

            # Create breast mask for sampling
            _, breast_mask = cv.threshold(image_gray, 10, 255, cv.THRESH_BINARY)

            # Sample lesion location
            lesion_bbox = sample_lesion_bbox(
                breast_bbox,
                breast_mask,
                min_size=min_lesion_size,
                max_size=max_lesion_size,
            )
            if lesion_bbox is None:
                print(f"Could not find valid lesion location in {img_path}")
                continue

            # Create mask for inpainting
            mask = create_lesion_mask(
                (image_gray.shape[1], image_gray.shape[0]), lesion_bbox
            )

            # Convert grayscale to RGB for the model
            image_rgb = Image.fromarray(image_gray).convert("RGB")

            # Generate lesion
            result = generate_lesion(
                pipe,
                image_rgb,
                mask,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            # Save result (preserve directory structure)
            relative_path = img_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(output_file), result)

            # Save mask
            mask_file = output_path / "masks" / relative_path
            mask_file.parent.mkdir(parents=True, exist_ok=True)
            mask.save(str(mask_file))

            # Record metadata
            metadata_records.append(
                {
                    "filename": str(relative_path),
                    "lesion_x1": lesion_bbox[0],
                    "lesion_y1": lesion_bbox[1],
                    "lesion_x2": lesion_bbox[2],
                    "lesion_y2": lesion_bbox[3],
                    "breast_x1": breast_bbox[0],
                    "breast_y1": breast_bbox[1],
                    "breast_x2": breast_bbox[2],
                    "breast_y2": breast_bbox[3],
                }
            )

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Save metadata
    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(output_path / "lesion_metadata.csv", index=False)
        print(f"\nSaved metadata to {output_path / 'lesion_metadata.csv'}")

    print(f"\nProcessed {len(metadata_records)} images successfully")
    print(f"Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic mammograms with lesions using inpainting"
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

    process_images(config)


if __name__ == "__main__":
    main()