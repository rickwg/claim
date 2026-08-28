"""
Generate synthetic mammograms with both a diffusion-inpainted lesion and a
geometric local-distortion (twirl or spherize) applied to disjoint regions.

For each healthy mammogram, produce a single combined image (label 1) with
both artifacts plus a single-modification healthy comparator (label 0)
matching the non-target artifact. The non-target artifact appears in both
classes — the distractor-in-both-classes setup. A `classification_target`
config flag selects which artifact is the discriminator. The discriminator
occupies a fixed position slot regardless of the target, so lesion and
distortion positions correspond across the two target settings when both are
generated with the same seed.

Three optional knobs vary the difficulty/confounding regime (defaults reproduce
the balanced, perfectly-separable setup above):
  - `discriminator_presence_prob` — fraction of label-1 images that actually
    carry the discriminator; the rest are label-1 with no discriminative signal
    (label noise that caps achievable accuracy below 1).
  - `distractor_pos_rate` / `distractor_neg_rate` — P(distractor present | label):
    equal rates give a class-independent distractor, unequal rates a
    correlated-but-not-causal confound.
  - `entangle_discriminator` — co-locate the discriminator on the distractor's
    footprint instead of using a disjoint slot.
Each record carries `discriminator_present` / `distractor_present` /
`lesion_present` / `distortion_present` flags plus the regime settings.

Outputs (under output_dir):
  - combined/{study_id}/{stem}_combined.png                    — unhealthy class
  - {lesion_only|distortion_only}/{study_id}/{stem}_<v>.png    — healthy class
  - masks/<variant>/{study_id}/...                             — discriminator mask | all-zero
  - ground_truth/<variant>/{study_id}/...                      — discriminator GT  | all-zero
  - distortion_ground_truth/<variant>/{study_id}/...           — circular distortion mask | all-zero
  - distortion_ground_truth_rect/<variant>/{study_id}/...      — lesion-shaped distortion mask | all-zero
  - lesion_ground_truth_rect/<variant>/{study_id}/...          — rectangular lesion mask | all-zero
  - dataset_metadata.jsonl                                     — 2 records per source image

Usage:
    python -m data.generate_combined --config config/generate_combined.yaml
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
from loguru import logger
from PIL import Image
from tqdm import tqdm

from data._common import get_breast_bbox, load_healthy_image_ids
from data.generate_lesions import (
    composite_lesion,
    create_feathered_mask,
    create_lesion_mask,
    extract_lesion_ground_truth,
    generate_lesion as run_inpainting,
    intensity_match,
    load_pipeline,
)
from data.generate_masked_distortion import (
    apply_spherize,
    apply_twirl,
    create_blend_mask,
)
from utils import dump_as_jsonl_file


SUPPORTED_DISTORTIONS = ("twirl", "spherize")
SUPPORTED_CLASSIFICATION_TARGETS = ("lesion", "distortion")
SUPPORTED_LESION_GT_SHAPES = ("rectangular", "ellipse")
SUPPORTED_SALIENCE_MODES = ("scalar", "unsharp")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _circle_intersects_bbox(
    center: tuple[int, int], radius: int, bbox: tuple[int, int, int, int]
) -> bool:
    """Return True iff a closed disk overlaps an axis-aligned rectangle."""
    cx, cy = center
    x1, y1, x2, y2 = bbox
    closest_x = max(x1, min(cx, x2))
    closest_y = max(y1, min(cy, y2))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= radius * radius


def _bboxes_disjoint(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    """Return True iff two axis-aligned bboxes do not overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1


def _bbox_centered_at(
    center: tuple[int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Axis-aligned bbox of the given size centered on a point."""
    cx, cy = center
    x1 = cx - width // 2
    y1 = cy - height // 2
    return x1, y1, x1 + width, y1 + height


def _bbox_within_breast(
    bbox: tuple[int, int, int, int], breast_mask: np.ndarray
) -> bool:
    """Return True iff the bbox lies fully inside breast tissue."""
    x1, y1, x2, y2 = bbox
    height, width = breast_mask.shape
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    region = breast_mask[y1:y2, x1:x2]
    return region.size > 0 and bool(np.all(region > 0))


def _disk_within_breast(
    center: tuple[int, int], radius: int, breast_mask: np.ndarray
) -> bool:
    """Return True iff the closed disk lies fully inside breast tissue."""
    cx, cy = center
    height, width = breast_mask.shape
    if cx - radius < 0 or cy - radius < 0 or cx + radius + 1 > width or cy + radius + 1 > height:
        return False
    y_offsets, x_offsets = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = x_offsets**2 + y_offsets**2 <= radius**2
    region = breast_mask[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
    return bool(np.all(region[disk] > 0))


def _sample_slot_center(
    breast_bbox: tuple[int, int, int, int],
    breast_mask: np.ndarray,
    lesion_width: int,
    lesion_height: int,
    distortion_radius: int,
    max_attempts: int,
) -> tuple[int, int] | None:
    """Sample a center able to host both a lesion bbox and a distortion disk."""
    x1_b, y1_b, x2_b, y2_b = breast_bbox
    margin = max((lesion_width + 1) // 2, (lesion_height + 1) // 2, distortion_radius)
    low_x, high_x = x1_b + margin, x2_b - margin
    low_y, high_y = y1_b + margin, y2_b - margin
    if low_x >= high_x or low_y >= high_y:
        return None

    for _ in range(max_attempts):
        cx = int(np.random.randint(low_x, high_x))
        cy = int(np.random.randint(low_y, high_y))
        bbox = _bbox_centered_at((cx, cy), lesion_width, lesion_height)
        if _bbox_within_breast(bbox, breast_mask) and _disk_within_breast(
            (cx, cy), distortion_radius, breast_mask
        ):
            return cx, cy

    return None


def sample_corresponding_regions(
    breast_bbox: tuple[int, int, int, int],
    breast_mask: np.ndarray,
    lesion_min_size: int,
    lesion_max_size: int,
    distortion_min_size: int,
    distortion_max_size: int,
    max_attempts: int,
    entangle: bool = False,
) -> dict | None:
    """Sample a discriminator slot and a distractor slot with target-independent positions.

    Each slot can host either artifact, and lesion bboxes, distortion disks, and
    lesion-shaped distortion ground truths all stay disjoint across the two
    slots, so swapping the classification target moves artifact types between
    slots while leaving both positions fixed.
    """
    for _ in range(max_attempts):
        lesion_width = int(np.random.randint(lesion_min_size, lesion_max_size))
        lesion_height = int(np.random.randint(lesion_min_size, lesion_max_size))
        distortion_radius = int(np.random.randint(
            distortion_min_size // 2, distortion_max_size // 2 + 1
        ))

        target_center = _sample_slot_center(
            breast_bbox, breast_mask,
            lesion_width, lesion_height, distortion_radius, max_attempts,
        )
        if target_center is None:
            continue

        if entangle:
            return {
                "target_center": target_center,
                "distractor_center": target_center,
                "lesion_size": (lesion_width, lesion_height),
                "distortion_radius": distortion_radius,
            }

        distractor_center = _sample_slot_center(
            breast_bbox, breast_mask,
            lesion_width, lesion_height, distortion_radius, max_attempts,
        )
        if distractor_center is None:
            continue

        target_bbox = _bbox_centered_at(target_center, lesion_width, lesion_height)
        distractor_bbox = _bbox_centered_at(distractor_center, lesion_width, lesion_height)
        distortion_clears_lesion_bbox = not _circle_intersects_bbox(
            distractor_center, distortion_radius, target_bbox
        ) and not _circle_intersects_bbox(
            target_center, distortion_radius, distractor_bbox
        )
        rectangles_disjoint = _bboxes_disjoint(target_bbox, distractor_bbox)
        if distortion_clears_lesion_bbox and rectangles_disjoint:
            return {
                "target_center": target_center,
                "distractor_center": distractor_center,
                "lesion_size": (lesion_width, lesion_height),
                "distortion_radius": distortion_radius,
            }

    return None


def merge_combined(
    original: np.ndarray,
    lesion_painted: np.ndarray,
    distorted: np.ndarray,
    m_l: np.ndarray,
    m_d: np.ndarray,
) -> np.ndarray:
    """Mask-aware merge: combined = original*(1-m_l-m_d) + lesion_painted*m_l + distorted*m_d.

    Asserts m_l · m_d ≈ 0 (disjoint regions). The pre-blend painted lesion and
    pre-blend distorted image are blended with their respective soft masks so
    that pixels outside the union of the two regions are bit-identical to the
    original — independent of how the single-modification images are saved.
    """
    overlap = float((m_l * m_d).max())
    if overlap > 1e-9:
        raise AssertionError(
            f"Lesion and distortion blend masks overlap (max product = {overlap:.6f}); "
            "regions must be disjoint."
        )
    weight_orig = 1.0 - m_l - m_d
    combined = (
        original.astype(np.float64) * weight_orig
        + lesion_painted.astype(np.float64) * m_l
        + distorted.astype(np.float64) * m_d
    )
    return np.clip(combined, 0, 255).astype(np.uint8)


def amplify_region_contrast(
    image: np.ndarray, region_mask: np.ndarray, gain: float
) -> np.ndarray:
    """Stretch intensity contrast about the region mean to raise its edge salience.

    Sobel is linear and the region mean is a scalar, so interior edge energy
    scales with `gain`; `gain=1.0` is the identity. Pixels outside the region
    and any empty region are returned unchanged.
    """
    region = region_mask > 0
    if not region.any():
        return image
    amplified = image.astype(np.float64)
    region_mean = amplified[region].mean()
    amplified[region] = region_mean + gain * (amplified[region] - region_mean)
    return np.clip(amplified, 0, 255).astype(np.uint8)


def sharpen_region_contrast(
    image: np.ndarray, region_mask: np.ndarray, gain: float, sigma: float
) -> np.ndarray:
    """Unsharp-mask the region: amplify only its high-frequency detail by `gain`.

    Preserves low-frequency brightness (the high-pass residual is scaled, the
    blurred base is not), so it raises edge salience with far less clipping than
    a full contrast stretch. `gain=1.0` is the identity; pixels outside the
    region and any empty region are returned unchanged.
    """
    region = region_mask > 0
    if not region.any():
        return image
    amplified = image.astype(np.float64)
    high_frequency = amplified - cv.GaussianBlur(amplified, (0, 0), sigma)
    amplified[region] = amplified[region] + (gain - 1.0) * high_frequency[region]
    return np.clip(amplified, 0, 255).astype(np.uint8)


def pick_healthy_variant(classification_target: str) -> str:
    return "distortion_only" if classification_target == "lesion" else "lesion_only"


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), array)


def _validate_config(config: dict) -> None:
    target = config.get("classification_target", "lesion")
    if target not in SUPPORTED_CLASSIFICATION_TARGETS:
        raise ValueError(
            f"classification_target must be one of {SUPPORTED_CLASSIFICATION_TARGETS}, got {target!r}"
        )

    distortion_type = config.get("distortion_type", "twirl")
    if distortion_type not in SUPPORTED_DISTORTIONS:
        raise ValueError(
            f"distortion_type must be one of {SUPPORTED_DISTORTIONS}, got {distortion_type!r}"
        )

    gt_shape = config.get("lesion_ground_truth_shape", "rectangular")
    if gt_shape not in SUPPORTED_LESION_GT_SHAPES:
        raise ValueError(
            f"lesion_ground_truth_shape must be one of {SUPPORTED_LESION_GT_SHAPES}, got {gt_shape!r}"
        )

    for rate_name in ("discriminator_presence_prob", "distractor_pos_rate", "distractor_neg_rate"):
        rate = config.get(rate_name, 1.0)
        if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"{rate_name} must be a probability in [0, 1], got {rate!r}")

    gain = config.get("lesion_salience_gain", 1.0)
    if not isinstance(gain, (int, float)) or isinstance(gain, bool) or float(gain) < 0.0:
        raise ValueError(f"lesion_salience_gain must be a non-negative number, got {gain!r}")

    salience_mode = config.get("lesion_salience_mode", "scalar")
    if salience_mode not in SUPPORTED_SALIENCE_MODES:
        raise ValueError(
            f"lesion_salience_mode must be one of {SUPPORTED_SALIENCE_MODES}, got {salience_mode!r}"
        )


def _resolve_output_dir(
    template: str, distortion_type: str, classification_target: str
) -> str:
    try:
        return template.format(
            distortion_type=distortion_type,
            classification_target=classification_target,
        )
    except KeyError as error:
        unknown = error.args[0]
        raise ValueError(
            f"Unsupported placeholder '{{{unknown}}}' in output_dir template "
            f"{template!r}. Only '{{distortion_type}}' and "
            f"'{{classification_target}}' are supported."
        ) from error


def _generate_distortion_artifacts(
    image_gray: np.ndarray,
    center: tuple[int, int],
    radius: int,
    distortion_type: str,
    twirl_angle: float,
    spherize_amount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (distortion_only_image, distorted_pre_blend, soft_blend_mask, binary_mask)."""
    if distortion_type == "twirl":
        distorted = apply_twirl(image_gray, center, radius, twirl_angle)
    else:
        distorted = apply_spherize(image_gray, center, radius, spherize_amount)

    blend_mask = create_blend_mask(image_gray.shape, center, radius)
    distortion_only = (
        image_gray.astype(np.float64) * (1 - blend_mask)
        + distorted.astype(np.float64) * blend_mask
    )
    distortion_only = np.clip(distortion_only, 0, 255).astype(np.uint8)

    binary_mask = np.zeros(image_gray.shape, dtype=np.uint8)
    binary_mask[blend_mask > 0] = 255
    return distortion_only, distorted, blend_mask, binary_mask


def _draw(probability: float) -> bool:
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    return bool(np.random.random() < probability)


def _resolve_presence(
    classification_target: str,
    discriminator_in_positive: bool,
    distractor_in_positive: bool,
    distractor_in_negative: bool,
) -> dict:
    """Map discriminator/distractor presence draws onto per-artifact presence.

    The discriminator drives the label, so it is never placed in the negative
    (label-0) image; the distractor may appear in either class at its own rate.
    """
    if classification_target == "lesion":
        return {
            "lesion_in_positive": discriminator_in_positive,
            "distortion_in_positive": distractor_in_positive,
            "lesion_in_negative": False,
            "distortion_in_negative": distractor_in_negative,
        }
    return {
        "distortion_in_positive": discriminator_in_positive,
        "lesion_in_positive": distractor_in_positive,
        "distortion_in_negative": False,
        "lesion_in_negative": distractor_in_negative,
    }


def _compose_image(
    original: np.ndarray,
    lesion_painted: np.ndarray,
    distorted_pre_blend: np.ndarray,
    lesion_blend_mask: np.ndarray,
    distortion_blend_mask: np.ndarray,
    lesion_present: bool,
    distortion_present: bool,
    entangle: bool,
    distortion_center: tuple[int, int],
    distortion_radius: int,
    distortion_type: str,
    twirl_angle: float,
    spherize_amount: float,
) -> np.ndarray:
    """Composite the requested artifacts onto the source image.

    Disjoint mode blends both artifacts with the mask-aware merge; an absent
    artifact contributes a zero blend mask, so pixels outside a present artifact
    stay identical to the source. Entangled mode layers the distortion on top of
    the lesioned base so the two artifacts share a footprint rather than
    occupying separate slots.
    """
    no_blend = np.zeros_like(distortion_blend_mask)
    lesion_mask = lesion_blend_mask if lesion_present else no_blend
    if not entangle:
        distortion_mask = distortion_blend_mask if distortion_present else no_blend
        return merge_combined(
            original=original,
            lesion_painted=lesion_painted,
            distorted=distorted_pre_blend,
            m_l=lesion_mask,
            m_d=distortion_mask,
        )
    base = merge_combined(
        original=original,
        lesion_painted=lesion_painted,
        distorted=distorted_pre_blend,
        m_l=lesion_mask,
        m_d=no_blend,
    )
    if not distortion_present:
        return base
    if distortion_type == "twirl":
        distorted_base = apply_twirl(base, distortion_center, distortion_radius, twirl_angle)
    else:
        distorted_base = apply_spherize(base, distortion_center, distortion_radius, spherize_amount)
    blended = (
        base.astype(np.float64) * (1 - distortion_blend_mask)
        + distorted_base.astype(np.float64) * distortion_blend_mask
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _select_masks(
    classification_target: str,
    lesion_present: bool,
    distortion_present: bool,
    lesion_mask_array: np.ndarray,
    lesion_gt: np.ndarray,
    distortion_binary_mask: np.ndarray,
    distortion_rect_mask: np.ndarray,
    zero_mask: np.ndarray,
) -> dict:
    """Pick the discriminator/auxiliary masks for one image given what it contains.

    A mask is all-zero when its artifact is absent, so the discriminator ground
    truth is empty for a label-1 image that drew no discriminator (label noise).
    """
    if classification_target == "lesion":
        discriminator_present = lesion_present
        discriminator_mask = lesion_mask_array
        discriminator_gt = lesion_gt
    else:
        discriminator_present = distortion_present
        discriminator_mask = distortion_binary_mask
        discriminator_gt = distortion_binary_mask
    return {
        "mask": discriminator_mask if discriminator_present else zero_mask,
        "ground_truth": discriminator_gt if discriminator_present else zero_mask,
        "distortion_ground_truth": distortion_binary_mask if distortion_present else zero_mask,
        "distortion_ground_truth_rect": distortion_rect_mask if distortion_present else zero_mask,
        "lesion_ground_truth_rect": lesion_mask_array if lesion_present else zero_mask,
    }


def process_images(config: dict):
    _validate_config(config)

    input_dir = config["input_dir"]
    output_dir_template = config["output_dir"]
    annotations_file = config.get("annotations_file")

    classification_target = config.get("classification_target", "lesion")
    distortion_type = config.get("distortion_type", "twirl")
    twirl_angle = config.get("twirl_angle", -237)
    spherize_amount = config.get("spherize_amount", -67)
    distortion_min_size = config.get("distortion_min_size", 60)
    distortion_max_size = config.get("distortion_max_size", 90)

    model_id = config.get("model_id", "Likalto4/inpainting_vindr_massbs16")
    device = config.get("device", "cuda")
    prompt = config.get("prompt", "a mammogram with a lesion")
    num_inference_steps = config.get("num_inference_steps", 40)
    guidance_scale = config.get("guidance_scale", 4.0)
    lesion_min_size = config.get("lesion_min_size", 30)
    lesion_max_size = config.get("lesion_max_size", 80)
    compositing = config.get("compositing", True)
    intensity_matching = config.get("intensity_matching", True)
    taper_pixels = config.get("taper_pixels", 15)
    intensity_match_dilate = config.get("intensity_match_dilate", 20)
    lesion_salience_gain = float(config.get("lesion_salience_gain", 1.0))
    lesion_salience_mode = config.get("lesion_salience_mode", "scalar")
    lesion_salience_sigma = float(config.get("lesion_salience_sigma", 3.0))

    lesion_gt_shape = config.get("lesion_ground_truth_shape", "rectangular")
    diff_threshold = config.get("diff_threshold", 5)
    morph_kernel_size = config.get("morph_kernel_size", 3)
    min_gt_fraction = config.get("min_gt_fraction", 0.6)

    disjoint_max_attempts = config.get("disjoint_max_attempts", 50)
    discriminator_presence_prob = config.get("discriminator_presence_prob", 1.0)
    distractor_pos_rate = config.get("distractor_pos_rate", 1.0)
    distractor_neg_rate = config.get("distractor_neg_rate", 1.0)
    entangle_discriminator = bool(config.get("entangle_discriminator", False))
    num_images = config.get("num_images")
    seed = config.get("seed")
    created = config.get("created", "")

    output_dir = _resolve_output_dir(
        output_dir_template, distortion_type, classification_target,
    )
    if created:
        created = f"{created}-{lesion_min_size}-{lesion_max_size}"
        output_dir = str(Path(output_dir) / created)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    healthy_variant = pick_healthy_variant(classification_target)
    logger.info(
        f"Generating combined ({distortion_type}, target={classification_target}) "
        f"for {len(image_files)} sources; healthy variant = {healthy_variant}"
    )
    logger.info(f"Loading inpainting model: {model_id}")
    pipe = load_pipeline(model_id, device)

    metadata_records = []
    skipped_disjoint = 0
    for img_path in tqdm(image_files, desc="Generating combined"):
        try:
            image_gray = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if image_gray is None:
                continue

            breast_bbox = get_breast_bbox(image_gray)
            if breast_bbox is None:
                continue

            _, breast_mask = cv.threshold(image_gray, 10, 255, cv.THRESH_BINARY)

            sampling_result = sample_corresponding_regions(
                breast_bbox=breast_bbox,
                breast_mask=breast_mask,
                lesion_min_size=lesion_min_size,
                lesion_max_size=lesion_max_size,
                distortion_min_size=distortion_min_size,
                distortion_max_size=distortion_max_size,
                max_attempts=disjoint_max_attempts,
                entangle=entangle_discriminator,
            )
            if sampling_result is None:
                skipped_disjoint += 1
                continue

            target_center = sampling_result["target_center"]
            distractor_center = sampling_result["distractor_center"]
            lesion_width, lesion_height = sampling_result["lesion_size"]
            distortion_radius = sampling_result["distortion_radius"]

            if classification_target == "lesion":
                lesion_center, distortion_center = target_center, distractor_center
            else:
                lesion_center, distortion_center = distractor_center, target_center
            lesion_bbox = _bbox_centered_at(lesion_center, lesion_width, lesion_height)

            lesion_mask_pil = create_lesion_mask(
                (image_gray.shape[1], image_gray.shape[0]), lesion_bbox
            )
            lesion_mask_array = np.array(lesion_mask_pil)
            image_rgb = Image.fromarray(image_gray).convert("RGB")
            raw_lesion = run_inpainting(
                pipe, image_rgb, lesion_mask_pil,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            if compositing:
                lesion_only = composite_lesion(
                    image_gray, raw_lesion, lesion_mask_array,
                    taper_pixels=taper_pixels,
                    dilate_radius=intensity_match_dilate,
                    intensity_matching=intensity_matching,
                )
            else:
                lesion_only = raw_lesion

            if intensity_matching:
                lesion_painted = intensity_match(
                    image_gray, raw_lesion, lesion_mask_array,
                    dilate_radius=intensity_match_dilate,
                )
            else:
                lesion_painted = raw_lesion
            if lesion_salience_gain != 1.0:
                if lesion_salience_mode == "unsharp":
                    lesion_painted = sharpen_region_contrast(
                        lesion_painted, lesion_mask_array,
                        lesion_salience_gain, lesion_salience_sigma,
                    )
                else:
                    lesion_painted = amplify_region_contrast(
                        lesion_painted, lesion_mask_array, lesion_salience_gain,
                    )
            lesion_blend_mask = create_feathered_mask(
                lesion_mask_array, taper_pixels=taper_pixels,
            )

            distortion_only, distorted_pre_blend, distortion_blend_mask, distortion_binary_mask = (
                _generate_distortion_artifacts(
                    image_gray=image_gray,
                    center=distortion_center,
                    radius=distortion_radius,
                    distortion_type=distortion_type,
                    twirl_angle=twirl_angle,
                    spherize_amount=spherize_amount,
                )
            )

            if lesion_gt_shape == "ellipse":
                lesion_gt = extract_lesion_ground_truth(
                    image_gray, lesion_only, lesion_bbox,
                    diff_threshold=diff_threshold,
                    morph_kernel_size=morph_kernel_size,
                    min_lesion_size=lesion_min_size,
                    min_gt_fraction=min_gt_fraction,
                )
                lesion_gt = cv.bitwise_and(lesion_gt, lesion_mask_array)
            else:
                lesion_gt = lesion_mask_array.copy()

            relative_path = img_path.relative_to(input_path)
            study_id = str(relative_path.parent)
            stem = relative_path.stem
            zero_mask = np.zeros(image_gray.shape, dtype=np.uint8)
            distortion_rect_bbox = _bbox_centered_at(
                distortion_center, lesion_width, lesion_height
            )
            distortion_rect_mask = np.array(
                create_lesion_mask(
                    (image_gray.shape[1], image_gray.shape[0]), distortion_rect_bbox
                )
            )

            discriminator_in_positive = _draw(discriminator_presence_prob)
            distractor_in_positive = _draw(distractor_pos_rate)
            distractor_in_negative = _draw(distractor_neg_rate)
            presence = _resolve_presence(
                classification_target,
                discriminator_in_positive,
                distractor_in_positive,
                distractor_in_negative,
            )
            positive_image = _compose_image(
                original=image_gray,
                lesion_painted=lesion_painted,
                distorted_pre_blend=distorted_pre_blend,
                lesion_blend_mask=lesion_blend_mask,
                distortion_blend_mask=distortion_blend_mask,
                lesion_present=presence["lesion_in_positive"],
                distortion_present=presence["distortion_in_positive"],
                entangle=entangle_discriminator,
                distortion_center=distortion_center,
                distortion_radius=distortion_radius,
                distortion_type=distortion_type,
                twirl_angle=twirl_angle,
                spherize_amount=spherize_amount,
            )
            negative_image = _compose_image(
                original=image_gray,
                lesion_painted=lesion_painted,
                distorted_pre_blend=distorted_pre_blend,
                lesion_blend_mask=lesion_blend_mask,
                distortion_blend_mask=distortion_blend_mask,
                lesion_present=presence["lesion_in_negative"],
                distortion_present=presence["distortion_in_negative"],
                entangle=entangle_discriminator,
                distortion_center=distortion_center,
                distortion_radius=distortion_radius,
                distortion_type=distortion_type,
                twirl_angle=twirl_angle,
                spherize_amount=spherize_amount,
            )
            positive_masks = _select_masks(
                classification_target,
                lesion_present=presence["lesion_in_positive"],
                distortion_present=presence["distortion_in_positive"],
                lesion_mask_array=lesion_mask_array,
                lesion_gt=lesion_gt,
                distortion_binary_mask=distortion_binary_mask,
                distortion_rect_mask=distortion_rect_mask,
                zero_mask=zero_mask,
            )
            negative_masks = _select_masks(
                classification_target,
                lesion_present=presence["lesion_in_negative"],
                distortion_present=presence["distortion_in_negative"],
                lesion_mask_array=lesion_mask_array,
                lesion_gt=lesion_gt,
                distortion_binary_mask=distortion_binary_mask,
                distortion_rect_mask=distortion_rect_mask,
                zero_mask=zero_mask,
            )

            combined_image_id = f"{stem}_combined"
            combined_image_path = output_path / "combined" / study_id / f"{combined_image_id}.png"
            combined_mask_path = output_path / "masks" / "combined" / study_id / f"{combined_image_id}.png"
            combined_gt_path = output_path / "ground_truth" / "combined" / study_id / f"{combined_image_id}.png"
            combined_dist_gt_path = (
                output_path / "distortion_ground_truth" / "combined" / study_id / f"{combined_image_id}.png"
            )
            combined_dist_gt_rect_path = (
                output_path / "distortion_ground_truth_rect" / "combined" / study_id / f"{combined_image_id}.png"
            )
            combined_lesion_gt_rect_path = (
                output_path / "lesion_ground_truth_rect" / "combined" / study_id / f"{combined_image_id}.png"
            )

            _save_image(combined_image_path, positive_image)
            _save_image(combined_mask_path, positive_masks["mask"])
            _save_image(combined_gt_path, positive_masks["ground_truth"])
            _save_image(combined_dist_gt_path, positive_masks["distortion_ground_truth"])
            _save_image(combined_dist_gt_rect_path, positive_masks["distortion_ground_truth_rect"])
            _save_image(combined_lesion_gt_rect_path, positive_masks["lesion_ground_truth_rect"])

            healthy_image_id = f"{stem}_{healthy_variant}"
            healthy_image_path = output_path / healthy_variant / study_id / f"{healthy_image_id}.png"
            healthy_mask_path = output_path / "masks" / healthy_variant / study_id / f"{healthy_image_id}.png"
            healthy_gt_path = output_path / "ground_truth" / healthy_variant / study_id / f"{healthy_image_id}.png"
            healthy_dist_gt_path = (
                output_path / "distortion_ground_truth" / healthy_variant / study_id / f"{healthy_image_id}.png"
            )
            healthy_dist_gt_rect_path = (
                output_path / "distortion_ground_truth_rect" / healthy_variant / study_id / f"{healthy_image_id}.png"
            )
            healthy_lesion_gt_rect_path = (
                output_path / "lesion_ground_truth_rect" / healthy_variant / study_id / f"{healthy_image_id}.png"
            )

            _save_image(healthy_image_path, negative_image)
            _save_image(healthy_mask_path, negative_masks["mask"])
            _save_image(healthy_gt_path, negative_masks["ground_truth"])
            _save_image(healthy_dist_gt_path, negative_masks["distortion_ground_truth"])
            _save_image(healthy_dist_gt_rect_path, negative_masks["distortion_ground_truth_rect"])
            _save_image(healthy_lesion_gt_rect_path, negative_masks["lesion_ground_truth_rect"])

            distortion_param = (
                {"twirl_angle": twirl_angle}
                if distortion_type == "twirl"
                else {"spherize_amount": spherize_amount}
            )
            common_lesion_fields = {
                "lesion_x1": int(lesion_bbox[0]),
                "lesion_y1": int(lesion_bbox[1]),
                "lesion_x2": int(lesion_bbox[2]),
                "lesion_y2": int(lesion_bbox[3]),
            }
            common_distortion_fields = {
                "distortion_center_x": int(distortion_center[0]),
                "distortion_center_y": int(distortion_center[1]),
                "distortion_radius": int(distortion_radius),
                **distortion_param,
            }
            common_slot_fields = {
                "discriminative_center_x": int(target_center[0]),
                "discriminative_center_y": int(target_center[1]),
                "distractor_center_x": int(distractor_center[0]),
                "distractor_center_y": int(distractor_center[1]),
            }
            generation_regime_fields = {
                "discriminator_presence_prob": float(discriminator_presence_prob),
                "distractor_pos_rate": float(distractor_pos_rate),
                "distractor_neg_rate": float(distractor_neg_rate),
                "distractor_label_correlation": float(distractor_pos_rate - distractor_neg_rate),
                "entangle_discriminator": bool(entangle_discriminator),
                "lesion_salience_gain": float(lesion_salience_gain),
                "lesion_salience_mode": str(lesion_salience_mode),
                "lesion_salience_sigma": float(lesion_salience_sigma),
            }

            combined_record = {
                "image_path": str(combined_image_path),
                "mask_path": str(combined_mask_path),
                "ground_truth_mask_path": str(combined_gt_path),
                "distortion_ground_truth_mask_path": str(combined_dist_gt_path),
                "distortion_ground_truth_rect_mask_path": str(combined_dist_gt_rect_path),
                "lesion_ground_truth_rect_mask_path": str(combined_lesion_gt_rect_path),
                "source_image_path": str(img_path),
                "image_id": combined_image_id,
                "source_image_id": stem,
                "subject_id": study_id,
                "label": 1,
                "variant": "combined",
                "classification_target": classification_target,
                "transformation": f"lesion+{distortion_type}",
                "model_id": model_id,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "discriminator_present": bool(discriminator_in_positive),
                "distractor_present": bool(distractor_in_positive),
                "lesion_present": bool(presence["lesion_in_positive"]),
                "distortion_present": bool(presence["distortion_in_positive"]),
                **common_lesion_fields,
                **common_distortion_fields,
                **common_slot_fields,
                **generation_regime_fields,
                "sibling_image_id": healthy_image_id,
            }

            healthy_record = {
                "image_path": str(healthy_image_path),
                "mask_path": str(healthy_mask_path),
                "ground_truth_mask_path": str(healthy_gt_path),
                "distortion_ground_truth_mask_path": str(healthy_dist_gt_path),
                "distortion_ground_truth_rect_mask_path": str(healthy_dist_gt_rect_path),
                "lesion_ground_truth_rect_mask_path": str(healthy_lesion_gt_rect_path),
                "source_image_path": str(img_path),
                "image_id": healthy_image_id,
                "source_image_id": stem,
                "subject_id": study_id,
                "label": 0,
                "variant": healthy_variant,
                "classification_target": classification_target,
                "discriminator_present": False,
                "distractor_present": bool(distractor_in_negative),
                "lesion_present": bool(presence["lesion_in_negative"]),
                "distortion_present": bool(presence["distortion_in_negative"]),
                **common_lesion_fields,
                **common_distortion_fields,
                **common_slot_fields,
                **generation_regime_fields,
                "sibling_image_id": combined_image_id,
            }
            if healthy_variant == "distortion_only":
                healthy_record["transformation"] = distortion_type
            else:
                healthy_record["transformation"] = "inpainting"
                healthy_record["model_id"] = model_id
                healthy_record["prompt"] = prompt
                healthy_record["num_inference_steps"] = num_inference_steps
                healthy_record["guidance_scale"] = guidance_scale

            metadata_records.append(combined_record)
            metadata_records.append(healthy_record)

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    if metadata_records:
        jsonl_path = str(output_path / "dataset_metadata.jsonl")
        dump_as_jsonl_file(data=metadata_records, file_path=jsonl_path)
        logger.info(f"Saved {len(metadata_records)} records to {jsonl_path}")

    n_sources = len(metadata_records) // 2
    if skipped_disjoint:
        logger.warning(
            f"Skipped {skipped_disjoint} sources after {disjoint_max_attempts} "
            "disjoint-sampling attempts"
        )
    logger.info(f"Processed {n_sources} sources → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined synthetic lesion + local-distortion mammograms."
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
