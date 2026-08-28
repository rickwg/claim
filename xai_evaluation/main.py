import os
from os.path import join
from pathlib import Path

import cv2 as cv
import numpy as np
from loguru import logger
from skimage.filters import sobel as skimage_sobel, sobel_h, sobel_v
from tqdm import tqdm

from common import EnvironmentVariables
from config.configuration import ExperimentConfig
from utils import (
    append_to_jsonl_file,
    dump_as_jsonl_file,
    generate_experiment_dir,
    load_config_file,
    load_jsonl_file,
)

DEFAULT_CONFIG_FILE_PATH = "config/experiments_config.yaml"


def mass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    a = np.sum(y_pred[0.0 < y_true])
    b = np.sum(y_pred)
    return np.divide(a, b, where=(0 != b))


def relative_importance(
    y_pred: np.ndarray, gt_mask: np.ndarray, reference_mask: np.ndarray | None = None
) -> float | None:
    """Attribution density inside the ground truth divided by density outside it.

    (ma / b) / (ima / (d - b)): ma and ima are the attribution mass inside the
    ground truth and inside the comparison region outside it; b is the ground-truth
    pixel count and d the comparison-region pixel count. The comparison region is the
    breast when a reference_mask is given (excluding the black background outside the
    breast, whose near-zero attribution would otherwise dilute the outside density),
    else the whole image. 1.0 = uniform, >1 = concentrated on the ground truth. The
    ground truth spans all available artifacts (one region for standalone data, both
    the discriminative and distractor regions for combined data).
    """
    inside = gt_mask > 0
    if reference_mask is not None:
        outside = (reference_mask > 0) & ~inside
    else:
        outside = ~inside
    ground_truth_pixels = float(np.count_nonzero(inside))
    outside_pixels = float(np.count_nonzero(outside))
    if ground_truth_pixels <= 0 or outside_pixels <= 0:
        return None
    ma = float(mass_accuracy(y_true=gt_mask, y_pred=y_pred))
    ima = float(mass_accuracy(y_true=outside.astype(np.float64), y_pred=y_pred))
    if ima <= 0:
        return None
    return (ma / ground_truth_pixels) / (ima / outside_pixels)


def enrichment(
    attribution: np.ndarray, region_mask: np.ndarray, reference_mask: np.ndarray
) -> float | None:
    """Mean attribution per region pixel divided by mean per reference pixel.

    Size-invariant companion to mass accuracy: 1.0 = no better than a uniform map
    over the reference region, >1 = concentrated in the region. The reference is
    the breast so that near-zero background pixels do not deflate the baseline.
    """
    region = region_mask > 0
    reference = reference_mask > 0
    region_area = float(np.count_nonzero(region))
    reference_area = float(np.count_nonzero(reference))
    if region_area <= 0 or reference_area <= 0:
        return None
    region_density = float(np.sum(attribution[region])) / region_area
    reference_density = float(np.sum(attribution[reference])) / reference_area
    if reference_density <= 0:
        return None
    return region_density / reference_density


def masked_correlation(
    first_map: np.ndarray, second_map: np.ndarray, reference_mask: np.ndarray
) -> float | None:
    """Pearson correlation between two maps over reference-region pixels only.

    Restricting to the breast avoids the shared near-zero background inflating the
    correlation between an attribution map and an edge-detector map.
    """
    reference = reference_mask > 0
    if first_map.shape != reference.shape or second_map.shape != reference.shape:
        return None
    first_values = first_map[reference].ravel()
    second_values = second_map[reference].ravel()
    if first_values.size < 2:
        return None
    if float(np.std(first_values)) == 0.0 or float(np.std(second_values)) == 0.0:
        return None
    return float(np.corrcoef(first_values, second_values)[0, 1])


def _load_experiment_config() -> ExperimentConfig:
    config_path = os.environ.get(
        EnvironmentVariables.CONFIG_FILE_PATH.value,
        DEFAULT_CONFIG_FILE_PATH,
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    return ExperimentConfig.from_dict(load_config_file(file_path=config_path))


def _resolve_project_root(project_dir: str) -> Path:
    project_root = Path(project_dir).expanduser()
    if not project_root.is_absolute():
        project_root = (Path.cwd() / project_root).resolve()
    else:
        project_root = project_root.resolve()
    return project_root


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _resolve_record_path(path_str: str, experiment_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        return (experiment_dir / path).resolve()
    resolved = path.resolve()
    if resolved.exists():
        return resolved
    experiment_name = experiment_dir.name
    parts = resolved.parts
    for i, part in enumerate(parts):
        if part == experiment_name and i + 1 < len(parts):
            candidate = (experiment_dir / Path(*parts[i + 1:])).resolve()
            if candidate.exists():
                return candidate
    return resolved


def _resolve_sample_path(data_dir: str, raw_path: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path
    return join(data_dir, raw_path.lstrip("/"))


def _build_processed_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record["tuple_id"]),
        str(record["method"]),
        str(record["image_path"]),
    )


def _load_existing_evaluation_records(
    path: Path,
    *,
    skip_corrupt_lines: bool = False,
) -> tuple[list[dict], set[tuple[str, str, str]]]:
    if not path.exists():
        return [], set()
    records = load_jsonl_file(file_path=str(path), skip_corrupt_lines=skip_corrupt_lines)
    processed_keys = {_build_processed_key(r) for r in records}
    return records, processed_keys


def _load_mask(mask_path: str) -> np.ndarray | None:
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv.threshold(mask, 127, 1, cv.THRESH_BINARY)
    return binary.astype(np.float64)


def _sample_dataset_name(record: dict) -> str | None:
    dataset_name = record.get("dataset_name")
    return str(dataset_name) if dataset_name is not None else None


def _xai_record_dataset_name(xai_record: dict) -> str | None:
    filters = xai_record.get("selected_generated_filters")
    if isinstance(filters, dict) and filters.get("dataset_name") is not None:
        return str(filters["dataset_name"])
    return None


def _load_sample_index(
    experiment_dir: Path,
    config: ExperimentConfig,
    xai_records: list[dict],
) -> dict:
    candidate_paths: list[Path] = []
    samples_filename = config.data.get("samples_filename", "training_split_metadata.jsonl")
    main_split_path = (experiment_dir / samples_filename).resolve()
    if main_split_path.exists():
        candidate_paths.append(main_split_path)
    for record in xai_records:
        samples_path = record.get("samples_path")
        if not samples_path:
            continue
        resolved = _resolve_record_path(str(samples_path), experiment_dir)
        if resolved.exists():
            candidate_paths.append(resolved)

    by_dataset_and_image: dict[tuple[str | None, str], dict] = {}
    by_image: dict[str, dict] = {}
    seen_paths: set[str] = set()
    for path in candidate_paths:
        path_str = str(path)
        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)
        for sample in load_jsonl_file(file_path=path_str):
            image_id = sample.get("image_id")
            if image_id is None:
                continue
            image_id = str(image_id)
            by_dataset_and_image.setdefault((_sample_dataset_name(sample), image_id), sample)
            by_image.setdefault(image_id, sample)
    return {"by_dataset_and_image": by_dataset_and_image, "by_image": by_image}


def _lookup_sample(sample_index: dict, xai_record: dict) -> dict | None:
    image_id = xai_record.get("image_id")
    if image_id is None:
        return None
    image_id = str(image_id)
    dataset_name = _xai_record_dataset_name(xai_record)
    if dataset_name is not None:
        match = sample_index["by_dataset_and_image"].get((dataset_name, image_id))
        if match is not None:
            return match
    return sample_index["by_image"].get(image_id)


def _resolve_existing_path(path_str: str, bases: list[Path]) -> str | None:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path) if path.exists() else None
    for base in bases:
        candidate = (base / path_str).resolve()
        if candidate.exists():
            return str(candidate)
    return None


def _load_rect_mask(sample: dict, field: str, bases: list[Path]) -> np.ndarray | None:
    raw_path = sample.get(field)
    if not raw_path:
        return None
    resolved = _resolve_existing_path(str(raw_path), bases)
    if resolved is None:
        return None
    return _load_mask(resolved)


def _matched_footprints(
    sample: dict, bases: list[Path]
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if str(sample.get("variant", "")) != "combined":
        return None, None, None
    classification_target = str(sample.get("classification_target", ""))
    if classification_target not in ("lesion", "distortion"):
        return None, None, None

    lesion_rect = _load_rect_mask(sample, "lesion_ground_truth_rect_mask_path", bases)
    distortion_rect = _load_rect_mask(sample, "distortion_ground_truth_rect_mask_path", bases)
    if lesion_rect is None or distortion_rect is None:
        return None, None, None

    if classification_target == "lesion":
        return lesion_rect, distortion_rect, "distortion"
    return distortion_rect, lesion_rect, "lesion"


DISTORTION_TRANSFORMATIONS = ("twirl", "spherize")
LESION_TRANSFORMATIONS = ("inpainting", "lesion")


def _load_grayscale_image(image_path: str) -> np.ndarray | None:
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image.astype(np.float64)


def _bounding_box_mask(binary_mask: np.ndarray) -> np.ndarray | None:
    rows, columns = np.nonzero(binary_mask)
    if rows.size == 0:
        return None
    rectangle = np.zeros_like(binary_mask)
    rectangle[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1] = 1.0
    return rectangle


def _region_mean(image: np.ndarray, rectangle: np.ndarray) -> float | None:
    if image.shape != rectangle.shape:
        return None
    area = float(np.sum(rectangle))
    if area <= 0:
        return None
    return float(np.sum(image * rectangle) / area)


def _saliency_score(
    generated_image: np.ndarray,
    source_image: np.ndarray,
    rectangle: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    generated_region_mean = _region_mean(generated_image, rectangle)
    source_region_mean = _region_mean(source_image, rectangle)
    if generated_region_mean is None or source_region_mean is None:
        return None, generated_region_mean, source_region_mean
    if source_region_mean <= 0:
        return None, generated_region_mean, source_region_mean
    return (
        abs(1.0 - generated_region_mean / source_region_mean),
        generated_region_mean,
        source_region_mean,
    )


def _edge_saliency_score(
    generated_edges: np.ndarray,
    source_edges: np.ndarray,
    rectangle: np.ndarray,
) -> float | None:
    """Edge-energy ratio in the region: mean|Sobel(generated)| / mean|Sobel(source)|.

    Unlike the intensity-shift score this stays non-zero for mean-preserving
    warps (a twirl rearranges pixels but adds edge structure), matching the
    paper's low-level notion of salience.
    """
    generated_region_mean = _region_mean(generated_edges, rectangle)
    source_region_mean = _region_mean(source_edges, rectangle)
    if generated_region_mean is None or source_region_mean is None:
        return None
    if source_region_mean <= 0:
        return None
    return generated_region_mean / source_region_mean


_ORIENTATION_HISTOGRAM_BINS = 12


def _region_orientation_statistics(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    region: np.ndarray,
) -> tuple[np.ndarray, complex, float]:
    magnitude = np.hypot(gradient_x[region], gradient_y[region])
    total_magnitude = float(np.sum(magnitude))
    axial_angle = np.mod(np.arctan2(gradient_y[region], gradient_x[region]), np.pi)
    histogram, _ = np.histogram(
        axial_angle,
        bins=_ORIENTATION_HISTOGRAM_BINS,
        range=(0.0, np.pi),
        weights=magnitude,
    )
    dominant_resultant = complex(np.sum(magnitude * np.exp(2j * axial_angle)))
    return histogram, dominant_resultant, total_magnitude


def _orientation_change_scores(
    generated_gradient_x: np.ndarray,
    generated_gradient_y: np.ndarray,
    source_gradient_x: np.ndarray,
    source_gradient_y: np.ndarray,
    rectangle: np.ndarray,
) -> tuple[float | None, float | None]:
    """Change in gradient *direction* between generated and clean image over the region.

    edge_salience tracks the change in gradient *magnitude* and is near-blind to a
    twirl, because a rotation preserves Sobel magnitude. This captures the orthogonal
    quantity. `divergence` is the total-variation distance between the magnitude-weighted
    axial orientation histograms (0 = identical orientation content, 1 = disjoint).
    `dominant_shift` is the rotation of the region's dominant orientation, in degrees
    (0-90), and is None when either region has no dominant orientation.
    """
    if generated_gradient_x.shape != rectangle.shape:
        return None, None
    region = rectangle > 0
    if not np.any(region):
        return None, None
    generated_histogram, generated_resultant, generated_total = _region_orientation_statistics(
        generated_gradient_x, generated_gradient_y, region
    )
    source_histogram, source_resultant, source_total = _region_orientation_statistics(
        source_gradient_x, source_gradient_y, region
    )
    if generated_total <= 0.0 or source_total <= 0.0:
        return None, None
    divergence = float(
        0.5 * np.sum(np.abs(generated_histogram / generated_total - source_histogram / source_total))
    )
    if abs(generated_resultant) <= 1e-12 or abs(source_resultant) <= 1e-12:
        return divergence, None
    dominant_shift = float(abs(np.degrees(np.angle(generated_resultant / source_resultant))) / 2.0)
    return divergence, dominant_shift


def _artifact_rectangles(sample: dict, bases: list[Path]) -> dict[str, np.ndarray]:
    rectangles: dict[str, np.ndarray] = {}
    if str(sample.get("variant", "")) == "combined":
        lesion_rectangle = _load_rect_mask(sample, "lesion_ground_truth_rect_mask_path", bases)
        distortion_rectangle = _load_rect_mask(sample, "distortion_ground_truth_rect_mask_path", bases)
        if lesion_rectangle is not None:
            rectangles["lesion"] = lesion_rectangle
        if distortion_rectangle is not None:
            rectangles["distortion"] = distortion_rectangle
        return rectangles

    transformation = str(sample.get("transformation", ""))
    if transformation in DISTORTION_TRANSFORMATIONS:
        artifact = "distortion"
    elif transformation in LESION_TRANSFORMATIONS:
        artifact = "lesion"
    else:
        return rectangles

    rectangle = _load_rect_mask(sample, "ground_truth_rect_mask_path", bases)
    if rectangle is None:
        raw_mask_path = sample.get("mask_path")
        resolved_mask_path = (
            _resolve_existing_path(str(raw_mask_path), bases) if raw_mask_path else None
        )
        if resolved_mask_path is not None:
            binary_mask = _load_mask(resolved_mask_path)
            if binary_mask is not None:
                rectangle = _bounding_box_mask(binary_mask)
    if rectangle is not None:
        rectangles[artifact] = rectangle
    return rectangles


def _empty_saliency_scores() -> dict:
    return {
        "lesion_saliency_score": None,
        "lesion_region_generated_mean": None,
        "lesion_region_source_mean": None,
        "lesion_edge_salience": None,
        "lesion_orientation_divergence": None,
        "lesion_orientation_shift": None,
        "distortion_saliency_score": None,
        "distortion_region_generated_mean": None,
        "distortion_region_source_mean": None,
        "distortion_edge_salience": None,
        "distortion_orientation_divergence": None,
        "distortion_orientation_shift": None,
    }


def _compute_saliency_scores(sample: dict, bases: list[Path]) -> dict:
    scores = _empty_saliency_scores()
    raw_generated_path = sample.get("image_path")
    raw_source_path = sample.get("source_image_path")
    if not raw_generated_path or not raw_source_path:
        return scores
    generated_path = _resolve_existing_path(str(raw_generated_path), bases)
    source_path = _resolve_existing_path(str(raw_source_path), bases)
    if generated_path is None or source_path is None:
        return scores
    generated_image = _load_grayscale_image(generated_path)
    source_image = _load_grayscale_image(source_path)
    if generated_image is None or source_image is None:
        return scores
    generated_normalized = generated_image / 255.0
    source_normalized = source_image / 255.0
    generated_edges = skimage_sobel(generated_normalized)
    source_edges = skimage_sobel(source_normalized)
    generated_gradient_x = sobel_v(generated_normalized)
    generated_gradient_y = sobel_h(generated_normalized)
    source_gradient_x = sobel_v(source_normalized)
    source_gradient_y = sobel_h(source_normalized)
    for artifact, rectangle in _artifact_rectangles(sample, bases).items():
        score, generated_region_mean, source_region_mean = _saliency_score(
            generated_image, source_image, rectangle
        )
        scores[f"{artifact}_saliency_score"] = score
        scores[f"{artifact}_region_generated_mean"] = generated_region_mean
        scores[f"{artifact}_region_source_mean"] = source_region_mean
        scores[f"{artifact}_edge_salience"] = _edge_saliency_score(
            generated_edges, source_edges, rectangle
        )
        orientation_divergence, orientation_shift = _orientation_change_scores(
            generated_gradient_x,
            generated_gradient_y,
            source_gradient_x,
            source_gradient_y,
            rectangle,
        )
        scores[f"{artifact}_orientation_divergence"] = orientation_divergence
        scores[f"{artifact}_orientation_shift"] = orientation_shift
    return scores


def _breast_reference_mask(
    sample: dict, bases: list[Path], target_shape: tuple[int, int]
) -> np.ndarray | None:
    raw_path = sample.get("image_path")
    if not raw_path:
        return None
    resolved = _resolve_existing_path(str(raw_path), bases)
    if resolved is None:
        return None
    image = _load_grayscale_image(resolved)
    if image is None:
        return None
    mask = (image > 10).astype(np.float64)
    if mask.shape != target_shape:
        mask = cv.resize(
            mask, (target_shape[1], target_shape[0]), interpolation=cv.INTER_NEAREST
        )
    return mask


def _load_edge_attribution(path_str: str, experiment_dir: Path) -> np.ndarray | None:
    if not path_str:
        return None
    resolved = str(_resolve_record_path(path_str, experiment_dir))
    if not os.path.exists(resolved):
        return None
    return np.abs(np.load(resolved)).astype(np.float64)


def main() -> None:
    config = _load_experiment_config()
    project_root = _resolve_project_root(project_dir=config.project_dir)
    experiment_dir = _resolve_path(
        path_str=generate_experiment_dir(config=config),
        project_root=project_root,
    )

    xai_dir = (experiment_dir / config.xai.get("output_dir", "xai")).resolve()
    xai_records_path = xai_dir / config.xai.get("xai_records", "xai_records.jsonl")
    if not xai_records_path.exists():
        raise FileNotFoundError(f"XAI records not found: {xai_records_path}")

    eval_dir = (experiment_dir / config.xai_evaluation.get("output_dir", "xai_evaluation")).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_records_path = (
        eval_dir / config.xai_evaluation.get("evaluation_records", "xai_evaluation_records.jsonl")
    ).resolve()
    intermediate_path = (
        eval_dir / config.xai_evaluation.get("intermediate_evaluation_records", "intermediate_evaluation_records.jsonl")
    ).resolve()

    eval_records, processed_keys = _load_existing_evaluation_records(eval_records_path)
    if not eval_records:
        eval_records, processed_keys = _load_existing_evaluation_records(
            intermediate_path, skip_corrupt_lines=True,
        )
    dump_as_jsonl_file(data=eval_records, file_path=str(intermediate_path))

    all_xai_records = load_jsonl_file(file_path=str(xai_records_path))
    xai_records = [
        r for r in all_xai_records
        if int(r.get("label", 0)) == 1 and str(r.get("split", "")) == "test"
    ]
    logger.info(
        f"Loaded {len(all_xai_records)} XAI records, "
        f"filtered to {len(xai_records)} (label=1, split=test)"
    )

    data_dir_raw = config.data.get("data_dir", "")
    if data_dir_raw:
        data_dir = str(_resolve_path(path_str=data_dir_raw, project_root=project_root))
    else:
        data_dir = str(project_root)

    sample_index = _load_sample_index(
        experiment_dir=experiment_dir, config=config, xai_records=xai_records,
    )
    mask_resolution_bases = [Path(data_dir), project_root, experiment_dir]

    edge_attribution_paths: dict[tuple[str, str], dict[str, str]] = {}
    for record in xai_records:
        method_name = str(record.get("method", ""))
        if method_name in ("sobel", "laplace"):
            edge_key = (
                str(_xai_record_dataset_name(record) or ""),
                str(record.get("image_id")),
            )
            edge_attribution_paths.setdefault(edge_key, {})[method_name] = str(
                record.get("attribution_path", "")
            )

    skipped = 0
    combined_scored = 0
    saliency_scored = 0
    saliency_cache: dict[tuple[str, str], dict] = {}
    breast_mask_cache: dict[tuple[str, str], np.ndarray | None] = {}
    edge_map_cache: dict[tuple[str, str], dict[str, np.ndarray | None]] = {}
    for xai_record in tqdm(xai_records, desc="Evaluating attributions", unit="attr"):
        key = _build_processed_key(xai_record)
        if key in processed_keys:
            skipped += 1
            continue

        attribution_path = str(xai_record.get("attribution_path", ""))
        resolved_attribution_path = str(_resolve_record_path(attribution_path, experiment_dir))
        if not os.path.exists(resolved_attribution_path):
            logger.warning(f"Attribution not found: {resolved_attribution_path}")
            continue

        attribution = np.load(resolved_attribution_path)

        image_path = str(xai_record["image_path"])
        mask_path = xai_record.get("mask_path")
        if not mask_path:
            logger.warning(f"No mask path for {image_path}")
            continue
        mask_path = str(mask_path)

        resolved_gt_path = str(_resolve_record_path(mask_path, experiment_dir))
        if not os.path.exists(resolved_gt_path):
            resolved_gt_path = _resolve_sample_path(data_dir, mask_path)
        if not os.path.exists(resolved_gt_path):
            logger.warning(f"Mask not found: {mask_path}")
            continue

        gt_mask = _load_mask(resolved_gt_path)
        if gt_mask is None:
            logger.warning(f"Failed to load mask: {resolved_gt_path}")
            continue

        a = np.abs(attribution).astype(np.float64)
        a_sum = np.sum(a)
        if a_sum > 0:
            a_normalized = a / a_sum
        else:
            a_normalized = a

        ma = float(mass_accuracy(y_true=gt_mask, y_pred=a_normalized))

        image_cache_key = (
            str(_xai_record_dataset_name(xai_record) or ""),
            str(xai_record.get("image_id")),
        )
        breast_mask = None

        discriminative_mask = None
        distractor_mask = None
        discriminative_mass_accuracy = None
        distractor_mass_accuracy = None
        discriminative_preference = None
        discriminative_enrichment = None
        distractor_enrichment = None
        discriminative_relative_importance = None
        distractor_relative_importance = None
        distractor_type = None
        classification_target = None
        sample = _lookup_sample(sample_index, xai_record)
        if sample is not None:
            classification_target = sample.get("classification_target")
            if image_cache_key not in breast_mask_cache:
                breast_mask_cache[image_cache_key] = _breast_reference_mask(
                    sample, mask_resolution_bases, a_normalized.shape
                )
            breast_mask = breast_mask_cache[image_cache_key]
            discriminative_mask, distractor_mask, distractor_type = _matched_footprints(
                sample=sample,
                bases=mask_resolution_bases,
            )
            if discriminative_mask is not None and distractor_mask is not None:
                discriminative_mass_accuracy = float(
                    mass_accuracy(y_true=discriminative_mask, y_pred=a_normalized)
                )
                distractor_mass_accuracy = float(
                    mass_accuracy(y_true=distractor_mask, y_pred=a_normalized)
                )
                denominator = discriminative_mass_accuracy + distractor_mass_accuracy
                if denominator > 0:
                    discriminative_preference = discriminative_mass_accuracy / denominator
                if breast_mask is not None:
                    discriminative_enrichment = enrichment(
                        a_normalized, discriminative_mask, breast_mask
                    )
                    distractor_enrichment = enrichment(
                        a_normalized, distractor_mask, breast_mask
                    )
                combined_artifact_region = (discriminative_mask > 0) | (distractor_mask > 0)
                clean_breast_reference = (
                    breast_mask.copy()
                    if breast_mask is not None
                    else np.ones_like(a_normalized, dtype=np.float64)
                )
                clean_breast_reference[combined_artifact_region] = 0.0
                discriminative_relative_importance = relative_importance(
                    y_pred=a_normalized, gt_mask=discriminative_mask, reference_mask=clean_breast_reference
                )
                distractor_relative_importance = relative_importance(
                    y_pred=a_normalized, gt_mask=distractor_mask, reference_mask=clean_breast_reference
                )
                combined_scored += 1

        mass_accuracy_gt_path = mask_path
        standalone_rect_mask = None
        if discriminative_mass_accuracy is not None:
            ma = discriminative_mass_accuracy
            discriminator_rect_field = (
                "distortion_ground_truth_rect_mask_path"
                if classification_target == "distortion"
                else "lesion_ground_truth_rect_mask_path"
            )
            discriminator_rect_path = sample.get(discriminator_rect_field)
            if discriminator_rect_path:
                mass_accuracy_gt_path = str(discriminator_rect_path)
        elif sample is not None:
            standalone_rect_mask = _load_rect_mask(
                sample, "ground_truth_rect_mask_path", mask_resolution_bases
            )
            if standalone_rect_mask is not None:
                ma = float(mass_accuracy(y_true=standalone_rect_mask, y_pred=a_normalized))
                mass_accuracy_gt_path = str(sample.get("ground_truth_rect_mask_path"))

        if discriminative_mask is not None and distractor_mask is not None:
            full_ground_truth_mask = ((discriminative_mask > 0) | (distractor_mask > 0)).astype(np.float64)
        elif standalone_rect_mask is not None:
            full_ground_truth_mask = standalone_rect_mask
        else:
            full_ground_truth_mask = gt_mask
        full_ground_truth_pixels = int(np.count_nonzero(full_ground_truth_mask > 0))
        full_ground_truth_mass_accuracy = float(
            mass_accuracy(y_true=full_ground_truth_mask, y_pred=a_normalized)
        )
        relative_importance_value = relative_importance(
            y_pred=a_normalized, gt_mask=full_ground_truth_mask, reference_mask=breast_mask
        )

        saliency_scores = _empty_saliency_scores()
        if sample is not None:
            cached_scores = saliency_cache.get(image_cache_key)
            if cached_scores is None:
                cached_scores = _compute_saliency_scores(sample, mask_resolution_bases)
                saliency_cache[image_cache_key] = cached_scores
            saliency_scores = cached_scores
            if (
                saliency_scores["lesion_saliency_score"] is not None
                or saliency_scores["distortion_saliency_score"] is not None
            ):
                saliency_scored += 1

        edge_corr_sobel = None
        edge_corr_laplace = None
        if breast_mask is not None:
            if image_cache_key not in edge_map_cache:
                paths = edge_attribution_paths.get(image_cache_key, {})
                edge_map_cache[image_cache_key] = {
                    "sobel": _load_edge_attribution(paths.get("sobel", ""), experiment_dir),
                    "laplace": _load_edge_attribution(paths.get("laplace", ""), experiment_dir),
                }
            edge_maps = edge_map_cache[image_cache_key]
            if edge_maps.get("sobel") is not None:
                edge_corr_sobel = masked_correlation(a, edge_maps["sobel"], breast_mask)
            if edge_maps.get("laplace") is not None:
                edge_corr_laplace = masked_correlation(a, edge_maps["laplace"], breast_mask)

        eval_record = {
            "tuple_id": xai_record["tuple_id"],
            "method": xai_record["method"],
            "model_name": xai_record.get("model_name"),
            "dataset_type": xai_record.get("dataset_type"),
            "dataset_variant_tag": xai_record.get("dataset_variant_tag"),
            "model_source": xai_record.get("model_source", "native"),
            "selected_generated_filters": xai_record.get("selected_generated_filters"),
            "repetition": xai_record.get("repetition"),
            "seed": xai_record.get("seed"),
            "image_id": xai_record.get("image_id"),
            "study_id": xai_record.get("study_id"),
            "split": xai_record.get("split"),
            "label": xai_record.get("label"),
            "image_path": image_path,
            "mask_path": mask_path,
            "ground_truth_mask_path": mass_accuracy_gt_path,
            "attribution_path": attribution_path,
            "prediction_logit": xai_record.get("prediction_logit"),
            "prediction_probability": xai_record.get("prediction_probability"),
            "mass_accuracy": ma,
            "classification_target": classification_target,
            "distractor_type": distractor_type,
            "discriminative_mass_accuracy": discriminative_mass_accuracy,
            "distractor_mass_accuracy": distractor_mass_accuracy,
            "discriminative_preference": discriminative_preference,
            "relative_importance": relative_importance_value,
            "discriminative_relative_importance": discriminative_relative_importance,
            "distractor_relative_importance": distractor_relative_importance,
            "full_ground_truth_mass_accuracy": full_ground_truth_mass_accuracy,
            "full_ground_truth_pixels": full_ground_truth_pixels,
            "discriminative_enrichment": discriminative_enrichment,
            "distractor_enrichment": distractor_enrichment,
            "edge_corr_sobel": edge_corr_sobel,
            "edge_corr_laplace": edge_corr_laplace,
            **saliency_scores,
        }

        eval_records.append(eval_record)
        processed_keys.add(key)
        append_to_jsonl_file(record=eval_record, file_path=str(intermediate_path))

    dump_as_jsonl_file(data=eval_records, file_path=str(eval_records_path))
    logger.info(
        f"Saved {len(eval_records)} evaluation records to {eval_records_path} "
        f"(skipped {skipped} already processed, "
        f"scored {combined_scored} combined records against a distractor footprint, "
        f"computed region saliency ratios for {saliency_scored} records)"
    )


if "__main__" == __name__:
    main()
