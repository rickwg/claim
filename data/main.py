"""
Create training split metadata from preprocessed and generated data.

Reads preprocessed healthy images and generated unhealthy images (synthetic
lesions), matches them by subject_id and image_id to form a binary classification dataset
(healthy=0, unhealthy=1), and outputs training_split_metadata.jsonl.

When no generated data is provided, falls back to using all preprocessed
samples with labels from annotations.

Pipeline:  preprocessing → data generation → main.py (this) → training
"""

import os
from pathlib import Path

import pandas as pd
from loguru import logger

from common import EnvironmentVariables
from config.configuration import ExperimentConfig
from utils import dump_as_jsonl_file, generate_experiment_dir, load_config_file, load_jsonl_file

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_OUTPUT_FILENAME = "training_split_metadata.jsonl"


def _collect_image_paths(root_dir: Path) -> dict[str, Path]:
    """Collect all images under root_dir, keyed by '{study_id}/{stem}'."""
    images = {}
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            rel = path.relative_to(root_dir)
            key = str(rel.with_suffix(""))
            images[key] = path
    return images


def _load_preprocessed_samples(
    image_dir: str,
    mask_dir: str,
    annotations_path: str,
    data_dir: str | None = None,
) -> list[dict]:
    """Build sample records from preprocessed images + annotations."""
    image_dir_path = Path(image_dir).expanduser().resolve()
    mask_dir_path = Path(mask_dir).expanduser().resolve()
    data_dir_path = Path(data_dir).expanduser().resolve() if data_dir else None

    if not image_dir_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir_path}")
    if not mask_dir_path.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir_path}")

    annotations_df = pd.read_csv(annotations_path)
    has_finding_set = set(
        annotations_df[
            annotations_df["finding_categories"] != "['No Finding']"
        ]["image_id"].unique()
    )
    image_meta = (
        annotations_df.drop_duplicates(subset=["image_id"])
        .set_index("image_id")[["split"]]
        .to_dict("index")
    )

    image_paths = _collect_image_paths(image_dir_path)
    mask_paths = _collect_image_paths(mask_dir_path)

    samples = []
    for key, img_path in image_paths.items():
        image_id = Path(key).name
        subject_id = Path(key).parent.name
        mask_path = mask_paths.get(key)
        if mask_path is None:
            continue

        meta = image_meta.get(image_id, {})
        label = 1 if image_id in has_finding_set else 0
        split = meta.get("split", "training")

        if data_dir_path:
            try:
                image_path_str = str(img_path.relative_to(data_dir_path))
                mask_path_str = str(mask_path.relative_to(data_dir_path))
            except ValueError:
                image_path_str = str(img_path)
                mask_path_str = str(mask_path)
        else:
            image_path_str = str(img_path)
            mask_path_str = str(mask_path)

        samples.append({
            "image_id": image_id,
            "subject_id": subject_id,
            "image_path": image_path_str,
            "mask_path": mask_path_str,
            "label": label,
            "split": split,
            "source": "preprocessed",
        })

    return samples


def _derive_dataset_name(raw_path: str) -> str:
    parent = Path(raw_path).parent
    parts = [p for p in parent.parts if p not in ("..", ".")]
    return "/".join(parts) if parts else "default"


def _load_generated_samples(
    metadata_paths: list[str],
    annotations_path: str,
    data_dir: str | None = None,
    dataset_names: list[str] | None = None,
) -> list[dict]:
    """Load generated samples from dataset_metadata.jsonl files and add split info."""
    annotations_df = pd.read_csv(annotations_path)
    image_meta = (
        annotations_df.drop_duplicates(subset=["image_id"])
        .set_index("image_id")[["split"]]
        .to_dict("index")
    )
    data_dir_path = Path(data_dir).resolve() if data_dir else None

    if isinstance(metadata_paths, str):
        metadata_paths = [metadata_paths]

    samples = []
    for i, path in enumerate(metadata_paths):
        if not os.path.exists(path):
            logger.warning(f"Generated metadata not found: {path}, skipping")
            continue
        dataset_name = dataset_names[i] if dataset_names else Path(path).parent.name
        records = load_jsonl_file(path)
        for record in records:
            image_id = record.get("image_id", Path(record["image_path"]).stem)
            record.setdefault("subject_id", Path(record["image_path"]).parent.name)
            record.setdefault("dataset_name", dataset_name)
            if data_dir_path:
                for path_key in (
                    "image_path",
                    "mask_path",
                    "source_image_path",
                    "ground_truth_mask_path",
                    "distortion_ground_truth_mask_path",
                    "distortion_ground_truth_rect_mask_path",
                    "lesion_ground_truth_rect_mask_path",
                    "ground_truth_rect_mask_path",
                ):
                    raw_path = record.get(path_key)
                    if not raw_path:
                        continue
                    record[path_key] = os.path.relpath(Path(raw_path).resolve(), data_dir_path)
            split_lookup_id = record.get("source_image_id", image_id)
            meta = image_meta.get(split_lookup_id, {})
            record.setdefault("split", meta.get("split", "training"))
            record.setdefault("source", "generated")
            samples.append(record)
        logger.info(f"Loaded {len(records)} generated records from {path} (dataset_name={dataset_name})")

    return samples


def create_training_split(
    image_dir: str,
    mask_dir: str,
    annotations_path: str,
    output_path: str,
    generated_metadata_paths: list[str] | None = None,
    label_filter: list[int] | None = None,
    data_dir: str | None = None,
    dataset_names: list[str] | None = None,
) -> dict:
    """Create training split JSONL from preprocessed + generated data.

    Args:
        image_dir: Absolute path to preprocessed images ({study_id}/{image_id}.png)
        mask_dir: Absolute path to masks ({study_id}/{image_id}.png)
        annotations_path: Absolute path to finding_annotations.csv
        output_path: Absolute output path for training_split_metadata.jsonl
        generated_metadata_paths: Absolute paths to dataset_metadata.jsonl from generation runs
        label_filter: Filter by label — [0]=healthy, [1]=unhealthy, null=all
        data_dir: Root directory; output paths are stored relative to this
        dataset_names: Names for each generated metadata path (derived from config paths)

    Returns:
        Summary dict with counts
    """
    preprocessed = _load_preprocessed_samples(image_dir, mask_dir, annotations_path, data_dir=data_dir)
    logger.info(f"Loaded {len(preprocessed)} preprocessed samples")

    generated = []
    if generated_metadata_paths:
        generated = _load_generated_samples(
            generated_metadata_paths, annotations_path, data_dir=data_dir, dataset_names=dataset_names,
        )
        logger.info(f"Loaded {len(generated)} generated samples total")

    if generated:
        generated_keys = {
            (s.get("subject_id"), s["image_id"])
            for s in generated if "image_id" in s
        }
        matched_healthy = [
            s for s in preprocessed
            if s["label"] == 0
            and (s.get("subject_id"), s.get("image_id")) in generated_keys
        ]
        unmatched_count = len(generated_keys) - len(matched_healthy)
        if unmatched_count > 0:
            logger.warning(
                f"{unmatched_count} generated samples have no matching healthy preprocessed image"
            )
        logger.info(
            f"Matched {len(matched_healthy)} healthy (label=0) to "
            f"{len(generated)} generated unhealthy (label=1) by subject_id and image_id"
        )
        all_samples = matched_healthy + generated
    else:
        all_samples = preprocessed

    if label_filter is not None:
        all_samples = [s for s in all_samples if s["label"] in label_filter]
        logger.info(f"Filtered to {len(all_samples)} samples with labels {label_filter}")

    if not all_samples:
        raise ValueError("No samples after filtering!")

    output_file_path = Path(output_path).expanduser().resolve()
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    dump_as_jsonl_file(data=all_samples, file_path=str(output_file_path))

    num_healthy = sum(s["label"] == 0 for s in all_samples)
    num_unhealthy = sum(s["label"] == 1 for s in all_samples)
    num_train = sum(s.get("split") == "training" for s in all_samples)
    num_test = sum(s.get("split") == "test" for s in all_samples)
    num_preprocessed = sum(s.get("source") == "preprocessed" for s in all_samples)
    num_generated = sum(s.get("source") == "generated" for s in all_samples)

    result = {
        "output_path": str(output_file_path),
        "num_samples": len(all_samples),
        "num_healthy": num_healthy,
        "num_unhealthy": num_unhealthy,
        "num_training": num_train,
        "num_test": num_test,
        "num_preprocessed": num_preprocessed,
        "num_generated": num_generated,
    }
    logger.info(
        f"Created {result['output_path']} — "
        f"{result['num_samples']} samples "
        f"(healthy={num_healthy}, unhealthy={num_unhealthy}, "
        f"preprocessed={num_preprocessed}, generated={num_generated}, "
        f"training={num_train}, test={num_test})"
    )
    return result


def _load_experiment_config() -> ExperimentConfig:
    config_path = os.environ.get(
        EnvironmentVariables.CONFIG_FILE_PATH.value,
        "config/experiments_config.yaml",
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    return ExperimentConfig.from_dict(load_config_file(file_path=config_path))


def _resolve_data_path(data_dir: Path, relative_path: str) -> str:
    return str((data_dir / relative_path).resolve())


def main() -> None:
    config = _load_experiment_config()
    data = config.data

    data_dir = Path(data["data_dir"]).expanduser().resolve()

    generated_metadata_paths = None
    dataset_names = None
    raw_gen_paths = data.get("generated_metadata_paths")
    if raw_gen_paths:
        if isinstance(raw_gen_paths, str):
            raw_gen_paths = [raw_gen_paths]
        generated_metadata_paths = [
            _resolve_data_path(data_dir, p) for p in raw_gen_paths
        ]
        dataset_names = [_derive_dataset_name(p) for p in raw_gen_paths]

    experiment_dir = generate_experiment_dir(config=config)
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(experiment_dir, data.get("samples_filename", DEFAULT_OUTPUT_FILENAME))

    result = create_training_split(
        image_dir=_resolve_data_path(data_dir, data["preprocessed_image_dir"]),
        mask_dir=_resolve_data_path(data_dir, data["preprocessed_mask_dir"]),
        annotations_path=_resolve_data_path(data_dir, data["annotations_path"]),
        output_path=output_path,
        generated_metadata_paths=generated_metadata_paths,
        label_filter=data.get("label_filter"),
        data_dir=str(data_dir),
        dataset_names=dataset_names,
    )


if __name__ == "__main__":
    main()
