from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image

from analyses._colors import (
    LESION_ARTIFACT,
    LESION_COLOR,
    artifact_color,
    distortion_family_from_name,
)
from config.configuration import ExperimentConfig
from utils import load_jsonl_file

COMBINED_PAIR_EXAMPLES = "combined_pair_examples"

ARTIFACT_COLOR_NAMES = {LESION_COLOR: "green"}
DEFAULT_ARTIFACT_COLOR_NAME = "orange"


def _artifact_color_name(artifact: str) -> str:
    return ARTIFACT_COLOR_NAMES.get(artifact_color(artifact), DEFAULT_ARTIFACT_COLOR_NAME)


def _sanitize_filename(text: str) -> str:
    value = text.strip() or "unknown"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _resolve_data_dir(config: ExperimentConfig) -> Path:
    data_dir = Path(config.data.get("data_dir", "")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()
    return data_dir


def _resolve_against_data_dir(data_dir: Path, raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    return str(path if path.is_absolute() else (data_dir / raw_path).resolve())


def _load_grayscale(path: str) -> np.ndarray | None:
    try:
        return np.array(Image.open(path).convert("L"))
    except (FileNotFoundError, OSError):
        return None


def _spread_pick(items: list, n_examples: int) -> list:
    if len(items) <= n_examples:
        return items
    if n_examples <= 1:
        return [items[0]]
    indices = sorted({round(i * (len(items) - 1) / (n_examples - 1)) for i in range(n_examples)})
    return [items[i] for i in indices]


def _lesion_footprint(record: dict) -> tuple | None:
    corners = (
        record.get("lesion_x1"), record.get("lesion_y1"),
        record.get("lesion_x2"), record.get("lesion_y2"),
    )
    if any(corner is None for corner in corners):
        return None
    return ("rect", tuple(int(c) for c in corners))


def _distortion_metric_rect(record: dict) -> tuple | None:
    center_x, center_y = record.get("distortion_center_x"), record.get("distortion_center_y")
    corners = ("lesion_x1", "lesion_y1", "lesion_x2", "lesion_y2")
    if center_x is None or center_y is None or any(record.get(corner) is None for corner in corners):
        return None
    width = int(record["lesion_x2"]) - int(record["lesion_x1"])
    height = int(record["lesion_y2"]) - int(record["lesion_y1"])
    x1 = int(center_x) - width // 2
    y1 = int(center_y) - height // 2
    return ("rect", (x1, y1, x1 + width, y1 + height))


def _roles(record: dict, dataset_name: str) -> tuple[str, str]:
    distortion = distortion_family_from_name(dataset_name)
    if str(record.get("classification_target", "")) == LESION_ARTIFACT:
        return LESION_ARTIFACT, distortion
    return distortion, LESION_ARTIFACT


def _draw_shape(axis, shape: tuple | None, color: str, linestyle: str) -> None:
    if shape is None:
        return
    kind, geometry = shape
    if kind == "rect":
        x1, y1, x2, y2 = geometry
        patch = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            edgecolor=color, facecolor="none", linewidth=1.6, linestyle=linestyle,
        )
    else:
        center_x, center_y, radius = geometry
        patch = mpatches.Circle(
            (center_x, center_y), radius,
            edgecolor=color, facecolor="none", linewidth=1.6, linestyle=linestyle,
        )
    axis.add_patch(patch)


def _draw_artifact(axis, record: dict, artifact_name: str, linestyle: str) -> None:
    footprint = (
        _lesion_footprint(record) if artifact_name == LESION_ARTIFACT
        else _distortion_metric_rect(record)
    )
    _draw_shape(axis, footprint, artifact_color(artifact_name), linestyle)


def _blank(axis, message: str) -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center")


def _render_pair_figure(
    pairs: list[tuple[dict, dict]],
    dataset_name: str,
    data_dir: Path,
) -> plt.Figure:
    discriminator_name, distractor_name = _roles(pairs[0][0], dataset_name)
    comparator_variant = str(pairs[0][1].get("variant", "comparator"))

    n_rows = len(pairs)
    figure, axes = plt.subplots(
        n_rows, 3,
        figsize=(12, 4 * n_rows),
        squeeze=False, constrained_layout=True,
    )
    figure.suptitle(
        f"{dataset_name}\n"
        f"discriminator = {discriminator_name} ({_artifact_color_name(discriminator_name)}, solid)   ·   "
        f"distractor = {distractor_name} ({_artifact_color_name(distractor_name)}, dashed)\n"
        f"rectangles = area-matched footprints scored by the metric   ·   "
        f"classes differ only inside the discriminator footprint",
        fontsize=10, fontweight="bold",
    )
    column_titles = ["label = 1  (combined)", f"label = 0  ({comparator_variant})", "| difference |"]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=10, fontweight="bold")

    for row, (combined_record, comparator_record) in enumerate(pairs):
        combined_image = _load_grayscale(
            _resolve_against_data_dir(data_dir, str(combined_record.get("image_path", "")))
        )
        comparator_image = _load_grayscale(
            _resolve_against_data_dir(data_dir, str(comparator_record.get("image_path", "")))
        )

        for column, image in ((0, combined_image), (1, comparator_image)):
            axis = axes[row, column]
            if image is None:
                _blank(axis, "Not found")
                continue
            axis.imshow(image, cmap="gray")
            _draw_artifact(axis, combined_record, discriminator_name, "-")
            _draw_artifact(axis, combined_record, distractor_name, "--")

        difference_axis = axes[row, 2]
        if combined_image is None or comparator_image is None or combined_image.shape != comparator_image.shape:
            _blank(difference_axis, "N/A")
        else:
            difference = np.abs(combined_image.astype(np.int32) - comparator_image.astype(np.int32))
            difference_axis.imshow(difference, cmap="inferno", vmin=0, vmax=max(int(difference.max()), 1))
            _draw_artifact(difference_axis, combined_record, discriminator_name, "-")
            _draw_artifact(difference_axis, combined_record, distractor_name, "--")
            changed_fraction = 100.0 * float(np.count_nonzero(difference)) / difference.size
            difference_axis.text(
                0.02, 0.98, f"max|Δ|={int(difference.max())}\nΔpx={changed_fraction:.2f}%",
                transform=difference_axis.transAxes, fontsize=8, color="white", va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
            )

        axes[row, 0].set_ylabel(str(combined_record.get("image_id", "")), fontsize=8)
        for column in range(3):
            axes[row, column].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False,
            )

    return figure


def combined_pair_examples(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
    n_examples: int = 4,
) -> list[Path] | None:
    samples_filename = config.data.get("samples_filename", "training_split_metadata.jsonl")
    split_path = (experiment_dir / samples_filename).resolve()
    if not split_path.exists():
        logger.warning(f"Training split not found for combined pair examples: {split_path}")
        return None

    records = load_jsonl_file(file_path=str(split_path))
    combined_records = [record for record in records if str(record.get("variant", "")) == "combined"]
    if not combined_records:
        logger.info("No combined records in training split; skipping combined pair examples.")
        return None

    data_dir = _resolve_data_dir(config)
    record_by_key = {
        (record.get("dataset_name"), record.get("image_id")): record for record in records
    }

    combined_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for record in combined_records:
        combined_by_dataset[str(record.get("dataset_name", ""))].append(record)

    output_paths: list[Path] = []
    for dataset_name, dataset_records in combined_by_dataset.items():
        picks = _spread_pick(
            sorted(dataset_records, key=lambda record: str(record.get("image_id", ""))),
            n_examples,
        )
        pairs = []
        for combined_record in picks:
            comparator = record_by_key.get(
                (combined_record.get("dataset_name"), combined_record.get("sibling_image_id"))
            )
            if comparator is not None:
                pairs.append((combined_record, comparator))
        if not pairs:
            logger.warning(f"No sibling comparators resolved for dataset {dataset_name}; skipping.")
            continue

        figure = _render_pair_figure(pairs, dataset_name, data_dir)
        output_path = analyses_dir / f"combined_pair_examples_{_sanitize_filename(dataset_name)}.png"
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        output_paths.append(output_path)
        logger.info(f"Saved combined pair examples to {output_path}")

    if not output_paths:
        return None
    logger.info(f"Generated {len(output_paths)} combined pair example plots")
    return output_paths
