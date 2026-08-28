from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from analyses._colors import artifact_color, dataset_artifacts_from_name
from analyses._utils import (
    _build_variant_label,
    _format_variant_tag,
    _load_xai_evaluation_records,
    _method_label,
    _resolve_record_path,
    _short_variant_label,
    _sort_methods,
)
from config.configuration import ExperimentConfig

XAI_ATTRIBUTION_EXAMPLES = "xai_attribution_examples"
XAI_ATTRIBUTION_OVERLAY_EXAMPLES = "xai_attribution_overlay_examples"


def _sanitize_filename(text: str) -> str:
    value = text.strip() or "unknown"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _select_examples(
    df: pd.DataFrame,
    n_examples: int = 3,
) -> list[tuple[str, str, float]]:
    image_ma = df.groupby("image_path")["mass_accuracy"].mean().sort_values()
    n_images = len(image_ma)
    if n_images == 0:
        return []

    if n_images >= n_examples:
        picks = [
            (0, "Worst"),
            (n_images // 2, "Median"),
            (n_images - 1, "Best"),
        ]
    elif n_images == 2:
        picks = [(0, "Worst"), (1, "Best")]
    else:
        picks = [(0, "Example")]

    return [
        (image_ma.index[idx], label, float(image_ma.iloc[idx]))
        for idx, label in picks
    ]


def _load_image_array(path: str) -> np.ndarray | None:
    try:
        return np.array(Image.open(path).convert("L"))
    except (FileNotFoundError, OSError):
        return None


def _render_attribution_figure(
    variant_df: pd.DataFrame,
    examples: list[tuple[str, str, float]],
    methods: list[str],
    figure_title: str,
    dataset_name: str,
    experiment_dir: Path,
) -> plt.Figure:
    n_cols = 2 + len(methods)
    n_rows = len(examples)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        figure_title, fontsize=12, fontweight="bold",
        color=artifact_color(dataset_artifacts_from_name(dataset_name)[0]),
    )

    col_labels = ["Image", "GT Mask"] + [_method_label(m) for m in methods]
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=10, fontweight="bold")

    for row, (image_path, pick_label, avg_ma) in enumerate(examples):
        sample_records = variant_df[variant_df["image_path"] == image_path]

        resolved_img = str(_resolve_record_path(str(image_path), experiment_dir))
        img = _load_image_array(resolved_img)
        if img is not None:
            axes[row, 0].imshow(img, cmap="gray")
        else:
            axes[row, 0].text(
                0.5, 0.5, "Not found",
                transform=axes[row, 0].transAxes, ha="center", va="center",
            )

        gt_path = sample_records.iloc[0].get("mask_path")
        if gt_path:
            resolved_gt = str(_resolve_record_path(str(gt_path), experiment_dir))
            gt_img = _load_image_array(resolved_gt)
            if gt_img is not None:
                axes[row, 1].imshow(gt_img, cmap="gray")
            else:
                axes[row, 1].text(
                    0.5, 0.5, "Not found",
                    transform=axes[row, 1].transAxes, ha="center", va="center",
                )
        else:
            axes[row, 1].text(
                0.5, 0.5, "No mask",
                transform=axes[row, 1].transAxes, ha="center", va="center",
            )

        for m_idx, method in enumerate(methods):
            col = 2 + m_idx
            method_records = sample_records[sample_records["method"] == method]
            if method_records.empty:
                axes[row, col].text(
                    0.5, 0.5, "N/A",
                    transform=axes[row, col].transAxes, ha="center", va="center",
                )
                continue

            attr_path = str(method_records.iloc[0].get("attribution_path", ""))
            resolved_attr = str(_resolve_record_path(attr_path, experiment_dir))
            try:
                attribution = np.abs(np.load(resolved_attr)).astype(np.float64)
                attr_max = attribution.max()
                if attr_max > 0:
                    attribution = attribution / attr_max
                axes[row, col].imshow(attribution, cmap="magma", vmin=0.0, vmax=1.0)
                ma = float(method_records.iloc[0].get("mass_accuracy", 0))
                axes[row, col].text(
                    0.02, 0.98, f"MA: {ma:.3f}",
                    transform=axes[row, col].transAxes,
                    fontsize=8, color="white", va="top",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
                )
            except (FileNotFoundError, OSError):
                axes[row, col].text(
                    0.5, 0.5, "Not found",
                    transform=axes[row, col].transAxes, ha="center", va="center",
                )

        axes[row, 0].set_ylabel(f"{pick_label}\nAvg MA: {avg_ma:.3f}", fontsize=9)
        for c in range(n_cols):
            axes[row, c].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False,
            )

    return fig


def xai_attribution_examples(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> list[Path] | None:
    eval_df = _load_xai_evaluation_records(config, experiment_dir)
    if eval_df.empty:
        return None

    methods = _sort_methods(eval_df["method"].unique())
    if not methods:
        return None

    eval_df = eval_df.copy()
    eval_df["_dataset_name"] = eval_df["selected_generated_filters"].map(
        lambda f: f.get("dataset_name", "") if isinstance(f, dict) else ""
    )
    eval_df["_variant_tag"] = eval_df["dataset_variant_tag"].fillna("full_dataset").astype(str)

    output_paths: list[Path] = []
    for (dataset_name, variant_tag), group_df in eval_df.groupby(
        ["_dataset_name", "_variant_tag"], dropna=False,
    ):
        examples = _select_examples(group_df)
        if not examples:
            continue

        name_part = _short_variant_label(str(dataset_name)) if dataset_name else ""
        tag_display = _format_variant_tag(str(variant_tag))
        figure_title = _build_variant_label(str(dataset_name), str(variant_tag))
        if name_part and tag_display:
            file_stem = f"{name_part}_{tag_display}"
        elif name_part:
            file_stem = name_part
        else:
            file_stem = str(variant_tag)

        fig = _render_attribution_figure(
            group_df, examples, methods, figure_title, str(dataset_name), experiment_dir,
        )

        output_path = analyses_dir / f"xai_attribution_examples_{_sanitize_filename(file_stem)}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        output_paths.append(output_path)
        logger.info(f"Saved attribution examples to {output_path}")

    if not output_paths:
        return None
    logger.info(f"Generated {len(output_paths)} attribution example plots")
    return output_paths


def _render_overlay_attribution_figure(
    variant_df: pd.DataFrame,
    examples: list[tuple[str, str, float]],
    methods: list[str],
    figure_title: str,
    dataset_name: str,
    experiment_dir: Path,
) -> plt.Figure:
    n_cols = 2 + len(methods)
    n_rows = len(examples)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        figure_title, fontsize=12, fontweight="bold",
        color=artifact_color(dataset_artifacts_from_name(dataset_name)[0]),
    )

    col_labels = ["Image", "GT Mask"] + [_method_label(m) for m in methods]
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=10, fontweight="bold")

    for row, (image_path, pick_label, avg_ma) in enumerate(examples):
        sample_records = variant_df[variant_df["image_path"] == image_path]

        resolved_img = str(_resolve_record_path(str(image_path), experiment_dir))
        img = _load_image_array(resolved_img)
        if img is not None:
            axes[row, 0].imshow(img, cmap="gray")
        else:
            axes[row, 0].text(
                0.5, 0.5, "Not found",
                transform=axes[row, 0].transAxes, ha="center", va="center",
            )

        gt_path = sample_records.iloc[0].get("mask_path")
        if gt_path:
            resolved_gt = str(_resolve_record_path(str(gt_path), experiment_dir))
            gt_img = _load_image_array(resolved_gt)
            if gt_img is not None:
                axes[row, 1].imshow(gt_img, cmap="gray")
            else:
                axes[row, 1].text(
                    0.5, 0.5, "Not found",
                    transform=axes[row, 1].transAxes, ha="center", va="center",
                )
        else:
            axes[row, 1].text(
                0.5, 0.5, "No mask",
                transform=axes[row, 1].transAxes, ha="center", va="center",
            )

        for m_idx, method in enumerate(methods):
            col = 2 + m_idx
            method_records = sample_records[sample_records["method"] == method]
            if method_records.empty:
                axes[row, col].text(
                    0.5, 0.5, "N/A",
                    transform=axes[row, col].transAxes, ha="center", va="center",
                )
                continue

            attr_path = str(method_records.iloc[0].get("attribution_path", ""))
            resolved_attr = str(_resolve_record_path(attr_path, experiment_dir))
            try:
                attribution = np.abs(np.load(resolved_attr)).astype(np.float64)
                p99 = np.percentile(attribution, 99)
                if p99 > 0:
                    attr_norm = np.clip(attribution, 0, p99) / p99
                else:
                    attr_norm = attribution

                if img is not None:
                    img_display = img
                    if attr_norm.shape != img_display.shape:
                        attr_norm = np.array(
                            Image.fromarray(attr_norm).resize(
                                (img_display.shape[1], img_display.shape[0]),
                                Image.BILINEAR,
                            )
                        )
                    axes[row, col].imshow(img_display, cmap="gray")
                    rgba = plt.cm.magma(attr_norm)
                    rgba[..., 3] = attr_norm
                    axes[row, col].imshow(rgba)
                else:
                    axes[row, col].imshow(attr_norm, cmap="magma", vmin=0.0, vmax=1.0)

                ma = float(method_records.iloc[0].get("mass_accuracy", 0))
                axes[row, col].text(
                    0.02, 0.98, f"MA: {ma:.3f}",
                    transform=axes[row, col].transAxes,
                    fontsize=8, color="white", va="top",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
                )
            except (FileNotFoundError, OSError):
                axes[row, col].text(
                    0.5, 0.5, "Not found",
                    transform=axes[row, col].transAxes, ha="center", va="center",
                )

        axes[row, 0].set_ylabel(f"{pick_label}\nAvg MA: {avg_ma:.3f}", fontsize=9)
        for c in range(n_cols):
            axes[row, c].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False,
            )

    return fig


def xai_attribution_overlay_examples(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> list[Path] | None:
    eval_df = _load_xai_evaluation_records(config, experiment_dir)
    if eval_df.empty:
        return None

    methods = _sort_methods(eval_df["method"].unique())
    if not methods:
        return None

    eval_df = eval_df.copy()
    eval_df["_dataset_name"] = eval_df["selected_generated_filters"].map(
        lambda f: f.get("dataset_name", "") if isinstance(f, dict) else ""
    )
    eval_df["_variant_tag"] = eval_df["dataset_variant_tag"].fillna("full_dataset").astype(str)

    output_paths: list[Path] = []
    for (dataset_name, variant_tag), group_df in eval_df.groupby(
        ["_dataset_name", "_variant_tag"], dropna=False,
    ):
        examples = _select_examples(group_df)
        if not examples:
            continue

        name_part = _short_variant_label(str(dataset_name)) if dataset_name else ""
        tag_display = _format_variant_tag(str(variant_tag))
        figure_title = _build_variant_label(str(dataset_name), str(variant_tag))
        if name_part and tag_display:
            file_stem = f"{name_part}_{tag_display}"
        elif name_part:
            file_stem = name_part
        else:
            file_stem = str(variant_tag)

        fig = _render_overlay_attribution_figure(
            group_df, examples, methods, figure_title, str(dataset_name), experiment_dir,
        )

        output_path = analyses_dir / f"xai_attribution_overlay_examples_{_sanitize_filename(file_stem)}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        output_paths.append(output_path)
        logger.info(f"Saved overlay attribution examples to {output_path}")

    if not output_paths:
        return None
    logger.info(f"Generated {len(output_paths)} overlay attribution example plots")
    return output_paths
