from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from analyses._colors import (
    DISTORTION_ARTIFACT,
    DISTORTION_COLOR,
    DISTORTION_FAMILY_ARTIFACTS,
    DISTRACTOR_HATCH,
    PLAIN_EDGE_COLOR,
    LESION_ARTIFACT,
    LESION_COLOR,
    MAGNITUDE_COLORMAP,
    SPHERIZE_ARTIFACT,
    TWIRL_ARTIFACT,
    accent_axis,
    apply_dataset_style,
    artifact_color,
    dataset_accent_color,
    dataset_artifacts,
    dataset_artifacts_from_name,
    dataset_legend_handle,
    dataset_style,
    distortion_family_from_name,
    magnitude_text_color,
)
from analyses._utils import (
    _build_variant_label,
    _dataset_ordered_labels,
    _format_x_label,
    _load_xai_evaluation_records,
    _method_color,
    _method_group_rank,
    _method_label,
    _method_labels,
    _resolve_category_and_parameter,
    _sort_methods,
)
from analyses._predictions import accuracy_by_dataset_and_model
from config.configuration import ExperimentConfig

XAI_MASS_ACCURACY_PLOT = "xai_mass_accuracy_plot"
XAI_MASS_ACCURACY_HEATMAP = "xai_mass_accuracy_heatmap"
XAI_MASS_ACCURACY_BOXPLOT = "xai_mass_accuracy_boxplot"
XAI_MASS_ACCURACY_BOXPLOT_BY_DATASET = "xai_mass_accuracy_boxplot_by_dataset"
XAI_RELATIVE_IMPORTANCE_BOXPLOT = "xai_relative_importance_boxplot"
XAI_RELATIVE_IMPORTANCE_BOXPLOT_LOG = "xai_relative_importance_boxplot_log"
XAI_DISCRIMINATOR_RELATIVE_IMPORTANCE_BOXPLOT = "xai_discriminator_relative_importance_boxplot"
XAI_DISCRIMINATOR_RELATIVE_IMPORTANCE_BOXPLOT_LOG = "xai_discriminator_relative_importance_boxplot_log"
XAI_DISTRACTOR_RELATIVE_IMPORTANCE_BOXPLOT = "xai_distractor_relative_importance_boxplot"
XAI_DISTRACTOR_RELATIVE_IMPORTANCE_BOXPLOT_LOG = "xai_distractor_relative_importance_boxplot_log"
XAI_RELATIVE_IMPORTANCE_DISC_VS_DISTRACTOR_BOXPLOT = "xai_relative_importance_disc_vs_distractor_boxplot"
XAI_RELATIVE_IMPORTANCE_DISC_VS_DISTRACTOR_BOXPLOT_LOG = "xai_relative_importance_disc_vs_distractor_boxplot_log"
XAI_RELATIVE_IMPORTANCE_DUMBBELL = "xai_relative_importance_dumbbell"
XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED = "xai_relative_importance_dumbbell_grouped"
XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_TWIRL = "xai_relative_importance_dumbbell_grouped_twirl"
XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_SPHERIZE = "xai_relative_importance_dumbbell_grouped_spherize"
XAI_RELATIVE_IMPORTANCE_DUMBBELL_MEAN = "xai_relative_importance_dumbbell_mean"
XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_MEAN = "xai_relative_importance_dumbbell_grouped_mean"
XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_TWIRL_MEAN = "xai_relative_importance_dumbbell_grouped_twirl_mean"
XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_SPHERIZE_MEAN = "xai_relative_importance_dumbbell_grouped_spherize_mean"
XAI_SEPARATION_INDEX = "xai_separation_index"
XAI_PERFORMANCE_COMPARISON = "xai_performance_comparison"
XAI_CONFIDENCE_VS_MASS_ACCURACY = "xai_confidence_vs_mass_accuracy"
XAI_DISCRIMINATIVE_VS_DISTRACTOR = "xai_discriminative_vs_distractor_plot"
XAI_SALIENCY_BOXPLOT_BY_DATASET = "xai_saliency_boxplot_by_dataset"
XAI_SALIENCY_LESION_VS_DISTORTION = "xai_saliency_lesion_vs_distortion"
XAI_EDGE_SALIENCE_BOXPLOT_BY_DATASET = "xai_edge_salience_boxplot_by_dataset"
XAI_EDGE_VS_ORIENTATION = "xai_edge_vs_orientation"
XAI_EDGE_CORRESPONDENCE = "xai_edge_correspondence"
XAI_SUMMARY = "xai_summary"

PAPER_AXIS_LABEL_FONTSIZE = 22
PAPER_TICK_LABEL_FONTSIZE = 18
PAPER_MINOR_TICK_LABEL_FONTSIZE = 14
PAPER_PANEL_TITLE_FONTSIZE = 20
PAPER_LEGEND_FONTSIZE = 16

_SALIENCY_ARTIFACTS = (
    (LESION_ARTIFACT, "lesion_saliency_score", LESION_COLOR),
    (DISTORTION_ARTIFACT, "distortion_saliency_score", DISTORTION_COLOR),
)
_SALIENCY_Y_LABEL = r"Intensity shift  $s_{\mathrm{int}}$"

_EDGE_SALIENCE_ARTIFACTS = (
    (LESION_ARTIFACT, "lesion_edge_salience", LESION_COLOR),
    (DISTORTION_ARTIFACT, "distortion_edge_salience", DISTORTION_COLOR),
)
_EDGE_SALIENCE_Y_LABEL = r"Edge salience  $s_{\mathrm{edge}}$"

_ORIENTATION_DIVERGENCE_ARTIFACTS = (
    (LESION_ARTIFACT, "lesion_orientation_divergence", LESION_COLOR),
    (DISTORTION_ARTIFACT, "distortion_orientation_divergence", DISTORTION_COLOR),
)
_ORIENTATION_SHIFT_ARTIFACTS = (
    (LESION_ARTIFACT, "lesion_orientation_shift", LESION_COLOR),
    (DISTORTION_ARTIFACT, "distortion_orientation_shift", DISTORTION_COLOR),
)
_ORIENTATION_DIVERGENCE_Y_LABEL = "Orientation divergence\n(TV distance, 0–1)"
_ORIENTATION_SHIFT_Y_LABEL = "Dominant shift\n[degrees]"
_EDGE_SALIENCE_SHORT_Y_LABEL = "Edge salience\n(|Sobel| gen / clean)"


def _stagger_labels(labels: list[str]) -> list[str]:
    return [label if i % 2 == 0 else f"\n{label}" for i, label in enumerate(labels)]


def _common_prefix_length(labels: list[str]) -> int:
    shortest = min(labels, key=len)
    for index, character in enumerate(shortest):
        if any(label[index] != character for label in labels):
            return index
    return len(shortest)


def _shorten_variant_labels(labels: list[str]) -> list[str]:
    if len(labels) < 2:
        return labels
    prefix_length = _common_prefix_length(labels)
    suffix_length = _common_prefix_length([label[::-1] for label in labels])
    shortened = []
    for label in labels:
        if prefix_length + suffix_length < len(label):
            core = label[prefix_length: len(label) - suffix_length].strip(" _-()")
        else:
            core = ""
        shortened.append(core or label)
    return shortened


def _build_xai_performance_dataframe(
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, record in eval_df.iterrows():
        selected_filters = record.get("selected_generated_filters")
        if not isinstance(selected_filters, dict):
            selected_filters = None
        variant_tag = str(record.get("dataset_variant_tag", "full_dataset"))

        category, parameter_name, parameter_value = _resolve_category_and_parameter(
            selected_filters=selected_filters,
            variant_tag=variant_tag,
        )

        dataset_name = ""
        if isinstance(selected_filters, dict):
            dataset_name = selected_filters.get("dataset_name", "")

        rows.append({
            "tuple_id": record.get("tuple_id"),
            "image_id": record.get("image_id"),
            "method": record.get("method"),
            "model_name": record.get("model_name"),
            "dataset_type": record.get("dataset_type"),
            "dataset_variant_tag": variant_tag,
            "dataset_name": dataset_name,
            "category": category,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "repetition": record.get("repetition"),
            "seed": record.get("seed"),
            "mass_accuracy": record.get("mass_accuracy"),
            "discriminative_mass_accuracy": record.get("discriminative_mass_accuracy"),
            "distractor_mass_accuracy": record.get("distractor_mass_accuracy"),
            "discriminative_preference": record.get("discriminative_preference"),
            "relative_importance": record.get("relative_importance"),
            "discriminative_relative_importance": record.get("discriminative_relative_importance"),
            "distractor_relative_importance": record.get("distractor_relative_importance"),
            "full_ground_truth_mass_accuracy": record.get("full_ground_truth_mass_accuracy"),
            "model_source": record.get("model_source", "native"),
            "discriminative_enrichment": record.get("discriminative_enrichment"),
            "distractor_enrichment": record.get("distractor_enrichment"),
            "edge_corr_sobel": record.get("edge_corr_sobel"),
            "edge_corr_laplace": record.get("edge_corr_laplace"),
            "distractor_type": record.get("distractor_type"),
            "classification_target": record.get("classification_target"),
            "lesion_saliency_score": record.get("lesion_saliency_score"),
            "distortion_saliency_score": record.get("distortion_saliency_score"),
            "lesion_edge_salience": record.get("lesion_edge_salience"),
            "distortion_edge_salience": record.get("distortion_edge_salience"),
            "lesion_orientation_divergence": record.get("lesion_orientation_divergence"),
            "distortion_orientation_divergence": record.get("distortion_orientation_divergence"),
            "lesion_orientation_shift": record.get("lesion_orientation_shift"),
            "distortion_orientation_shift": record.get("distortion_orientation_shift"),
            "prediction_probability": record.get("prediction_probability"),
            "prediction_logit": record.get("prediction_logit"),
        })

    return pd.DataFrame(rows)


def _build_xai_method_series(
    xai_df: pd.DataFrame,
    category: str,
) -> dict[str, pd.DataFrame]:
    cat_df = xai_df[xai_df["category"] == category].copy()
    if cat_df.empty:
        return {}

    methods = _sort_methods(cat_df["method"].unique())
    result: dict[str, pd.DataFrame] = {}

    for method in methods:
        method_df = cat_df[cat_df["method"] == method]
        numeric_df = method_df[method_df["parameter_value"].notna()].copy()

        if not numeric_df.empty:
            aggregated = (
                numeric_df.groupby("parameter_value", as_index=False)
                .agg(
                    mass_accuracy=("mass_accuracy", "mean"),
                    num_samples=("mass_accuracy", "count"),
                )
                .sort_values("parameter_value")
                .reset_index(drop=True)
            )
            aggregated["x_value"] = aggregated["parameter_value"].astype(float)
            aggregated["x_label"] = aggregated["x_value"].map(_format_x_label)
            result[method] = aggregated
            continue

        if method_df.empty:
            continue

        fallback = pd.DataFrame([{
            "x_value": 0.0,
            "x_label": "all",
            "mass_accuracy": method_df["mass_accuracy"].mean(skipna=True),
            "num_samples": int(len(method_df)),
        }])
        result[method] = fallback

    return result


def _prepare_xai_data(
    config: ExperimentConfig,
    experiment_dir: Path,
) -> pd.DataFrame:
    eval_df = _load_xai_evaluation_records(config, experiment_dir)
    if eval_df.empty:
        return pd.DataFrame()
    return _build_xai_performance_dataframe(eval_df)


def _plot_xai_metric_line(
    axis,
    *,
    method_series: dict[str, pd.DataFrame],
    category: str,
    y_limits: tuple[float, float] = (0.0, 1.0),
) -> None:
    if not method_series:
        axis.text(
            0.5, 0.5, f"No {category} runs",
            transform=axis.transAxes, ha="center", va="center",
        )
        axis.set_title(f"{category.capitalize()} — Mass Accuracy")
        axis.set_ylabel("Mass Accuracy")
        axis.set_ylim(*y_limits)
        axis.grid(alpha=0.3)
        return

    all_x_values: list[float] = sorted({
        xv for series_df in method_series.values() for xv in series_df["x_value"]
    })
    x_pos = {xv: i for i, xv in enumerate(all_x_values)}
    x_labels = {}
    for series_df in method_series.values():
        for xv, xl in zip(series_df["x_value"], series_df["x_label"]):
            x_labels[xv] = xl

    for method_name, series_df in method_series.items():
        values = series_df["mass_accuracy"]
        if not values.notna().any():
            continue
        positions = [x_pos[xv] for xv in series_df["x_value"]]
        axis.plot(positions, values, marker="o", linewidth=2, label=_method_label(method_name))

    tick_positions = list(range(len(all_x_values)))
    if tick_positions:
        axis.set_xticks(tick_positions)
        axis.set_xticklabels([x_labels.get(xv, str(xv)) for xv in all_x_values])
    axis.set_title(f"{category.capitalize()} — Mass Accuracy")
    axis.set_ylabel("Mass Accuracy")
    axis.set_ylim(*y_limits)
    axis.grid(alpha=0.3)
    axis.legend(fontsize="small")


def xai_mass_accuracy_plot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    all_categories = [
        ("twirl", "Twirl Angle"),
        ("spherize", "Spherize Amount"),
        ("lesion", "Lesion Dataset"),
    ]
    present = set(xai_df["category"].unique())
    categories = [c for c in all_categories if c[0] in present] or all_categories
    n_cols = len(categories)

    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    for col, (category, x_label) in enumerate(categories):
        method_series = _build_xai_method_series(xai_df, category)
        _plot_xai_metric_line(axes[col], method_series=method_series, category=category)
        axes[col].set_xlabel(x_label)

    output_path = analyses_dir / "xai_mass_accuracy_plot.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved XAI mass accuracy plot to {output_path}")
    return output_path


def _resolve_variant_columns(
    xai_df: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    xai_df = xai_df.copy()
    xai_df["_variant_label"] = xai_df.apply(
        lambda r: _build_variant_label(
            str(r.get("dataset_name", "")),
            str(r.get("dataset_variant_tag", "")),
        ),
        axis=1,
    )
    labels = _dataset_ordered_labels(xai_df, "_variant_label")
    return [(label, xai_df[xai_df["_variant_label"] == label]) for label in labels]


def xai_mass_accuracy_heatmap(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    variants = _resolve_variant_columns(xai_df)
    if not variants:
        return None

    methods = _sort_methods(xai_df["method"].unique())
    if not methods:
        return None

    n_cols = len(variants)
    figure, axes = plt.subplots(
        1, n_cols, figsize=(max(3, 2 * len(methods)) * n_cols, max(4, len(methods) * 0.8 + 2)),
        squeeze=False, constrained_layout=True,
    )
    axes = axes[0]
    last_im = None

    for col, (variant_label, variant_df) in enumerate(variants):
        data = np.full((len(methods), 1), np.nan)
        for row_idx, method in enumerate(methods):
            method_df = variant_df[variant_df["method"] == method]
            ma_values = method_df["mass_accuracy"].dropna()
            if not ma_values.empty:
                data[row_idx, 0] = ma_values.mean()

        im = axes[col].imshow(data, aspect="auto", cmap=MAGNITUDE_COLORMAP, vmin=0.0, vmax=1.0)
        last_im = im

        for row_idx in range(len(methods)):
            val = data[row_idx, 0]
            if not np.isnan(val):
                axes[col].text(
                    0, row_idx, f"{val:.3f}", ha="center", va="center", fontsize=9,
                    color=magnitude_text_color(val),
                )

        axes[col].set_xticks([])
        axes[col].set_yticks(range(len(methods)))
        axes[col].set_yticklabels(_method_labels(methods) if col == 0 else [])
        _accent_dataset_panel(axes[col], variant_df, variant_label, fontsize=9)

    if last_im is not None:
        figure.colorbar(last_im, ax=axes, shrink=0.6, label="Mass Accuracy")

    output_path = analyses_dir / "xai_mass_accuracy_heatmap.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved XAI mass accuracy heatmap to {output_path}")
    return output_path


def xai_mass_accuracy_boxplot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    variants = _resolve_variant_columns(xai_df)
    if not variants:
        return None

    n_cols = len(variants)
    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]

    for col, (variant_label, variant_df) in enumerate(variants):
        methods = _sort_methods(variant_df["method"].unique())
        box_data = []
        box_labels = []
        present_methods = []
        for method in methods:
            values = variant_df.loc[variant_df["method"] == method, "mass_accuracy"].dropna()
            if not values.empty:
                box_data.append(values.values)
                box_labels.append(_method_label(method))
                present_methods.append(method)

        if not box_data:
            axes[col].text(
                0.5, 0.5, "No data",
                transform=axes[col].transAxes, ha="center", va="center",
            )
            _accent_dataset_panel(axes[col], variant_df, variant_label, fontsize=9)
            continue

        bp = axes[col].boxplot(box_data, patch_artist=True, showfliers=False, zorder=3)
        colors = [_method_color(m, present_methods) for m in present_methods]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        staggered_labels = [
            label if i % 2 == 0 else f"\n{label}"
            for i, label in enumerate(box_labels)
        ]
        axes[col].set_xticks(range(1, len(box_labels) + 1))
        axes[col].set_xticklabels(staggered_labels, fontsize=8)
        axes[col].set_ylim(0.0, 1.0)
        axes[col].set_ylabel("Mass Accuracy")
        _accent_dataset_panel(axes[col], variant_df, variant_label, fontsize=9)
        axes[col].set_axisbelow(True)
        axes[col].grid(alpha=0.3, which="both", zorder=0)
        axes[col].tick_params(axis="x", rotation=0)

    output_path = analyses_dir / "xai_mass_accuracy_boxplot.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved XAI mass accuracy boxplot to {output_path}")
    return output_path


_ARTIFACT_SYMBOL_LETTERS = {LESION_ARTIFACT: "L", TWIRL_ARTIFACT: "T", SPHERIZE_ARTIFACT: "S"}


def _artifact_letters(artifacts: tuple[str | None, str | None]) -> str:
    letters = "".join(
        _ARTIFACT_SYMBOL_LETTERS[artifact]
        for artifact in artifacts
        if artifact in _ARTIFACT_SYMBOL_LETTERS
    )
    return letters or "?"


def _dataset_symbol_letters_from_name(dataset_name: str) -> str:
    return _artifact_letters(dataset_artifacts_from_name(dataset_name))


def _dataset_symbol_letters(row) -> str:
    return _artifact_letters(dataset_artifacts(row))


def _dataset_symbol(letters: str) -> str:
    return rf"$\mathcal{{D}}_{{{letters}}}$"


def _model_symbol(letters: str) -> str:
    return rf"$f_{{\mathcal{{D}}_{{{letters}}}}}$"


def _analysis_dataset_label(row) -> str:
    return _dataset_symbol(_dataset_symbol_letters(row))


def _model_dataset_label(row) -> str:
    model_source = str(row.get("model_source", "native"))
    if model_source.startswith("lesion_only"):
        return _model_symbol("L")
    if model_source.startswith("distortion_only"):
        return _model_symbol(_ARTIFACT_SYMBOL_LETTERS.get(_distortion_family_label(row), "T"))
    return _model_symbol(_dataset_symbol_letters(row))


def _distortion_family_label(row) -> str:
    dataset_name = str(row.get("dataset_name", ""))
    if any(family in dataset_name for family in DISTORTION_FAMILY_ARTIFACTS):
        return distortion_family_from_name(dataset_name)
    category = str(row.get("category", ""))
    return category if category in DISTORTION_FAMILY_ARTIFACTS else DISTORTION_ARTIFACT


def _discriminator_and_distractor_artifacts(row) -> tuple[str, str]:
    discriminator, distractor = dataset_artifacts(row)
    return (
        discriminator or _distortion_family_label(row),
        distractor or LESION_ARTIFACT,
    )


def _relative_importance_panel_title(variant_df: pd.DataFrame) -> str:
    row = variant_df.iloc[0]
    return f"analysis dataset: {_analysis_dataset_label(row)}\nmodel dataset: {_model_dataset_label(row)}"


def _accent_dataset_panel(axis, variant_df: pd.DataFrame, title: str, *, fontsize) -> None:
    row = variant_df.iloc[0]
    axis.set_title(title, fontsize=fontsize, color=dataset_accent_color(row))
    accent_axis(axis, row)


def _color_dataset_tick_labels(
    axis,
    labelled_frames: list[tuple[str, pd.DataFrame]],
    accent_by_label: dict[str, str],
) -> None:
    for tick_label, (label, _) in zip(axis.get_xticklabels(), labelled_frames):
        tick_label.set_color(accent_by_label.get(label, PLAIN_EDGE_COLOR))


def _box_upper_whisker(values: np.ndarray) -> float:
    q1, q3 = np.percentile(values, [25, 75])
    upper_cap = q3 + 1.5 * (q3 - q1)
    within = values[values <= upper_cap]
    return float(within.max()) if within.size else float(values.max())


def _box_lower_whisker(values: np.ndarray) -> float:
    q1, q3 = np.percentile(values, [25, 75])
    lower_cap = q1 - 1.5 * (q3 - q1)
    within = values[values >= lower_cap]
    return float(within.min()) if within.size else float(values.min())


def _relative_importance_boxplot(
    config: ExperimentConfig,
    experiment_dir: Path,
    analyses_dir: Path,
    *,
    log_scale: bool,
    output_name: str,
    value_column: str = "relative_importance",
    combined_only: bool = False,
    y_label_base: str = "Relative Importance",
    figure_title: str | None = None,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty or value_column not in xai_df.columns:
        return None

    if combined_only:
        xai_df = xai_df[xai_df[value_column].notna()].copy()
        if xai_df.empty:
            logger.info(f"No records with {value_column}; skipping {output_name}.")
            return None

    variants = _resolve_variant_columns(xai_df)
    if not variants:
        return None

    panels = []
    upper_whiskers = []
    lower_whiskers = []
    for variant_label, variant_df in variants:
        box_data = []
        box_labels = []
        present_methods = []
        for method in _sort_methods(variant_df["method"].unique()):
            values = variant_df.loc[variant_df["method"] == method, value_column].dropna()
            if log_scale:
                values = values[values > 0]
            if not values.empty:
                box_data.append(values.values)
                box_labels.append(_method_label(method))
                present_methods.append(method)
                upper_whiskers.append(_box_upper_whisker(values.values))
                lower_whiskers.append(_box_lower_whisker(values.values))
        panels.append((variant_df, box_data, box_labels, present_methods))

    y_max = max(upper_whiskers) * 1.1 if upper_whiskers else 1.0
    if log_scale:
        y_min = max(min(min(lower_whiskers) * 0.9, 0.9), 1e-6) if lower_whiskers else 0.5
    else:
        y_min = 0.0
    y_label = f"{y_label_base} (log scale, 1 = uniform)" if log_scale else f"{y_label_base} (1 = uniform)"

    n_cols = len(panels)
    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    if figure_title:
        figure.suptitle(figure_title, fontsize=11)

    for col, (variant_df, box_data, box_labels, present_methods) in enumerate(panels):
        if log_scale:
            axes[col].set_yscale("log")
        axes[col].set_ylim(y_min, y_max)
        _accent_dataset_panel(
            axes[col], variant_df, _relative_importance_panel_title(variant_df), fontsize=9,
        )
        if not box_data:
            axes[col].text(
                0.5, 0.5, "No data",
                transform=axes[col].transAxes, ha="center", va="center",
            )
            continue

        bp = axes[col].boxplot(box_data, patch_artist=True, showfliers=False, zorder=3)
        colors = [_method_color(m, present_methods) for m in present_methods]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        axes[col].axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=2)
        staggered_labels = [
            label if i % 2 == 0 else f"\n{label}"
            for i, label in enumerate(box_labels)
        ]
        axes[col].set_xticks(range(1, len(box_labels) + 1))
        axes[col].set_xticklabels(staggered_labels, fontsize=8)
        axes[col].set_ylabel(y_label)
        axes[col].set_axisbelow(True)
        axes[col].grid(alpha=0.3, which="both" if log_scale else "major", zorder=0)
        axes[col].tick_params(axis="x", rotation=0)

    output_path = analyses_dir / output_name
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved {output_name} to {output_path}")
    return output_path


def xai_relative_importance_boxplot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_boxplot(
        config, experiment_dir, analyses_dir,
        log_scale=False, output_name="xai_relative_importance_boxplot.png",
    )


def xai_relative_importance_boxplot_log(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_boxplot(
        config, experiment_dir, analyses_dir,
        log_scale=True, output_name="xai_relative_importance_boxplot_log.png",
    )


_DISCRIMINATOR_RI_TITLE = (
    "Discriminator relative importance  "
    "(discriminator in the numerator; clean-breast denominator, both artifacts excluded)"
)
_DISTRACTOR_RI_TITLE = (
    "Distractor relative importance  "
    "(distractor in the numerator; clean-breast denominator, both artifacts excluded)"
)


def xai_discriminator_relative_importance_boxplot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_boxplot(
        config, experiment_dir, analyses_dir,
        log_scale=False, output_name="xai_discriminator_relative_importance_boxplot.png",
        value_column="discriminative_relative_importance", combined_only=True,
        y_label_base="Discriminator Relative Importance", figure_title=_DISCRIMINATOR_RI_TITLE,
    )


def xai_discriminator_relative_importance_boxplot_log(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_boxplot(
        config, experiment_dir, analyses_dir,
        log_scale=True, output_name="xai_discriminator_relative_importance_boxplot_log.png",
        value_column="discriminative_relative_importance", combined_only=True,
        y_label_base="Discriminator Relative Importance", figure_title=_DISCRIMINATOR_RI_TITLE,
    )


def xai_distractor_relative_importance_boxplot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_boxplot(
        config, experiment_dir, analyses_dir,
        log_scale=False, output_name="xai_distractor_relative_importance_boxplot.png",
        value_column="distractor_relative_importance", combined_only=True,
        y_label_base="Distractor Relative Importance", figure_title=_DISTRACTOR_RI_TITLE,
    )


def xai_distractor_relative_importance_boxplot_log(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_boxplot(
        config, experiment_dir, analyses_dir,
        log_scale=True, output_name="xai_distractor_relative_importance_boxplot_log.png",
        value_column="distractor_relative_importance", combined_only=True,
        y_label_base="Distractor Relative Importance", figure_title=_DISTRACTOR_RI_TITLE,
    )


def _relative_importance_disc_vs_distractor_boxplot(
    config: ExperimentConfig,
    experiment_dir: Path,
    analyses_dir: Path,
    *,
    log_scale: bool,
    output_name: str,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty or "discriminative_relative_importance" not in xai_df.columns:
        return None

    combined_df = xai_df[xai_df["discriminative_relative_importance"].notna()].copy()
    if combined_df.empty:
        logger.info(f"No records with per-region relative importance; skipping {output_name}.")
        return None

    variants = [
        (label, variant_df)
        for label, variant_df in _resolve_variant_columns(combined_df)
        if variant_df["discriminative_relative_importance"].notna().any()
    ]
    if not variants:
        return None

    panels = []
    upper_whiskers = []
    lower_whiskers = []
    for variant_label, variant_df in variants:
        entries = []
        for method in _sort_methods(variant_df["method"].unique()):
            method_df = variant_df[variant_df["method"] == method]
            discriminator_values = method_df["discriminative_relative_importance"].dropna()
            distractor_values = method_df["distractor_relative_importance"].dropna()
            if log_scale:
                discriminator_values = discriminator_values[discriminator_values > 0]
                distractor_values = distractor_values[distractor_values > 0]
            if discriminator_values.empty and distractor_values.empty:
                continue
            entries.append((method, discriminator_values.values, distractor_values.values))
            for values in (discriminator_values.values, distractor_values.values):
                if values.size:
                    upper_whiskers.append(_box_upper_whisker(values))
                    lower_whiskers.append(_box_lower_whisker(values))
        panels.append((variant_df, entries))

    y_max = max(upper_whiskers) * 1.1 if upper_whiskers else 1.0
    if log_scale:
        y_min = max(min(min(lower_whiskers) * 0.9, 0.9), 1e-6) if lower_whiskers else 0.5
    else:
        y_min = 0.0
    y_label = (
        "Relative Importance (log scale, 1 = uniform)"
        if log_scale
        else "Relative Importance (1 = uniform)"
    )

    n_cols = len(panels)
    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    figure.suptitle(
        "Relative importance per region — discriminator vs distractor "
        "(each artifact in the numerator; shared clean-breast denominator, both artifacts excluded)",
        fontsize=10,
    )

    for col, (variant_df, entries) in enumerate(panels):
        axis = axes[col]
        if log_scale:
            axis.set_yscale("log")
        axis.set_ylim(y_min, y_max)
        _accent_dataset_panel(
            axis, variant_df, _relative_importance_panel_title(variant_df), fontsize=9,
        )
        discriminator_artifact, distractor_artifact = _discriminator_and_distractor_artifacts(
            variant_df.iloc[0]
        )
        discriminator_color = artifact_color(discriminator_artifact)
        distractor_color = artifact_color(distractor_artifact)
        if not entries:
            axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center")
            continue

        discriminator_data, discriminator_positions = [], []
        distractor_data, distractor_positions = [], []
        centers, labels = [], []
        for index, (method, discriminator_values, distractor_values) in enumerate(entries):
            centers.append(index)
            labels.append(_method_label(method))
            if discriminator_values.size:
                discriminator_data.append(discriminator_values)
                discriminator_positions.append(index - 0.19)
            if distractor_values.size:
                distractor_data.append(distractor_values)
                distractor_positions.append(index + 0.19)

        if discriminator_data:
            discriminator_box = axis.boxplot(
                discriminator_data, positions=discriminator_positions, widths=0.34,
                patch_artist=True, showfliers=False, zorder=3,
            )
            for patch in discriminator_box["boxes"]:
                patch.set_facecolor(discriminator_color)
                patch.set_alpha(0.85)
            for median in discriminator_box["medians"]:
                median.set_color("black")
        if distractor_data:
            distractor_box = axis.boxplot(
                distractor_data, positions=distractor_positions, widths=0.34,
                patch_artist=True, showfliers=False, zorder=3,
            )
            for patch in distractor_box["boxes"]:
                patch.set_facecolor(distractor_color)
                patch.set_alpha(0.85)
            for median in distractor_box["medians"]:
                median.set_color("black")

        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=2)
        axis.set_xticks(centers)
        axis.set_xticklabels(_stagger_labels(labels), fontsize=8)
        axis.set_xlim(-0.6, len(entries) - 0.4)
        axis.set_ylabel(y_label)
        axis.set_axisbelow(True)
        axis.grid(alpha=0.3, which="both" if log_scale else "major", zorder=0)
        axis.legend(
            handles=[
                mpatches.Patch(
                    color=discriminator_color, alpha=0.85,
                    label=f"discriminator: {discriminator_artifact}",
                ),
                mpatches.Patch(
                    color=distractor_color, alpha=0.85,
                    label=f"distractor: {distractor_artifact}",
                ),
            ],
            fontsize="small",
        )

    output_path = analyses_dir / output_name
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved {output_name} to {output_path}")
    return output_path


def xai_relative_importance_disc_vs_distractor_boxplot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_disc_vs_distractor_boxplot(
        config, experiment_dir, analyses_dir,
        log_scale=False, output_name="xai_relative_importance_disc_vs_distractor_boxplot.png",
    )


def xai_relative_importance_disc_vs_distractor_boxplot_log(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_disc_vs_distractor_boxplot(
        config, experiment_dir, analyses_dir,
        log_scale=True, output_name="xai_relative_importance_disc_vs_distractor_boxplot_log.png",
    )


_REFERENCE_METHOD_ORDER = ("sobel", "laplace", "random")


def _positive_values(values: pd.Series) -> np.ndarray:
    positive = values.dropna()
    return positive[positive > 0].values


class _CenterStatistic(NamedTuple):
    name: str
    of_values: Callable[[np.ndarray], float]
    axis_label: str


_MEDIAN_CENTER = _CenterStatistic(
    name="median",
    of_values=lambda values: float(np.median(values)),
    axis_label=r"Median $\mathrm{RI}$ (log scale, 1 = uniform)",
)
_MEAN_CENTER = _CenterStatistic(
    name="mean",
    of_values=lambda values: float(np.mean(values)),
    axis_label=r"Mean $\mathrm{RI}$ (log scale, 1 = uniform)",
)


def _positive_center(values: pd.Series, center_statistic: _CenterStatistic) -> float | None:
    positive = _positive_values(values)
    return center_statistic.of_values(positive) if positive.size else None


def _standard_deviation(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def _reference_method_rank(method: str) -> int:
    if method in _REFERENCE_METHOD_ORDER:
        return _REFERENCE_METHOD_ORDER.index(method)
    return len(_REFERENCE_METHOD_ORDER)


def _methods_ordered_by_separation(
    combined_df: pd.DataFrame,
    center_statistic: _CenterStatistic,
) -> list[str]:
    separations = {}
    for method in combined_df["method"].unique():
        method_df = combined_df[combined_df["method"] == method]
        discriminator = _positive_center(
            method_df["discriminative_relative_importance"], center_statistic
        )
        distractor = _positive_center(
            method_df["distractor_relative_importance"], center_statistic
        )
        if discriminator is None or distractor is None:
            continue
        separations[method] = discriminator / distractor
    canonical_rank = {method: rank for rank, method in enumerate(_sort_methods(separations.keys()))}
    attribution_methods = [method for method in separations if _method_group_rank(method) == 0]
    reference_methods = [method for method in separations if _method_group_rank(method) != 0]
    return sorted(
        attribution_methods,
        key=lambda method: (-separations[method], canonical_rank[method]),
    ) + sorted(
        reference_methods,
        key=lambda method: (_reference_method_rank(method), canonical_rank[method]),
    )


class _DumbbellEntry(NamedTuple):
    rank: int
    method: str
    discriminator_center: float
    distractor_center: float
    discriminator_standard_deviation: float
    distractor_standard_deviation: float
    discriminator_sample_count: int
    distractor_sample_count: int


def _dumbbell_entries(
    variant_df: pd.DataFrame,
    method_order: list[str],
    center_statistic: _CenterStatistic,
) -> list[_DumbbellEntry]:
    entries = []
    for rank, method in enumerate(method_order):
        method_df = variant_df[variant_df["method"] == method]
        discriminator = _positive_values(method_df["discriminative_relative_importance"])
        distractor = _positive_values(method_df["distractor_relative_importance"])
        if not discriminator.size or not distractor.size:
            continue
        entries.append(_DumbbellEntry(
            rank=rank,
            method=method,
            discriminator_center=center_statistic.of_values(discriminator),
            distractor_center=center_statistic.of_values(distractor),
            discriminator_standard_deviation=_standard_deviation(discriminator),
            distractor_standard_deviation=_standard_deviation(distractor),
            discriminator_sample_count=int(discriminator.size),
            distractor_sample_count=int(distractor.size),
        ))
    return entries


def _analysis_group_frames(combined_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    grouped = combined_df.copy()
    grouped["_analysis_label"] = grouped.apply(_analysis_dataset_label, axis=1)
    labels = _dataset_ordered_labels(grouped, "_analysis_label")
    return [(label, grouped[grouped["_analysis_label"] == label]) for label in labels]


def _model_source_frames(group_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    sources = sorted(
        group_df["model_source"].astype(str).unique(),
        key=lambda source: (source != "native", source),
    )
    return [(source, group_df[group_df["model_source"].astype(str) == source]) for source in sources]


def _model_training_dataset_names(config: ExperimentConfig) -> dict[str, str]:
    training_dataset_names = {}
    for spec in config.xai.get("cross_evaluation") or []:
        if not isinstance(spec, dict):
            continue
        model_dataset_name = spec.get("model_dataset_name")
        if not model_dataset_name:
            continue
        model_source = str(spec.get("name") or model_dataset_name)
        training_dataset_names[model_source] = str(model_dataset_name)
    return training_dataset_names


def _model_label(row, training_dataset_names: dict[str, str]) -> str:
    model_source = str(row.get("model_source", "native"))
    if model_source in training_dataset_names:
        return _model_symbol(
            _dataset_symbol_letters_from_name(training_dataset_names[model_source])
        )
    return _model_dataset_label(row)


def _model_test_accuracy(
    row,
    *,
    accuracy_by_dataset: dict[tuple[str, str], float],
) -> float | None:
    return accuracy_by_dataset.get(
        (str(row.get("dataset_name", "")), str(row.get("model_source", "native")))
    )


def _model_legend_label(
    row,
    *,
    accuracy_by_dataset: dict[str, float],
    training_dataset_names: dict[str, str],
) -> str:
    label = _model_label(row, training_dataset_names)
    accuracy = _model_test_accuracy(row, accuracy_by_dataset=accuracy_by_dataset)
    if accuracy is None:
        return label
    return f"{label}  (test acc {accuracy:.2%})"


def _grouped_column_offsets(count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    span = 0.22
    step = 2 * span / (count - 1)
    return [-span + index * step for index in range(count)]


_BELOW_UNIFORM_COLOR = "#eeeeee"
_DUMBBELL_CONNECTOR_COLOR = "#9e9e9e"
_ROW_SEPARATOR_COLOR = "#dddddd"
_MODEL_SOURCE_LINE_STYLES = ("solid", "dashed", "dashdot", "dotted")
_LABELED_LOG_MANTISSAS = (2.0, 3.0, 5.0)


def _plain_log_tick_label(value: float, _position: int) -> str:
    return f"{value:g}"


def _labeled_log_minor_tick(value: float, _position: int) -> str:
    if value <= 0:
        return ""
    mantissa = value / 10 ** np.floor(np.log10(value))
    if not any(np.isclose(mantissa, labeled) for labeled in _LABELED_LOG_MANTISSAS):
        return ""
    return f"{value:g}"


def _label_log_value_axis_mid_decade_ticks(axis) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(_plain_log_tick_label))
    axis.yaxis.set_minor_formatter(FuncFormatter(_labeled_log_minor_tick))
    axis.tick_params(axis="y", which="major", labelsize=PAPER_TICK_LABEL_FONTSIZE)
    axis.tick_params(
        axis="y", which="minor",
        labelsize=PAPER_MINOR_TICK_LABEL_FONTSIZE, labelcolor="#555555",
    )


def _dumbbell_panels_per_combination(
    combined_df: pd.DataFrame,
    method_order: list[str],
    center_statistic: _CenterStatistic,
) -> list[tuple[str, list]]:
    panels = []
    for _, variant_df in _resolve_variant_columns(combined_df):
        entries = _dumbbell_entries(variant_df, method_order, center_statistic)
        if entries:
            panels.append((_relative_importance_panel_title(variant_df), [(variant_df, entries)]))
    return panels


def _dumbbell_panels_per_analysis_dataset(
    combined_df: pd.DataFrame,
    method_order: list[str],
    center_statistic: _CenterStatistic,
) -> list[tuple[str, list]]:
    panels = []
    for analysis_label, group_df in _analysis_group_frames(combined_df):
        series = [
            (source_df, _dumbbell_entries(source_df, method_order, center_statistic))
            for _, source_df in _model_source_frames(group_df)
        ]
        series = [(source_df, entries) for source_df, entries in series if entries]
        if series:
            panels.append((f"analysis dataset: {analysis_label}", series))
    return panels


def _dumbbell_table_frame(
    panels: list[tuple[str, list]],
    *,
    center_statistic: _CenterStatistic,
    accuracy_by_dataset: dict[str, float],
    training_dataset_names: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for _, series in panels:
        for source_df, entries in series:
            row = source_df.iloc[0]
            discriminator_artifact, distractor_artifact = _discriminator_and_distractor_artifacts(row)
            for entry in entries:
                rows.append({
                    "analysis_dataset": _analysis_dataset_label(row),
                    "model_dataset": _model_label(row, training_dataset_names),
                    "model_source": str(row.get("model_source", "native")),
                    "model_test_accuracy": _model_test_accuracy(
                        row, accuracy_by_dataset=accuracy_by_dataset,
                    ),
                    "dataset_name": str(row.get("dataset_name", "")),
                    "dataset_variant_tag": str(row.get("dataset_variant_tag", "")),
                    "method": entry.method,
                    "method_label": _method_label(entry.method),
                    "method_rank": entry.rank,
                    "discriminator_artifact": discriminator_artifact,
                    "distractor_artifact": distractor_artifact,
                    f"{center_statistic.name}_discriminator_relative_importance": (
                        entry.discriminator_center
                    ),
                    "std_discriminator_relative_importance": (
                        entry.discriminator_standard_deviation
                    ),
                    f"{center_statistic.name}_distractor_relative_importance": (
                        entry.distractor_center
                    ),
                    "std_distractor_relative_importance": entry.distractor_standard_deviation,
                    "discriminator_over_distractor_ratio": (
                        entry.discriminator_center / entry.distractor_center
                    ),
                    "num_discriminator_samples": entry.discriminator_sample_count,
                    "num_distractor_samples": entry.distractor_sample_count,
                })
    return pd.DataFrame(rows)


def _distortion_family_of_row(row) -> str:
    discriminator, distractor = dataset_artifacts(row)
    for artifact in (discriminator, distractor):
        if artifact in DISTORTION_FAMILY_ARTIFACTS:
            return artifact
    return DISTORTION_ARTIFACT


def _relative_importance_dumbbell(
    config: ExperimentConfig,
    experiment_dir: Path,
    analyses_dir: Path,
    *,
    group_by_analysis_dataset: bool,
    output_name: str,
    distortion_family: str | None = None,
    center_statistic: _CenterStatistic = _MEDIAN_CENTER,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty or "discriminative_relative_importance" not in xai_df.columns:
        return None

    all_combined_df = xai_df[
        xai_df["discriminative_relative_importance"].notna()
        & xai_df["distractor_relative_importance"].notna()
    ].copy()
    if all_combined_df.empty:
        logger.info(f"No records with per-region relative importance; skipping {output_name}.")
        return None

    combined_df = all_combined_df
    if distortion_family is not None:
        combined_df = all_combined_df[
            all_combined_df.apply(_distortion_family_of_row, axis=1) == distortion_family
        ].copy()
    if combined_df.empty:
        logger.info(f"No {distortion_family} records with relative importance; skipping {output_name}.")
        return None

    method_order = _methods_ordered_by_separation(all_combined_df, center_statistic)
    if not method_order:
        return None

    accuracy_by_dataset = accuracy_by_dataset_and_model(
        config=config, experiment_dir=experiment_dir,
    )
    training_dataset_names = _model_training_dataset_names(config)

    build_panels = (
        _dumbbell_panels_per_analysis_dataset
        if group_by_analysis_dataset
        else _dumbbell_panels_per_combination
    )
    panels = build_panels(combined_df, method_order, center_statistic)
    if not panels:
        return None

    bounds = [
        bound
        for _, series in build_panels(all_combined_df, method_order, center_statistic)
        for _, entries in series
        for entry in entries
        for bound in (entry.discriminator_center, entry.distractor_center)
    ]
    value_min = min(min(bounds) * 0.7, 0.7)
    value_max = max(max(bounds) * 1.4, 1.4)

    n_cols = len(panels)
    max_series_per_panel = max(len(series) for _, series in panels)
    method_column_scale = 1 + 0.35 * (max_series_per_panel - 1)
    figure, axes = plt.subplots(
        1, n_cols,
        figsize=((0.62 * len(method_order) * method_column_scale + 1.8) * n_cols, 7.5),
        squeeze=False, constrained_layout=True,
    )
    axes = axes[0]

    for col, (panel_title, series) in enumerate(panels):
        axis = axes[col]
        axis.set_yscale("log")
        axis.set_ylim(value_min, value_max)
        _label_log_value_axis_mid_decade_ticks(axis)
        axis.axhspan(value_min, 1.0, color=_BELOW_UNIFORM_COLOR, zorder=0)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=2)
        for boundary in range(len(method_order) - 1):
            axis.axvline(boundary + 0.5, color=_ROW_SEPARATOR_COLOR, linewidth=0.6, zorder=1)

        offsets = _grouped_column_offsets(len(series))
        for series_index, (source_df, entries) in enumerate(series):
            line_style = _MODEL_SOURCE_LINE_STYLES[series_index % len(_MODEL_SOURCE_LINE_STYLES)]
            discriminator, distractor = _discriminator_and_distractor_artifacts(source_df.iloc[0])
            for entry in entries:
                position = entry.rank + offsets[series_index]
                axis.plot(
                    [position, position], [entry.distractor_center, entry.discriminator_center],
                    color=_DUMBBELL_CONNECTOR_COLOR, linewidth=2, linestyle=line_style, zorder=3,
                )
                for center, artifact in (
                    (entry.distractor_center, distractor),
                    (entry.discriminator_center, discriminator),
                ):
                    axis.plot(
                        position, center, marker="o", markersize=8, color=artifact_color(artifact),
                        markeredgecolor="white", markeredgewidth=0.8, zorder=5,
                    )

        discriminator_artifact, distractor_artifact = _discriminator_and_distractor_artifacts(
            series[0][0].iloc[0]
        )
        legend_handles = [
            Line2D(
                [], [], marker="o", linestyle="none", markersize=8,
                color=artifact_color(discriminator_artifact),
                label=f"discriminator: {discriminator_artifact}",
            ),
            Line2D(
                [], [], marker="o", linestyle="none", markersize=8,
                color=artifact_color(distractor_artifact),
                label=f"distractor: {distractor_artifact}",
            ),
        ]
        legend_handles += [
            Line2D(
                [], [], linestyle=_MODEL_SOURCE_LINE_STYLES[index % len(_MODEL_SOURCE_LINE_STYLES)],
                linewidth=2, color=_DUMBBELL_CONNECTOR_COLOR,
                label=_model_legend_label(
                    source_df.iloc[0],
                    accuracy_by_dataset=accuracy_by_dataset,
                    training_dataset_names=training_dataset_names,
                ),
            )
            for index, (source_df, _) in enumerate(series)
        ]

        axis.set_xticks(range(len(method_order)))
        axis.set_xticklabels(
            _method_labels(method_order), fontsize=PAPER_TICK_LABEL_FONTSIZE,
            rotation=30, ha="right", rotation_mode="anchor",
        )
        axis.set_xlim(-0.7, len(method_order) - 0.3)
        axis.set_ylabel(center_statistic.axis_label, fontsize=PAPER_AXIS_LABEL_FONTSIZE)
        _accent_dataset_panel(
            axis, series[0][0], panel_title, fontsize=PAPER_PANEL_TITLE_FONTSIZE,
        )
        axis.set_axisbelow(True)
        axis.grid(alpha=0.3, axis="y", which="both", zorder=0)
        axis.legend(handles=legend_handles, fontsize=PAPER_LEGEND_FONTSIZE, loc="upper right")

    table_path = analyses_dir / f"{Path(output_name).stem}.csv"
    _dumbbell_table_frame(
        panels,
        center_statistic=center_statistic,
        accuracy_by_dataset=accuracy_by_dataset,
        training_dataset_names=training_dataset_names,
    ).to_csv(table_path, index=False)
    logger.info(f"Saved {table_path.name} to {table_path}")

    output_path = analyses_dir / output_name
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved {output_name} to {output_path}")
    return output_path


def xai_relative_importance_dumbbell(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_dumbbell(
        config, experiment_dir, analyses_dir,
        group_by_analysis_dataset=False,
        output_name="xai_relative_importance_dumbbell.png",
    )


def xai_relative_importance_dumbbell_grouped(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_dumbbell(
        config, experiment_dir, analyses_dir,
        group_by_analysis_dataset=True,
        output_name="xai_relative_importance_dumbbell_grouped.png",
    )


def xai_relative_importance_dumbbell_grouped_twirl(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_dumbbell(
        config, experiment_dir, analyses_dir,
        group_by_analysis_dataset=True,
        output_name="xai_relative_importance_dumbbell_grouped_twirl.png",
        distortion_family=TWIRL_ARTIFACT,
    )


def xai_relative_importance_dumbbell_grouped_spherize(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_dumbbell(
        config, experiment_dir, analyses_dir,
        group_by_analysis_dataset=True,
        output_name="xai_relative_importance_dumbbell_grouped_spherize.png",
        distortion_family=SPHERIZE_ARTIFACT,
    )


def xai_relative_importance_dumbbell_mean(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_dumbbell(
        config, experiment_dir, analyses_dir,
        group_by_analysis_dataset=False,
        output_name="xai_relative_importance_dumbbell_mean.png",
        center_statistic=_MEAN_CENTER,
    )


def xai_relative_importance_dumbbell_grouped_mean(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_dumbbell(
        config, experiment_dir, analyses_dir,
        group_by_analysis_dataset=True,
        output_name="xai_relative_importance_dumbbell_grouped_mean.png",
        center_statistic=_MEAN_CENTER,
    )


def xai_relative_importance_dumbbell_grouped_twirl_mean(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_dumbbell(
        config, experiment_dir, analyses_dir,
        group_by_analysis_dataset=True,
        output_name="xai_relative_importance_dumbbell_grouped_twirl_mean.png",
        distortion_family=TWIRL_ARTIFACT,
        center_statistic=_MEAN_CENTER,
    )


def xai_relative_importance_dumbbell_grouped_spherize_mean(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    return _relative_importance_dumbbell(
        config, experiment_dir, analyses_dir,
        group_by_analysis_dataset=True,
        output_name="xai_relative_importance_dumbbell_grouped_spherize_mean.png",
        distortion_family=SPHERIZE_ARTIFACT,
        center_statistic=_MEAN_CENTER,
    )


def _paired_log2_ratios(cell_df: pd.DataFrame) -> np.ndarray:
    pairs = cell_df[[
        "discriminative_relative_importance",
        "distractor_relative_importance",
    ]].dropna()
    positive_pairs = pairs[
        (pairs["discriminative_relative_importance"] > 0)
        & (pairs["distractor_relative_importance"] > 0)
    ]
    if positive_pairs.empty:
        return np.empty(0)
    return np.log2(
        positive_pairs["discriminative_relative_importance"].values
        / positive_pairs["distractor_relative_importance"].values
    )


def _per_seed_separation_indices(method_df: pd.DataFrame) -> list[float]:
    if "seed" not in method_df.columns:
        return []
    indices = []
    for _, seed_df in method_df.groupby("seed"):
        log2_ratios = _paired_log2_ratios(seed_df)
        if log2_ratios.size:
            indices.append(float(np.median(log2_ratios)))
    return indices


def _separation_index_statistics(method_df: pd.DataFrame) -> dict | None:
    log2_ratios = _paired_log2_ratios(method_df)
    if not log2_ratios.size:
        return None
    first_quartile, median, third_quartile = np.percentile(log2_ratios, [25, 50, 75])
    per_seed_indices = _per_seed_separation_indices(method_df)
    return {
        "separation_index_log2": float(median),
        "separation_index_log2_q1": float(first_quartile),
        "separation_index_log2_q3": float(third_quartile),
        "median_paired_ratio": float(2.0 ** median),
        "separation_index_log2_seed_mean": (
            float(np.mean(per_seed_indices)) if per_seed_indices else None
        ),
        "separation_index_log2_seed_sd": (
            float(np.std(per_seed_indices, ddof=1)) if len(per_seed_indices) > 1 else None
        ),
        "num_seeds": len(per_seed_indices),
        "num_pairs": int(log2_ratios.size),
    }


def xai_separation_index(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty or "discriminative_relative_importance" not in xai_df.columns:
        return None

    combined_df = xai_df[
        xai_df["discriminative_relative_importance"].notna()
        & xai_df["distractor_relative_importance"].notna()
    ].copy()
    if combined_df.empty:
        logger.info("No records with per-region relative importance; skipping xai_separation_index.")
        return None

    method_order = _methods_ordered_by_separation(combined_df)
    if not method_order:
        return None

    training_dataset_names = _model_training_dataset_names(config)

    rows = []
    for _, group_df in _analysis_group_frames(combined_df):
        for _, source_df in _model_source_frames(group_df):
            row = source_df.iloc[0]
            discriminator_artifact, distractor_artifact = _discriminator_and_distractor_artifacts(row)
            for rank, method in enumerate(method_order):
                statistics = _separation_index_statistics(source_df[source_df["method"] == method])
                if statistics is None:
                    continue
                rows.append({
                    "analysis_dataset": _analysis_dataset_label(row),
                    "model_dataset": _model_label(row, training_dataset_names),
                    "model_source": str(row.get("model_source", "native")),
                    "dataset_name": str(row.get("dataset_name", "")),
                    "dataset_variant_tag": str(row.get("dataset_variant_tag", "")),
                    "method": method,
                    "method_label": _method_label(method),
                    "method_rank": rank,
                    "discriminator_artifact": discriminator_artifact,
                    "distractor_artifact": distractor_artifact,
                    **statistics,
                })

    if not rows:
        return None

    output_path = analyses_dir / "xai_separation_index.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"Saved xai_separation_index.csv to {output_path}")
    return output_path


def xai_performance_comparison(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty or "relative_importance" not in xai_df.columns or "model_source" not in xai_df.columns:
        return None

    combined = xai_df[xai_df["dataset_name"].astype(str).str.contains("combined")].copy()
    if combined.empty:
        return None
    combined["is_cross"] = combined["model_source"].astype(str).ne("native")
    combined["analysis"] = combined.apply(_analysis_dataset_label, axis=1)

    conditions = [
        (False, "native model — trained on the same dataset"),
        (True, "cross model — trained without the distractor"),
    ]

    panels = []
    all_means = []
    for analysis in sorted(combined["analysis"].unique()):
        subset = combined[combined["analysis"] == analysis]
        methods = _sort_methods(subset["method"].unique())
        condition_means = []
        for is_cross, label in conditions:
            means = [
                float(subset.loc[(subset["method"] == m) & (subset["is_cross"] == is_cross), "relative_importance"].mean())
                for m in methods
            ]
            condition_means.append((label, means))
            all_means.extend([value for value in means if value == value])
        panels.append((analysis, subset, methods, condition_means))

    y_max = max(all_means) * 1.1 if all_means else 1.0

    n_cols = len(panels)
    figure, axes = plt.subplots(1, n_cols, figsize=(9 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    width = 0.38
    for col, (analysis, subset, methods, condition_means) in enumerate(panels):
        x = np.arange(len(methods))
        accent = dataset_accent_color(subset.iloc[0])
        for offset, (label, means) in enumerate(condition_means):
            is_cross = conditions[offset][0]
            axes[col].bar(
                x + (offset - 0.5) * width, means, width, label=label, zorder=3,
                color=accent, alpha=0.85,
                hatch=DISTRACTOR_HATCH if is_cross else None,
                edgecolor=PLAIN_EDGE_COLOR, linewidth=0.8,
            )
        axes[col].axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=2)
        staggered_labels = [
            lbl if i % 2 == 0 else f"\n{lbl}" for i, lbl in enumerate(_method_label(m) for m in methods)
        ]
        axes[col].set_xticks(x)
        axes[col].set_xticklabels(staggered_labels, fontsize=8)
        axes[col].set_ylim(0.0, y_max)
        axes[col].set_ylabel("mean relative importance (1 = uniform)")
        _accent_dataset_panel(axes[col], subset, f"analysis dataset: {analysis}", fontsize=11)
        axes[col].legend(fontsize=8)
        axes[col].set_axisbelow(True)
        axes[col].grid(alpha=0.3, axis="y", zorder=0)

    output_path = analyses_dir / "xai_performance_comparison.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved XAI performance comparison to {output_path}")
    return output_path


def _dataset_and_model_label(row) -> str:
    return f"{_analysis_dataset_label(row)}  ·  {_model_dataset_label(row)}"


def _resolve_dataset_symbol_columns(xai_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frame = xai_df.copy()
    frame["_dataset_symbol_label"] = frame.apply(_dataset_and_model_label, axis=1)
    labels = _dataset_ordered_labels(frame, "_dataset_symbol_label")
    return [(label, frame[frame["_dataset_symbol_label"] == label]) for label in labels]


def xai_mass_accuracy_boxplot_by_dataset(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    variants = _resolve_dataset_symbol_columns(xai_df)
    if not variants:
        return None

    method_scores: dict[str, float] = {}
    for method in xai_df["method"].unique():
        max_mean = float("-inf")
        for _, variant_df in variants:
            values = variant_df.loc[variant_df["method"] == method, "mass_accuracy"].dropna()
            if values.empty:
                continue
            mean_value = float(values.mean())
            if mean_value > max_mean:
                max_mean = mean_value
        if max_mean != float("-inf"):
            method_scores[method] = max_mean

    def _sort_key(method: str) -> tuple:
        group = _method_group_rank(method)
        if group == 0:
            return (group, -method_scores[method], method)
        return (group, method)

    methods = sorted(method_scores, key=_sort_key)
    if not methods:
        return None

    n_variants = len(variants)
    group_width = 0.8
    box_width = group_width / max(n_variants, 1)

    fig_width = max(10.0, 1.5 * len(methods) + 0.5 * max(n_variants, 1))
    figure, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)

    legend_handles = []
    for v_idx, (variant_label, variant_df) in enumerate(variants):
        box_data = []
        positions = []
        offset = (v_idx - (n_variants - 1) / 2) * box_width
        for m_idx, method in enumerate(methods):
            values = variant_df.loc[variant_df["method"] == method, "mass_accuracy"].dropna()
            if values.empty:
                continue
            box_data.append(values.values)
            positions.append(m_idx + 1 + offset)

        style = dataset_style(variant_df.iloc[0])
        legend_handles.append(dataset_legend_handle(variant_label, style))
        if not box_data:
            continue

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,
            zorder=3,
        )
        for patch in bp["boxes"]:
            apply_dataset_style(patch, style)

    labels = [_method_label(m) for m in methods]
    staggered_labels = [
        label if i % 2 == 0 else f"\n{label}"
        for i, label in enumerate(labels)
    ]
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(staggered_labels, fontsize=8)
    ax.set_xlim(0.5, len(methods) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mass Accuracy")
    ax.set_title("Mass Accuracy by Method and Dataset")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, which="both", zorder=0)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize="small",
        title="fill = discriminator · hatch = distractor",
        ncols=2,
    )

    output_path = analyses_dir / "xai_mass_accuracy_boxplot_by_dataset.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved XAI mass accuracy boxplot by dataset to {output_path}")
    return output_path


def xai_confidence_vs_mass_accuracy(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    variants = _resolve_variant_columns(xai_df)
    if not variants:
        return None

    n_cols = len(variants)
    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]

    for col, (variant_label, variant_df) in enumerate(variants):
        scored_df = variant_df.dropna(subset=["prediction_probability", "mass_accuracy"])
        if scored_df.empty:
            axes[col].text(
                0.5, 0.5, "No data",
                transform=axes[col].transAxes, ha="center", va="center",
            )
            _accent_dataset_panel(axes[col], variant_df, variant_label, fontsize=9)
            continue

        methods = _sort_methods(scored_df["method"].unique())

        for method in methods:
            method_df = scored_df[scored_df["method"] == method]
            x = method_df["prediction_probability"].values
            y = method_df["mass_accuracy"].values
            color = _method_color(method, methods)
            axes[col].scatter(x, y, alpha=0.4, s=20, color=color, label=_method_label(method))

            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                x_fit = np.linspace(x.min(), x.max(), 50)
                axes[col].plot(x_fit, np.polyval(coeffs, x_fit), color=color, linewidth=2)

        axes[col].set_xlim(0.0, 1.0)
        axes[col].set_ylim(0.0, 1.0)
        axes[col].set_xlabel("Prediction Probability")
        axes[col].set_ylabel("Mass Accuracy")
        _accent_dataset_panel(axes[col], variant_df, variant_label, fontsize=9)
        axes[col].grid(alpha=0.3)
        axes[col].legend(fontsize="small")

    output_path = analyses_dir / "xai_confidence_vs_mass_accuracy.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved confidence vs mass accuracy plot to {output_path}")
    return output_path


def xai_discriminative_vs_distractor_plot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty or "distractor_mass_accuracy" not in xai_df.columns:
        return None

    combined_df = xai_df[xai_df["distractor_mass_accuracy"].notna()].copy()
    if combined_df.empty:
        logger.info("No combined-dataset records with a distractor footprint; skipping plot.")
        return None

    variants = [
        (label, variant_df)
        for label, variant_df in _resolve_variant_columns(combined_df)
        if variant_df["distractor_mass_accuracy"].notna().any()
    ]
    if not variants:
        return None

    accuracy_by_dataset = accuracy_by_dataset_and_model(
        config=config, experiment_dir=experiment_dir,
    )

    n_cols = len(variants)
    figure, axes = plt.subplots(
        2, n_cols,
        figsize=(max(8.0, 1.3 * len(combined_df["method"].unique()) + 2.0) * n_cols, 10),
        squeeze=False, constrained_layout=True,
    )

    for col, (variant_label, variant_df) in enumerate(variants):
        present_methods: list[str] = []
        discriminative_means: list[float] = []
        distractor_means: list[float] = []
        preference_means: list[float] = []
        for method in _sort_methods(variant_df["method"].unique()):
            method_df = variant_df[variant_df["method"] == method]
            distractor_values = method_df["distractor_mass_accuracy"].dropna()
            if distractor_values.empty:
                continue
            discriminative_values = method_df["discriminative_mass_accuracy"].dropna()
            preference_values = method_df["discriminative_preference"].dropna()
            present_methods.append(method)
            discriminative_means.append(
                float(discriminative_values.mean()) if not discriminative_values.empty else np.nan
            )
            distractor_means.append(float(distractor_values.mean()))
            preference_means.append(
                float(preference_values.mean()) if not preference_values.empty else np.nan
            )

        target_values = variant_df["classification_target"].dropna()
        distractor_type_values = variant_df["distractor_type"].dropna()
        subtitle_bits = []
        if not target_values.empty:
            subtitle_bits.append(f"discriminator={target_values.iloc[0]}")
        if not distractor_type_values.empty:
            subtitle_bits.append(f"distractor={distractor_type_values.iloc[0]}")
        subtitle = ", ".join(subtitle_bits)

        dataset_names = variant_df["dataset_name"].dropna()
        dataset_name = str(dataset_names.iloc[0]) if not dataset_names.empty else ""
        model_accuracy = accuracy_by_dataset.get(dataset_name)

        positions = np.arange(len(present_methods))
        staggered_labels = _stagger_labels(_method_labels(present_methods))
        bar_width = 0.4

        discriminator_artifact, distractor_artifact = _discriminator_and_distractor_artifacts(
            variant_df.iloc[0]
        )

        top_axis = axes[0][col]
        top_axis.bar(
            positions - bar_width / 2, discriminative_means, bar_width,
            label=f"discriminator: {discriminator_artifact}",
            color=artifact_color(discriminator_artifact),
        )
        top_axis.bar(
            positions + bar_width / 2, distractor_means, bar_width,
            label=f"distractor: {distractor_artifact}",
            color=artifact_color(distractor_artifact),
            hatch=DISTRACTOR_HATCH, edgecolor=PLAIN_EDGE_COLOR, linewidth=0.8,
        )
        if model_accuracy is not None:
            top_axis.axhline(
                model_accuracy, color=PLAIN_EDGE_COLOR, linestyle="-", linewidth=1.6, zorder=5,
                label=f"model test acc = {model_accuracy:.3f}",
            )
        top_axis.set_xticks(positions)
        top_axis.set_xticklabels(staggered_labels, fontsize=8)
        top_axis.set_ylim(0.0, 1.0)
        top_axis.set_ylabel("Mean Mass Accuracy\n(area-matched rect footprints)")
        _accent_dataset_panel(top_axis, variant_df, f"{variant_label}\n{subtitle}", fontsize=9)
        top_axis.set_axisbelow(True)
        top_axis.grid(alpha=0.3, axis="y")
        top_axis.legend(fontsize="small")

        bottom_axis = axes[1][col]
        colors = [_method_color(method, present_methods) for method in present_methods]
        bottom_axis.bar(positions, preference_means, 0.6, color=colors, alpha=0.85)
        bottom_axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        bottom_axis.set_xticks(positions)
        bottom_axis.set_xticklabels(staggered_labels, fontsize=8)
        bottom_axis.set_ylim(0.0, 1.0)
        bottom_axis.set_ylabel("Discriminative Preference")
        bottom_axis.set_title("disc / (disc + distractor)   —   0.5 = neutral", fontsize=8)
        bottom_axis.set_axisbelow(True)
        bottom_axis.grid(alpha=0.3, axis="y")

    output_path = analyses_dir / "xai_discriminative_vs_distractor.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved discriminative-vs-distractor plot to {output_path}")
    return output_path


def xai_edge_correspondence(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty or "edge_corr_sobel" not in xai_df.columns:
        return None

    scored_df = xai_df[xai_df["edge_corr_sobel"].notna()].copy()
    if scored_df.empty:
        logger.info("No attribution-vs-edge correspondence scores; skipping plot.")
        return None

    variants = [
        (label, variant_df)
        for label, variant_df in _resolve_variant_columns(scored_df)
        if variant_df["edge_corr_sobel"].notna().any()
    ]
    if not variants:
        return None

    n_cols = len(variants)
    figure, axes = plt.subplots(
        1, n_cols,
        figsize=(max(8.0, 1.3 * len(scored_df["method"].unique()) + 2.0) * n_cols, 5.5),
        squeeze=False, constrained_layout=True,
    )

    lowest = 0.0
    for col, (variant_label, variant_df) in enumerate(variants):
        present_methods: list[str] = []
        sobel_means: list[float] = []
        laplace_means: list[float] = []
        for method in _sort_methods(variant_df["method"].unique()):
            method_df = variant_df[variant_df["method"] == method]
            sobel_values = method_df["edge_corr_sobel"].dropna()
            if sobel_values.empty:
                continue
            laplace_values = method_df["edge_corr_laplace"].dropna()
            present_methods.append(method)
            sobel_means.append(float(sobel_values.mean()))
            laplace_means.append(
                float(laplace_values.mean()) if not laplace_values.empty else np.nan
            )

        lowest = min([lowest, *sobel_means, *[v for v in laplace_means if not np.isnan(v)]])

        target_values = variant_df["classification_target"].dropna()
        distractor_type_values = variant_df["distractor_type"].dropna()
        subtitle_bits = []
        if not target_values.empty:
            subtitle_bits.append(f"discriminator={target_values.iloc[0]}")
        if not distractor_type_values.empty:
            subtitle_bits.append(f"distractor={distractor_type_values.iloc[0]}")
        subtitle = ", ".join(subtitle_bits)

        positions = np.arange(len(present_methods))
        staggered_labels = _stagger_labels(_method_labels(present_methods))
        bar_width = 0.4

        edge_filter_order = _sort_methods(["sobel", "laplace"])
        axis = axes[0][col]
        axis.bar(
            positions - bar_width / 2, sobel_means, bar_width,
            label="vs Sobel", color=_method_color("sobel", edge_filter_order),
        )
        axis.bar(
            positions + bar_width / 2, laplace_means, bar_width,
            label="vs Laplace", color=_method_color("laplace", edge_filter_order),
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(staggered_labels, fontsize=8)
        axis.set_ylabel("Mean Pearson r with edge map\n(over breast pixels)")
        _accent_dataset_panel(axis, variant_df, f"{variant_label}\n{subtitle}", fontsize=9)
        axis.set_axisbelow(True)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize="small")

    for col in range(n_cols):
        axes[0][col].set_ylim(min(-0.05, lowest - 0.05), 1.0)

    output_path = analyses_dir / "xai_edge_correspondence.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved attribution-vs-edge correspondence plot to {output_path}")
    return output_path


def _artifact_dataset_label(row) -> str:
    return _dataset_symbol(_dataset_symbol_letters(row))


def _wrapped_tick_labels(labels: list[str]) -> list[str]:
    return [label.replace(" (", "\n(") for label in labels]


def _artifact_dataset_frames(xai_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    grouped = xai_df.copy()
    grouped["_artifact_dataset_label"] = grouped.apply(_artifact_dataset_label, axis=1)
    labels = _dataset_ordered_labels(grouped, "_artifact_dataset_label")
    return [(label, grouped[grouped["_artifact_dataset_label"] == label]) for label in labels]


def _positive_values_for_log_axis(values: pd.Series, label: str, column: str) -> np.ndarray:
    positive = values[values > 0]
    dropped = len(values) - len(positive)
    if dropped:
        logger.info(f"{column} for {label!r}: {dropped}/{len(values)} non-positive values omitted on log axis.")
    return positive.values


def _saliency_per_image(variant_df: pd.DataFrame) -> pd.DataFrame:
    if "image_id" not in variant_df.columns:
        return pd.DataFrame()
    columns = ["image_id"] + [
        column for _, column, _ in _SALIENCY_ARTIFACTS if column in variant_df.columns
    ]
    if len(columns) == 1:
        return pd.DataFrame()
    return variant_df[columns].drop_duplicates(subset=["image_id"]).reset_index(drop=True)


def xai_saliency_boxplot_by_dataset(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    dataset_frames = _artifact_dataset_frames(xai_df)
    accent_by_label = {
        label: dataset_accent_color(frame.iloc[0]) for label, frame in dataset_frames
    }
    per_image_by_variant = [(label, _saliency_per_image(frame)) for label, frame in dataset_frames]
    per_image_by_variant = [(label, df) for label, df in per_image_by_variant if not df.empty]
    if not per_image_by_variant:
        logger.info("No saliency scores available; skipping saliency boxplot.")
        return None

    present_artifacts = [
        (artifact, column, color)
        for artifact, column, color in _SALIENCY_ARTIFACTS
        if any(
            column in df.columns and df[column].notna().any()
            for _, df in per_image_by_variant
        )
    ]
    if not present_artifacts:
        return None

    n_variants = len(per_image_by_variant)
    n_artifacts = len(present_artifacts)
    box_width = 0.8 / max(n_artifacts, 1)

    fig_width = max(8.0, 1.6 * n_variants + 2.0)
    figure, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)

    legend_handles = [
        mpatches.Patch(facecolor=color, alpha=0.85, label=artifact)
        for artifact, _, color in present_artifacts
    ]

    for a_idx, (_, column, color) in enumerate(present_artifacts):
        offset = (a_idx - (n_artifacts - 1) / 2) * box_width
        box_data = []
        positions = []
        for v_idx, (label, df) in enumerate(per_image_by_variant):
            if column not in df.columns:
                continue
            values = _positive_values_for_log_axis(df[column].dropna(), label, column)
            if not len(values):
                continue
            box_data.append(values)
            positions.append(v_idx + 1 + offset)
        if not box_data:
            continue
        bp = ax.boxplot(
            box_data, positions=positions, widths=box_width * 0.9,
            patch_artist=True, showfliers=False, zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

    ax.set_yscale("log")
    ax.set_xticks(range(1, n_variants + 1))
    ax.set_xticklabels(
        _wrapped_tick_labels([label for label, _ in per_image_by_variant]),
        fontsize=PAPER_TICK_LABEL_FONTSIZE,
    )
    _color_dataset_tick_labels(ax, per_image_by_variant, accent_by_label)
    ax.set_xlim(0.5, n_variants + 0.5)
    ax.set_ylabel(_SALIENCY_Y_LABEL, fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=PAPER_TICK_LABEL_FONTSIZE)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, axis="y", which="major", zorder=0)
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=PAPER_LEGEND_FONTSIZE,
        title="Artifact", title_fontsize=PAPER_LEGEND_FONTSIZE,
    )

    output_path = analyses_dir / "xai_saliency_boxplot_by_dataset.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved saliency boxplot by dataset to {output_path}")
    return output_path


def xai_saliency_lesion_vs_distortion(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None
    if not {"lesion_saliency_score", "distortion_saliency_score"}.issubset(xai_df.columns):
        return None

    paired_variants = []
    for label, variant_df in _artifact_dataset_frames(xai_df):
        per_image = _saliency_per_image(variant_df)
        if per_image.empty:
            continue
        if not {"lesion_saliency_score", "distortion_saliency_score"}.issubset(per_image.columns):
            continue
        paired = per_image.dropna(subset=["lesion_saliency_score", "distortion_saliency_score"])
        positive = paired[
            (paired["lesion_saliency_score"] > 0) & (paired["distortion_saliency_score"] > 0)
        ]
        if len(positive) < len(paired):
            logger.info(
                f"{label!r}: {len(paired) - len(positive)}/{len(paired)} image pairs omitted on log axes."
            )
        if positive.empty:
            continue
        paired_variants.append((label, positive))

    if not paired_variants:
        logger.info("No images with both lesion and distortion saliency; skipping scatter.")
        return None

    all_values = np.concatenate([
        np.concatenate([
            paired["lesion_saliency_score"].values, paired["distortion_saliency_score"].values,
        ])
        for _, paired in paired_variants
    ])
    axis_min = float(all_values.min()) * 0.7
    axis_max = float(all_values.max()) * 1.4

    n_cols = len(paired_variants)
    figure, axes = plt.subplots(
        1, n_cols, figsize=(5.5 * n_cols, 5.5), squeeze=False, constrained_layout=True,
    )
    axes = axes[0]

    for col, (label, paired) in enumerate(paired_variants):
        axis = axes[col]
        lesion = paired["lesion_saliency_score"].values
        distortion = paired["distortion_saliency_score"].values
        axis.scatter(
            lesion, distortion, alpha=0.35, s=20,
            color=PLAIN_EDGE_COLOR, edgecolors="none", zorder=3,
        )
        axis.plot(
            [axis_min, axis_max], [axis_min, axis_max],
            color="black", linestyle="--", linewidth=1.0, zorder=2,
        )
        axis.scatter(
            [lesion.mean()], [distortion.mean()],
            marker="X", s=160, color=DISTORTION_COLOR, edgecolors="black", zorder=5,
            label=f"mean ({lesion.mean():.3g}, {distortion.mean():.3g})",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.set_aspect("equal")
        axis.set_xlabel("Lesion saliency (log scale)", color=LESION_COLOR)
        axis.set_ylabel("Distortion saliency (log scale)", color=DISTORTION_COLOR)
        axis.set_axisbelow(True)
        axis.grid(alpha=0.3, zorder=0)
        axis.legend(fontsize="small", loc="upper left", title=f"{label}  ·  n={len(paired)}")

    output_path = analyses_dir / "xai_saliency_lesion_vs_distortion.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved lesion-vs-distortion saliency scatter to {output_path}")
    return output_path


def _edge_salience_per_image(variant_df: pd.DataFrame) -> pd.DataFrame:
    if "image_id" not in variant_df.columns:
        return pd.DataFrame()
    columns = ["image_id"] + [
        column for _, column, _ in _EDGE_SALIENCE_ARTIFACTS if column in variant_df.columns
    ]
    if len(columns) == 1:
        return pd.DataFrame()
    return variant_df[columns].drop_duplicates(subset=["image_id"]).reset_index(drop=True)


def xai_edge_salience_boxplot_by_dataset(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    dataset_frames = _artifact_dataset_frames(xai_df)
    accent_by_label = {
        label: dataset_accent_color(frame.iloc[0]) for label, frame in dataset_frames
    }
    per_image_by_variant = [
        (label, _edge_salience_per_image(frame)) for label, frame in dataset_frames
    ]
    per_image_by_variant = [(label, df) for label, df in per_image_by_variant if not df.empty]
    if not per_image_by_variant:
        logger.info("No edge salience scores available; skipping edge-salience boxplot.")
        return None

    present_artifacts = [
        (artifact, column, color)
        for artifact, column, color in _EDGE_SALIENCE_ARTIFACTS
        if any(
            column in df.columns and df[column].notna().any()
            for _, df in per_image_by_variant
        )
    ]
    if not present_artifacts:
        return None

    n_variants = len(per_image_by_variant)
    n_artifacts = len(present_artifacts)
    box_width = 0.8 / max(n_artifacts, 1)

    fig_width = max(8.0, 1.6 * n_variants + 2.0)
    figure, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)

    legend_handles = [
        mpatches.Patch(facecolor=color, alpha=0.85, label=artifact)
        for artifact, _, color in present_artifacts
    ]

    for a_idx, (_, column, color) in enumerate(present_artifacts):
        offset = (a_idx - (n_artifacts - 1) / 2) * box_width
        box_data = []
        positions = []
        for v_idx, (_, df) in enumerate(per_image_by_variant):
            if column not in df.columns:
                continue
            values = df[column].dropna()
            if values.empty:
                continue
            box_data.append(values.values)
            positions.append(v_idx + 1 + offset)
        if not box_data:
            continue
        bp = ax.boxplot(
            box_data, positions=positions, widths=box_width * 0.9,
            patch_artist=True, showfliers=False, zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xticks(range(1, n_variants + 1))
    ax.set_xticklabels(
        _wrapped_tick_labels([label for label, _ in per_image_by_variant]),
        fontsize=PAPER_TICK_LABEL_FONTSIZE,
    )
    _color_dataset_tick_labels(ax, per_image_by_variant, accent_by_label)
    ax.set_xlim(0.5, n_variants + 0.5)
    ax.set_ylabel(_EDGE_SALIENCE_Y_LABEL, fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=PAPER_TICK_LABEL_FONTSIZE)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, axis="y", zorder=0)
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=PAPER_LEGEND_FONTSIZE,
        title="Artifact", title_fontsize=PAPER_LEGEND_FONTSIZE,
    )

    output_path = analyses_dir / "xai_edge_salience_boxplot_by_dataset.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved edge-salience boxplot by dataset to {output_path}")
    return output_path


def _artifact_metrics_per_image(variant_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if "image_id" not in variant_df.columns:
        return pd.DataFrame()
    present_columns = ["image_id"] + [column for column in columns if column in variant_df.columns]
    if len(present_columns) == 1:
        return pd.DataFrame()
    return variant_df[present_columns].drop_duplicates(subset=["image_id"]).reset_index(drop=True)


def _plot_artifact_metric_boxes(
    axis,
    per_image_by_variant: list[tuple[str, pd.DataFrame]],
    artifacts: tuple[tuple[str, str, str], ...],
    *,
    y_label: str,
    title: str,
    reference_line: float | None = None,
    y_bottom: float | None = None,
    y_top: float | None = None,
) -> list[tuple[str, str, str]]:
    present_artifacts = [
        (artifact, column, color)
        for artifact, column, color in artifacts
        if any(column in df.columns and df[column].notna().any() for _, df in per_image_by_variant)
    ]
    n_variants = len(per_image_by_variant)
    if not present_artifacts:
        axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center")
        axis.set_ylabel(y_label)
        axis.set_title(title, fontsize=9)
        return present_artifacts

    n_artifacts = len(present_artifacts)
    box_width = 0.8 / max(n_artifacts, 1)
    for a_idx, (_, column, color) in enumerate(present_artifacts):
        offset = (a_idx - (n_artifacts - 1) / 2) * box_width
        box_data = []
        positions = []
        for v_idx, (_, df) in enumerate(per_image_by_variant):
            if column not in df.columns:
                continue
            values = df[column].dropna()
            if values.empty:
                continue
            box_data.append(values.values)
            positions.append(v_idx + 1 + offset)
        if not box_data:
            continue
        bp = axis.boxplot(
            box_data, positions=positions, widths=box_width * 0.9,
            patch_artist=True, showfliers=False, zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

    if reference_line is not None:
        axis.axhline(reference_line, color="black", linestyle="--", linewidth=1.0, zorder=2)
    labels = _shorten_variant_labels([label for label, _ in per_image_by_variant])
    axis.set_xticks(range(1, n_variants + 1))
    axis.set_xticklabels(_stagger_labels(labels), fontsize=8)
    axis.set_xlim(0.5, n_variants + 0.5)
    if y_bottom is not None or y_top is not None:
        axis.set_ylim(bottom=y_bottom, top=y_top)
    axis.set_ylabel(y_label)
    axis.set_title(title, fontsize=9)
    axis.set_axisbelow(True)
    axis.grid(alpha=0.3, axis="y", zorder=0)
    return present_artifacts


def xai_edge_vs_orientation(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    metric_columns = [
        column
        for _, column, _ in (
            *_EDGE_SALIENCE_ARTIFACTS,
            *_ORIENTATION_DIVERGENCE_ARTIFACTS,
            *_ORIENTATION_SHIFT_ARTIFACTS,
        )
    ]
    per_image_by_variant = [
        (label, _artifact_metrics_per_image(variant_df, metric_columns))
        for label, variant_df in _resolve_variant_columns(xai_df)
    ]
    per_image_by_variant = [(label, df) for label, df in per_image_by_variant if not df.empty]
    if not per_image_by_variant:
        logger.info("No saliency scores available; skipping edge-vs-orientation plot.")
        return None

    has_orientation = any(
        column in df.columns and df[column].notna().any()
        for _, df in per_image_by_variant
        for _, column, _ in _ORIENTATION_DIVERGENCE_ARTIFACTS
    )
    if not has_orientation:
        logger.info("No orientation scores present; skipping edge-vs-orientation plot.")
        return None

    n_variants = len(per_image_by_variant)
    fig_width = max(9.0, 1.7 * n_variants + 2.0)
    figure, axes = plt.subplots(
        3, 1, figsize=(fig_width, 12), constrained_layout=True, sharex=True,
    )

    _plot_artifact_metric_boxes(
        axes[0], per_image_by_variant, _EDGE_SALIENCE_ARTIFACTS,
        y_label=_EDGE_SALIENCE_SHORT_Y_LABEL,
        title="Gradient magnitude channel — edge salience  (1 = clean-level edge energy; a rotation preserves it)",
        reference_line=1.0,
    )
    present_artifacts = _plot_artifact_metric_boxes(
        axes[1], per_image_by_variant, _ORIENTATION_DIVERGENCE_ARTIFACTS,
        y_label=_ORIENTATION_DIVERGENCE_Y_LABEL,
        title="Gradient orientation channel — distribution change  (0 = identical orientation content)",
        y_bottom=0.0,
    )
    _plot_artifact_metric_boxes(
        axes[2], per_image_by_variant, _ORIENTATION_SHIFT_ARTIFACTS,
        y_label=_ORIENTATION_SHIFT_Y_LABEL,
        title="Gradient orientation channel — dominant rotation  (twirl: low here + high divergence = a spatially-varying swirl)",
        y_bottom=0.0,
    )

    if present_artifacts:
        axes[0].legend(
            handles=[
                mpatches.Patch(facecolor=color, alpha=0.85, label=artifact)
                for artifact, _, color in present_artifacts
            ],
            loc="upper right", fontsize="small", title="Injected artifact",
        )
    figure.suptitle(
        "Where the injected artifact lives: magnitude (Sobel-visible) vs orientation (Sobel-magnitude-blind)",
        fontsize=11,
    )

    output_path = analyses_dir / "xai_edge_vs_orientation.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved edge-vs-orientation plot to {output_path}")
    return output_path


def xai_summary(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path | None:
    xai_df = _prepare_xai_data(config, experiment_dir)
    if xai_df.empty:
        return None

    group_cols = [
        "tuple_id", "method", "dataset_type", "dataset_name",
        "dataset_variant_tag", "category", "parameter_name", "parameter_value",
    ]
    existing_cols = [c for c in group_cols if c in xai_df.columns]

    summary_df = (
        xai_df.groupby(existing_cols, as_index=False, dropna=False)
        .agg(
            mean_mass_accuracy=("mass_accuracy", "mean"),
            std_mass_accuracy=("mass_accuracy", "std"),
            median_mass_accuracy=("mass_accuracy", "median"),
            mean_discriminative_mass_accuracy=("discriminative_mass_accuracy", "mean"),
            mean_distractor_mass_accuracy=("distractor_mass_accuracy", "mean"),
            mean_discriminative_preference=("discriminative_preference", "mean"),
            mean_relative_importance=("relative_importance", "mean"),
            median_relative_importance=("relative_importance", "median"),
            mean_discriminative_relative_importance=("discriminative_relative_importance", "mean"),
            median_discriminative_relative_importance=("discriminative_relative_importance", "median"),
            mean_distractor_relative_importance=("distractor_relative_importance", "mean"),
            median_distractor_relative_importance=("distractor_relative_importance", "median"),
            mean_discriminative_enrichment=("discriminative_enrichment", "mean"),
            mean_distractor_enrichment=("distractor_enrichment", "mean"),
            mean_lesion_edge_salience=("lesion_edge_salience", "mean"),
            mean_distortion_edge_salience=("distortion_edge_salience", "mean"),
            mean_lesion_orientation_divergence=("lesion_orientation_divergence", "mean"),
            mean_distortion_orientation_divergence=("distortion_orientation_divergence", "mean"),
            mean_lesion_orientation_shift=("lesion_orientation_shift", "mean"),
            mean_distortion_orientation_shift=("distortion_orientation_shift", "mean"),
            mean_edge_corr_sobel=("edge_corr_sobel", "mean"),
            mean_edge_corr_laplace=("edge_corr_laplace", "mean"),
            num_distractor_samples=("distractor_mass_accuracy", "count"),
            mean_prediction_probability=("prediction_probability", "mean"),
            num_samples=("mass_accuracy", "count"),
        )
        .sort_values(["category", "method", "parameter_value"])
        .reset_index(drop=True)
    )

    csv_path = analyses_dir / "xai_evaluation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved XAI evaluation summary ({len(summary_df)} rows) to {csv_path}")

    plot_group_cols = ["dataset_name", "dataset_variant_tag"]
    plot_existing = [c for c in plot_group_cols if c in xai_df.columns]
    if not plot_existing:
        return csv_path

    plot_df = (
        xai_df.groupby(plot_existing + ["method"], as_index=False, dropna=False)
        .agg(mean_mass_accuracy=("mass_accuracy", "mean"))
    )
    plot_df["variant_label"] = plot_df.apply(
        lambda r: _build_variant_label(
            str(r.get("dataset_name", "")),
            str(r.get("dataset_variant_tag", "")),
        ),
        axis=1,
    )
    label_colors = {
        str(row["variant_label"]): dataset_accent_color(row)
        for _, row in plot_df.iterrows()
    }

    method_scores: dict[str, float] = (
        plot_df.groupby("method")["mean_mass_accuracy"].max().to_dict()
    )

    def _summary_sort_key(method: str) -> tuple:
        group = _method_group_rank(method)
        if group == 0:
            return (group, -method_scores.get(method, float("-inf")), method)
        return (group, method)

    methods = sorted(plot_df["method"].unique(), key=_summary_sort_key)
    variant_labels = _dataset_ordered_labels(plot_df, "variant_label")
    if not methods or not variant_labels:
        return csv_path

    data = np.full((len(variant_labels), len(methods)), np.nan)
    label_to_row = {label: i for i, label in enumerate(variant_labels)}
    method_to_col = {m: i for i, m in enumerate(methods)}

    for _, row in plot_df.iterrows():
        r = label_to_row.get(row["variant_label"])
        c = method_to_col.get(row["method"])
        if r is not None and c is not None:
            data[r, c] = row["mean_mass_accuracy"]

    fig_width = 0.9 * len(methods) + 3.5
    fig_height = 0.28 * len(variant_labels) + 1.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    im = ax.imshow(data, aspect="equal", cmap=MAGNITUDE_COLORMAP, vmin=0.0, vmax=1.0)
    for r_idx in range(len(variant_labels)):
        for c_idx in range(len(methods)):
            val = data[r_idx, c_idx]
            if not np.isnan(val):
                ax.text(
                    c_idx, r_idx, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color=magnitude_text_color(val),
                )

    method_label_texts = _method_labels(methods)
    staggered_method_labels = [
        label if i % 2 == 0 else f"\n{label}"
        for i, label in enumerate(method_label_texts)
    ]
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(staggered_method_labels, rotation=0, ha="center", fontsize=7)
    ax.set_yticks(range(len(variant_labels)))
    ax.set_yticklabels(variant_labels, fontsize=7)
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(label_colors.get(tick_label.get_text(), PLAIN_EDGE_COLOR))
    ax.set_title("Mass Accuracy by Dataset Variant and XAI Method", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="Mean Mass Accuracy")
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Mean Mass Accuracy", fontsize=7)

    plot_path = analyses_dir / "xai_evaluation_summary.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved XAI evaluation summary plot to {plot_path}")

    return csv_path
