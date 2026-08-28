from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from analyses._colors import (
    MAGNITUDE_COLORMAP,
    PLAIN_EDGE_COLOR,
    artifact_color,
    dataset_artifacts_from_name,
    magnitude_text_color,
)
from analyses._utils import (
    _format_x_label,
    _hyperparameter_variant_label,
    _lesion_size_order,
    _resolve_category_and_parameter,
    _resolve_record_path,
    _short_variant_label,
)
from config.configuration import ExperimentConfig
from utils import (
    generate_training_dir,
    load_training_records,
)

MODEL_ACCURACY_PLOT = "model_accuracy_plot"
MODEL_ACCURACY_ALL_SPLITS_PLOT = "model_accuracy_all_splits_plot"
MODEL_ACCURACY_GROUPED_BAR_PLOT = "model_accuracy_grouped_bar_plot"
MODEL_ACCURACY_HEATMAP = "model_accuracy_heatmap"
MODEL_LOSS_PLOT = "model_loss_plot"
HYPERPARAMETER_SUMMARY = "hyperparameter_summary"

_HYPERPARAMETER_AXIS_LABELS = {
    "twirl": "Twirl Angle",
    "spherize": "Spherize Amount",
    "lesion": "Lesion Dataset",
}


def _last_non_null(series: pd.Series) -> float | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return float(non_null.iloc[-1])


def _extract_metrics_from_log(metrics_path: Path) -> dict[str, float | None]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Lightning metrics log not found: {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        raise ValueError(f"Lightning metrics log is empty: {metrics_path}")

    available_columns = set(metrics_df.columns)
    metric_values = {}
    for metric_name in ("test_acc", "test_loss", "train_acc", "train_loss", "val_acc", "val_loss"):
        csv_column = "train_loss_epoch" if metric_name == "train_loss" else metric_name
        if csv_column in available_columns:
            metric_values[metric_name] = _last_non_null(metrics_df[csv_column])
        else:
            metric_values[metric_name] = None
    return metric_values


def _format_filter_label(filters: dict | None, fallback: str) -> str:
    if not isinstance(filters, dict) or not filters:
        return fallback
    ordered_items = sorted(filters.items(), key=lambda item: item[0])
    return ", ".join(f"{key}={value}" for key, value in ordered_items)


def _dataset_series_label(dataset_name: str) -> str:
    base_label = _short_variant_label(dataset_name)
    discriminator, distractor = dataset_artifacts_from_name(dataset_name)
    if distractor is None:
        return base_label
    return f"{base_label} ({discriminator})"


def _series_dataset_name(series_df: pd.DataFrame) -> str:
    if "dataset_name" not in series_df.columns:
        return ""
    dataset_names = series_df["dataset_name"].dropna()
    return str(dataset_names.iloc[0]) if len(dataset_names) else ""


def _series_color(series_df: pd.DataFrame) -> str:
    discriminator, _ = dataset_artifacts_from_name(_series_dataset_name(series_df))
    return artifact_color(discriminator)


def _has_distractor(series_df: pd.DataFrame) -> bool:
    return dataset_artifacts_from_name(_series_dataset_name(series_df))[1] is not None


def _build_category_series(
    performance_df: pd.DataFrame,
    category: str,
) -> dict[str, pd.DataFrame]:
    category_df = performance_df[performance_df["category"] == category].copy()
    if category_df.empty:
        return {}

    variant_names = sorted(category_df["dataset_name"].unique(), key=_lesion_size_order)
    result: dict[str, pd.DataFrame] = {}

    for dataset_name in variant_names:
        variant_df = category_df[category_df["dataset_name"] == dataset_name]
        label = _dataset_series_label(dataset_name)

        numeric_df = variant_df[variant_df["parameter_value"].notna()].copy()
        if not numeric_df.empty:
            aggregated = (
                numeric_df.groupby("parameter_value", as_index=False)
                .agg(
                    test_acc=("test_acc", "mean"),
                    test_loss=("test_loss", "mean"),
                    train_acc=("train_acc", "mean"),
                    train_loss=("train_loss", "mean"),
                    val_acc=("val_acc", "mean"),
                    val_loss=("val_loss", "mean"),
                    num_runs=("variant_tag", "count"),
                )
                .sort_values("parameter_value")
                .reset_index(drop=True)
            )
            aggregated["x_value"] = aggregated["parameter_value"].astype(float)
            aggregated["x_label"] = aggregated["x_value"].map(_format_x_label)
            aggregated["dataset_name"] = dataset_name
            result[label] = aggregated
            continue

        if variant_df.empty:
            continue

        fallback = pd.DataFrame([
            {
                "dataset_name": dataset_name,
                "x_value": 0.0,
                "x_label": "all",
                "test_acc": variant_df["test_acc"].mean(skipna=True),
                "test_loss": variant_df["test_loss"].mean(skipna=True),
                "train_acc": variant_df["train_acc"].mean(skipna=True),
                "train_loss": variant_df["train_loss"].mean(skipna=True),
                "val_acc": variant_df["val_acc"].mean(skipna=True),
                "val_loss": variant_df["val_loss"].mean(skipna=True),
                "num_runs": int(len(variant_df)),
            }
        ])
        result[label] = fallback

    return result


def _plot_metric_line(
    axis,
    *,
    category_series: dict[str, pd.DataFrame],
    metric_name: str,
    category: str,
    metric_label: str,
    primary_split_label: str = "Test",
    overlays: list[tuple[str, str, str]] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    if not category_series:
        axis.text(
            0.5,
            0.5,
            f"No {category} runs",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_title(f"{category.capitalize()} — {metric_label}")
        axis.set_ylabel(metric_label)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        axis.grid(alpha=0.3)
        return

    overlays = overlays or []

    all_x_values: list[float] = sorted({
        xv
        for series_df in category_series.values()
        for xv in series_df["x_value"]
    })
    x_pos = {xv: i for i, xv in enumerate(all_x_values)}
    x_labels = {}
    for series_df in category_series.values():
        for xv, xl in zip(series_df["x_value"], series_df["x_label"]):
            x_labels[xv] = xl

    single_variant = len(category_series) == 1

    def _legend_label(variant_label: str, split_label: str) -> str:
        if single_variant:
            return split_label
        return f"{variant_label} ({split_label.lower()})"

    for variant_label, series_df in category_series.items():
        metric_values = series_df[metric_name]
        if not metric_values.notna().any():
            continue
        positions = [x_pos[xv] for xv in series_df["x_value"]]
        color = _series_color(series_df)
        axis.plot(
            positions,
            metric_values,
            marker="s" if _has_distractor(series_df) else "o",
            linestyle="--" if _has_distractor(series_df) else "-",
            color=color,
            linewidth=2,
            label=_legend_label(variant_label, primary_split_label),
        )

        for overlay_metric, overlay_split_label, linestyle in overlays:
            if overlay_metric not in series_df.columns:
                continue
            overlay_values = series_df[overlay_metric]
            if not overlay_values.notna().any():
                continue
            axis.plot(
                positions,
                overlay_values,
                marker="s",
                linewidth=1.5,
                linestyle=linestyle,
                color=color,
                alpha=0.6,
                label=_legend_label(variant_label, overlay_split_label),
            )

    tick_positions = list(range(len(all_x_values)))
    if tick_positions:
        axis.set_xticks(tick_positions)
        axis.set_xticklabels([x_labels.get(xv, str(xv)) for xv in all_x_values])
    axis.set_title(f"{category.capitalize()} — {metric_label}")
    axis.set_ylabel(metric_label)
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    axis.grid(alpha=0.3)
    axis.legend(fontsize="small")


def _build_model_performance_dataframe(
    config: ExperimentConfig,
    training_records_path: Path,
    experiment_dir: Path,
) -> pd.DataFrame:
    if not training_records_path.exists():
        raise FileNotFoundError(f"Training records not found: {training_records_path}")

    records = load_training_records(file_path=str(training_records_path))
    if not records:
        raise ValueError(f"No training records found: {training_records_path}")

    rows = []
    for record in records:
        dataset_meta_data = record.dataset_meta_data or {}
        variant_tag = str(dataset_meta_data.get("dataset_variant_tag", "full_dataset"))
        selected_filters = dataset_meta_data.get("selected_generated_filters")
        category, parameter_name, parameter_value = _resolve_category_and_parameter(
            selected_filters=selected_filters,
            variant_tag=variant_tag,
        )

        metrics_path = _resolve_record_path(path_str=record.training_log_path, experiment_dir=experiment_dir)
        metrics = _extract_metrics_from_log(metrics_path=metrics_path)

        dataset_name = ""
        if isinstance(selected_filters, dict):
            dataset_name = selected_filters.get("dataset_name", "")

        rows.append({
            "variant_tag": variant_tag,
            "variant_label": _format_filter_label(filters=selected_filters, fallback=variant_tag),
            "dataset_name": dataset_name,
            "model_name": record.model_name,
            "seed": record.seed,
            "repetition": record.repetition,
            "num_total_samples": dataset_meta_data.get("num_total_samples"),
            "category": category,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "test_acc": metrics["test_acc"],
            "test_loss": metrics["test_loss"],
            "train_acc": metrics["train_acc"],
            "train_loss": metrics["train_loss"],
            "val_acc": metrics["val_acc"],
            "val_loss": metrics["val_loss"],
            "model_path": record.model_path,
            "training_log_path": str(metrics_path),
        })

    performance_df = pd.DataFrame(rows)
    if performance_df.empty:
        raise ValueError(f"Could not build model performance dataframe from: {training_records_path}")
    return performance_df


def _training_records_path(config: ExperimentConfig, experiment_dir: Path) -> Path:
    training_dir = Path(
        generate_training_dir(base_dir=str(experiment_dir), training_config=config.training)
    ).resolve()
    training_records_name = config.training.get("training_records", "training_records.jsonl")
    return (training_dir / training_records_name).resolve()


def _mean_test_accuracy_by_dataset(performance_df: pd.DataFrame) -> dict[str, float]:
    scored = performance_df.dropna(subset=["test_acc"])
    grouped = scored.groupby("dataset_name")["test_acc"].mean()
    return {str(name): float(value) for name, value in grouped.items() if str(name)}


def _model_accuracy_from_summary_csv(analyses_dir: Path) -> dict[str, float]:
    summary_path = (analyses_dir / "model_performance_summary.csv").resolve()
    if not summary_path.exists():
        return {}
    summary_df = pd.read_csv(summary_path)
    if not {"dataset_name", "test_acc"}.issubset(summary_df.columns):
        return {}
    return _mean_test_accuracy_by_dataset(summary_df)


def model_accuracy_by_dataset(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> dict[str, float]:
    try:
        performance_df = _build_model_performance_dataframe(
            config=config,
            training_records_path=_training_records_path(config, experiment_dir),
            experiment_dir=experiment_dir,
        )
        accuracy_by_dataset = _mean_test_accuracy_by_dataset(performance_df)
        if accuracy_by_dataset:
            return accuracy_by_dataset
    except (FileNotFoundError, ValueError) as error:
        logger.info(f"Falling back to model_performance_summary.csv for model accuracy: {error}")
    return _model_accuracy_from_summary_csv(analyses_dir)


def _prepare_performance_data(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    training_records_path = _training_records_path(config, experiment_dir)

    performance_df = _build_model_performance_dataframe(
        config=config,
        training_records_path=training_records_path,
        experiment_dir=experiment_dir,
    )

    if not performance_df["test_acc"].notna().any():
        raise ValueError(
            f"No test_acc values found in training logs referenced by {training_records_path}"
        )

    performance_df = performance_df.sort_values(
        by=["category", "parameter_value", "variant_label", "model_name", "repetition", "seed"],
    ).reset_index(drop=True)

    summary_csv_path = analyses_dir / "model_performance_summary.csv"
    performance_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved model performance summary to {summary_csv_path}")

    twirl_series = _build_category_series(performance_df=performance_df, category="twirl")
    spherize_series = _build_category_series(performance_df=performance_df, category="spherize")
    lesion_series = _build_category_series(performance_df=performance_df, category="lesion")

    return twirl_series, spherize_series, lesion_series


def model_accuracy_plot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path:
    twirl_series, spherize_series, lesion_series = _prepare_performance_data(
        config, experiment_dir=experiment_dir, analyses_dir=analyses_dir,
    )

    all_categories = [
        ("twirl", twirl_series, "Twirl Angle"),
        ("spherize", spherize_series, "Spherize Amount"),
        ("lesion", lesion_series, "Lesion Dataset"),
    ]
    categories = [c for c in all_categories if c[1]] or all_categories
    n_cols = len(categories)

    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    for col, (category, series, x_label) in enumerate(categories):
        _plot_metric_line(
            axes[col],
            category_series=series,
            metric_name="test_acc",
            category=category,
            metric_label="Accuracy",
            y_limits=(0.0, 1.0),
            overlays=[("train_acc", "Train", "dotted")],
        )
        axes[col].set_xlabel(x_label)

    output_path = analyses_dir / "model_accuracy_plot.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved model accuracy plot to {output_path}")
    return output_path


def model_accuracy_all_splits_plot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path:
    twirl_series, spherize_series, lesion_series = _prepare_performance_data(
        config, experiment_dir=experiment_dir, analyses_dir=analyses_dir,
    )

    all_categories = [
        ("twirl", twirl_series, "Twirl Angle"),
        ("spherize", spherize_series, "Spherize Amount"),
        ("lesion", lesion_series, "Lesion Dataset"),
    ]
    categories = [c for c in all_categories if c[1]] or all_categories
    n_cols = len(categories)

    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    for col, (category, series, x_label) in enumerate(categories):
        _plot_metric_line(
            axes[col],
            category_series=series,
            metric_name="test_acc",
            category=category,
            metric_label="Accuracy",
            y_limits=(0.0, 1.0),
            overlays=[
                ("train_acc", "Train", "dotted"),
                ("val_acc", "Val", "dashed"),
            ],
        )
        axes[col].set_xlabel(x_label)

    output_path = analyses_dir / "model_accuracy_all_splits_plot.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved model accuracy all-splits plot to {output_path}")
    return output_path


def _plot_accuracy_grouped_bars(
    axis,
    *,
    category_series: dict[str, pd.DataFrame],
    category: str,
    x_label: str,
    metric_name: str = "test_acc",
) -> None:
    if not category_series:
        axis.text(
            0.5, 0.5, f"No {category} runs",
            transform=axis.transAxes, ha="center", va="center",
        )
        axis.set_title(f"{category.capitalize()} — Test Accuracy by Size")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3, axis="y")
        return

    x_label_map: dict[float, str] = {}
    for series_df in category_series.values():
        for xv, xl in zip(series_df["x_value"], series_df["x_label"]):
            x_label_map[xv] = xl
    x_values = sorted(x_label_map.keys())
    x_labels_list = [x_label_map[xv] for xv in x_values]

    variant_labels = list(category_series.keys())
    n_variants = len(variant_labels)
    bar_width = 0.8 / max(n_variants, 1)
    x_positions = np.arange(len(x_values))

    for i, variant_label in enumerate(variant_labels):
        series_df = category_series[variant_label]
        offset = (i - (n_variants - 1) / 2) * bar_width
        heights: list[float] = []
        for xv in x_values:
            row = series_df[series_df["x_value"] == xv]
            if row.empty or metric_name not in row.columns:
                heights.append(np.nan)
                continue
            val = row[metric_name].iloc[0]
            heights.append(float(val) if pd.notna(val) else np.nan)
        axis.bar(
            x_positions + offset,
            heights,
            width=bar_width,
            label=variant_label,
            color=_series_color(series_df),
            hatch="///" if _has_distractor(series_df) else None,
            edgecolor=PLAIN_EDGE_COLOR,
            linewidth=0.8,
            zorder=3,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(x_labels_list)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"{category.capitalize()} — Test Accuracy by Size")
    axis.set_axisbelow(True)
    axis.grid(alpha=0.3, axis="y", zorder=0)
    axis.legend(fontsize="small")


def model_accuracy_grouped_bar_plot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path:
    twirl_series, spherize_series, lesion_series = _prepare_performance_data(
        config, experiment_dir=experiment_dir, analyses_dir=analyses_dir,
    )

    all_categories = [
        ("twirl", twirl_series, "Twirl Angle"),
        ("spherize", spherize_series, "Spherize Amount"),
        ("lesion", lesion_series, "Lesion Dataset"),
    ]
    categories = [c for c in all_categories if c[1]] or all_categories
    n_cols = len(categories)

    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    for col, (category, series, x_label) in enumerate(categories):
        _plot_accuracy_grouped_bars(
            axes[col],
            category_series=series,
            category=category,
            x_label=x_label,
        )

    output_path = analyses_dir / "model_accuracy_grouped_bar_plot.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved model accuracy grouped bar plot to {output_path}")
    return output_path


def _plot_accuracy_heatmap(
    axis,
    *,
    category_series: dict[str, pd.DataFrame],
    category: str,
    x_label: str,
):
    if not category_series:
        axis.text(
            0.5, 0.5, f"No {category} runs",
            transform=axis.transAxes, ha="center", va="center",
        )
        axis.set_title(f"{category.capitalize()} — Test Accuracy")
        return None

    x_label_map: dict[float, str] = {}
    for series_df in category_series.values():
        for xv, xl in zip(series_df["x_value"], series_df["x_label"]):
            x_label_map[xv] = xl
    x_values = sorted(x_label_map.keys())
    x_labels_list = [x_label_map[xv] for xv in x_values]
    x_to_col = {xv: i for i, xv in enumerate(x_values)}

    variant_labels = list(category_series.keys())

    data = np.full((len(variant_labels), len(x_values)), np.nan)
    for row_idx, variant_label in enumerate(variant_labels):
        series_df = category_series[variant_label]
        for _, row in series_df.iterrows():
            col_idx = x_to_col.get(row["x_value"])
            if col_idx is not None and pd.notna(row["test_acc"]):
                data[row_idx, col_idx] = row["test_acc"]

    im = axis.imshow(data, aspect="auto", cmap=MAGNITUDE_COLORMAP, vmin=0.0, vmax=1.0)

    for row_idx in range(len(variant_labels)):
        for col_idx in range(len(x_values)):
            val = data[row_idx, col_idx]
            if not np.isnan(val):
                axis.text(
                    col_idx, row_idx, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color=magnitude_text_color(val),
                )

    axis.set_xticks(range(len(x_labels_list)))
    axis.set_xticklabels(x_labels_list)
    axis.set_yticks(range(len(variant_labels)))
    axis.set_yticklabels(variant_labels)
    for tick_label, variant_label in zip(axis.get_yticklabels(), variant_labels):
        tick_label.set_color(_series_color(category_series[variant_label]))
    axis.set_xlabel(x_label)
    axis.set_title(f"{category.capitalize()} — Test Accuracy")

    return im


def model_accuracy_heatmap(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path:
    twirl_series, spherize_series, lesion_series = _prepare_performance_data(
        config, experiment_dir=experiment_dir, analyses_dir=analyses_dir,
    )

    all_categories = [
        ("twirl", twirl_series, "Twirl Angle"),
        ("spherize", spherize_series, "Spherize Amount"),
        ("lesion", lesion_series, "Lesion Dataset"),
    ]
    categories = [c for c in all_categories if c[1]] or all_categories
    n_cols = len(categories)

    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    last_im = None
    for col, (category, series, x_label) in enumerate(categories):
        im = _plot_accuracy_heatmap(
            axes[col],
            category_series=series,
            category=category,
            x_label=x_label,
        )
        if im is not None:
            last_im = im

    if last_im is not None:
        figure.colorbar(last_im, ax=axes, shrink=0.6, label="Test Accuracy")

    output_path = analyses_dir / "model_accuracy_heatmap.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved model accuracy heatmap to {output_path}")
    return output_path


def model_loss_plot(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path:
    twirl_series, spherize_series, lesion_series = _prepare_performance_data(
        config, experiment_dir=experiment_dir, analyses_dir=analyses_dir,
    )

    all_categories = [
        ("twirl", twirl_series, "Twirl Angle"),
        ("spherize", spherize_series, "Spherize Amount"),
        ("lesion", lesion_series, "Lesion Dataset"),
    ]
    categories = [c for c in all_categories if c[1]] or all_categories
    n_cols = len(categories)

    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    for col, (category, series, x_label) in enumerate(categories):
        _plot_metric_line(
            axes[col],
            category_series=series,
            metric_name="test_loss",
            category=category,
            metric_label="Loss",
            overlays=[("train_loss", "Train", "dotted")],
        )
        axes[col].set_xlabel(x_label)

    output_path = analyses_dir / "model_loss_plot.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved model loss plot to {output_path}")
    return output_path


def _extract_hp_from_record(record) -> dict[str, object]:
    model_params = record.model_params if isinstance(record.model_params, dict) else {}
    training_params = model_params.get("training_params", {})
    if not isinstance(training_params, dict):
        training_params = {}
    return {
        "batch_size": int(record.batch_size),
        "lr": float(training_params.get("lr", 0)),
        "weight_decay": float(training_params.get("weight_decay", 0)),
        "base_channels": int(model_params.get("base_channels", 32)),
    }


def _format_hp_label(hp: dict) -> str:
    return f"bs={hp['batch_size']} lr={hp['lr']:g} wd={hp['weight_decay']:g} ch={hp['base_channels']}"


def _plot_hyperparameter_heatmap(
    axis,
    *,
    cat_df: pd.DataFrame,
    hp_order: list[str],
    category: str,
    cat_title: str,
):
    num_hp = len(hp_order)

    if cat_df.empty:
        axis.text(0.5, 0.5, f"No {category} runs", transform=axis.transAxes, ha="center", va="center")
        axis.set_title(f"{cat_title} — Test Accuracy by Hyperparameters")
        return None

    variant_keys = sorted(
        cat_df[["dataset_name", "parameter_value"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda key: (
            _lesion_size_order(str(key[0])),
            float("inf") if pd.isna(key[1]) else float(key[1]),
        ),
    )

    data = np.full((num_hp, len(variant_keys)), np.nan)
    for hp_idx, hp_label in enumerate(hp_order):
        for var_idx, (var_name, var_value) in enumerate(variant_keys):
            value_mask = (
                cat_df["parameter_value"].isna()
                if pd.isna(var_value)
                else cat_df["parameter_value"] == var_value
            )
            subset = cat_df[
                (cat_df["hp_label"] == hp_label)
                & (cat_df["dataset_name"] == var_name)
                & value_mask
            ]
            if not subset.empty and subset["test_acc"].notna().any():
                data[hp_idx, var_idx] = subset["test_acc"].mean()

    valid_rows = ~np.all(np.isnan(data), axis=1)
    if not valid_rows.any():
        axis.text(0.5, 0.5, f"No {category} data", transform=axis.transAxes, ha="center", va="center")
        axis.set_title(f"{cat_title} — Test Accuracy by Hyperparameters")
        return None

    filtered_data = data[valid_rows]
    filtered_hp_labels = [hp_order[i] for i in range(num_hp) if valid_rows[i]]

    im = axis.imshow(filtered_data, aspect="auto", cmap=MAGNITUDE_COLORMAP, vmin=0.0, vmax=1.0)

    for row_idx in range(filtered_data.shape[0]):
        for col_idx in range(filtered_data.shape[1]):
            val = filtered_data[row_idx, col_idx]
            if not np.isnan(val):
                axis.text(
                    col_idx, row_idx, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color=magnitude_text_color(val),
                )

    has_sub_parameter = any(not pd.isna(value) for _, value in variant_keys)
    column_labels = [_hyperparameter_variant_label(name, value) for name, value in variant_keys]
    axis.set_xticks(range(len(variant_keys)))
    axis.set_xticklabels(column_labels, rotation=0, ha="center", fontsize=8)
    for tick_label, (variant_name, _) in zip(axis.get_xticklabels(), variant_keys):
        tick_label.set_color(artifact_color(dataset_artifacts_from_name(str(variant_name))[0]))
    axis.set_yticks(range(len(filtered_hp_labels)))
    axis.set_yticklabels(filtered_hp_labels, fontsize=8)

    for boundary in range(1, len(variant_keys)):
        if variant_keys[boundary][0] != variant_keys[boundary - 1][0]:
            axis.axvline(boundary - 0.5, color="white", linewidth=2)

    parameter_axis_label = _HYPERPARAMETER_AXIS_LABELS.get(category, "Dataset")
    subtitle = (
        f"(ticks: lesion size / {parameter_axis_label})"
        if has_sub_parameter
        else "(ticks: lesion size)"
    )
    axis.set_title(f"{cat_title} — Test Accuracy by Hyperparameters\n{subtitle}")
    return im


def hyperparameter_summary(
    config: ExperimentConfig,
    *,
    experiment_dir: Path,
    analyses_dir: Path,
) -> Path:
    training_dir = Path(
        generate_training_dir(base_dir=str(experiment_dir), training_config=config.training)
    ).resolve()
    training_records_name = config.training.get("training_records", "training_records.jsonl")
    training_records_path = (training_dir / training_records_name).resolve()

    if not training_records_path.exists():
        raise FileNotFoundError(f"Training records not found: {training_records_path}")

    records = load_training_records(file_path=str(training_records_path))
    if not records:
        raise ValueError(f"No training records found: {training_records_path}")

    rows = []
    for record in records:
        dataset_meta_data = record.dataset_meta_data or {}
        variant_tag = str(dataset_meta_data.get("dataset_variant_tag", "full_dataset"))
        selected_filters = dataset_meta_data.get("selected_generated_filters")
        category, _, parameter_value = _resolve_category_and_parameter(
            selected_filters=selected_filters,
            variant_tag=variant_tag,
        )
        dataset_name = ""
        if isinstance(selected_filters, dict):
            dataset_name = selected_filters.get("dataset_name", "")

        metrics_path = _resolve_record_path(path_str=record.training_log_path, experiment_dir=experiment_dir)
        metrics = _extract_metrics_from_log(metrics_path=metrics_path)
        hp = _extract_hp_from_record(record)

        rows.append({
            "variant_tag": variant_tag,
            "dataset_name": dataset_name,
            "parameter_value": parameter_value,
            "category": category,
            "hp_label": _format_hp_label(hp),
            **hp,
            "test_acc": metrics["test_acc"],
            "test_loss": metrics["test_loss"],
            "val_acc": metrics["val_acc"],
            "val_loss": metrics["val_loss"],
        })

    df = pd.DataFrame(rows)

    detail_csv_path = analyses_dir / "hyperparameter_detail.csv"
    df.sort_values("test_acc", ascending=False).to_csv(detail_csv_path, index=False)
    logger.info(f"Saved per-run hyperparameter detail to {detail_csv_path}")

    agg_df = (
        df.groupby("hp_label", as_index=False)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            mean_val_acc=("val_acc", "mean"),
            mean_test_loss=("test_loss", "mean"),
            num_runs=("variant_tag", "count"),
        )
        .sort_values("mean_test_acc", ascending=False)
        .reset_index(drop=True)
    )
    summary_csv_path = analyses_dir / "hyperparameter_summary.csv"
    agg_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved aggregated hyperparameter summary to {summary_csv_path}")

    hp_order = agg_df["hp_label"].tolist()
    all_categories = [
        ("twirl", "Twirl"),
        ("spherize", "Spherize"),
        ("lesion", "Lesion"),
    ]
    present = set(df["category"].unique())
    categories = [c for c in all_categories if c[0] in present] or all_categories
    n_cols = len(categories)

    num_hp = len(hp_order)
    fig_height = max(6, num_hp * 0.45 + 2)
    figure, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols + 1, fig_height), squeeze=False, constrained_layout=True)
    axes = axes[0]

    last_im = None
    for col, (category, cat_title) in enumerate(categories):
        cat_df = df[df["category"] == category]
        im = _plot_hyperparameter_heatmap(
            axes[col],
            cat_df=cat_df,
            hp_order=hp_order,
            category=category,
            cat_title=cat_title,
        )
        if im is not None:
            last_im = im

    if last_im is not None:
        figure.colorbar(last_im, ax=axes, shrink=0.6, label="Test Accuracy")

    output_path = analyses_dir / "hyperparameter_summary.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved hyperparameter summary plot to {output_path}")
    return output_path
