from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from analyses._colors import (
    apply_dataset_style,
    dataset_style,
)
from analyses._predictions import (
    NATIVE_MODEL_SOURCE,
    POOLED_COUNT_COLUMNS,
    pooled_prediction_summary,
)
from analyses.xai_evaluation import (
    PAPER_AXIS_LABEL_FONTSIZE,
    PAPER_MINOR_TICK_LABEL_FONTSIZE,
    PAPER_TICK_LABEL_FONTSIZE,
    _dataset_symbol,
    _dataset_symbol_letters_from_name,
    _model_label,
    _model_training_dataset_names,
)
from config.configuration import ExperimentConfig

MODEL_TRANSFER_ACCURACY_SUMMARY = "model_transfer_accuracy_summary"
MODEL_TRANSFER_ACCURACY_PLOT = "model_transfer_accuracy_plot"


def _transfer_accuracy_frame(config: ExperimentConfig, experiment_dir: Path) -> pd.DataFrame:
    summary = pooled_prediction_summary(config=config, experiment_dir=experiment_dir)
    training_dataset_names = _model_training_dataset_names(config)
    summary["analysis_dataset"] = summary["dataset_name"].map(
        lambda name: _dataset_symbol(_dataset_symbol_letters_from_name(name))
    )
    summary["model_dataset"] = summary.apply(
        lambda row: _model_label(row, training_dataset_names), axis=1
    )
    return summary.sort_values(["dataset_name", "model_source"]).reset_index(drop=True)


_SUMMARY_COLUMNS = [
    "analysis_dataset",
    "model_dataset",
    "model_source",
    "dataset_name",
    "dataset_variant_tag",
    "true_positive_rate",
    "true_negative_rate_generated",
    "true_negative_rate_preprocessed",
    "false_positive_rate_generated",
    "false_positive_rate_preprocessed",
    "accuracy_test_population",
    "accuracy_generated_negatives",
    "accuracy_all_negatives",
    "accuracy_test_population_seed_mean",
    "accuracy_test_population_seed_sd",
    "num_seeds",
    *POOLED_COUNT_COLUMNS,
]


def model_transfer_accuracy_summary(
    config: ExperimentConfig,
    experiment_dir: Path,
    analyses_dir: Path,
) -> None:
    summary_df = _transfer_accuracy_frame(config=config, experiment_dir=experiment_dir)
    output_path = analyses_dir / f"{MODEL_TRANSFER_ACCURACY_SUMMARY}.csv"
    summary_df[_SUMMARY_COLUMNS].to_csv(output_path, index=False)
    logger.info(f"Saved transfer accuracy summary with {len(summary_df)} rows to {output_path}")


def _transferred_datasets(summary_df: pd.DataFrame) -> list[str]:
    model_source_counts = summary_df.groupby("dataset_name")["model_source"].nunique()
    return sorted(model_source_counts[model_source_counts > 1].index)


def _style_row(row) -> dict:
    return {
        "dataset_name": row["dataset_name"],
        "model_source": row["model_source"],
    }


_GROUP_WIDTH = 0.7


def _native_first(dataset_rows: pd.DataFrame) -> pd.DataFrame:
    return dataset_rows.assign(
        is_native=dataset_rows["model_source"].eq(NATIVE_MODEL_SOURCE)
    ).sort_values("is_native", ascending=False)


def _draw_grouped_metric(axis, summary_df: pd.DataFrame, dataset_names: list[str], metric: str) -> None:
    bar_positions = []
    model_labels = []
    for dataset_index, dataset_name in enumerate(dataset_names):
        dataset_rows = _native_first(summary_df[summary_df["dataset_name"] == dataset_name])
        offsets = np.linspace(-_GROUP_WIDTH / 2, _GROUP_WIDTH / 2, 2 * len(dataset_rows) + 1)[1::2]
        for offset, (_, row) in zip(offsets, dataset_rows.iterrows()):
            position = dataset_index + offset
            bar_positions.append(position)
            model_labels.append(row["model_dataset"])
            value = row[metric]
            if pd.isna(value):
                continue
            bars = axis.bar(position, value, width=_GROUP_WIDTH / len(dataset_rows) * 0.82)
            apply_dataset_style(bars[0], dataset_style(_style_row(row)))

    axis.set_xticks(bar_positions, minor=True)
    axis.set_xticklabels(model_labels, minor=True, fontsize=PAPER_MINOR_TICK_LABEL_FONTSIZE)
    axis.tick_params(axis="x", which="minor", length=0)
    axis.set_xticks(range(len(dataset_names)))
    axis.set_xticklabels(
        [_dataset_symbol(_dataset_symbol_letters_from_name(name)) for name in dataset_names],
        fontsize=PAPER_TICK_LABEL_FONTSIZE,
    )
    axis.tick_params(axis="x", which="major", length=0, pad=30)
    axis.set_xlim(-0.5, len(dataset_names) - 0.5)
    axis.tick_params(axis="y", labelsize=PAPER_TICK_LABEL_FONTSIZE)
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)


def model_transfer_accuracy_plot(
    config: ExperimentConfig,
    experiment_dir: Path,
    analyses_dir: Path,
) -> None:
    summary_df = _transfer_accuracy_frame(config=config, experiment_dir=experiment_dir)
    dataset_names = _transferred_datasets(summary_df)
    if not dataset_names:
        logger.warning("No dataset carries both a native and a transferred model; skipping the transfer plot.")
        return

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    _draw_grouped_metric(axes[0], summary_df, dataset_names, "accuracy_test_population")
    axes[0].set_ylabel("Test accuracy", fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    axes[0].set_ylim(0.0, 1.05)

    _draw_grouped_metric(axes[1], summary_df, dataset_names, "false_positive_rate_generated")
    axes[1].set_ylabel("FPR on artifact-bearing healthy", fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    axes[1].set_ylim(0.0, 1.05)

    figure.tight_layout()
    output_path = analyses_dir / f"{MODEL_TRANSFER_ACCURACY_PLOT}.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    logger.info(f"Saved transfer accuracy plot to {output_path}")
