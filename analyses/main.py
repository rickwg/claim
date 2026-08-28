import os
from pathlib import Path

from loguru import logger

from analyses._colors import apply_figure_style
from analyses.dataset import (
    COMBINED_PAIR_EXAMPLES,
    combined_pair_examples,
)
from analyses.model import (
    HYPERPARAMETER_SUMMARY,
    MODEL_ACCURACY_ALL_SPLITS_PLOT,
    MODEL_ACCURACY_GROUPED_BAR_PLOT,
    MODEL_ACCURACY_HEATMAP,
    MODEL_ACCURACY_PLOT,
    MODEL_LOSS_PLOT,
    hyperparameter_summary,
    model_accuracy_all_splits_plot,
    model_accuracy_grouped_bar_plot,
    model_accuracy_heatmap,
    model_accuracy_plot,
    model_loss_plot,
)
from analyses.prediction import (
    MODEL_TRANSFER_ACCURACY_PLOT,
    MODEL_TRANSFER_ACCURACY_SUMMARY,
    model_transfer_accuracy_plot,
    model_transfer_accuracy_summary,
)
from analyses.xai import (
    XAI_ATTRIBUTION_EXAMPLES,
    XAI_ATTRIBUTION_OVERLAY_EXAMPLES,
    xai_attribution_examples,
    xai_attribution_overlay_examples,
)
from analyses.xai_evaluation import (
    XAI_CONFIDENCE_VS_MASS_ACCURACY,
    XAI_DISCRIMINATIVE_VS_DISTRACTOR,
    XAI_MASS_ACCURACY_BOXPLOT,
    XAI_MASS_ACCURACY_BOXPLOT_BY_DATASET,
    XAI_MASS_ACCURACY_HEATMAP,
    XAI_MASS_ACCURACY_PLOT,
    XAI_RELATIVE_IMPORTANCE_BOXPLOT,
    XAI_RELATIVE_IMPORTANCE_BOXPLOT_LOG,
    XAI_PERFORMANCE_COMPARISON,
    XAI_SALIENCY_BOXPLOT_BY_DATASET,
    XAI_SALIENCY_LESION_VS_DISTORTION,
    XAI_EDGE_SALIENCE_BOXPLOT_BY_DATASET,
    XAI_EDGE_VS_ORIENTATION,
    XAI_EDGE_CORRESPONDENCE,
    XAI_SUMMARY,
    xai_confidence_vs_mass_accuracy,
    xai_discriminative_vs_distractor_plot,
    xai_edge_correspondence,
    xai_edge_salience_boxplot_by_dataset,
    xai_edge_vs_orientation,
    xai_mass_accuracy_boxplot,
    xai_mass_accuracy_boxplot_by_dataset,
    xai_mass_accuracy_heatmap,
    xai_mass_accuracy_plot,
    xai_relative_importance_boxplot,
    xai_relative_importance_boxplot_log,
    XAI_DISCRIMINATOR_RELATIVE_IMPORTANCE_BOXPLOT,
    XAI_DISCRIMINATOR_RELATIVE_IMPORTANCE_BOXPLOT_LOG,
    XAI_DISTRACTOR_RELATIVE_IMPORTANCE_BOXPLOT,
    XAI_DISTRACTOR_RELATIVE_IMPORTANCE_BOXPLOT_LOG,
    XAI_RELATIVE_IMPORTANCE_DISC_VS_DISTRACTOR_BOXPLOT,
    XAI_RELATIVE_IMPORTANCE_DISC_VS_DISTRACTOR_BOXPLOT_LOG,
    XAI_RELATIVE_IMPORTANCE_DUMBBELL,
    XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED,
    XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_SPHERIZE,
    XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_SPHERIZE_MEAN,
    XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_MEAN,
    XAI_RELATIVE_IMPORTANCE_DUMBBELL_MEAN,
    XAI_SEPARATION_INDEX,
    XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_TWIRL,
    XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_TWIRL_MEAN,
    xai_discriminator_relative_importance_boxplot,
    xai_discriminator_relative_importance_boxplot_log,
    xai_distractor_relative_importance_boxplot,
    xai_distractor_relative_importance_boxplot_log,
    xai_relative_importance_disc_vs_distractor_boxplot,
    xai_relative_importance_disc_vs_distractor_boxplot_log,
    xai_relative_importance_dumbbell,
    xai_relative_importance_dumbbell_grouped,
    xai_relative_importance_dumbbell_grouped_spherize,
    xai_relative_importance_dumbbell_grouped_spherize_mean,
    xai_relative_importance_dumbbell_grouped_mean,
    xai_relative_importance_dumbbell_mean,
    xai_separation_index,
    xai_relative_importance_dumbbell_grouped_twirl,
    xai_relative_importance_dumbbell_grouped_twirl_mean,
    xai_performance_comparison,
    xai_saliency_boxplot_by_dataset,
    xai_saliency_lesion_vs_distortion,
    xai_summary,
)
from common import EnvironmentVariables
from config.configuration import ExperimentConfig
from utils import generate_experiment_dir, load_config_file

DEFAULT_CONFIG_FILE_PATH = "config/experiments_config.yaml"


def _load_experiment_config() -> ExperimentConfig:
    config_path = os.environ.get(
        EnvironmentVariables.CONFIG_FILE_PATH.value,
        DEFAULT_CONFIG_FILE_PATH,
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    return ExperimentConfig.from_dict(load_config_file(file_path=config_path))


def _normalize_analysis_name(analysis_name: str) -> str:
    return analysis_name.strip().lstrip("_")


def _resolve_analyses(analyses_config: dict) -> list[str]:
    requested = analyses_config.get("analyses")
    if requested is None:
        return []
    if not isinstance(requested, list):
        raise ValueError("analyses.analyses must be a list of analysis names.")

    normalized = []
    for item in requested:
        if not isinstance(item, str):
            raise ValueError(f"Invalid analysis name: {item}")
        normalized.append(_normalize_analysis_name(item))
    return normalized


def main() -> None:
    apply_figure_style()
    config = _load_experiment_config()
    experiment_dir = Path(generate_experiment_dir(config=config)).resolve()
    analyses_subdir = config.analyses.get("output_dir", "analyses")
    analyses_dir = (experiment_dir / analyses_subdir).resolve()
    analyses_dir.mkdir(parents=True, exist_ok=True)

    available_analyses = {
        MODEL_ACCURACY_PLOT: model_accuracy_plot,
        MODEL_ACCURACY_ALL_SPLITS_PLOT: model_accuracy_all_splits_plot,
        MODEL_ACCURACY_GROUPED_BAR_PLOT: model_accuracy_grouped_bar_plot,
        MODEL_ACCURACY_HEATMAP: model_accuracy_heatmap,
        MODEL_LOSS_PLOT: model_loss_plot,
        HYPERPARAMETER_SUMMARY: hyperparameter_summary,
        MODEL_TRANSFER_ACCURACY_SUMMARY: model_transfer_accuracy_summary,
        MODEL_TRANSFER_ACCURACY_PLOT: model_transfer_accuracy_plot,
        XAI_MASS_ACCURACY_PLOT: xai_mass_accuracy_plot,
        XAI_MASS_ACCURACY_HEATMAP: xai_mass_accuracy_heatmap,
        XAI_MASS_ACCURACY_BOXPLOT: xai_mass_accuracy_boxplot,
        XAI_MASS_ACCURACY_BOXPLOT_BY_DATASET: xai_mass_accuracy_boxplot_by_dataset,
        XAI_RELATIVE_IMPORTANCE_BOXPLOT: xai_relative_importance_boxplot,
        XAI_RELATIVE_IMPORTANCE_BOXPLOT_LOG: xai_relative_importance_boxplot_log,
        XAI_DISCRIMINATOR_RELATIVE_IMPORTANCE_BOXPLOT: xai_discriminator_relative_importance_boxplot,
        XAI_DISCRIMINATOR_RELATIVE_IMPORTANCE_BOXPLOT_LOG: xai_discriminator_relative_importance_boxplot_log,
        XAI_DISTRACTOR_RELATIVE_IMPORTANCE_BOXPLOT: xai_distractor_relative_importance_boxplot,
        XAI_DISTRACTOR_RELATIVE_IMPORTANCE_BOXPLOT_LOG: xai_distractor_relative_importance_boxplot_log,
        XAI_RELATIVE_IMPORTANCE_DISC_VS_DISTRACTOR_BOXPLOT: xai_relative_importance_disc_vs_distractor_boxplot,
        XAI_RELATIVE_IMPORTANCE_DISC_VS_DISTRACTOR_BOXPLOT_LOG: xai_relative_importance_disc_vs_distractor_boxplot_log,
        XAI_RELATIVE_IMPORTANCE_DUMBBELL: xai_relative_importance_dumbbell,
        XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED: xai_relative_importance_dumbbell_grouped,
        XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_TWIRL: xai_relative_importance_dumbbell_grouped_twirl,
        XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_SPHERIZE: xai_relative_importance_dumbbell_grouped_spherize,
        XAI_RELATIVE_IMPORTANCE_DUMBBELL_MEAN: xai_relative_importance_dumbbell_mean,
        XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_MEAN: xai_relative_importance_dumbbell_grouped_mean,
        XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_TWIRL_MEAN: xai_relative_importance_dumbbell_grouped_twirl_mean,
        XAI_RELATIVE_IMPORTANCE_DUMBBELL_GROUPED_SPHERIZE_MEAN: xai_relative_importance_dumbbell_grouped_spherize_mean,
        XAI_SEPARATION_INDEX: xai_separation_index,
        XAI_PERFORMANCE_COMPARISON: xai_performance_comparison,
        XAI_CONFIDENCE_VS_MASS_ACCURACY: xai_confidence_vs_mass_accuracy,
        XAI_DISCRIMINATIVE_VS_DISTRACTOR: xai_discriminative_vs_distractor_plot,
        XAI_SALIENCY_BOXPLOT_BY_DATASET: xai_saliency_boxplot_by_dataset,
        XAI_SALIENCY_LESION_VS_DISTORTION: xai_saliency_lesion_vs_distortion,
        XAI_EDGE_SALIENCE_BOXPLOT_BY_DATASET: xai_edge_salience_boxplot_by_dataset,
        XAI_EDGE_VS_ORIENTATION: xai_edge_vs_orientation,
        XAI_EDGE_CORRESPONDENCE: xai_edge_correspondence,
        XAI_SUMMARY: xai_summary,
        XAI_ATTRIBUTION_EXAMPLES: xai_attribution_examples,
        XAI_ATTRIBUTION_OVERLAY_EXAMPLES: xai_attribution_overlay_examples,
        COMBINED_PAIR_EXAMPLES: combined_pair_examples,
    }
    requested_analyses = _resolve_analyses(config.analyses)
    if not requested_analyses:
        logger.info("No analyses configured in analyses.analyses.")
        return

    for analysis_name in requested_analyses:
        analysis_func = available_analyses.get(analysis_name)
        if analysis_func is None:
            logger.warning(
                f"Skipping unsupported analysis '{analysis_name}'. "
                f"Supported analyses: {sorted(available_analyses.keys())}"
            )
            continue
        logger.info(f"Running analysis: {analysis_name}")
        analysis_func(
            config=config,
            experiment_dir=experiment_dir,
            analyses_dir=analyses_dir,
        )


if __name__ == "__main__":
    main()
