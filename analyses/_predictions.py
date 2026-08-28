import json
from pathlib import Path, PurePosixPath

import pandas as pd
from loguru import logger

from config.configuration import ExperimentConfig

NATIVE_MODEL_SOURCE = "native"
GENERATED_SAMPLE_SOURCE = "generated"
PREPROCESSED_SAMPLE_SOURCE = "preprocessed"
UNHEALTHY_DECISION_THRESHOLD = 0.5

POOLED_COUNT_COLUMNS = [
    "num_positive",
    "num_true_positive",
    "num_negative_generated",
    "num_false_positive_generated",
    "num_negative_preprocessed",
    "num_false_positive_preprocessed",
]

_prediction_counts_cache: dict[tuple[str, str], pd.DataFrame] = {}


def _sample_sources_by_relative_path(samples_path: Path) -> dict[str, str]:
    if not samples_path.exists():
        raise FileNotFoundError(f"Training split metadata not found: {samples_path}")

    sources = {}
    with samples_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            image_path = record.get("image_path")
            source = record.get("source")
            if image_path and source:
                sources[str(image_path)] = str(source)
    if not sources:
        raise ValueError(f"No sample sources found in {samples_path}")

    logger.info(f"Loaded {len(sources)} sample sources from {samples_path}")
    return sources


def _sample_source(image_path: str, sources: dict[str, str]) -> str | None:
    path_parts = PurePosixPath(str(image_path)).parts
    for first_part_index in range(len(path_parts)):
        source = sources.get("/".join(path_parts[first_part_index:]))
        if source is not None:
            return source
    return None


def _negative_type(source: str | None) -> str:
    if source == GENERATED_SAMPLE_SOURCE:
        return "generated"
    if source == PREPROCESSED_SAMPLE_SOURCE:
        return "preprocessed"
    return "unknown"


def _blank_tuple_counts() -> dict:
    return {column: 0 for column in POOLED_COUNT_COLUMNS}


def _per_tuple_prediction_counts(xai_records_path: Path, sources: dict[str, str]) -> pd.DataFrame:
    if not xai_records_path.exists():
        raise FileNotFoundError(f"XAI records not found: {xai_records_path}")

    counts_by_tuple = {}
    identity_by_tuple = {}
    scored_sample_keys = set()
    unresolved_source_count = 0

    with xai_records_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("split", "")) != "test":
                continue

            tuple_id = str(record.get("tuple_id", ""))
            sample_key = (tuple_id, str(record.get("image_path", "")))
            if sample_key in scored_sample_keys:
                continue
            scored_sample_keys.add(sample_key)

            if tuple_id not in counts_by_tuple:
                counts_by_tuple[tuple_id] = _blank_tuple_counts()
                selected_filters = record.get("selected_generated_filters") or {}
                identity_by_tuple[tuple_id] = {
                    "dataset_variant_tag": str(record.get("dataset_variant_tag", "")),
                    "dataset_name": str(selected_filters.get("dataset_name", "")),
                    "model_source": str(record.get("model_source", NATIVE_MODEL_SOURCE)),
                    "repetition": record.get("repetition"),
                    "seed": record.get("seed"),
                }

            counts = counts_by_tuple[tuple_id]
            label = int(record.get("label", 0))
            predicted_unhealthy = (
                float(record.get("prediction_probability", 0.0)) >= UNHEALTHY_DECISION_THRESHOLD
            )

            if label == 1:
                counts["num_positive"] += 1
                counts["num_true_positive"] += int(predicted_unhealthy)
                continue

            negative_type = _negative_type(_sample_source(record.get("image_path", ""), sources))
            if negative_type == "unknown":
                unresolved_source_count += 1
                continue
            counts[f"num_negative_{negative_type}"] += 1
            counts[f"num_false_positive_{negative_type}"] += int(predicted_unhealthy)

    if unresolved_source_count:
        logger.warning(
            f"Could not resolve the sample source for {unresolved_source_count} negative XAI records; "
            f"they are excluded from the prediction summary."
        )
    if not counts_by_tuple:
        raise ValueError(f"No test-split XAI records found in {xai_records_path}")

    logger.info(
        f"Aggregated {len(scored_sample_keys)} unique test predictions "
        f"across {len(counts_by_tuple)} model/dataset tuples."
    )
    return pd.DataFrame([
        {"tuple_id": tuple_id, **identity_by_tuple[tuple_id], **counts}
        for tuple_id, counts in counts_by_tuple.items()
    ])


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.where(denominator > 0).div(denominator.where(denominator > 0))


def with_derived_rates(frame: pd.DataFrame) -> pd.DataFrame:
    derived = frame.copy()
    derived["true_positive_rate"] = _ratio(derived["num_true_positive"], derived["num_positive"])
    derived["false_positive_rate_generated"] = _ratio(
        derived["num_false_positive_generated"], derived["num_negative_generated"]
    )
    derived["false_positive_rate_preprocessed"] = _ratio(
        derived["num_false_positive_preprocessed"], derived["num_negative_preprocessed"]
    )
    derived["true_negative_rate_generated"] = 1.0 - derived["false_positive_rate_generated"]
    derived["true_negative_rate_preprocessed"] = 1.0 - derived["false_positive_rate_preprocessed"]

    num_correct_generated = (
        derived["num_true_positive"]
        + derived["num_negative_generated"]
        - derived["num_false_positive_generated"]
    )
    num_scored_generated = derived["num_positive"] + derived["num_negative_generated"]
    derived["accuracy_generated_negatives"] = _ratio(num_correct_generated, num_scored_generated)

    num_correct_all = num_correct_generated + (
        derived["num_negative_preprocessed"] - derived["num_false_positive_preprocessed"]
    )
    num_scored_all = num_scored_generated + derived["num_negative_preprocessed"]
    derived["accuracy_all_negatives"] = _ratio(num_correct_all, num_scored_all)

    carries_generated_negatives = derived["num_negative_generated"] > 0
    num_negative = derived["num_negative_generated"].where(
        carries_generated_negatives, derived["num_negative_preprocessed"]
    )
    num_false_positive = derived["num_false_positive_generated"].where(
        carries_generated_negatives, derived["num_false_positive_preprocessed"]
    )
    derived["accuracy_test_population"] = _ratio(
        derived["num_true_positive"] + num_negative - num_false_positive,
        derived["num_positive"] + num_negative,
    )
    return derived


def _resolved_input_paths(config: ExperimentConfig, experiment_dir: Path) -> tuple[Path, Path]:
    xai_dir = (experiment_dir / config.xai.get("output_dir", "xai")).resolve()
    xai_records_path = (xai_dir / config.xai.get("xai_records", "xai_records.jsonl")).resolve()
    samples_path = (
        experiment_dir / config.data.get("samples_filename", "training_split_metadata.jsonl")
    ).resolve()
    return xai_records_path, samples_path


def per_tuple_prediction_counts(config: ExperimentConfig, experiment_dir: Path) -> pd.DataFrame:
    xai_records_path, samples_path = _resolved_input_paths(config, experiment_dir)
    cache_key = (str(xai_records_path), str(samples_path))
    if cache_key not in _prediction_counts_cache:
        _prediction_counts_cache[cache_key] = _per_tuple_prediction_counts(
            xai_records_path=xai_records_path,
            sources=_sample_sources_by_relative_path(samples_path=samples_path),
        )
    return _prediction_counts_cache[cache_key]


def pooled_prediction_summary(config: ExperimentConfig, experiment_dir: Path) -> pd.DataFrame:
    per_tuple_rates = with_derived_rates(
        per_tuple_prediction_counts(config=config, experiment_dir=experiment_dir)
    )
    group_columns = ["dataset_variant_tag", "dataset_name", "model_source"]

    pooled = with_derived_rates(
        per_tuple_rates.groupby(group_columns, as_index=False)[POOLED_COUNT_COLUMNS].sum()
    )
    seed_statistics = per_tuple_rates.groupby(group_columns, as_index=False).agg(
        accuracy_test_population_seed_mean=("accuracy_test_population", "mean"),
        accuracy_test_population_seed_sd=("accuracy_test_population", lambda values: values.std(ddof=1)),
        num_seeds=("tuple_id", "nunique"),
    )
    return pooled.merge(seed_statistics, on=group_columns, how="left")


def accuracy_by_dataset_and_model(config: ExperimentConfig, experiment_dir: Path) -> dict[tuple[str, str], float]:
    try:
        summary = pooled_prediction_summary(config=config, experiment_dir=experiment_dir)
    except (FileNotFoundError, ValueError) as error:
        logger.info(f"Analysis-dataset accuracy unavailable, legends will omit it: {error}")
        return {}
    return {
        (str(row["dataset_name"]), str(row["model_source"])): float(row["accuracy_test_population"])
        for _, row in summary.iterrows()
        if pd.notna(row["accuracy_test_population"])
    }
