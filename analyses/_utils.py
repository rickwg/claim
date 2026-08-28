import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from analyses._colors import (
    ATTRIBUTION_METHOD_COLORMAP,
    BASELINE_METHOD_COLOR,
    EDGE_FILTER_METHOD_COLORMAP,
)
from config.configuration import ExperimentConfig
from utils import load_jsonl_file

_METHOD_DISPLAY_NAMES = {
    "saliency": "Saliency",
    "input_x_gradient": "Input×Grad",
    "integrated_gradient": "Int. Grad",
    "kernel_shap": "Kernel SHAP",
    "lime": "LIME",
    "lrp": "LRP",
    "guided_backprop": "Guided BP",
    "deep_lift": "DeepLIFT",
    "gradient_shap": "Grad SHAP",
    "sobel": "Sobel",
    "laplace": "Laplace",
    "random": "Random",
}

_ATTRIBUTION_METHODS = {
    "saliency",
    "input_x_gradient",
    "integrated_gradient",
    "kernel_shap",
    "lime",
    "lrp",
    "guided_backprop",
    "deep_lift",
    "gradient_shap",
}
_EDGE_FILTER_METHODS = {"sobel", "laplace"}
_BASELINE_METHODS = {"random"}


def _method_group_rank(method: str) -> int:
    if method in _ATTRIBUTION_METHODS:
        return 0
    if method in _EDGE_FILTER_METHODS:
        return 1
    if method in _BASELINE_METHODS:
        return 2
    return 3


def _method_sort_key(method: str) -> tuple[int, str]:
    return (_method_group_rank(method), method)


def _sort_methods(methods) -> list[str]:
    return sorted(set(methods), key=_method_sort_key)


def _method_color(method: str, method_order: list[str]):
    group = _method_group_rank(method)
    same_group = [m for m in method_order if _method_group_rank(m) == group]
    if method in same_group:
        idx = same_group.index(method)
    else:
        idx = 0
    n = max(len(same_group), 1)
    t = 0.0 if n == 1 else idx / (n - 1)
    if group == 0:
        return ATTRIBUTION_METHOD_COLORMAP(0.35 + 0.60 * t)
    if group == 1:
        return EDGE_FILTER_METHOD_COLORMAP(0.55 + 0.35 * t)
    if group == 2:
        return BASELINE_METHOD_COLOR
    return plt.cm.Greys(0.4)


def _method_colors(methods: list[str]) -> list:
    return [_method_color(m, methods) for m in methods]


def _method_label(method: str) -> str:
    return _METHOD_DISPLAY_NAMES.get(method, method)


def _method_labels(methods: list[str]) -> list[str]:
    return [_method_label(m) for m in methods]


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
            candidate = (experiment_dir / Path(*parts[i + 1 :])).resolve()
            if candidate.exists():
                return candidate
    return resolved


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_category_and_parameter(
    *,
    selected_filters: dict | None,
    variant_tag: str,
) -> tuple[str, str | None, float | None]:
    if isinstance(selected_filters, dict):
        if "twirl_angle" in selected_filters:
            return "twirl", "twirl_angle", _safe_float(selected_filters.get("twirl_angle"))
        if "spherize_amount" in selected_filters:
            return "spherize", "spherize_amount", _safe_float(selected_filters.get("spherize_amount"))
        transformation = selected_filters.get("transformation")
        if isinstance(transformation, str) and transformation in {"twirl", "spherize"}:
            return transformation, "transformation", None
        dataset_name = selected_filters.get("dataset_name")
        if isinstance(dataset_name, str):
            if "lesion" in dataset_name:
                return "lesion", "dataset_name", None
            if "spherize" in dataset_name:
                return "spherize", "dataset_name", None
            if "twirl" in dataset_name:
                return "twirl", "dataset_name", None

    if variant_tag.startswith("twirl_angle-"):
        raw_value = variant_tag.removeprefix("twirl_angle-").replace("m", "-")
        return "twirl", "twirl_angle", _safe_float(raw_value)
    if variant_tag.startswith("spherize_amount-"):
        raw_value = variant_tag.removeprefix("spherize_amount-").replace("m", "-")
        return "spherize", "spherize_amount", _safe_float(raw_value)
    if variant_tag.startswith("transformation-"):
        transformation = variant_tag.removeprefix("transformation-")
        if transformation in {"twirl", "spherize"}:
            return transformation, "transformation", None
    if "lesion" in variant_tag:
        return "lesion", "dataset_name", None

    return "other", None, None


def _format_x_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}"


_DATASET_DISPLAY_NAMES = {
    "synthetic_twirl": "Twirl",
    "synthetic_spherize": "Spherize",
    "synthetic_lesions": "Diffusion Lesions",
}

_VARIANT_TAG_DISPLAY_NAMES = {
    "twirl_angle": "angle",
    "spherize_amount": "amount",
    "dataset_name": "",
    "transformation": "",
}

_COMBINED_CONDITION_DISPLAY_NAMES = {
    "a1p100": "Presence 100%",
    "a1p090": "Presence 90%",
    "a1p075": "Presence 75%",
    "a1p060": "Presence 60%",
    "a2c8020": "Confound 0.8/0.2",
    "a2c9010": "Confound 0.9/0.1",
    "bentangled": "Entangled",
}


def _combined_condition_code(dataset_name: str) -> str | None:
    return next(
        (code for code in _COMBINED_CONDITION_DISPLAY_NAMES if code in dataset_name),
        None,
    )


def _combined_condition_label(dataset_name: str) -> str | None:
    code = _combined_condition_code(dataset_name)
    return _COMBINED_CONDITION_DISPLAY_NAMES[code] if code else None


def _dataset_ordered_labels(frame: pd.DataFrame, label_column: str) -> list[str]:
    labels = list(frame[label_column].unique())
    if "dataset_name" not in frame.columns:
        return sorted(labels)
    order_key = (
        frame.groupby(label_column)["dataset_name"]
        .apply(lambda names: str(names.dropna().min()) if names.notna().any() else "")
        .to_dict()
    )
    return sorted(labels, key=lambda label: (order_key.get(label, ""), label))


def _dataset_display_name(dataset_name: str) -> str:
    if not dataset_name:
        return "unknown"
    segments = [p for p in dataset_name.split("/") if p]
    candidate = next(
        (s for s in segments if s.startswith("synthetic_")), segments[0] if segments else dataset_name,
    )
    for prefix, display in _DATASET_DISPLAY_NAMES.items():
        if candidate.startswith(prefix):
            return display
    return candidate.removeprefix("synthetic_")


def _format_variant_tag(variant_tag: str) -> str:
    if not variant_tag or variant_tag == "full_dataset":
        return ""
    if "-" not in variant_tag:
        return variant_tag
    key, _, raw_value = variant_tag.partition("-")
    display_key = _VARIANT_TAG_DISPLAY_NAMES.get(key, key)
    value = raw_value.replace("m", "-") if raw_value and raw_value[0] == "m" else raw_value
    if not display_key:
        return value
    return f"{display_key}={value}"


_LESION_SIZE_PATTERN = re.compile(r"(\d+-\d+)(?:-[a-zA-Z]+)*$")


def _extract_lesion_size(dataset_name: str) -> str | None:
    match = _LESION_SIZE_PATTERN.search(dataset_name)
    return match.group(1) if match else None


def _lesion_size_order(dataset_name: str) -> tuple[int, int, int, str]:
    size = _extract_lesion_size(dataset_name)
    if not size:
        return (1, 0, 0, dataset_name)
    low, high = (int(part) for part in size.split("-"))
    return (0, low, high, dataset_name)


def _short_variant_label(dataset_name: str) -> str:
    display = _dataset_display_name(dataset_name)
    size = _extract_lesion_size(dataset_name)
    return f"{display} {size}" if size else display


def _hyperparameter_variant_label(dataset_name: str, parameter_value: float | None) -> str:
    size = _extract_lesion_size(dataset_name) or _dataset_display_name(dataset_name)
    if parameter_value is None or pd.isna(parameter_value):
        return size
    return f"{size}\n{_format_x_label(float(parameter_value))}"


def _full_dataset_label(dataset_name: str) -> str:
    return dataset_name or "unknown"


def _build_variant_label(dataset_name: str, variant_tag: str) -> str:
    condition_label = _combined_condition_label(dataset_name)
    if condition_label:
        return condition_label
    name = _dataset_display_name(dataset_name) if dataset_name else ""
    tag = _format_variant_tag(variant_tag)
    if name and tag:
        return f"{name} ({tag})"
    if name:
        return name
    return tag or "unknown"


def _load_xai_evaluation_records(
    config: ExperimentConfig,
    experiment_dir: Path,
) -> pd.DataFrame:
    eval_subdir = config.xai_evaluation.get("output_dir", "xai_evaluation")
    eval_dir = (experiment_dir / eval_subdir).resolve()
    eval_records_name = config.xai_evaluation.get(
        "evaluation_records", "xai_evaluation_records.jsonl"
    )
    eval_records_path = (eval_dir / eval_records_name).resolve()
    if not eval_records_path.exists():
        logger.warning(f"XAI evaluation records not found: {eval_records_path}")
        return pd.DataFrame()

    records = load_jsonl_file(file_path=str(eval_records_path))
    if not records:
        logger.warning(f"No records in XAI evaluation file: {eval_records_path}")
        return pd.DataFrame()

    logger.info(f"Loaded {len(records)} XAI evaluation records from {eval_records_path}")
    return pd.DataFrame(records)
