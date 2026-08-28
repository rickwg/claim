import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import (
    DeepLift,
    GradientShap,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
    KernelShap,
    Lime,
    LRP,
    Saliency,
)
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule, EpsilonRule
from loguru import logger
from PIL import Image
from skimage.filters import laplace as skimage_laplace, sobel as skimage_sobel
from skimage.segmentation import slic
from tqdm.auto import tqdm
from torchvision import transforms

from common import EnvironmentVariables, TrainingRecord
from config.configuration import ExperimentConfig
from training.main import (
    LitBinaryClassifier,
    _filter_samples_for_generated_filters,
    _resolve_model_blocks,
    _resolve_sample_image_path,
    patchify,
    unpatchify,
)
from utils import (
    append_to_jsonl_file,
    dump_as_jsonl_file,
    generate_data_dir,
    generate_experiment_dir,
    generate_training_dir,
    load_config_file,
    load_jsonl_file,
    load_training_records,
    set_random_states,
)

DEFAULT_CONFIG_FILE_PATH = "config/experiments_config.yaml"
SUPPORTED_METHODS = {
    "saliency",
    "input_x_gradient",
    "integrated_gradient",
    "kernel_shap",
    "lime",
    "lrp",
    "guided_backprop",
    "deep_lift",
    "gradient_shap",
    "random",
    "sobel",
    "laplace",
}
GRADIENT_BASED_METHODS = {
    "saliency",
    "input_x_gradient",
    "integrated_gradient",
    "lrp",
    "guided_backprop",
    "deep_lift",
    "gradient_shap",
}
PERTURBATION_METHODS = {"kernel_shap", "lime"}
LEGACY_LAYER_PREFIXES = {
    "model.layer1.": "model.layers.0.",
    "model.layer2.": "model.layers.1.",
    "model.layer3.": "model.layers.2.",
    "model.layer4.": "model.layers.3.",
}
TO_TENSOR = transforms.ToTensor()


class _AlphaBetaRule(Alpha1_Beta0_Rule):
    def __init__(self, alpha: float = 2.0, beta: float = -1.0, set_bias_to_zero: bool = False):
        super().__init__(set_bias_to_zero=set_bias_to_zero)
        self.alpha = alpha
        self.beta = beta

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            positive_weights = module.weight.data.clamp(min=0)
            negative_weights = module.weight.data.clamp(max=0)
            module.weight.data = self.alpha * positive_weights + self.beta * negative_weights
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


def _prepare_model_for_lrp(model: torch.nn.Module, alpha: float, beta: float) -> None:
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.rule = _AlphaBetaRule(alpha=alpha, beta=beta)
        elif isinstance(module, torch.nn.Identity):
            module.rule = EpsilonRule()


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
            candidate = (experiment_dir / Path(*parts[i + 1 :])).resolve()
            if candidate.exists():
                return candidate
    return resolved


def _resolve_data_dir(config: ExperimentConfig, project_root: Path) -> Path:
    data_dir = Path(generate_data_dir(config=config)).expanduser()
    if not data_dir.is_absolute():
        data_dir = (project_root / data_dir).resolve()
    else:
        data_dir = data_dir.resolve()
    return data_dir


def _resolve_image_data_dir(config: ExperimentConfig, project_root: Path, fallback_data_dir: Path) -> Path:
    configured_data_dir = config.data.get("data_dir")
    if not isinstance(configured_data_dir, str) or not configured_data_dir.strip():
        return fallback_data_dir

    data_dir = Path(configured_data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (project_root / data_dir).resolve()
    else:
        data_dir = data_dir.resolve()
    return data_dir


def _resolve_device(device_preference: str) -> torch.device:
    normalized = (device_preference or "auto").lower()
    if normalized == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "mps":
        use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if use_mps else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "gpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if use_mps else "cpu")
    raise ValueError(f"Unsupported xai.device value: {device_preference}")


def _resolve_model_init_kwargs(
    *,
    training_record: TrainingRecord,
    checkpoint_hparams: dict,
) -> dict[str, object]:
    model_params = training_record.model_params if isinstance(training_record.model_params, dict) else {}
    training_params = model_params.get("training_params")
    if not isinstance(training_params, dict):
        training_params = {}

    lr = float(checkpoint_hparams.get("lr", training_params.get("lr", 1e-3)))
    weight_decay = float(checkpoint_hparams.get("weight_decay", training_params.get("weight_decay", 0.0)))
    base_channels = int(checkpoint_hparams.get("base_channels", model_params.get("base_channels", 32)))
    raw_blocks = checkpoint_hparams.get("blocks", model_params.get("blocks", [2, 2, 2, 2]))
    blocks = _resolve_model_blocks({"blocks": raw_blocks})
    in_channels = int(checkpoint_hparams.get("in_channels", model_params.get("in_channels", 1)))
    use_patchify = bool(checkpoint_hparams.get("use_patchify", model_params.get("use_patchify", False)))
    stem = str(checkpoint_hparams.get("stem", model_params.get("stem", "standard")))
    head_pooling = str(checkpoint_hparams.get("head_pooling", model_params.get("head_pooling", "avg")))
    model_name = str(checkpoint_hparams.get("model_name", training_record.model_name or ""))
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "base_channels": base_channels,
        "blocks": blocks,
        "in_channels": in_channels,
        "use_patchify": use_patchify,
        "stem": stem,
        "head_pooling": head_pooling,
        "model_name": model_name,
    }


def _uses_legacy_checkpoint_layer_naming(state_dict: dict) -> bool:
    has_legacy_prefix = any(
        key.startswith(legacy_prefix)
        for key in state_dict
        for legacy_prefix in LEGACY_LAYER_PREFIXES
    )
    has_modern_prefix = any(key.startswith("model.layers.") for key in state_dict)
    return has_legacy_prefix and not has_modern_prefix


def _remap_legacy_checkpoint_state_dict(state_dict: dict) -> tuple[dict, int]:
    remapped_state_dict: dict = {}
    remapped_keys_count = 0
    for key, value in state_dict.items():
        remapped_key = key
        for legacy_prefix, modern_prefix in LEGACY_LAYER_PREFIXES.items():
            if key.startswith(legacy_prefix):
                remapped_key = f"{modern_prefix}{key[len(legacy_prefix):]}"
                remapped_keys_count += 1
                break
        remapped_state_dict[remapped_key] = value
    return remapped_state_dict, remapped_keys_count


def _load_lit_model_for_xai(
    *,
    checkpoint_path: Path,
    training_record: TrainingRecord,
    device: torch.device,
) -> LitBinaryClassifier:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a valid state_dict.")

    if _uses_legacy_checkpoint_layer_naming(state_dict):
        remapped_state_dict, remapped_keys_count = _remap_legacy_checkpoint_state_dict(state_dict)
        logger.warning(
            f"Detected legacy checkpoint parameter naming for {checkpoint_path.name}; "
            f"remapping {remapped_keys_count} layer keys (layer1..4 -> layers.0..3)."
        )
        checkpoint_hparams = checkpoint.get("hyper_parameters") or {}
        if not isinstance(checkpoint_hparams, dict):
            raise ValueError(
                f"Checkpoint hyper_parameters must be a mapping, got: {checkpoint_hparams}"
            )
        model_init_kwargs = _resolve_model_init_kwargs(
            training_record=training_record,
            checkpoint_hparams=checkpoint_hparams,
        )
        lit_model = LitBinaryClassifier(**model_init_kwargs)
        lit_model.load_state_dict(remapped_state_dict, strict=True)
    else:
        try:
            lit_model = LitBinaryClassifier.load_from_checkpoint(str(checkpoint_path), map_location=device)
        except RuntimeError as error:
            raise RuntimeError(
                f"Failed to load checkpoint at {checkpoint_path}. "
                "Model architecture and checkpoint parameters appear incompatible."
            ) from error

    lit_model.to(device)
    lit_model.eval()
    return lit_model


def _resolve_xai_methods(xai_config: dict) -> list[str]:
    raw_methods = xai_config.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("xai.methods must be a non-empty list of method names.")

    methods: list[str] = []
    for raw_method in raw_methods:
        if not isinstance(raw_method, str):
            raise ValueError(f"Invalid xai method value: {raw_method}")
        method_name = raw_method.strip().lower()
        if method_name not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported xai method '{raw_method}'. "
                f"Supported methods: {sorted(SUPPORTED_METHODS)}"
            )
        if method_name in methods:
            continue
        methods.append(method_name)
    return methods


def _resolve_splits(xai_config: dict) -> list[str]:
    raw_splits = xai_config.get("splits", ["test"])
    if isinstance(raw_splits, str):
        normalized = raw_splits.strip()
        if not normalized:
            raise ValueError("xai.splits must not be an empty string.")
        return [normalized]
    if not isinstance(raw_splits, list) or not raw_splits:
        raise ValueError("xai.splits must be a non-empty string or list of split names.")
    if not all(isinstance(split, str) and split.strip() for split in raw_splits):
        raise ValueError("xai.splits must contain only non-empty strings.")
    return [split.strip() for split in raw_splits]


def _resolve_samples_path(
    *,
    dataset_meta_data: dict,
    fallback_samples_filename: str,
    project_root: Path,
    data_dir: Path,
) -> Path:
    raw_samples_path = dataset_meta_data.get("samples_path")
    if raw_samples_path:
        candidate = Path(str(raw_samples_path)).expanduser()
    else:
        candidate = Path(fallback_samples_filename).expanduser()

    if candidate.is_absolute():
        return candidate.resolve()

    project_candidate = (project_root / candidate).resolve()
    if project_candidate.exists():
        return project_candidate

    data_candidate = (data_dir / candidate).resolve()
    if data_candidate.exists():
        return data_candidate

    return project_candidate


def _resolve_optional_sample_path(image_data_dir: str, raw_path: object) -> str | None:
    if raw_path is None:
        return None
    if pd.isna(raw_path):
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    return _resolve_sample_image_path(data_dir=image_data_dir, image_path=text)


def _load_samples_for_training_record(
    *,
    record: TrainingRecord,
    config: ExperimentConfig,
    project_root: Path,
    data_dir: Path,
    image_data_dir: Path,
    selected_splits: list[str],
) -> tuple[list[dict], Path]:
    dataset_meta_data = record.dataset_meta_data or {}
    samples_filename = config.data.get("samples_filename", "dataset_metadata.jsonl")
    samples_path = _resolve_samples_path(
        dataset_meta_data=dataset_meta_data,
        fallback_samples_filename=samples_filename,
        project_root=project_root,
        data_dir=data_dir,
    )
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples metadata file not found: {samples_path}")

    samples = load_jsonl_file(file_path=str(samples_path))
    if not samples:
        raise ValueError(f"No samples found in metadata file: {samples_path}")

    sample_df = pd.DataFrame(samples)
    required_columns = {"image_path", "label"}
    missing_columns = required_columns.difference(sample_df.columns)
    if missing_columns:
        raise ValueError(
            f"Samples metadata at {samples_path} is missing required columns: {sorted(missing_columns)}"
        )

    sample_df = sample_df.dropna(subset=["image_path", "label"]).copy()
    sample_df["image_path"] = sample_df["image_path"].map(
        lambda path: _resolve_sample_image_path(str(image_data_dir), str(path))
    )
    sample_df = sample_df[sample_df["image_path"].map(os.path.exists)].reset_index(drop=True)
    if sample_df.empty:
        raise ValueError(f"No valid image paths after preprocessing in metadata file: {samples_path}")

    sample_df["label"] = sample_df["label"].astype(int)
    if "mask_path" in sample_df.columns:
        sample_df["mask_path"] = sample_df["mask_path"].map(
            lambda path: _resolve_optional_sample_path(str(image_data_dir), path)
        )
    else:
        sample_df["mask_path"] = None

    raw_selected_filters = dataset_meta_data.get("selected_generated_filters")
    if raw_selected_filters is None:
        selected_filters: dict[str, object] = {}
    elif isinstance(raw_selected_filters, dict) and not raw_selected_filters:
        selected_filters = {}
    elif not isinstance(raw_selected_filters, dict):
        raise ValueError(
            "dataset_meta_data.selected_generated_filters must be a mapping when present, "
            f"got: {raw_selected_filters}"
        )
    else:
        selected_filters = raw_selected_filters
    if selected_filters:
        sample_df = _filter_samples_for_generated_filters(
            sample_df=sample_df,
            samples_path=str(samples_path),
            generated_filters=selected_filters,
        )

    if "split" in sample_df.columns:
        split_filtered = sample_df[sample_df["split"].astype(str).isin(selected_splits)].reset_index(drop=True)
        if split_filtered.empty:
            logger.warning(
                f"No samples matched xai.splits={selected_splits} for tuple {record.model_name}. "
                f"Falling back to all splits from {samples_path}."
            )
        else:
            sample_df = split_filtered

    if sample_df.empty:
        raise ValueError(
            f"No samples available for XAI after applying dataset/split filters for {record.model_name}"
        )

    return sample_df.to_dict("records"), samples_path


def _sanitize_for_path(text: str) -> str:
    value = text.strip() or "unknown"
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _build_tuple_id(record: TrainingRecord) -> str:
    dataset_meta_data = record.dataset_meta_data or {}
    dataset_type = str(dataset_meta_data.get("dataset_type", "dataset"))
    variant_tag = str(dataset_meta_data.get("dataset_variant_tag", "full_dataset"))
    return _sanitize_for_path(
        f"{dataset_type}-{record.model_name}-{variant_tag}-rep{record.repetition}-seed{record.seed}"
    )


def _resolve_image_id(sample: dict) -> str:
    image_id = sample.get("image_id")
    if image_id is not None and str(image_id).strip():
        return str(image_id).strip()

    image_path = sample.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(f"Could not resolve image_id for sample: {sample}")
    return Path(image_path).stem


def _build_sample_stem(sample: dict) -> str:
    image_id = _resolve_image_id(sample)
    image_path = str(sample.get("image_path", ""))
    split = str(sample.get("split", "unknown"))
    digest = hashlib.sha1(f"{image_path}|{split}".encode("utf-8")).hexdigest()[:10]
    study_id = str(sample.get("study_id", "")).strip()
    prefix = f"{study_id}-{image_id}" if study_id else image_id
    return _sanitize_for_path(f"{prefix}-{split}-{digest}")


@dataclass(frozen=True)
class _AttributionSettings:
    baseline_value: float
    integrated_gradients_steps: int
    integrated_gradients_internal_batch_size: int
    gradient_shap_n_samples: int
    gradient_shap_stdevs: float
    perturbation_n_samples: int
    perturbations_per_eval: int
    superpixel_n_segments: int
    superpixel_compactness: float
    lrp_alpha: float
    lrp_beta: float
    patch_size: int | None
    original_image_size: int | None
    normalize_input: bool
    normalize_mean: float
    normalize_std: float


def _load_image_tensor(image_path: str, settings: _AttributionSettings) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    tensor = TO_TENSOR(image)
    if settings.normalize_input:
        tensor = transforms.functional.normalize(
            tensor, mean=[settings.normalize_mean], std=[settings.normalize_std]
        )
    if settings.patch_size is not None:
        tensor = patchify(tensor, settings.patch_size)
    return tensor


def _load_input_batch(
    *,
    image_paths: list[str],
    device: torch.device,
    settings: _AttributionSettings,
) -> torch.Tensor:
    sample_tensors = [_load_image_tensor(image_path=path, settings=settings) for path in image_paths]
    return torch.stack(sample_tensors).to(device)


def _compute_feature_mask(
    *,
    image_path: str,
    device: torch.device,
    settings: _AttributionSettings,
) -> torch.Tensor:
    image = np.array(Image.open(image_path).convert("L"))
    segments = slic(
        image,
        n_segments=settings.superpixel_n_segments,
        compactness=settings.superpixel_compactness,
        channel_axis=None,
        start_label=0,
    )
    mask = torch.from_numpy(segments).long().unsqueeze(0)
    if settings.patch_size is not None:
        mask = patchify(mask, settings.patch_size)
    return mask.unsqueeze(0).to(device)


def _compute_predictions(*, model: torch.nn.Module, input_batch: torch.Tensor) -> list[tuple[float, float]]:
    with torch.no_grad():
        logits = model(input_batch).reshape(-1)
    if logits.numel() != input_batch.shape[0]:
        raise ValueError(
            f"Expected one logit per input sample, got {logits.numel()} logits "
            f"for {input_batch.shape[0]} samples."
        )
    probabilities = torch.sigmoid(logits)
    return [
        (float(logit), float(probability))
        for logit, probability in zip(logits.tolist(), probabilities.tolist(), strict=True)
    ]


def _to_attribution_map(attribution: torch.Tensor) -> np.ndarray:
    array = attribution.detach().cpu().numpy()
    while array.ndim > 2:
        array = np.abs(array).sum(axis=0)
    attribution_map = np.abs(array).astype(np.float32)
    if attribution_map.ndim != 2:
        raise ValueError(f"Expected a 2D attribution map, got shape {attribution_map.shape}")
    return attribution_map


def _image_array_for_edge_filter(sample_tensor: torch.Tensor, settings: _AttributionSettings) -> np.ndarray:
    detached = sample_tensor.detach()
    if settings.patch_size is not None and settings.original_image_size is not None:
        image_tensor = unpatchify(detached, settings.original_image_size, settings.patch_size)
    else:
        image_tensor = detached
    return image_tensor.squeeze().cpu().numpy()


def _build_attributors(*, model: torch.nn.Module) -> dict[str, object]:
    return {
        "saliency": Saliency(model),
        "input_x_gradient": InputXGradient(model),
        "integrated_gradient": IntegratedGradients(model),
        "kernel_shap": KernelShap(model),
        "lime": Lime(model),
        "lrp": LRP(model),
        "guided_backprop": GuidedBackprop(model),
        "deep_lift": DeepLift(model),
        "gradient_shap": GradientShap(model),
    }


def _attribution_map_from_sample_attribution(
    sample_attribution: torch.Tensor,
    settings: _AttributionSettings,
) -> np.ndarray:
    if settings.patch_size is not None and settings.original_image_size is not None:
        sample_attribution = unpatchify(
            sample_attribution, settings.original_image_size, settings.patch_size
        )
    return _to_attribution_map(sample_attribution)


def _compute_gradient_attribution_batch(
    *,
    method_name: str,
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    attributors: dict[str, object],
    settings: _AttributionSettings,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    batch = input_batch.detach().clone().requires_grad_(True)

    if method_name == "saliency":
        return attributors["saliency"].attribute(batch, abs=True)
    if method_name == "input_x_gradient":
        return attributors["input_x_gradient"].attribute(batch)
    if method_name == "integrated_gradient":
        baseline = torch.full_like(batch, fill_value=settings.baseline_value)
        return attributors["integrated_gradient"].attribute(
            batch,
            baselines=baseline,
            n_steps=settings.integrated_gradients_steps,
            internal_batch_size=settings.integrated_gradients_internal_batch_size,
        )
    if method_name == "lrp":
        _prepare_model_for_lrp(model, alpha=settings.lrp_alpha, beta=settings.lrp_beta)
        return attributors["lrp"].attribute(batch)
    if method_name == "guided_backprop":
        return attributors["guided_backprop"].attribute(batch)
    if method_name == "deep_lift":
        baseline = torch.full_like(batch, fill_value=settings.baseline_value)
        return attributors["deep_lift"].attribute(batch, baselines=baseline)
    if method_name == "gradient_shap":
        baseline = torch.full_like(batch, fill_value=settings.baseline_value)
        return attributors["gradient_shap"].attribute(
            batch,
            baselines=baseline,
            n_samples=settings.gradient_shap_n_samples,
            stdevs=settings.gradient_shap_stdevs,
        )
    raise ValueError(f"Unsupported gradient-based method: {method_name}")


def _compute_perturbation_attribution(
    *,
    method_name: str,
    input_tensor: torch.Tensor,
    feature_mask: torch.Tensor,
    attributors: dict[str, object],
    settings: _AttributionSettings,
) -> np.ndarray:
    baseline = torch.full_like(input_tensor, fill_value=settings.baseline_value)
    attribution = attributors[method_name].attribute(
        input_tensor,
        baselines=baseline,
        feature_mask=feature_mask,
        n_samples=settings.perturbation_n_samples,
        perturbations_per_eval=settings.perturbations_per_eval,
    )
    return _attribution_map_from_sample_attribution(attribution.squeeze(0), settings)


def _compute_model_free_attribution(
    *,
    method_name: str,
    sample_tensor: torch.Tensor,
    settings: _AttributionSettings,
) -> np.ndarray:
    if method_name == "sobel":
        image = _image_array_for_edge_filter(sample_tensor, settings)
        return _to_attribution_map(torch.from_numpy(skimage_sobel(image)))
    if method_name == "laplace":
        image = _image_array_for_edge_filter(sample_tensor, settings)
        return _to_attribution_map(torch.from_numpy(skimage_laplace(image)))
    if method_name == "random":
        if settings.patch_size is not None and settings.original_image_size is not None:
            attribution_height = attribution_width = settings.original_image_size
        else:
            attribution_height, attribution_width = int(sample_tensor.shape[-2]), int(sample_tensor.shape[-1])
        return np.random.uniform(0.0, 1.0, size=(attribution_height, attribution_width)).astype(np.float32)
    raise ValueError(f"Unsupported model-free method: {method_name}")


def _chunked_samples(samples: list[dict], chunk_size: int) -> list[list[dict]]:
    return [samples[start : start + chunk_size] for start in range(0, len(samples), chunk_size)]


def _build_processed_key(record: dict) -> tuple[str, str, str, str]:
    return (
        str(record["tuple_id"]),
        str(record["method"]).strip().lower(),
        str(record["image_path"]),
        str(record.get("split", "unknown")),
    )


def _load_existing_xai_records(
    xai_records_path: Path,
    *,
    skip_corrupt_lines: bool = False,
) -> tuple[list[dict], set[tuple[str, str, str, str]]]:
    if not xai_records_path.exists():
        return [], set()

    records = load_jsonl_file(file_path=str(xai_records_path), skip_corrupt_lines=skip_corrupt_lines)
    processed_keys = {_build_processed_key(record) for record in records}
    return records, processed_keys


def _dataset_name_of_training_record(training_record: TrainingRecord) -> str | None:
    dataset_meta_data = training_record.dataset_meta_data or {}
    selected_filters = dataset_meta_data.get("selected_generated_filters") or {}
    return selected_filters.get("dataset_name")


def _find_training_records_by_repetition(
    training_records: list[TrainingRecord],
    dataset_name: str,
    role: str,
) -> dict[int, TrainingRecord]:
    matches = [
        training_record
        for training_record in training_records
        if _dataset_name_of_training_record(training_record) == dataset_name
    ]
    if not matches:
        available_dataset_names = sorted(
            {
                name
                for training_record in training_records
                if (name := _dataset_name_of_training_record(training_record)) is not None
            }
        )
        raise ValueError(
            f"cross_evaluation {role} dataset_name '{dataset_name}' matched no training records. "
            f"Available dataset_names: {available_dataset_names}"
        )

    records_by_repetition: dict[int, TrainingRecord] = {}
    for training_record in matches:
        repetition = int(training_record.repetition)
        if repetition in records_by_repetition:
            raise ValueError(
                f"cross_evaluation {role} dataset_name '{dataset_name}' matched multiple training "
                f"records for repetition {repetition}; expected exactly one per repetition."
            )
        records_by_repetition[repetition] = training_record
    return records_by_repetition


def _sanitize_model_source(model_source: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", model_source).strip("_")


def _build_cross_evaluation_records(
    cross_specs: list,
    training_records: list[TrainingRecord],
) -> list[TrainingRecord]:
    cross_records: list[TrainingRecord] = []
    for spec in cross_specs:
        if not isinstance(spec, dict):
            raise ValueError(f"Each xai.cross_evaluation entry must be a mapping, got: {spec}")
        model_dataset_name = spec.get("model_dataset_name")
        data_dataset_name = spec.get("data_dataset_name")
        if not model_dataset_name or not data_dataset_name:
            raise ValueError(
                "Each xai.cross_evaluation entry requires 'model_dataset_name' and 'data_dataset_name'."
            )
        model_records_by_repetition = _find_training_records_by_repetition(
            training_records, model_dataset_name, "model"
        )
        data_records_by_repetition = _find_training_records_by_repetition(
            training_records, data_dataset_name, "data"
        )
        model_source = str(spec.get("name") or _sanitize_model_source(model_dataset_name))

        paired_repetitions = sorted(set(model_records_by_repetition) & set(data_records_by_repetition))
        if not paired_repetitions:
            raise ValueError(
                f"cross_evaluation '{model_source}' has no repetition present for both the model "
                f"dataset '{model_dataset_name}' (repetitions {sorted(model_records_by_repetition)}) and the "
                f"data dataset '{data_dataset_name}' (repetitions {sorted(data_records_by_repetition)})."
            )
        unpaired_repetitions = sorted(
            (set(model_records_by_repetition) | set(data_records_by_repetition)) - set(paired_repetitions)
        )
        if unpaired_repetitions:
            logger.warning(
                f"cross_evaluation '{model_source}': skipping repetitions {unpaired_repetitions}, which are "
                f"missing on one side (model dataset '{model_dataset_name}' has "
                f"{sorted(model_records_by_repetition)}, data dataset '{data_dataset_name}' has "
                f"{sorted(data_records_by_repetition)})."
            )

        for repetition in paired_repetitions:
            model_record = model_records_by_repetition[repetition]
            data_record = data_records_by_repetition[repetition]
            cross_dataset_meta_data = dict(data_record.dataset_meta_data or {})
            base_variant_tag = str(cross_dataset_meta_data.get("dataset_variant_tag", "full_dataset"))
            cross_dataset_meta_data["dataset_variant_tag"] = f"{base_variant_tag}__xmodel-{model_source}"
            cross_dataset_meta_data["model_source"] = model_source
            cross_records.append(
                TrainingRecord(
                    batch_size=model_record.batch_size,
                    epochs=model_record.epochs,
                    model_name=model_record.model_name,
                    model_path=model_record.model_path,
                    training_log_path=model_record.training_log_path,
                    repetition=model_record.repetition,
                    seed=model_record.seed,
                    dataset_meta_data=cross_dataset_meta_data,
                    model_params=model_record.model_params,
                )
            )
    return cross_records


def main() -> None:
    config = _load_experiment_config()
    set_random_states(config.seed)
    project_root = _resolve_project_root(project_dir=config.project_dir)
    experiment_dir = _resolve_path(path_str=generate_experiment_dir(config=config), project_root=project_root)
    training_dir = _resolve_path(
        path_str=generate_training_dir(base_dir=str(experiment_dir), training_config=config.training),
        project_root=project_root,
    )
    xai_dir = (experiment_dir / config.xai.get("output_dir", "xai")).resolve()
    xai_dir.mkdir(parents=True, exist_ok=True)

    training_records_name = config.training.get("training_records", "training_records.jsonl")
    training_records_path = (training_dir / training_records_name).resolve()
    if not training_records_path.exists():
        raise FileNotFoundError(f"Training records not found: {training_records_path}")

    training_records = load_training_records(file_path=str(training_records_path))
    if not training_records:
        raise ValueError(f"No training records found: {training_records_path}")

    cross_evaluation_specs = config.xai.get("cross_evaluation")
    if isinstance(cross_evaluation_specs, list) and cross_evaluation_specs:
        cross_evaluation_records = _build_cross_evaluation_records(
            cross_specs=cross_evaluation_specs,
            training_records=training_records,
        )
        logger.info(
            f"Cross-evaluation enabled: applying {len(cross_evaluation_records)} model(s) trained on one "
            f"dataset to the test split of another dataset in the same training records."
        )
        training_records = training_records + cross_evaluation_records

    methods = _resolve_xai_methods(config.xai)
    selected_splits = _resolve_splits(config.xai)
    baseline_value = float(config.xai.get("integrated_gradient_baseline_value", 0.0))
    ig_n_steps = int(config.xai.get("integrated_gradients_steps", 32))
    if ig_n_steps <= 0:
        raise ValueError("xai.integrated_gradients_steps must be a positive integer.")
    gradient_shap_n_samples = int(config.xai.get("gradient_shap_n_samples", 5))
    if gradient_shap_n_samples <= 0:
        raise ValueError("xai.gradient_shap_n_samples must be a positive integer.")
    gradient_shap_stdevs = float(config.xai.get("gradient_shap_stdevs", 0.0))
    if gradient_shap_stdevs < 0:
        raise ValueError("xai.gradient_shap_stdevs must be non-negative.")
    perturbation_n_samples = int(config.xai.get("perturbation_n_samples", 256))
    perturbations_per_eval = int(config.xai.get("perturbations_per_eval", 64))
    if perturbations_per_eval <= 0:
        raise ValueError("xai.perturbations_per_eval must be a positive integer.")
    superpixel_n_segments = int(config.xai.get("superpixel_n_segments", 100))
    superpixel_compactness = float(config.xai.get("superpixel_compactness", 10.0))
    lrp_alpha = float(config.xai.get("lrp_alpha", 2.0))
    lrp_beta = float(config.xai.get("lrp_beta", -1.0))
    batch_size = int(config.xai.get("batch_size", 8))
    if batch_size <= 0:
        raise ValueError("xai.batch_size must be a positive integer.")
    raw_ig_internal_batch_size = config.xai.get("integrated_gradients_internal_batch_size")
    integrated_gradients_internal_batch_size = (
        int(raw_ig_internal_batch_size) if raw_ig_internal_batch_size is not None else 4 * batch_size
    )
    if integrated_gradients_internal_batch_size <= 0:
        raise ValueError("xai.integrated_gradients_internal_batch_size must be a positive integer.")
    raw_max_samples = config.xai.get("max_samples")
    max_samples = int(raw_max_samples) if raw_max_samples is not None else None

    xai_records_path = (xai_dir / config.xai.get("xai_records", "xai_records.jsonl")).resolve()
    intermediate_records_path = (
        xai_dir / config.xai.get("intermediate_xai_records", "intermediate_xai_records.jsonl")
    ).resolve()
    xai_records, processed_keys = _load_existing_xai_records(xai_records_path=xai_records_path)
    if not xai_records:
        xai_records, processed_keys = _load_existing_xai_records(
            xai_records_path=intermediate_records_path,
            skip_corrupt_lines=True,
        )
    dump_as_jsonl_file(data=xai_records, file_path=str(intermediate_records_path))

    data_dir = _resolve_data_dir(config=config, project_root=project_root)
    image_data_dir = _resolve_image_data_dir(
        config=config,
        project_root=project_root,
        fallback_data_dir=data_dir,
    )
    device_preference = config.xai.get("device", config.training.get("device", "auto"))
    device = _resolve_device(device_preference=str(device_preference))
    logger.info(f"Running XAI on device: {device}")

    logger.info(
        f"Computing attributions for {len(training_records)} training records with methods={methods}, "
        f"splits={selected_splits}, and batch_size={batch_size}"
    )
    with tqdm(training_records, desc="XAI tuples", unit="tuple", dynamic_ncols=True) as tuple_progress:
        for training_record in tuple_progress:
            tuple_id = _build_tuple_id(training_record)
            tuple_progress.set_postfix_str(tuple_id, refresh=False)

            dataset_meta_data = training_record.dataset_meta_data or {}
            raw_selected_filters = dataset_meta_data.get("selected_generated_filters")
            selected_filters = raw_selected_filters if isinstance(raw_selected_filters, dict) else {}
            dataset_variant_tag = str(dataset_meta_data.get("dataset_variant_tag", "full_dataset"))
            dataset_type = str(dataset_meta_data.get("dataset_type", "dataset"))
            record_model_params = training_record.model_params if isinstance(training_record.model_params, dict) else {}
            use_patchify = bool(record_model_params.get("use_patchify", False))
            settings = _AttributionSettings(
                baseline_value=baseline_value,
                integrated_gradients_steps=ig_n_steps,
                integrated_gradients_internal_batch_size=integrated_gradients_internal_batch_size,
                gradient_shap_n_samples=gradient_shap_n_samples,
                gradient_shap_stdevs=gradient_shap_stdevs,
                perturbation_n_samples=perturbation_n_samples,
                perturbations_per_eval=perturbations_per_eval,
                superpixel_n_segments=superpixel_n_segments,
                superpixel_compactness=superpixel_compactness,
                lrp_alpha=lrp_alpha,
                lrp_beta=lrp_beta,
                patch_size=int(record_model_params.get("image_size", 0)) if use_patchify else None,
                original_image_size=int(record_model_params.get("original_image_size", 512)) if use_patchify else None,
                normalize_input=bool(record_model_params.get("normalize_input", False)),
                normalize_mean=float(record_model_params.get("normalize_mean", 0.5)),
                normalize_std=float(record_model_params.get("normalize_std", 0.5)),
            )

            if not training_record.model_path:
                raise ValueError(f"Missing model_path for tuple '{tuple_id}' in training records.")
            checkpoint_path = _resolve_record_path(path_str=training_record.model_path, experiment_dir=experiment_dir)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found for tuple '{tuple_id}': {checkpoint_path}")

            logger.info(f"Loading model for tuple '{tuple_id}' from {checkpoint_path}")
            lit_model = _load_lit_model_for_xai(
                checkpoint_path=checkpoint_path,
                training_record=training_record,
                device=device,
            )
            model = lit_model.model.to(device)
            model.eval()

            attributors = _build_attributors(model=model)

            samples, samples_path = _load_samples_for_training_record(
                record=training_record,
                config=config,
                project_root=project_root,
                data_dir=data_dir,
                image_data_dir=image_data_dir,
                selected_splits=selected_splits,
            )
            if max_samples is not None and len(samples) > max_samples:
                samples = samples[:max_samples]
                logger.info(
                    f"Tuple '{tuple_id}' limited to {max_samples} of available samples "
                    f"for XAI (samples_path={samples_path})"
                )
            else:
                logger.info(f"Tuple '{tuple_id}' has {len(samples)} samples for XAI (samples_path={samples_path})")

            unsupported_methods: set[str] = set()
            total_attribution_tasks = len(samples) * len(methods)
            with tqdm(
                total=total_attribution_tasks,
                desc=f"{tuple_id} attributions",
                unit="attr",
                leave=False,
                dynamic_ncols=True,
            ) as attribution_progress:
                for batch_samples in _chunked_samples(samples, batch_size):
                    pending_by_sample: list[tuple[dict, set[str]]] = []
                    already_processed_count = 0
                    for sample in batch_samples:
                        image_path = str(sample["image_path"])
                        split = str(sample.get("split", "unknown"))
                        pending_methods = {
                            method_name for method_name in methods
                            if (tuple_id, method_name, image_path, split) not in processed_keys
                        }
                        already_processed_count += len(methods) - len(pending_methods)
                        if pending_methods:
                            pending_by_sample.append((sample, pending_methods))
                    if already_processed_count:
                        attribution_progress.update(already_processed_count)
                    if not pending_by_sample:
                        continue

                    active_samples = [sample for sample, _ in pending_by_sample]
                    input_batch = _load_input_batch(
                        image_paths=[str(sample["image_path"]) for sample in active_samples],
                        device=device,
                        settings=settings,
                    )
                    predictions = _compute_predictions(model=model, input_batch=input_batch)
                    feature_masks: dict[int, torch.Tensor] = {}

                    for method_name in methods:
                        sample_indices = [
                            index
                            for index, (_, pending_methods) in enumerate(pending_by_sample)
                            if method_name in pending_methods
                        ]
                        if not sample_indices:
                            continue
                        if method_name in unsupported_methods:
                            attribution_progress.update(len(sample_indices))
                            continue

                        computed_maps: list[tuple[int, np.ndarray]] = []
                        try:
                            if method_name in GRADIENT_BASED_METHODS:
                                attribution_batch = _compute_gradient_attribution_batch(
                                    method_name=method_name,
                                    model=model,
                                    input_batch=input_batch[sample_indices],
                                    attributors=attributors,
                                    settings=settings,
                                )
                                computed_maps = [
                                    (index, _attribution_map_from_sample_attribution(sample_attribution, settings))
                                    for index, sample_attribution in zip(
                                        sample_indices, attribution_batch, strict=True
                                    )
                                ]
                            elif method_name in PERTURBATION_METHODS:
                                for index in sample_indices:
                                    if index not in feature_masks:
                                        feature_masks[index] = _compute_feature_mask(
                                            image_path=str(active_samples[index]["image_path"]),
                                            device=device,
                                            settings=settings,
                                        )
                                    attribution_map = _compute_perturbation_attribution(
                                        method_name=method_name,
                                        input_tensor=input_batch[index : index + 1],
                                        feature_mask=feature_masks[index],
                                        attributors=attributors,
                                        settings=settings,
                                    )
                                    computed_maps.append((index, attribution_map))
                            else:
                                for index in sample_indices:
                                    attribution_map = _compute_model_free_attribution(
                                        method_name=method_name,
                                        sample_tensor=input_batch[index],
                                        settings=settings,
                                    )
                                    computed_maps.append((index, attribution_map))
                        except torch.cuda.OutOfMemoryError as error:
                            raise RuntimeError(
                                f"CUDA out of memory while computing '{method_name}' attributions with "
                                f"xai.batch_size={batch_size}. Reduce xai.batch_size, "
                                f"xai.integrated_gradients_internal_batch_size, or xai.perturbations_per_eval."
                            ) from error
                        except Exception as error:
                            unsupported_methods.add(method_name)
                            logger.warning(
                                f"Skipping XAI method '{method_name}' for tuple '{tuple_id}' "
                                f"(model_name={training_record.model_name}): {type(error).__name__}: {error}. "
                                f"This method is skipped for the remaining samples of this tuple."
                            )
                            attribution_progress.update(len(sample_indices) - len(computed_maps))

                        for index, attribution_map in computed_maps:
                            sample = active_samples[index]
                            prediction_logit, prediction_probability = predictions[index]
                            sample_stem = _build_sample_stem(sample)
                            mask_path = sample.get("mask_path")
                            method_dir = (xai_dir / "attributions" / tuple_id / method_name).resolve()
                            method_dir.mkdir(parents=True, exist_ok=True)
                            attribution_npy_path = (method_dir / f"{sample_stem}.npy").resolve()
                            np.save(attribution_npy_path, attribution_map)

                            xai_record = {
                                "tuple_id": tuple_id,
                                "method": method_name,
                                "model_name": training_record.model_name,
                                "model_path": str(checkpoint_path),
                                "dataset_type": dataset_type,
                                "dataset_variant_tag": dataset_variant_tag,
                                "model_source": str(dataset_meta_data.get("model_source", "native")),
                                "selected_generated_filters": selected_filters,
                                "samples_path": str(samples_path),
                                "repetition": int(training_record.repetition),
                                "seed": int(training_record.seed),
                                "image_id": _resolve_image_id(sample),
                                "study_id": sample.get("study_id"),
                                "split": str(sample.get("split", "unknown")),
                                "label": int(sample["label"]),
                                "image_path": str(sample["image_path"]),
                                "mask_path": str(mask_path) if isinstance(mask_path, str) else None,
                                "attribution_path": str(attribution_npy_path),
                                "prediction_logit": prediction_logit,
                                "prediction_probability": prediction_probability,
                                "attribution_min": float(np.min(attribution_map)),
                                "attribution_max": float(np.max(attribution_map)),
                                "attribution_sum": float(np.sum(attribution_map)),
                            }
                            xai_records.append(xai_record)
                            processed_keys.add(_build_processed_key(xai_record))
                            append_to_jsonl_file(record=xai_record, file_path=str(intermediate_records_path))
                            attribution_progress.update(1)

    dump_as_jsonl_file(data=xai_records, file_path=str(xai_records_path))
    logger.info(f"Saved {len(xai_records)} XAI records to {xai_records_path}")


if __name__ == "__main__":
    main()
