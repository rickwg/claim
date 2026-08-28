import math
import os
from dataclasses import asdict
from itertools import product
from os.path import join
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models as torchvision_models, transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B4_Weights,
    ResNet18_Weights,
)

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ImportError:  # pragma: no cover - compatibility fallback
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

from common import EnvironmentVariables, LIGHTNING_LOGS_FILENAME, TrainingRecord
from config.configuration import ExperimentConfig
from utils import (
    dump_as_jsonl_file,
    generate_data_dir,
    generate_experiment_dir,
    generate_training_dir,
    load_config_file,
    load_jsonl_file,
    load_training_records,
    set_random_states,
)


def _resolve_sample_image_path(
    data_dir: str,
    image_path: str,
) -> str:
    if os.path.isabs(image_path):
        return image_path

    return join(data_dir, image_path.lstrip("/"))


def _resolve_samples_path(data_dir: str, samples_filename: str) -> str:
    if os.path.isabs(samples_filename):
        return samples_filename
    return join(data_dir, samples_filename)


def _resolve_configured_data_dir(config: ExperimentConfig, fallback_data_dir: str) -> str:
    configured_data_dir = config.data.get("data_dir")
    if not isinstance(configured_data_dir, str) or not configured_data_dir.strip():
        return fallback_data_dir

    data_dir_path = Path(configured_data_dir).expanduser()
    if data_dir_path.is_absolute():
        return str(data_dir_path.resolve())

    project_dir_path = Path(config.project_dir).expanduser()
    if not project_dir_path.is_absolute():
        project_dir_path = (Path.cwd() / project_dir_path).resolve()
    else:
        project_dir_path = project_dir_path.resolve()
    return str((project_dir_path / data_dir_path).resolve())


def _parse_numeric_value(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for '{field_name}': {value}") from error


def _format_filter_tag_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:g}".replace("-", "m").replace(".", "p")

    text = str(value).strip()
    if not text:
        return "empty"

    sanitized = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized)


def _series_is_numeric(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    return pd.to_numeric(values, errors="coerce").notna().all()


def _try_resolve_filter_field_name(
    raw_field_name: str,
    available_columns: set[str],
) -> str | None:
    if raw_field_name in available_columns:
        return raw_field_name

    aliases = {
        "twirl_angles": "twirl_angle",
        "spherize_amounts": "spherize_amount",
    }
    alias = aliases.get(raw_field_name)
    if alias and alias in available_columns:
        return alias

    if raw_field_name.endswith("s") and raw_field_name[:-1] in available_columns:
        return raw_field_name[:-1]

    return None


def _discover_all_filter_values(
    sample_df: pd.DataFrame,
    samples_path: str,
    field_name: str,
) -> list[object]:
    if field_name not in sample_df.columns:
        raise ValueError(
            f"JSONL metadata file at {samples_path} does not contain '{field_name}'."
        )

    generated_df = sample_df
    if "source" in sample_df.columns:
        generated_rows = sample_df[sample_df["source"].astype(str) == "generated"]
        if not generated_rows.empty:
            generated_df = generated_rows

    values = generated_df[field_name].dropna()
    if values.empty:
        raise ValueError(
            f"No non-null values found for '{field_name}' in metadata file at {samples_path}."
        )

    if _series_is_numeric(values):
        numeric_values = pd.to_numeric(values, errors="coerce")
        return sorted({float(v) for v in numeric_values.tolist() if pd.notna(v)})
    return sorted(set(values.tolist()), key=lambda value: str(value))


def _resolve_parameter_values(
    sample_df: pd.DataFrame,
    samples_path: str,
    field_name: str,
    raw_value_spec: object,
) -> list[object]:
    if raw_value_spec is None:
        raise ValueError(
            f"Parameterization for '{field_name}' cannot be null. "
            "Use a scalar, list, or 'all'."
        )

    if isinstance(raw_value_spec, str) and raw_value_spec.strip().lower() == "all":
        return _discover_all_filter_values(
            sample_df=sample_df,
            samples_path=samples_path,
            field_name=field_name,
        )

    values = raw_value_spec if isinstance(raw_value_spec, (list, tuple, set)) else [raw_value_spec]
    if not values:
        raise ValueError(
            f"Parameterization list for '{field_name}' is empty. Provide at least one value."
        )

    normalized_values: list[object] = []
    if _series_is_numeric(sample_df[field_name]):
        for value in values:
            normalized_values.append(_parse_numeric_value(value=value, field_name=field_name))
    else:
        normalized_values.extend(values)
    return normalized_values


def _build_field_match_mask(series: pd.Series, field_name: str, field_value: object) -> pd.Series:
    if _series_is_numeric(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_value = _parse_numeric_value(value=field_value, field_name=field_name)
        return numeric_series.sub(numeric_value).abs() < 1e-8
    return series.astype(str) == str(field_value)


def _filter_samples_for_generated_filters(
    sample_df: pd.DataFrame,
    samples_path: str,
    generated_filters: dict[str, object],
) -> pd.DataFrame:
    filtered_df = sample_df.copy()
    for field_name, field_value in generated_filters.items():
        if field_name not in filtered_df.columns:
            raise ValueError(
                f"JSONL metadata file at {samples_path} does not contain '{field_name}', "
                f"but this field is configured for filtering."
            )

        match_mask = _build_field_match_mask(
            series=filtered_df[field_name],
            field_name=field_name,
            field_value=field_value,
        )
        filter_repr = f"{field_name}={field_value}"

        if "source" in filtered_df.columns:
            generated_mask = filtered_df["source"].astype(str).eq("generated")
            matched_generated = generated_mask & match_mask
            if int(matched_generated.sum()) == 0:
                raise ValueError(
                    f"No generated samples found for filter '{filter_repr}' "
                    f"in metadata file at {samples_path}."
                )
            filtered_df = filtered_df[(~generated_mask) | match_mask]
        else:
            if int(match_mask.sum()) == 0:
                raise ValueError(
                    f"No samples found for filter '{filter_repr}' "
                    f"in metadata file at {samples_path}."
                )
            filtered_df = filtered_df[match_mask | filtered_df[field_name].isna()]

        filtered_df = filtered_df.reset_index(drop=True)
        if filtered_df.empty:
            raise ValueError(
                f"Filtering by '{filter_repr}' produced an empty dataset from {samples_path}."
            )

    return filtered_df


def _partition_contains_generated_negatives(sample_df: pd.DataFrame) -> bool:
    if "source" not in sample_df.columns:
        return False
    generated_mask = sample_df["source"].astype(str).eq("generated")
    return bool((sample_df.loc[generated_mask, "label"] == 0).any())


def _keep_only_generated_rows(sample_df: pd.DataFrame) -> pd.DataFrame:
    generated_mask = sample_df["source"].astype(str).eq("generated")
    return sample_df[generated_mask].reset_index(drop=True)


def _resolve_dataset_variants(
    training_config: dict,
    data_dir: str,
    samples_filename: str,
) -> list[dict]:
    raw_parameterizations = training_config.get("dataset_parameterizations")
    if training_config.get("twirl_angle") not in {None, "", "none", "None"}:
        raise ValueError(
            "training.twirl_angle is no longer supported. "
            "Use training.dataset_parameterizations.twirl_angle instead."
        )

    if raw_parameterizations is None:
        return [{"tag": "", "description": "full_dataset", "generated_filters": None}]

    if not isinstance(raw_parameterizations, dict):
        raise ValueError(
            "training.dataset_parameterizations must be a mapping of metadata field to value/list/'all'."
        )

    samples_path = _resolve_samples_path(data_dir=data_dir, samples_filename=samples_filename)
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Could not find JSONL metadata file at {samples_path}")

    samples = load_jsonl_file(samples_path)
    if not samples:
        raise ValueError(f"No samples were found in JSONL metadata file at {samples_path}")
    sample_df = pd.DataFrame(samples)
    available_columns = set(sample_df.columns)

    parameterized_fields = []
    for raw_field_name, raw_value_spec in raw_parameterizations.items():
        if not isinstance(raw_field_name, str):
            raise ValueError(
                f"Dataset parameterization field names must be strings, got: {raw_field_name}"
            )
        field_name = _try_resolve_filter_field_name(
            raw_field_name=raw_field_name,
            available_columns=available_columns,
        )
        if field_name is None:
            logger.info(
                "Skipping parameterization field '{}' — not present in {}",
                raw_field_name,
                samples_path,
            )
            continue
        parameter_values = _resolve_parameter_values(
            sample_df=sample_df,
            samples_path=samples_path,
            field_name=field_name,
            raw_value_spec=raw_value_spec,
        )
        parameterized_fields.append((field_name, parameter_values))

    if not parameterized_fields:
        logger.info("All parameterization fields were skipped — using full dataset")
        return [{"tag": "", "description": "full_dataset", "generated_filters": None}]

    parameterized_fields.sort(key=lambda entry: (0 if entry[0] == "dataset_name" else 1, entry[0]))

    generated_df = sample_df
    if "source" in sample_df.columns:
        gen_rows = sample_df[sample_df["source"].astype(str) == "generated"]
        if not gen_rows.empty:
            generated_df = gen_rows

    field_names = [entry[0] for entry in parameterized_fields]
    value_lists = [entry[1] for entry in parameterized_fields]

    seen: set[tuple] = set()
    dataset_variants = []
    for combination in product(*value_lists):
        applicable_filters: dict[str, object] = {}
        subset = generated_df
        skip = False
        for field_name, field_value in zip(field_names, combination):
            if field_name not in subset.columns:
                continue
            non_null = subset[field_name].dropna()
            if non_null.empty:
                continue
            match_mask = _build_field_match_mask(subset[field_name], field_name, field_value)
            matched_subset = subset[match_mask]
            if matched_subset.empty:
                skip = True
                break
            subset = matched_subset
            applicable_filters[field_name] = field_value

        if skip or not applicable_filters:
            continue

        key = tuple(sorted(applicable_filters.items(), key=lambda item: str(item)))
        if key in seen:
            continue
        seen.add(key)

        tag = "-".join(
            f"{fn}-{_format_filter_tag_value(fv)}" for fn, fv in applicable_filters.items()
        )
        description = ", ".join(f"{fn}={fv}" for fn, fv in applicable_filters.items())
        dataset_variants.append({
            "tag": tag,
            "description": description,
            "generated_filters": dict(applicable_filters),
        })

    if not dataset_variants:
        raise ValueError("No dataset variants resolved from training.dataset_parameterizations.")

    return dataset_variants


def _split_train_validation(dataframe: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(dataframe) < 2:
        return dataframe.reset_index(drop=True), dataframe.reset_index(drop=True)

    val_count = max(1, int(round(len(dataframe) * val_size)))
    val_count = min(val_count, len(dataframe) - 1)
    if val_count <= 0:
        return dataframe.reset_index(drop=True), dataframe.sample(n=1, random_state=seed).reset_index(drop=True)

    stratify = None
    if dataframe["label"].nunique() > 1:
        label_counts = dataframe["label"].value_counts()
        if label_counts.min() >= 2 and val_count >= dataframe["label"].nunique():
            stratify = dataframe["label"]

    train_df, val_df = train_test_split(
        dataframe,
        test_size=val_count,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def patchify(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, h, w = tensor.shape
    patches_h = h // patch_size
    patches_w = w // patch_size
    return (
        tensor
        .view(1, patches_h, patch_size, patches_w, patch_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(patches_h * patches_w, 1, patch_size, patch_size)
    )


def unpatchify(patches: torch.Tensor, original_size: int, patch_size: int) -> torch.Tensor:
    patches_per_side = original_size // patch_size
    return (
        patches
        .view(patches_per_side, patches_per_side, 1, patch_size, patch_size)
        .permute(2, 0, 3, 1, 4)
        .reshape(1, original_size, original_size)
    )


class ImageClassificationDataset(Dataset):
    def __init__(self, samples: list[dict], image_size: int, use_patchify: bool = False,
                 normalize_input: bool = False, normalize_mean: float = 0.5, normalize_std: float = 0.5):
        self.samples = samples
        self.image_size = image_size
        self.use_patchify = use_patchify
        transform_list = [transforms.ToTensor()]
        if normalize_input:
            transform_list.append(transforms.Normalize(mean=[normalize_mean], std=[normalize_std]))
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("L")
        image = self.transform(image)
        if self.use_patchify:
            image = patchify(image, self.image_size)
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return image, label


class JsonlDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_data_dir: str,
        batch_size: int,
        num_workers: int,
        val_size: float,
        samples_filename: str,
        image_size: int,
        train_validation_split_seed: int,
        generated_filters: dict[str, object] | None = None,
        dataset_variant_tag: str | None = None,
        use_patchify: bool = False,
        normalize_input: bool = False,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
        exclude_preprocessed_when_self_contained: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_data_dir = image_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.samples_filename = samples_filename
        self.image_size = image_size
        self.train_validation_split_seed = train_validation_split_seed
        self.generated_filters = generated_filters
        self.dataset_variant_tag = dataset_variant_tag
        self.use_patchify = use_patchify
        self.normalize_input = normalize_input
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.exclude_preprocessed_when_self_contained = exclude_preprocessed_when_self_contained
        self.dataset_meta_data: dict = {}
        self._has_logged_subdataset_summary = False
        self.train_set: ImageClassificationDataset | None = None
        self.val_set: ImageClassificationDataset | None = None
        self.test_set: ImageClassificationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        samples_path = _resolve_samples_path(data_dir=self.data_dir, samples_filename=self.samples_filename)
        if not os.path.exists(samples_path):
            raise FileNotFoundError(f"Could not find JSONL metadata file at {samples_path}")

        samples = load_jsonl_file(samples_path)
        if not samples:
            raise ValueError(f"No samples were found in JSONL metadata file at {samples_path}")

        sample_df = pd.DataFrame(samples)
        required_columns = {"image_path", "label"}
        missing_columns = required_columns.difference(sample_df.columns)
        if missing_columns:
            raise ValueError(
                f"JSONL metadata file at {samples_path} is missing required columns: {sorted(missing_columns)}"
            )

        sample_df = sample_df.dropna(subset=["image_path", "label"]).copy()
        sample_df["image_path"] = sample_df["image_path"].astype(str).map(
            lambda path: _resolve_sample_image_path(self.image_data_dir, path)
        )
        sample_df["label"] = sample_df["label"].astype(int)
        sample_df = sample_df[sample_df["image_path"].map(os.path.exists)].reset_index(drop=True)
        if sample_df.empty:
            raise ValueError(f"No valid image paths from metadata file at {samples_path}")

        if self.generated_filters:
            sample_df = _filter_samples_for_generated_filters(
                sample_df=sample_df,
                samples_path=samples_path,
                generated_filters=self.generated_filters,
            )
            if self.exclude_preprocessed_when_self_contained and _partition_contains_generated_negatives(sample_df):
                preprocessed_count = int(sample_df["source"].astype(str).eq("preprocessed").sum())
                sample_df = _keep_only_generated_rows(sample_df)
                logger.info(
                    f"Self-contained partition '{self.dataset_variant_tag}': dropped {preprocessed_count} "
                    f"preprocessed-healthy rows so the in-both-classes distractor stays non-predictive."
                )

        if "split" in sample_df.columns:
            train_pool = sample_df[sample_df["split"] == "training"]
            test_pool = sample_df[sample_df["split"] == "test"]
        else:
            train_pool = sample_df
            test_pool = pd.DataFrame(columns=sample_df.columns)

        if train_pool.empty:
            train_pool = sample_df
        train_df, val_df = _split_train_validation(
            dataframe=train_pool,
            val_size=self.val_size,
            seed=self.train_validation_split_seed,
        )

        if test_pool.empty:
            test_df = val_df.copy()
            if test_df.empty:
                test_df = train_df.copy()
        else:
            test_df = test_pool.reset_index(drop=True)

        dataset_kwargs = {
            "image_size": self.image_size,
            "use_patchify": self.use_patchify,
            "normalize_input": self.normalize_input,
            "normalize_mean": self.normalize_mean,
            "normalize_std": self.normalize_std,
        }
        self.train_set = ImageClassificationDataset(train_df.to_dict("records"), **dataset_kwargs)
        self.val_set = ImageClassificationDataset(val_df.to_dict("records"), **dataset_kwargs)
        self.test_set = ImageClassificationDataset(test_df.to_dict("records"), **dataset_kwargs)

        positive_ratio = float(sample_df["label"].mean())
        self.dataset_meta_data = {
            "data_dir": self.image_data_dir,
            "samples_path": samples_path,
            "num_total_samples": len(sample_df),
            "num_train_samples": len(train_df),
            "num_val_samples": len(val_df),
            "num_test_samples": len(test_df),
            "positive_ratio": positive_ratio,
            "dataset_variant_tag": self.dataset_variant_tag,
            "selected_generated_filters": self.generated_filters or {},
        }
        if "dataset_name" in sample_df.columns:
            self.dataset_meta_data["datasets"] = (
                sample_df["dataset_name"].dropna().value_counts().to_dict()
            )

        if not self._has_logged_subdataset_summary:
            subdataset_name = self.dataset_variant_tag or "full_dataset"
            selected_filters = self.generated_filters or {}
            logger.info(
                f"Using subdataset '{subdataset_name}' with {len(sample_df)} samples "
                f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}), "
                f"generated_filters={selected_filters}"
            )
            self._has_logged_subdataset_summary = True

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            raise RuntimeError("Data module has not been set up.")
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_set is None:
            raise RuntimeError("Data module has not been set up.")
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_set is None:
            raise RuntimeError("Data module has not been set up.")
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


STEM_TYPES = {"standard", "gentle", "gentle_strided"}
HEAD_POOLING_TYPES = {"avg", "max", "gem"}


def _build_stem(stem_type: str, in_channels: int, base_channels: int) -> nn.Module:
    normalized_stem = (stem_type or "standard").lower()
    if normalized_stem not in STEM_TYPES:
        supported = ", ".join(sorted(STEM_TYPES))
        raise ValueError(f"Unknown stem '{stem_type}'. Supported stems: {supported}")
    if in_channels > 1 and normalized_stem == "standard":
        normalized_stem = "gentle"
    if normalized_stem == "gentle":
        return nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False),
        )
    if normalized_stem == "gentle_strided":
        return nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False),
        )
    return nn.Sequential(
        nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(base_channels),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )


class GlobalMaxPool2d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.amax(dim=(-2, -1), keepdim=True)


class GeMPool2d(nn.Module):
    def __init__(self, initial_power: float = 3.0, epsilon: float = 1e-6):
        super().__init__()
        self.power = nn.Parameter(torch.tensor(float(initial_power)))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.clamp(min=self.epsilon).pow(self.power).mean(dim=(-2, -1), keepdim=True)
        return pooled.pow(1.0 / self.power)


def _build_head_pooling(head_pooling: str) -> nn.Module:
    normalized_pooling = (head_pooling or "avg").lower()
    if normalized_pooling == "avg":
        return nn.AdaptiveAvgPool2d((1, 1))
    if normalized_pooling == "max":
        return GlobalMaxPool2d()
    if normalized_pooling == "gem":
        return GeMPool2d()
    supported = ", ".join(sorted(HEAD_POOLING_TYPES))
    raise ValueError(f"Unknown head_pooling '{head_pooling}'. Supported poolings: {supported}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.ReLU(inplace=False)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu_out(out)


class ResNetBackbone(nn.Module):
    def __init__(self, base_channels: int = 32, blocks: list[int] | None = None,
                 in_channels: int = 1, stem: str = "standard", head_pooling: str = "avg"):
        super().__init__()
        blocks = blocks or [2, 2, 2, 2]
        normalized_blocks = [int(v) for v in blocks]
        if not normalized_blocks:
            raise ValueError("Model blocks configuration cannot be empty.")
        if any(block_count <= 0 for block_count in normalized_blocks):
            raise ValueError(
                f"All block counts must be positive integers, got: {normalized_blocks}"
            )

        self.stem = _build_stem(stem_type=stem, in_channels=in_channels, base_channels=base_channels)

        self.layers = nn.ModuleList()
        input_channels = base_channels
        for stage_index, block_count in enumerate(normalized_blocks):
            output_channels = base_channels * (2 ** stage_index)
            stride = 1 if stage_index == 0 else 2
            self.layers.append(
                self._make_layer(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_blocks=block_count,
                    stride=stride,
                )
            )
            input_channels = output_channels

        self.pool = _build_head_pooling(head_pooling=head_pooling)
        self.embedding_dim = input_channels

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return self.pool(x).flatten(1)


class ResNetLikeCNN(nn.Module):
    def __init__(self, base_channels: int = 32, blocks: list[int] | None = None, in_channels: int = 1,
                 stem: str = "standard", head_pooling: str = "avg"):
        super().__init__()
        self.backbone = ResNetBackbone(
            base_channels=base_channels, blocks=blocks, in_channels=in_channels,
            stem=stem, head_pooling=head_pooling,
        )
        self.classifier = nn.Linear(self.backbone.embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x)).squeeze(1)

    @property
    def head_module(self) -> nn.Module:
        return self.classifier


class PatchAggregateClassifier(nn.Module):
    def __init__(self, base_channels: int = 32, blocks: list[int] | None = None, head_pooling: str = "avg"):
        super().__init__()
        self.backbone = ResNetBackbone(base_channels=base_channels, blocks=blocks, in_channels=1,
                                       stem="gentle", head_pooling=head_pooling)
        self.classifier = nn.Linear(self.backbone.embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        patches = x.view(B * N, C, H, W)
        embeddings = self.backbone(patches)
        embeddings = embeddings.view(B, N, -1)
        image_embedding = embeddings.mean(dim=1)
        return self.classifier(image_embedding).squeeze(1)

    @property
    def head_module(self) -> nn.Module:
        return self.classifier


PRETRAINED_ARCHITECTURES = {"convnext_tiny", "resnet18", "efficientnet_b4"}


def _instantiate_pretrained_backbone(model_constructor, weights) -> nn.Module:
    if weights is None:
        return model_constructor(weights=None)
    try:
        return model_constructor(weights=weights)
    except Exception as error:
        logger.warning(
            f"Pretrained weights unavailable ({error}); falling back to random initialization. "
            "This is expected during XAI reconstruction, where checkpoint weights are loaded afterwards."
        )
        return model_constructor(weights=None)


def _build_pretrained_backbone(arch: str, dropout: float, pretrained: bool) -> nn.Module:
    normalized_arch = arch.lower()
    if normalized_arch == "convnext_tiny":
        backbone = _instantiate_pretrained_backbone(
            torchvision_models.convnext_tiny,
            ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None,
        )
        head_in_features = backbone.classifier[-1].in_features
        head_layers = list(backbone.classifier.children())[:-1]
        if dropout > 0.0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(head_in_features, 1))
        backbone.classifier = nn.Sequential(*head_layers)
        return backbone
    if normalized_arch == "resnet18":
        backbone = _instantiate_pretrained_backbone(
            torchvision_models.resnet18,
            ResNet18_Weights.DEFAULT if pretrained else None,
        )
        head_in_features = backbone.fc.in_features
        backbone.fc = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in_features, 1))
            if dropout > 0.0 else nn.Linear(head_in_features, 1)
        )
        return backbone
    if normalized_arch == "efficientnet_b4":
        backbone = _instantiate_pretrained_backbone(
            torchvision_models.efficientnet_b4,
            EfficientNet_B4_Weights.DEFAULT if pretrained else None,
        )
        head_in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout if dropout > 0.0 else 0.4),
            nn.Linear(head_in_features, 1),
        )
        return backbone
    supported = ", ".join(sorted(PRETRAINED_ARCHITECTURES))
    raise ValueError(f"Unknown arch '{arch}'. Supported architectures: {supported}")


class PretrainedMammoClassifier(nn.Module):
    def __init__(self, arch: str = "convnext_tiny", dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.arch = arch
        self.backbone = _build_pretrained_backbone(arch=arch, dropout=dropout, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x).flatten()

    @property
    def head_module(self) -> nn.Module:
        return self.backbone.fc if self.arch.lower() == "resnet18" else self.backbone.classifier


SUPPORTED_MODELS = {"ResNetLikeCNN", "PatchAggregateClassifier", "PretrainedMammoClassifier"}


def _build_model(
    model_name: str, base_channels: int, blocks: list[int], in_channels: int = 1,
    arch: str = "convnext_tiny", dropout: float = 0.0, pretrained: bool = True,
    stem: str = "standard", head_pooling: str = "avg",
) -> nn.Module:
    if model_name == "PatchAggregateClassifier":
        return PatchAggregateClassifier(base_channels=base_channels, blocks=blocks, head_pooling=head_pooling)
    if model_name == "ResNetLikeCNN":
        return ResNetLikeCNN(base_channels=base_channels, blocks=blocks, in_channels=in_channels,
                             stem=stem, head_pooling=head_pooling)
    if model_name == "PretrainedMammoClassifier":
        return PretrainedMammoClassifier(arch=arch, dropout=dropout, pretrained=pretrained)
    supported = ", ".join(sorted(SUPPORTED_MODELS))
    raise ValueError(f"Unknown model_name '{model_name}'. Supported models: {supported}")


def _split_head_and_backbone_parameters(model: nn.Module) -> tuple[list, list]:
    head_parameter_ids = {id(parameter) for parameter in model.head_module.parameters()}
    backbone_parameters = []
    head_parameters = []
    for parameter in model.parameters():
        if id(parameter) in head_parameter_ids:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    return backbone_parameters, head_parameters


def _two_phase_lr_lambda(base_lr: float, freeze_epochs: int, warmup_epochs: int,
                         total_epochs: int, lr_min: float):
    minimum_factor = lr_min / base_lr
    phase_two_epochs = max(1, total_epochs - freeze_epochs)
    warmup = max(1, warmup_epochs)

    def lr_factor(epoch: int) -> float:
        if epoch < freeze_epochs:
            return 1.0
        phase_two_epoch = epoch - freeze_epochs
        if phase_two_epoch < warmup:
            return (phase_two_epoch + 1) / warmup
        progress = (phase_two_epoch - warmup) / max(1, phase_two_epochs - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return lr_factor


class LitBinaryClassifier(L.LightningModule):
    def __init__(self, lr: float, weight_decay: float, base_channels: int, blocks: list[int],
                 in_channels: int = 1, use_patchify: bool = False, model_name: str = "",
                 arch: str = "convnext_tiny", dropout: float = 0.0, pretrained: bool = True,
                 finetuning: dict | None = None, stem: str = "standard", head_pooling: str = "avg"):
        super().__init__()
        self.save_hyperparameters()
        resolved_model_name = model_name or ("PatchAggregateClassifier" if use_patchify else "ResNetLikeCNN")
        self.model = _build_model(
            model_name=resolved_model_name,
            base_channels=base_channels,
            blocks=blocks,
            in_channels=in_channels,
            arch=arch,
            dropout=dropout,
            pretrained=pretrained,
            stem=stem,
            head_pooling=head_pooling,
        )
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == y).float().mean()
        self.log(
            f"{stage}_loss",
            loss,
            on_epoch=True,
            on_step=(stage == "train"),
            prog_bar=(stage != "train"),
            batch_size=x.shape[0],
        )
        self.log(
            f"{stage}_acc",
            accuracy,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch=batch, stage="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch=batch, stage="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch=batch, stage="test")

    def on_train_epoch_start(self) -> None:
        finetuning = self.hparams.finetuning
        if not (isinstance(finetuning, dict) and finetuning.get("enabled", False)):
            return
        freeze_epochs = int(finetuning.get("freeze_epochs", 5))
        backbone_unfrozen = self.current_epoch >= freeze_epochs
        backbone_parameters, _ = _split_head_and_backbone_parameters(self.model)
        for parameter in backbone_parameters:
            parameter.requires_grad_(backbone_unfrozen)
        if not backbone_unfrozen:
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

    def configure_optimizers(self) -> dict:
        finetuning = self.hparams.finetuning
        if isinstance(finetuning, dict) and finetuning.get("enabled", False):
            return self._configure_two_phase_optimizers(finetuning)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _configure_two_phase_optimizers(self, finetuning: dict) -> dict:
        if not isinstance(self.model, PretrainedMammoClassifier):
            logger.warning(
                "Two-phase fine-tuning is enabled for a from-scratch model "
                f"('{type(self.model).__name__}'). It only benefits pretrained backbones; the "
                "head-only warmup phase is meaningless when the backbone is randomly initialized."
            )
        freeze_epochs = int(finetuning.get("freeze_epochs", 5))
        warmup_epochs = int(finetuning.get("warmup_epochs", 3))
        lr_head = float(finetuning.get("lr_head", 1e-3))
        lr_backbone = float(finetuning.get("lr_backbone", 3e-4))
        lr_min = float(finetuning.get("lr_min", 1e-6))
        total_epochs = int(self.trainer.max_epochs)

        backbone_parameters, head_parameters = _split_head_and_backbone_parameters(self.model)
        optimizer = torch.optim.AdamW([
            {"params": backbone_parameters, "lr": lr_backbone, "weight_decay": self.hparams.weight_decay},
            {"params": head_parameters, "lr": lr_head, "weight_decay": 0.0},
        ])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                _two_phase_lr_lambda(lr_backbone, freeze_epochs, warmup_epochs, total_epochs, lr_min),
                _two_phase_lr_lambda(lr_head, freeze_epochs, warmup_epochs, total_epochs, lr_min),
            ],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def _build_wandb_logger(
    wandb_config: dict,
    experiment_name: str,
    run_name: str,
    training_params: dict,
    model_params: dict,
    dataset_variant_description: str,
    seed: int,
) -> WandbLogger | None:
    if not wandb_config.get("enabled", False):
        return None

    wandb_logger = WandbLogger(
        project=wandb_config.get("project", "claim"),
        entity=wandb_config.get("entity"),
        name=run_name,
        group=experiment_name,
        tags=wandb_config.get("tags", []),
        config={
            "training_params": training_params,
            "model_params": model_params,
            "dataset_variant": dataset_variant_description,
            "seed": seed,
        },
    )
    return wandb_logger


def _resolve_trainer_accelerator(device_preference: str) -> str:
    normalized = (device_preference or "auto").lower()
    if normalized == "cuda":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if normalized == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if normalized in {"cpu", "gpu", "auto"}:
        return normalized
    return "auto"


def _generate_training_parameter_sets(
    training_config: dict, base_training_params: dict, base_model_params: dict,
) -> list[tuple[dict, dict]]:
    if not training_config.get("use_hyperparameter_tuning", False):
        return [(base_training_params, base_model_params)]

    tuning_config = training_config.get("hyperparameter_tuning", {})
    search_space = {
        "batch_size": tuning_config.get("batch_sizes"),
        "lr": tuning_config.get("learning_rates"),
        "weight_decay": tuning_config.get("weight_decays"),
    }
    model_search_space = {
        "base_channels": tuning_config.get("base_channels"),
    }
    active_space = {key: values for key, values in search_space.items() if values}
    active_model_space = {key: values for key, values in model_search_space.items() if values}
    if not active_space and not active_model_space:
        return [(base_training_params, base_model_params)]

    all_names = list(active_space.keys()) + list(active_model_space.keys())
    all_values = [active_space[k] for k in active_space] + [active_model_space[k] for k in active_model_space]
    training_param_names = set(active_space.keys())

    parameter_sets = []
    for combination in product(*all_values):
        updated_training = dict(base_training_params)
        updated_model = dict(base_model_params)
        for name, value in zip(all_names, combination):
            if name in training_param_names:
                updated_training[name] = value
            else:
                updated_model[name] = value
        parameter_sets.append((updated_training, updated_model))
    return parameter_sets


def _resolve_model_blocks(model_params: dict) -> list[int]:
    raw_blocks = model_params.get("blocks", [2, 2, 2, 2])
    if raw_blocks is None:
        return [2, 2, 2, 2]
    if not isinstance(raw_blocks, (list, tuple)):
        raise ValueError(
            f"model_params.blocks must be a list of positive integers, got: {raw_blocks}"
        )

    blocks = [int(value) for value in raw_blocks]
    if not blocks:
        raise ValueError("model_params.blocks cannot be empty.")
    if any(value <= 0 for value in blocks):
        raise ValueError(
            f"model_params.blocks must contain only positive integers, got: {blocks}"
        )
    return blocks


def _load_experiment_config() -> ExperimentConfig:
    config_path = os.environ.get(
        EnvironmentVariables.CONFIG_FILE_PATH.value,
        "config/experiments_config.yaml",
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    return ExperimentConfig.from_dict(load_config_file(file_path=config_path))


def _dump_records(records: list[TrainingRecord], output_path: str) -> None:
    dump_as_jsonl_file(data=[asdict(record) for record in records], file_path=output_path)


def main() -> None:
    load_dotenv()
    config = _load_experiment_config()
    set_random_states(config.seed)

    experiment_dir = generate_experiment_dir(config=config)
    experiment_dir_resolved = str(Path(experiment_dir).resolve())
    training_dir = generate_training_dir(base_dir=experiment_dir, training_config=config.training)
    models_dir = join(training_dir, "models")
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    training_records_path = join(training_dir, config.training["training_records"])
    intermediate_records_path = join(
        training_dir, config.training["intermediate_training_records"]
    )
    if os.path.exists(intermediate_records_path):
        records = load_training_records(intermediate_records_path)
    elif os.path.exists(training_records_path):
        records = load_training_records(training_records_path)
    else:
        records = []

    completed_run_names = {
        Path(record.training_log_path).parent.name for record in records
    }
    if completed_run_names:
        logger.info(
            f"Resuming: {len(completed_run_names)} completed runs found, will be skipped"
        )

    data_dir = generate_data_dir(config=config)
    image_data_dir = _resolve_configured_data_dir(config=config, fallback_data_dir=data_dir)
    samples_filename = config.data.get("samples_filename", "dataset_metadata.jsonl")
    dataset_variants = _resolve_dataset_variants(
        training_config=config.training,
        data_dir=experiment_dir,
        samples_filename=samples_filename,
    )
    if len(dataset_variants) == 1 and not dataset_variants[0]["generated_filters"]:
        logger.info("Running training without generated-sample filtering")
    else:
        variant_descriptions = [variant["description"] for variant in dataset_variants]
        logger.info(f"Running training for dataset variants: {variant_descriptions}")

    model_configs = config.training.get("models", [])
    if not model_configs:
        raise ValueError("No model configuration found in config.training.models")

    repetition_count = int(config.training.get("num_training_repetitions", 1))
    configured_gradient_clip = config.training.get("gradient_clip_val")
    gradient_clip_val = float(configured_gradient_clip) if configured_gradient_clip else None

    total_training_runs = 0
    for mc in model_configs:
        ps = _generate_training_parameter_sets(
            training_config=config.training,
            base_training_params=dict(mc.get("training_params", {})),
            base_model_params=dict(mc.get("model_params", {})),
        )
        total_training_runs += len(ps)
    total_training_runs *= len(dataset_variants) * repetition_count

    if config.training.get("use_hyperparameter_tuning", False):
        logger.info(
            f"Total models to train: {total_training_runs} "
            f"(hyperparameter tuning enabled, {len(dataset_variants)} dataset variants, "
            f"{len(model_configs)} model configs, {repetition_count} repetitions)"
        )
    else:
        logger.info(
            f"Total models to train: {total_training_runs} "
            f"({len(dataset_variants)} dataset variants, {len(model_configs)} model configs, "
            f"{repetition_count} repetitions)"
        )

    for variant in dataset_variants:
        variant_tag = variant["tag"]
        variant_description = variant["description"]
        generated_filters = variant["generated_filters"]
        for model_config in model_configs:
            dataset_type = model_config.get("dataset_type", "vindrmammo")
            model_name = model_config.get("model_name", "ResNetLikeCNN")
            base_model_params = dict(model_config.get("model_params", {}))
            base_training_params = dict(model_config.get("training_params", {}))
            parameter_sets = _generate_training_parameter_sets(
                training_config=config.training,
                base_training_params=base_training_params,
                base_model_params=base_model_params,
            )

            for parameter_set_index, (training_params, model_params) in enumerate(parameter_sets):
                batch_size = int(training_params.get("batch_size", 8))
                epochs = int(training_params.get("epochs", 10))
                lr = float(training_params.get("lr", 1e-3))
                weight_decay = float(training_params.get("weight_decay", 0.0))
                blocks = _resolve_model_blocks(model_params=model_params)
                image_size = int(model_params.get("image_size", 224))
                use_patchify = bool(model_params.get("use_patchify", False))
                in_channels = 1
                base_channels = int(model_params.get("base_channels", 32))
                normalize_input = bool(model_params.get("normalize_input", False))
                normalize_mean = float(model_params.get("normalize_mean", 0.5))
                normalize_std = float(model_params.get("normalize_std", 0.5))
                arch = str(model_params.get("arch", "convnext_tiny"))
                dropout = float(model_params.get("dropout", 0.0))
                pretrained = bool(model_params.get("pretrained", True))
                stem = str(model_params.get("stem", "standard"))
                head_pooling = str(model_params.get("head_pooling", "avg"))
                finetuning = training_params.get("finetuning")

                for repetition in range(repetition_count):
                    variant_suffix = f"-{variant_tag}" if variant_tag else ""
                    run_name = (
                        f"{dataset_type}-{model_name}{variant_suffix}-params{parameter_set_index}-rep{repetition}"
                    )

                    if run_name in completed_run_names:
                        logger.info(f"Skipping already completed run: {run_name}")
                        continue

                    run_seed = int(config.seed + repetition)
                    set_random_states(seed=run_seed)
                    datamodule = JsonlDataModule(
                        data_dir=experiment_dir,
                        image_data_dir=image_data_dir,
                        batch_size=batch_size,
                        num_workers=int(config.training.get("num_workers", 0)),
                        val_size=float(config.training.get("val_size", 0.2)),
                        samples_filename=samples_filename,
                        image_size=image_size,
                        train_validation_split_seed=int(config.seed),
                        generated_filters=generated_filters,
                        dataset_variant_tag=variant_tag or "full_dataset",
                        use_patchify=use_patchify,
                        normalize_input=normalize_input,
                        normalize_mean=normalize_mean,
                        normalize_std=normalize_std,
                        exclude_preprocessed_when_self_contained=bool(
                            config.training.get("exclude_preprocessed_when_self_contained", False)
                        ),
                    )
                    model = LitBinaryClassifier(
                        model_name=model_name,
                        lr=lr,
                        weight_decay=weight_decay,
                        base_channels=base_channels,
                        blocks=blocks,
                        in_channels=in_channels,
                        use_patchify=use_patchify,
                        arch=arch,
                        dropout=dropout,
                        pretrained=pretrained,
                        finetuning=finetuning,
                        stem=stem,
                        head_pooling=head_pooling,
                    )

                    logger.info(f"Starting training run: {run_name} ({variant_description})")

                    csv_logger = CSVLogger(
                        save_dir=training_dir,
                        name="lightning_logs",
                        version=run_name,
                    )
                    wandb_config = config.training.get("wandb", {})
                    wandb_logger = _build_wandb_logger(
                        wandb_config=wandb_config,
                        experiment_name=config.experiment_name,
                        run_name=run_name,
                        training_params=training_params,
                        model_params=model_params,
                        dataset_variant_description=variant_description,
                        seed=run_seed,
                    )
                    loggers = [csv_logger]
                    if wandb_logger:
                        loggers.append(wandb_logger)

                    checkpoint_callback = ModelCheckpoint(
                        dirpath=models_dir,
                        filename=f"{run_name}-best",
                        monitor="val_loss",
                        mode="min",
                        save_top_k=1,
                    )
                    trainer = L.Trainer(
                        max_epochs=epochs,
                        accelerator=_resolve_trainer_accelerator(config.training.get("device", "auto")),
                        devices=1,
                        logger=loggers,
                        callbacks=[checkpoint_callback],
                        deterministic=True,
                        gradient_clip_val=gradient_clip_val,
                        log_every_n_steps=10,
                        enable_progress_bar=True,
                    )
                    trainer.fit(model=model, datamodule=datamodule)
                    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

                    if wandb_logger:
                        wandb_logger.experiment.finish()

                    records.append(
                        TrainingRecord(
                            batch_size=batch_size,
                            epochs=epochs,
                            model_name=model_name,
                            model_path=os.path.relpath(checkpoint_callback.best_model_path, experiment_dir_resolved),
                            training_log_path=os.path.relpath(join(csv_logger.log_dir, LIGHTNING_LOGS_FILENAME), experiment_dir_resolved),
                            repetition=repetition,
                            seed=run_seed,
                            dataset_meta_data={
                                **datamodule.dataset_meta_data,
                                "dataset_type": dataset_type,
                            },
                            model_params={
                                **model_params,
                                "in_channels": in_channels,
                                "training_params": training_params,
                            },
                        )
                    )
                    _dump_records(records=records, output_path=intermediate_records_path)

    _dump_records(records=records, output_path=training_records_path)


if __name__ == "__main__":
    main()
