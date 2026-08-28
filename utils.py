import json
import os
import platform
from datetime import datetime
from os.path import join, split
from pathlib import Path
import random
from typing import Any, Dict, List
import pickle

import numpy as np
import torch
import yaml
from PIL import Image
from numpy.random import Generator
from torch.utils.data import TensorDataset, random_split

from common import DATA_CONFIG_FILENAME, TrainingRecord
from config.configuration import ExperimentConfig

CLUSTER_PLATFORM_NAME = "#102-Ubuntu"
CLUSTER_DATA_DIR = "/mnt"


def load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as file:
        return pickle.load(file)


def dump_as_pickle(data: Any, file_path: str) -> None:
    output_dir = join(*split(file_path)[:-1])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_json_file(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        file = json.load(f)
    return file


def dump_as_json_file(data: Dict, file_path: str) -> None:
    output_dir = join(*split(file_path)[:-1])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(obj=data, fp=f, indent=4)


def dump_as_jsonl_file(data: list[dict], file_path: str) -> None:
    output_dir = join(*split(file_path)[:-1])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def append_to_jsonl_file(record: dict, file_path: str) -> None:
    output_dir = join(*split(file_path)[:-1])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        json.dump(record, f)
        f.write("\n")


def load_jsonl_file(file_path: str, *, skip_corrupt_lines: bool = False) -> list[dict]:
    with open(file_path, "r") as f:
        if not skip_corrupt_lines:
            return [json.loads(line) for line in f]
        output = []
        for line in f:
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return output


def load_yaml_file(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        file = yaml.safe_load(f)
    return file


def dump_as_yaml_file(data: Dict, file_path: str) -> None:
    output_dir = join(*split(file_path)[:-1])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_config_file(file_path: str) -> Dict:
    """Load config file supporting both JSON and YAML formats."""
    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
        return load_yaml_file(file_path)
    elif file_path.endswith('.json'):
        return load_json_file(file_path)
    else:
        # Try YAML first, then JSON
        try:
            return load_yaml_file(file_path)
        except yaml.YAMLError:
            return load_json_file(file_path)



def is_image_data(x: np.ndarray) -> bool:
    return ".png" in str(x[0]) or ".jpg" in str(x[0]) or ".jpeg" in str(x[0])


def load_gray_scale_image(relative_path: str, config: ExperimentConfig) -> Image.Image:
    image_path = join(generate_data_dir(config), relative_path)
    return Image.open(image_path).convert("L")


def to_float_array(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return a.astype(np.float32)


def load_image_data(x: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    output = list()
    for path in x:
        img = load_gray_scale_image(relative_path=path, config=config)
        img = to_float_array(a=np.array(img))
        output.append(np.array(img))
    return np.array(output)


def load_dataset(
    file_path: str, return_ground_truth_explanation: bool = False
) -> tuple:
    x, y, explanation = load_pickle(file_path=file_path)
    return (x, y, explanation) if return_ground_truth_explanation else (x, y)


def load_training_records(file_path: str) -> list[TrainingRecord]:
    with open(file_path, "r") as f:
        output = [TrainingRecord(**json.loads(line)) for line in f]
    return output


def today_formatted() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def append_date(s: str) -> str:
    return f"{s}-{today_formatted()}"


def get_root_dir_based_on_platform() -> str:
    return CLUSTER_DATA_DIR if CLUSTER_PLATFORM_NAME in platform.version() else ""


def is_hydra():
    return os.environ.get("SLURM_WORKING_CLUSTER", "").startswith("hydra")


def generate_experiment_dir(config: ExperimentConfig) -> str:
    return join(
        config.artifacts_dir,
        f"{config.experiment_name}-{config.created}",
    )


def _resolve_path_from_project_dir(path_str: str, project_dir: str) -> str:
    expanded = os.path.expanduser(path_str)
    if os.path.isabs(expanded):
        return expanded
    return join(project_dir, expanded)


def generate_data_dir(config: ExperimentConfig) -> str:
    data_dir = config.data.get("data_dir") or config.data.get("base_dir")
    if isinstance(data_dir, str) and data_dir.strip():
        return _resolve_path_from_project_dir(path_str=data_dir, project_dir=config.project_dir)

    if is_hydra():
        d = join(
            "/mnt/data",
            config.data["data_scenario"],
        )
    else:
        d = join(config.data["data_dir"], config.data["data_scenario"])

    return d


def generate_datasets_paths(config: ExperimentConfig) -> list[tuple]:
    data_dir = generate_data_dir(config=config)
    dataset_config = load_config_file(file_path=join(data_dir, DATA_CONFIG_FILENAME))
    output = list()
    for dataset in dataset_config["datasets"]:
        training = join(
            data_dir,
            dataset["dataset_type"],
            dataset_config["output_filenames"]["training"],
        )
        test = join(
            data_dir,
            dataset["dataset_type"],
            dataset_config["output_filenames"]["test"],
        )
        meta_data = join(
            data_dir,
            dataset["dataset_type"],
            dataset_config["output_filenames"]["meta_data"],
        )

        output.append((training, test, meta_data))
    return output


def generate_dataset_paths(config: ExperimentConfig, dataset_type: str) -> tuple:
    data_dir = generate_data_dir(config=config)
    dataset_config = load_config_file(file_path=join(data_dir, DATA_CONFIG_FILENAME))
    training = join(
        data_dir,
        dataset_type,
        dataset_config["output_filenames"]["training"],
    )
    test = join(
        data_dir,
        dataset_type,
        dataset_config["output_filenames"]["test"],
    )
    meta_data = join(
        data_dir,
        dataset_type,
        dataset_config["output_filenames"]["meta_data"],
    )

    return training, test, meta_data


def generate_training_dir(base_dir: str, training_config: dict) -> str:
    return join(
        base_dir,
        training_config["output_dir"],
    )


def generate_xai_dir(config: ExperimentConfig) -> str:
    return config.xai["output_dir"]


def generate_evaluation_dir(config: ExperimentConfig) -> str:
    return config.xai_evaluation["output_dir"]


def generate_analyses_dir(config: ExperimentConfig) -> str:
    return config.analyses["output_dir"]


def generate_dataset_type(dataset_config: dict) -> str:
    return (
        f"{dataset_config['dataset_type']}"
        f"-i{dataset_config['params']['num_informative_features']}"
        f"-s{dataset_config['params']['num_suppressor_features']}"
    )


def get_dataset_type(dataset_config: dict) -> str:
    return dataset_config["dataset_type"].split("-")[0]


def set_random_states(seed: int) -> Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed)


def create_train_val_split(data: TensorDataset, val_size: float) -> List:
    num_samples = len(data)
    num_val_samples = int(val_size * num_samples)
    num_train_samples = num_samples - num_val_samples
    return random_split(dataset=data, lengths=[num_train_samples, num_val_samples])


def numpy_to_tensor(a: np.array) -> np.array:
    return torch.from_numpy(a).float()


def tensor_to_numpy(a: torch.Tensor) -> np.array:
    try:
        if isinstance(a, np.ndarray):
            output = a
        else:
            output = a.numpy()
    except RuntimeError:
        output = a.detach().numpy()
    return output
