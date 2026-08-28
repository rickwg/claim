from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum

DATA_CONFIG_FILENAME = "data_config.yaml"
EXPERIMENT_CONFIG_FILENAME = "experiment_config.yaml"
EXPERIMENT_CONFIG_PATH = "configs/experiment_config.yaml"
DATASET_METADATA_FILENAME = "dataset_metadata.json"
LIGHTNING_LOGS_FILENAME = "metrics.csv"
TETRIS_DATASET_TYPE = "tetris"



class EnvironmentVariables(Enum):
    CONFIG_FILE_PATH = "FILE_PATH_TO_EXPERIMENT_CONFIG"
    PREPROCESS_CONFIG_FILE_PATH = "FILE_PATH_TO_PREPROCESS_CONFIG"
    DATA_CONFIG_FILE_PATH = "FILE_PATH_TO_DATA_CONFIG"


@dataclass
class TrainingRecord:
    batch_size: int
    epochs: int
    model_name: str
    model_path: str
    training_log_path: str
    repetition: int
    seed: int
    dataset_meta_data: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)

