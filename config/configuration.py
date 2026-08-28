from dataclasses import field, dataclass


@dataclass
class ExperimentConfig:
    seed: int
    created: str
    artifacts_dir: str
    project_dir: str
    experiment_name: str
    data: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    xai: dict = field(default_factory=dict)
    xai_evaluation: dict = field(default_factory=dict)
    analyses: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        return cls(**config_dict)

    # Backward compatibility alias
    @classmethod
    def get(cls, input_conf: dict) -> 'ExperimentConfig':
        return cls.from_dict(input_conf)
