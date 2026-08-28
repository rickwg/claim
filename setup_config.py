"""
Generate pipeline setup configuration bundles.

Supported setup modes:
  - data: generate local + cluster configs for data generation stages.
  - experiment: generate local + cluster configs for training split + experiments.

Usage:
    python setup_config.py --setup data
    python setup_config.py --setup experiment
    python setup_config.py --setup experiment --config path/to/experiments_config.yaml

By default each mode bundles its built-in config list (DATA_SETUP_CONFIGS /
EXPERIMENT_SETUP_CONFIGS). Pass --config/-c with one or more paths to override
that list (e.g. to bundle an integration-test config).
"""

import argparse
from datetime import datetime
from pathlib import Path

import yaml


# Data-generation setup (preprocessing + synthetic generation)
DATA_SETUP_CONFIGS = [
    "config/preprocess_config_mame.yaml",
    "config/preprocess_config.yaml",
    "config/generate_masked_distortion.yaml",
    "config/generate_lesions.yaml",
    "config/generate_combined.yaml",
]

# Experiment setup (training split + experiment runs)
EXPERIMENT_SETUP_CONFIGS = [
    "config/experiments_config.yaml",
]

# Cluster container bind mounts (from scripts/*_run_prebuild*.sh):
#   $DATA_DIR:/mnt/data
#   $ARTIFACT_DIR:/mnt/artifacts
#   $PROJECT_DIR:/workdir
CLUSTER_MOUNT_DATA = "/mnt/data"
CLUSTER_MOUNT_ARTIFACTS = "/mnt/artifacts"
CLUSTER_MOUNT_WORKDIR = "/workdir"

# Local path prefixes (relative to project root) → cluster mount points.
# More specific prefixes must come before less specific ones.
# Edit these to match your project layout.
CLUSTER_PATH_MAPS = [
    ("data/vindrmammo_data", CLUSTER_MOUNT_DATA),
    ("vindrmammo_data", CLUSTER_MOUNT_DATA),
    ("artifacts", CLUSTER_MOUNT_ARTIFACTS),
]

# Config keys whose values are file/directory paths to remap
PATH_KEYS = {
    "input_dir", "output_dir", "mask_output_dir",
    "image_dir", "mask_dir",
    "preprocessed_image_dir", "preprocessed_mask_dir",
    "annotations_file", "annotations_path",
    "output_path",
    "artifacts_dir", "project_dir",
    "data_dir", "base_dir",
}

# Config keys whose values are lists of paths
PATH_LIST_KEYS = {
    "generated_metadata_paths",
}

CLUSTER_CONFIG_OVERRIDES = {
    "generate_masked_distortion.yaml": {
        "input_dir": f"{CLUSTER_MOUNT_DATA}/mame_preprocessed",
        "output_dir": f"{CLUSTER_MOUNT_DATA}/synthetic_{{transformation}}",
    },
    "generate_lesions.yaml": {
        "input_dir": f"{CLUSTER_MOUNT_DATA}/mame_preprocessed",
        "output_dir": f"{CLUSTER_MOUNT_DATA}/synthetic_lesions",
    },
    "generate_combined.yaml": {
        "input_dir": f"{CLUSTER_MOUNT_DATA}/mame_preprocessed",
        "output_dir": f"{CLUSTER_MOUNT_DATA}/synthetic_combined/{{distortion_type}}/{{classification_target}}",
    },
}


def _today_formatted() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_yaml(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _remap_path(path_str: str, project_root: Path) -> str:
    """Remap a local path to its cluster container mount equivalent.

    Resolves the path to absolute (using project_root for relative paths),
    then checks against CLUSTER_PATH_MAPS. Returns the original string
    if no mapping matches.
    """
    if not path_str or path_str == ".":
        return CLUSTER_MOUNT_WORKDIR

    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()

    for local_prefix, cluster_mount in CLUSTER_PATH_MAPS:
        local_abs = (project_root / local_prefix).resolve()
        try:
            remainder = p.relative_to(local_abs)
            suffix = str(remainder)
            if suffix == ".":
                return cluster_mount
            return f"{cluster_mount}/{suffix}"
        except ValueError:
            continue

    return path_str


def _remap_config(config: dict, project_root: Path) -> dict:
    """Recursively remap path-valued keys in a config dict for cluster."""
    result = {}
    for key, value in config.items():
        if key in PATH_KEYS and isinstance(value, str):
            result[key] = _remap_path(value, project_root)
        elif key in PATH_LIST_KEYS and isinstance(value, list):
            result[key] = [
                _remap_path(v, project_root) if isinstance(v, str) else v
                for v in value
            ]
        elif isinstance(value, dict):
            result[key] = _remap_config(value, project_root)
        else:
            result[key] = value
    return result


def process_configs(
    configs: list[str],
    environment: str,
    project_root: Path,
    output_dir: Path | None = None,
    created: str | None = None,
) -> tuple[Path, int]:
    """Process config files and save them for one environment."""
    timestamp = created or _today_formatted()
    target_dir = output_dir or (project_root / "artifacts" / "configs" / f"{environment}-{timestamp}")
    target_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for config_path in configs:
        if not Path(config_path).exists():
            print(f"  SKIP  {config_path} (not found)")
            continue

        config = _load_yaml(config_path) or {}
        config["created"] = timestamp

        if environment == "cluster":
            config = _remap_config(config, project_root)
            config_name = Path(config_path).name
            path_overrides = CLUSTER_CONFIG_OVERRIDES.get(config_name, {})
            for override_key, override_value in path_overrides.items():
                if isinstance(override_value, str):
                    config[override_key] = override_value

        out_path = target_dir / Path(config_path).name
        _save_yaml(config, str(out_path))
        print(f"  {config_path} → {out_path}")
        processed_count += 1

    return target_dir, processed_count


def _path_relative_to_project_root(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _set_experiment_samples_defaults(
    experiments_config_path: Path,
    *,
    samples_filename: str = "training_split_metadata.jsonl",
    data_dir: str | None = None,
) -> None:
    if not Path(experiments_config_path).exists():
        return
    config = _load_yaml(str(experiments_config_path)) or {}
    data_config = config.get("data", {})
    if not isinstance(data_config, dict):
        data_config = {}
    if isinstance(data_dir, str) and data_dir.strip():
        data_config["data_dir"] = data_dir
    data_config["samples_filename"] = samples_filename
    config["data"] = data_config
    _save_yaml(config, str(experiments_config_path))


def create_data_setup(
    project_root: Path, configs: list[str] | None = None,
) -> dict[str, tuple[Path, int]]:
    """Create local + cluster data-generation setup configs."""
    configs = configs or DATA_SETUP_CONFIGS
    timestamp = _today_formatted()
    output_root = project_root / "artifacts" / "data"
    outputs: dict[str, tuple[Path, int]] = {}
    for environment in ("local", "cluster"):
        print(f"\nData setup ({environment}):")
        environment_output_dir = output_root / f"{environment}-{timestamp}"
        outputs[environment] = process_configs(
            configs=configs,
            environment=environment,
            project_root=project_root,
            output_dir=environment_output_dir,
            created=timestamp,
        )
    return outputs


def _read_experiment_name(config_path: str) -> str:
    config = _load_yaml(config_path) or {}
    experiment_name = config.get("experiment_name")
    if isinstance(experiment_name, str) and experiment_name.strip():
        return experiment_name.strip()
    return "claim"


def create_experiment_setup(
    project_root: Path, configs: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Bundle each experiment config under its own `<experiment_name>-<timestamp>`
    directory so the local + cluster configs share the experiment directory that
    the pipeline stages resolve from `experiment_name` + `created`
    (see utils.generate_experiment_dir), instead of a separate `claim-<timestamp>`."""
    configs = configs or EXPERIMENT_SETUP_CONFIGS
    timestamp = _today_formatted()
    setups: list[tuple[str, Path]] = []

    for config_path in configs:
        if not Path(config_path).exists():
            print(f"  SKIP  {config_path} (not found)")
            continue

        experiment_name = _read_experiment_name(config_path)
        setup_dir = project_root / "artifacts" / f"{experiment_name}-{timestamp}"
        setup_dir.mkdir(parents=True, exist_ok=True)
        config_name = Path(config_path).name

        print(f"\nExperiment setup '{config_name}' → {setup_dir.name} (local):")
        process_configs(
            configs=[config_path],
            environment="local",
            project_root=project_root,
            output_dir=setup_dir,
            created=timestamp,
        )
        _set_experiment_samples_defaults(
            experiments_config_path=setup_dir / config_name,
            samples_filename="training_split_metadata.jsonl",
        )

        cluster_dir = setup_dir / "cluster"
        print(f"Experiment setup '{config_name}' → {setup_dir.name}/cluster (cluster):")
        process_configs(
            configs=[config_path],
            environment="cluster",
            project_root=project_root,
            output_dir=cluster_dir,
            created=timestamp,
        )
        _set_experiment_samples_defaults(
            experiments_config_path=cluster_dir / config_name,
            data_dir=CLUSTER_MOUNT_DATA,
            samples_filename="training_split_metadata.jsonl",
        )

        setups.append((config_path, setup_dir))

    return setups


def main():
    parser = argparse.ArgumentParser(
        description="Generate local+cluster setup configuration bundles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python setup_config.py --setup data
  python setup_config.py --setup experiment
  python setup_config.py --setup experiment --config config/experiments_config.integration.yaml
  python setup_config.py --setup data -c config/preprocess_config_mame.yaml config/generate_lesions.yaml

By default each mode bundles its built-in config list. Pass --config/-c with one
or more paths to override that list.

cluster path mappings (edit CLUSTER_PATH_MAPS in this script to customize):
  data/vindrmammo_data/  →  /mnt/data/
  vindrmammo_data/       →  /mnt/data/
  artifacts/             →  /mnt/artifacts/
  . (project root)       →  /workdir
""",
    )
    parser.add_argument(
        "--setup", choices=["data", "experiment"], required=True,
        help="Setup mode: data (local+cluster generation), experiment (claim local+cluster)",
    )
    parser.add_argument(
        "--config", "-c", dest="configs", nargs="+", metavar="PATH", default=None,
        help="One or more config YAML paths to bundle. "
             "Defaults to the built-in config list for the chosen --setup mode.",
    )
    args = parser.parse_args()

    if args.configs:
        missing_configs = [path for path in args.configs if not Path(path).exists()]
        if missing_configs:
            parser.error("config file(s) not found: " + ", ".join(missing_configs))

    project_root = Path.cwd().resolve()

    print(f"Project root: {project_root}")
    print()

    if args.setup == "data":
        print(f"Mode: {args.setup}")
        outputs = create_data_setup(project_root, configs=args.configs)
        local_dir, local_count = outputs["local"]
        cluster_dir, cluster_count = outputs["cluster"]
        print(
            f"\nData setup complete:\n"
            f"  local   ({local_count} config(s)) → {local_dir}\n"
            f"  cluster ({cluster_count} config(s)) → {cluster_dir}"
        )
        return

    print(f"Mode: {args.setup}")
    setups = create_experiment_setup(project_root, configs=args.configs)
    if not setups:
        print("\nNo experiment configs were bundled.")
        return
    print(f"\nExperiment setup complete ({len(setups)} config(s)):")
    for config_path, setup_dir in setups:
        config_name = Path(config_path).name
        print(
            f"  {config_name} → {setup_dir}\n"
            f"      local   → {setup_dir / config_name}\n"
            f"      cluster → {setup_dir / 'cluster' / config_name}"
        )


if __name__ == "__main__":
    main()
