import argparse
import os
import shlex
import subprocess
from pathlib import Path

import yaml

SCRIPT_PATH = Path(__file__).resolve().parent
BUILD_SCRIPT = SCRIPT_PATH / "build.sh"
RUN_GPU_SCRIPT = SCRIPT_PATH / "gpu_run_prebuild.sh"
RUN_GPU_SCRIPT_SQFS = SCRIPT_PATH / "gpu_run_prebuild_sqfs.sh"
RUN_CPU_SCRIPT = SCRIPT_PATH / "cpu_run_prebuild.sh"
RUN_CPU_SCRIPT_SQFS = SCRIPT_PATH / "cpu_run_prebuild_sqfs.sh"
TEST_GPU_SCRIPT = SCRIPT_PATH / "gpu_test_prebuild.sh"
TEST_GPU_SCRIPT_SQFS = SCRIPT_PATH / "gpu_test_prebuild_sqfs.sh"
TEST_CPU_SCRIPT = SCRIPT_PATH / "cpu_test_prebuild.sh"
TEST_CPU_SCRIPT_SQFS = SCRIPT_PATH / "cpu_test_prebuild_sqfs.sh"
# Fallback cluster data directory when neither --data-dir nor the config sets one.
DATA_DIR_CLUSTER = os.environ.get("CLUSTER_DATA_DIR", "")
RUN_SCRIPTS = {
    "gpu": RUN_GPU_SCRIPT,
    "gpu-sqfs": RUN_GPU_SCRIPT_SQFS,
    "cpu": RUN_CPU_SCRIPT,
    "cpu-sqfs": RUN_CPU_SCRIPT_SQFS,
    "gpu-test": TEST_GPU_SCRIPT,
    "gpu-test-sqfs": TEST_GPU_SCRIPT_SQFS,
    "cpu-test": TEST_CPU_SCRIPT,
    "cpu-test-sqfs": TEST_CPU_SCRIPT_SQFS,
}

EXPERIMENT_MODE_ALIASES = {
    "evaluation": "xai_evaluation",
    "xai-evaluation": "xai_evaluation",
}
EXPERIMENT_MODES = {"training", "xai", "xai_evaluation", "analyses", "full"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a Hydra cluster job")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to a YAML config file (required unless --mode build).",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        help=(
            "Job mode: build, training, xai, xai_evaluation, analyses, full, "
            "preprocess, generate_masked_distortion, generate_lesions, "
            "generate_combined, data"
        ),
    )
    parser.add_argument(
        "--device",
        required=False,
        type=str,
        choices=tuple(RUN_SCRIPTS.keys()),
        default="gpu",
        help="Execution device for non-build jobs.",
    )
    parser.add_argument(
        "--mail",
        required=False,
        type=str,
        default="",
        help="Value for sbatch --mail-user.",
    )
    parser.add_argument(
        "--project-dir",
        required=False,
        type=str,
        default="",
        help="Override project_dir from YAML config.",
    )
    parser.add_argument(
        "--data-dir",
        required=False,
        type=str,
        default="",
        help="Override data_dir from YAML config.",
    )
    parser.add_argument(
        "--sqfs-file",
        required=False,
        type=str,
        default="",
        help="Sqfs dataset filename (e.g. mri-2025-08-25.sqfs). Required when --device gpu-sqfs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command instead of submitting.",
    )
    return parser.parse_args()


def _resolve_local_config_path(config_path_arg: str) -> Path:
    config_path = Path(config_path_arg).expanduser()
    if config_path.is_absolute():
        return config_path.resolve(strict=True)
    cwd_candidate = (Path.cwd() / config_path).resolve(strict=False)
    if cwd_candidate.exists():
        return cwd_candidate.resolve(strict=True)
    project_candidate = (SCRIPT_PATH.parent / config_path).resolve(strict=False)
    if project_candidate.exists():
        return project_candidate.resolve(strict=True)
    raise FileNotFoundError(f"Config file not found: {config_path_arg}")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return config


def _get_config_string(config: dict, *field_names: str) -> str:
    for field_name in field_names:
        value = config.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Config field '{field_name}' must be a string.")
        value = value.strip()
        if value:
            return value
    return ""


def _host_path_to_container_path(
    path: Path,
    *,
    project_dir: Path,
    data_dir: Path,
    artifact_dir: Path,
) -> str:
    host_path = path.expanduser().resolve(strict=False)
    mount_sources = (
        (project_dir.expanduser().resolve(strict=False), "/workdir"),
        (data_dir.expanduser().resolve(strict=False), "/mnt/data"),
        (artifact_dir.expanduser().resolve(strict=False), "/mnt/artifacts"),
    )
    for source_path, container_root in mount_sources:
        try:
            rel = host_path.relative_to(source_path)
            rel_posix = rel.as_posix()
            return container_root if rel_posix == "." else f"{container_root}/{rel_posix}"
        except ValueError:
            continue
    host_path_str = host_path.as_posix()
    if host_path_str.startswith("/workdir/") or host_path_str.startswith("/mnt/"):
        return host_path_str
    raise ValueError(
        "Config path is outside mounted directories. "
        "Pass a config path under project_dir/data_dir/artifact_dir."
    )


def _resolve_container_config_path(
    config_path_arg: str,
    *,
    local_config_path: Path,
    project_dir: Path,
    data_dir: Path,
    artifact_dir: Path,
) -> str:
    config_arg_path = Path(config_path_arg).expanduser()
    if not config_arg_path.is_absolute():
        return config_arg_path.as_posix()
    return _host_path_to_container_path(
        local_config_path,
        project_dir=project_dir,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
    )


def _normalize_mode(mode: str) -> str:
    mode_normalized = mode.strip()
    return EXPERIMENT_MODE_ALIASES.get(mode_normalized, mode_normalized)


def _resolve_container_command(mode: str, *, container_config_path: str) -> list[str]:
    normalized_mode = _normalize_mode(mode)
    match normalized_mode:
        case m if m in EXPERIMENT_MODES:
            return ["run_experiments.py", "--mode", m]
        case "preprocess" | "preprocessing":
            return ["run_data_preprocessing.py"]
        case "generate_masked_distortion":
            return [
                "python",
                "-m",
                "data.generate_masked_distortion",
                "--config",
                container_config_path,
            ]
        case "generate_lesions":
            return [
                "python",
                "-m",
                "data.generate_lesions",
                "--config",
                container_config_path,
            ]
        case "generate_combined":
            return [
                "python",
                "-m",
                "data.generate_combined",
                "--config",
                container_config_path,
            ]
        case "data" | "training_split":
            return [
                "env",
                f"FILE_PATH_TO_DATA_CONFIG={container_config_path}",
                "python",
                "-m",
                "data.main",
            ]
        case _:
            raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = _parse_args()
    normalized_mode = _normalize_mode(args.mode)
    if normalized_mode != "build" and not args.config:
        raise ValueError("--config is required unless --mode build.")

    local_config_path = _resolve_local_config_path(args.config) if args.config else None
    config = _load_yaml(local_config_path) if local_config_path else {}

    project_dir = args.project_dir or _get_config_string(config, "project_dir")
    if not project_dir:
        raise ValueError("Missing project_dir. Set it in YAML or pass --project-dir.")
    project_dir_path = Path(project_dir).expanduser().resolve(strict=False)

    sbatch_command = ["sbatch"]
    if args.mail:
        sbatch_command.append(f"--mail-user={args.mail}")

    if normalized_mode == "build":
        sbatch_command.extend([str(BUILD_SCRIPT), project_dir_path.as_posix()])
    else:
        data_dir = args.data_dir or _get_config_string(config, "data_dir") or DATA_DIR_CLUSTER
        if not data_dir:
            raise ValueError(
                "Missing data_dir. Set it in YAML, pass --data-dir, "
                "or export CLUSTER_DATA_DIR."
            )

        data_dir_path = Path(data_dir).expanduser().resolve(strict=False)
        artifact_dir_path = (project_dir_path / "artifacts").resolve(strict=False)
        container_config_path = _resolve_container_config_path(
            args.config,
            local_config_path=local_config_path,
            project_dir=project_dir_path,
            data_dir=data_dir_path,
            artifact_dir=artifact_dir_path,
        )
        container_command = _resolve_container_command(
            normalized_mode,
            container_config_path=container_config_path,
        )
        run_script = RUN_SCRIPTS[args.device]

        is_sqfs = args.device in ("gpu-sqfs", "cpu-sqfs", "gpu-test-sqfs", "cpu-test-sqfs")
        if is_sqfs:
            sqfs_file = args.sqfs_file or _get_config_string(config, "sqfs_file")
            if not sqfs_file:
                raise ValueError("--sqfs-file is required when --device gpu-sqfs.")
            data_arg = sqfs_file
        else:
            data_arg = data_dir_path.as_posix()

        sbatch_command.extend(
            [
                str(run_script),
                project_dir_path.as_posix(),
                data_arg,
                artifact_dir_path.as_posix(),
                container_config_path,
                *container_command,
            ]
        )

    print(shlex.join(sbatch_command))
    if args.dry_run:
        return
    subprocess.run(sbatch_command, check=True)


if __name__ == "__main__":
    main()
