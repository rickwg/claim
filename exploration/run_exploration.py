"""
Exploration pipeline orchestrator.

Runs the full anomaly-generation and evaluation pipeline under a single
timestamped output directory:

  1. generate       – Generate anomalies (masked distortions and/or inpainted lesions)
  2. visualize      – Visualise original vs manipulated images
  3. metrics        – Compute low-level distortion metrics
  4. metrics-viz    – Visualise the metrics

Usage:
    # Full pipeline (new run)
    python exploration/run_exploration.py --config config/exploration.yaml

    # Re-run only metrics + visualisation on an existing run
    python exploration/run_exploration.py --config config/exploration.yaml \\
        --run-dir artifacts/exploration/run_20260304_155821 \\
        --steps metrics metrics-viz

    # Only generate, skip everything else
    python exploration/run_exploration.py --config config/exploration.yaml --steps generate
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Ensure the exploration directory is importable
_EXPLORATION_DIR = Path(__file__).resolve().parent
if str(_EXPLORATION_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORATION_DIR))

ALL_STEPS = ["generate", "visualize", "metrics", "metrics-viz"]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-anomaly pipeline
# ---------------------------------------------------------------------------


def _run_masked_distortions(run_dir: Path, config: dict, steps: list[str]):
    """Generate masked distortions, visualise, compute & visualise metrics.

    Supports parameter sweeps: if twirl_angles / spherize_amounts are lists,
    each value gets its own subdirectory and a combined sweep visualisation is
    produced at the end.
    """
    from generate_masked_distortion import process_images
    from visualize_masked_distortions import visualize_all as viz_distortions
    from anomaly_distortion_metrics import process_all as compute_metrics
    from visualize_anomaly_metrics import visualize_all as viz_metrics
    from visualize_anomaly_metrics import plot_parameter_sweep

    base = run_dir / "masked_distortions"
    md_cfg = config.get("masked_distortions", {})
    transformations = md_cfg.get("transformations", ["twirl", "spherize"])

    # Build parameter lists (backward compatible with scalar values)
    twirl_angles = md_cfg.get("twirl_angles", None)
    if twirl_angles is None:
        twirl_angles = [md_cfg.get("twirl_angle", -137)]
    elif not isinstance(twirl_angles, list):
        twirl_angles = [twirl_angles]

    spherize_amounts = md_cfg.get("spherize_amounts", None)
    if spherize_amounts is None:
        spherize_amounts = [md_cfg.get("spherize_amount", -37)]
    elif not isinstance(spherize_amounts, list):
        spherize_amounts = [spherize_amounts]

    # Build list of (transformation, param_name, param_value) runs
    runs: list[tuple[str, str, float]] = []
    if "twirl" in transformations:
        for angle in twirl_angles:
            runs.append(("twirl", "twirl_angle", angle))
    if "spherize" in transformations:
        for amount in spherize_amounts:
            runs.append(("spherize", "spherize_amount", amount))

    sweep_frames: list[pd.DataFrame] = []
    seed = config.get("seed")

    for transformation, param_name, param_value in runs:
        label = f"{transformation}_{param_value}"
        param_base = base / label
        generated_dir = param_base / "generated"
        viz_dir = param_base / "visualizations"
        metrics_dir = param_base / "metrics"
        metrics_viz_dir = param_base / "metrics_visualizations"

        print(f"\n{'=' * 60}")
        print(f"  Parameter run: {label}")
        print(f"{'=' * 60}")

        # --- 1. Generate ---
        if "generate" in steps:
            print(f"\n  Generating {transformation} ({param_name}={param_value})")
            gen_config = {
                "input_dir": config["input_dir"],
                "output_dir": str(generated_dir),
                "annotations_file": config.get("annotations_file"),
                "transformations": [transformation],
                "twirl_angle": param_value if param_name == "twirl_angle" else md_cfg.get("twirl_angle", -137),
                "spherize_amount": param_value if param_name == "spherize_amount" else md_cfg.get("spherize_amount", -37),
                "min_lesion_size": md_cfg.get("min_lesion_size", 60),
                "max_lesion_size": md_cfg.get("max_lesion_size", 90),
                "num_images": config.get("num_images"),
                "seed": seed,
            }
            process_images(gen_config)

        # --- 2. Visualise distortions ---
        if "visualize" in steps:
            print(f"\n  Visualising {label}")
            viz_config = {
                "original_dir": config["input_dir"],
                "generated_dir": str(generated_dir),
                "output_dir": str(viz_dir),
                "metadata_path": None,
                "figsize": config.get("visualization", {}).get("figsize_distortions", [12, 4]),
                "dpi": config.get("visualization", {}).get("dpi", 150),
                "num_images": config.get("num_images"),
            }
            viz_distortions(viz_config)

        # --- 3. Compute metrics ---
        if "metrics" in steps:
            print(f"\n  Computing metrics for {label}")
            met_config = {
                "original_dir": config["input_dir"],
                "generated_dir": str(generated_dir),
                "output_dir": str(metrics_dir),
                "metadata_path": None,
                "num_images": config.get("num_images"),
            }
            compute_metrics(met_config)

        # --- 4. Visualise metrics ---
        if "metrics-viz" in steps:
            print(f"\n  Visualising metrics for {label}")
            met_viz_config = {
                "metrics_dir": str(metrics_dir),
                "original_dir": config["input_dir"],
                "generated_dir": str(generated_dir),
                "output_dir": str(metrics_viz_dir),
                **config.get("metrics_visualization", {}),
            }
            viz_metrics(met_viz_config)

        # Collect for sweep
        metrics_csv = metrics_dir / "metrics.csv"
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            df["param_name"] = param_name
            df["param_value"] = param_value
            sweep_frames.append(df)

    # --- 5. Parameter sweep visualisation ---
    if sweep_frames and "metrics-viz" in steps and len(runs) > 1:
        print(f"\n{'=' * 60}")
        print("  Generating parameter sweep visualisation")
        print(f"{'=' * 60}")
        sweep_dir = base / "parameter_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(sweep_frames, ignore_index=True)
        combined.to_csv(sweep_dir / "sweep_metrics.csv", index=False)
        print(f"  Saved combined sweep metrics → {sweep_dir / 'sweep_metrics.csv'}")

        met_viz_cfg = config.get("metrics_visualization", {})
        plot_parameter_sweep(
            combined,
            sweep_dir,
            figsize=tuple(met_viz_cfg.get("figsize_bar", [14, 6])),
            dpi=met_viz_cfg.get("dpi", 150),
        )


def _run_generated_lesions(run_dir: Path, config: dict, steps: list[str]):
    """Generate inpainted lesions, visualise, compute & visualise metrics."""
    from generate_lesions import process_images
    from visualize_lesions import visualize_all as viz_lesions
    from anomaly_distortion_metrics import process_all as compute_metrics
    from visualize_anomaly_metrics import visualize_all as viz_metrics

    base = run_dir / "generated_lesions"
    generated_dir = base / "generated"
    viz_dir = base / "visualizations"
    metrics_dir = base / "metrics"
    metrics_viz_dir = base / "metrics_visualizations"

    gl_cfg = config.get("generated_lesions", {})

    # --- 1. Generate ---
    if "generate" in steps:
        print("\n" + "=" * 60)
        print("  Generating inpainted lesions")
        print("=" * 60)
        gen_config = {
            "input_dir": config["input_dir"],
            "output_dir": str(generated_dir),
            "annotations_file": config.get("annotations_file"),
            "model_id": gl_cfg.get("model_id", "Likalto4/inpainting_vindr_massbs16"),
            "device": gl_cfg.get("device", "cpu"),
            "prompt": gl_cfg.get("prompt", "a mammogram with a lesion"),
            "num_inference_steps": gl_cfg.get("num_inference_steps", 40),
            "guidance_scale": gl_cfg.get("guidance_scale", 4.0),
            "min_lesion_size": gl_cfg.get("min_lesion_size", 20),
            "max_lesion_size": gl_cfg.get("max_lesion_size", 40),
            "num_images": config.get("num_images"),
            "seed": config.get("seed"),
        }
        process_images(gen_config)

    # --- 2. Visualise lesions ---
    if "visualize" in steps:
        print("\n" + "=" * 60)
        print("  Visualising generated lesions")
        print("=" * 60)
        viz_cfg = config.get("visualization", {})
        viz_config = {
            "original_dir": config["input_dir"],
            "generated_dir": str(generated_dir),
            "output_dir": str(viz_dir),
            "metadata_path": None,
            "threshold": viz_cfg.get("threshold", 10),
            "figsize": viz_cfg.get("figsize_lesions", [24, 4]),
            "dpi": viz_cfg.get("dpi", 150),
            "seamless_padding": viz_cfg.get("seamless_padding", 5),
            "num_images": config.get("num_images"),
        }
        viz_lesions(viz_config)

    # --- 3. Compute metrics ---
    if "metrics" in steps:
        print("\n" + "=" * 60)
        print("  Computing lesion metrics")
        print("=" * 60)
        met_config = {
            "original_dir": config["input_dir"],
            "generated_dir": str(generated_dir),
            "output_dir": str(metrics_dir),
            "metadata_path": None,
            "num_images": config.get("num_images"),
        }
        compute_metrics(met_config)

    # --- 4. Visualise metrics ---
    if "metrics-viz" in steps:
        print("\n" + "=" * 60)
        print("  Visualising lesion metrics")
        print("=" * 60)
        met_viz_config = {
            "metrics_dir": str(metrics_dir),
            "original_dir": config["input_dir"],
            "generated_dir": str(generated_dir),
            "output_dir": str(metrics_viz_dir),
            **config.get("metrics_visualization", {}),
        }
        viz_metrics(met_viz_config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ANOMALY_RUNNERS = {
    "masked_distortions": _run_masked_distortions,
    "generated_lesions": _run_generated_lesions,
}


def run_exploration(config: dict, run_dir: Path | None = None, steps: list[str] | None = None):
    """Execute the exploration pipeline.

    Args:
        config: Consolidated configuration dictionary.
        run_dir: Existing run directory to reuse. If None, a new timestamped
                 directory is created.
        steps: Pipeline steps to execute. Defaults to all steps.
               Available: "generate", "visualize", "metrics", "metrics-viz".
    """
    if steps is None:
        steps = list(ALL_STEPS)

    if run_dir is None:
        output_dir = Path(config.get("output_dir", "./artifacts/exploration"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save consolidated config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    anomalies = config.get("anomalies", ["masked_distortions"])
    print(f"Exploration run: {run_dir}")
    print(f"Anomaly types:   {anomalies}")
    print(f"Steps:           {steps}")

    for anomaly in anomalies:
        runner = ANOMALY_RUNNERS.get(anomaly)
        if runner is None:
            print(f"Unknown anomaly type: {anomaly!r} — skipping")
            continue
        runner(run_dir, config, steps)

    print("\n" + "=" * 60)
    print(f"  Exploration complete → {run_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full exploration pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Full pipeline (new run)
  python exploration/run_exploration.py --config config/exploration.yaml

  # Re-run metrics + visualisation on an existing run
  python exploration/run_exploration.py --config config/exploration.yaml \\
      --run-dir artifacts/exploration/run_20260304_155821 \\
      --steps metrics metrics-viz

  # Only generate, skip downstream steps
  python exploration/run_exploration.py --config config/exploration.yaml --steps generate
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to consolidated YAML configuration file",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Reuse an existing run directory instead of creating a new one",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=None,
        help=f"Pipeline steps to run (default: all). Choices: {ALL_STEPS}",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    run_dir = Path(args.run_dir) if args.run_dir else None

    print("Configuration:")
    print("-" * 40)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print("-" * 40)

    run_exploration(config, run_dir=run_dir, steps=args.steps)


if __name__ == "__main__":
    main()
