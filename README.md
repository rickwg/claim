# claim

## Setup

### Build the Apptainer container

```bash
# On the cluster (via SLURM)
python scripts/submit_hydra_job.py --mode build --project-dir /path/to/project

# Locally
apptainer build --fakeroot apptainerfile.sif apptainerfile.def
```

`apptainerfile.def` installs dependencies with `uv sync --extra data-download`
into `/opt/venv`, and runtime uses `uv run --no-sync`, so `vindrmammo_py` comes
from the built image rather than a mounted `/workdir/.venv`.

The container runs arbitrary Python scripts via `uv`:

```bash
apptainer run apptainerfile.sif <script.py> [args...]
```

### Generate environment-specific configs

Use `setup_config.py` to create timestamped setup bundles.
Each setup command generates both local and cluster config variants, with cluster
paths remapped to container mount points (`/mnt/data`, `/mnt/artifacts`, `/workdir`).

```bash
# Data setup — creates both local + cluster config bundles for
# preprocessing and synthetic data generation configs
python setup_config.py --setup data

# Experiment setup — creates artifacts/{experiment_name}-{timestamp}
# with local configs at root and cluster configs under /cluster
python setup_config.py --setup experiment

# Override the bundled config list with --config/-c
python setup_config.py --setup data -c config/generate_combined.yaml
```

Output directories:

- Data setup: `artifacts/data/local-{timestamp}/` and `artifacts/data/cluster-{timestamp}/`
- Experiment setup: `artifacts/{experiment_name}-{timestamp}/` (cluster variants under
  `cluster/`). `experiment_name` is read from each bundled config — one directory per config.

Edit `CLUSTER_PATH_MAPS` in `setup_config.py` to customize the path mappings.
For experiment setups, generated `experiments_config.yaml` is prefilled with
`data.samples_filename: training_split_metadata.jsonl` pointing to the setup root.

## VinDr-Mammo Download (DICOM + PNG)

Use `data/download_vindrmammo.py` to download VinDr-Mammo metadata + DICOM files
from PhysioNet and convert downloaded DICOM files to PNG.

Prerequisites:

- Install download dependencies: `uv sync --extra data-download`
- Set PhysioNet credentials in your environment (or `.env`):

```bash
export PHYSIONET_USERNAME=<your_physionet_username>
export PHYSIONET_PASSWORD=<your_physionet_password>
```

### Run download locally

```bash
uv sync --extra data-download
uv run python -m data.download_vindrmammo \
    --data-dir /absolute/path/to/vindrmammo_data \
    --max-images 0
```

`--max-images 0` downloads all images listed in metadata. Use smaller values (for
example `--max-images 100`) for smoke tests.

### Run download on Hydra cluster (CPU, via SLURM)

```bash
# Credentials must be available in the submitted job environment
export PHYSIONET_USERNAME=<your_physionet_username>
export PHYSIONET_PASSWORD=<your_physionet_password>

sbatch scripts/cpu_run_prebuild.sh \
    $PROJECT_DIR $DATA_DIR $ARTIFACT_DIR \
    config/preprocess_config.yaml \
    python -m data.download_vindrmammo --data-dir /mnt/data/vindrmammo_data --max-images 0
```

Notes:

- The generic SLURM wrapper requires a config path argument (`$4`) even though
  `data.download_vindrmammo` does not directly consume it.
- Rebuild `apptainerfile.sif` after pulling these changes (and whenever Python
  dependencies change) so `/opt/venv` stays in sync.

## Pipeline Overview

The data pipeline has three stages:

1. **Preprocessing** — DICOM/PNG → normalized images + masks
2. **Data generation** — synthetic lesion creation (distortion or inpainting)
3. **Training split** — combine preprocessed + generated data → `training_split_metadata.jsonl`

```
preprocessing → data generation (optional) → training split → training
```

## 1. Data Preprocessing

Preprocessing applies laterality flipping, resizing, normalization, and generates
masks for all images (all-zero masks for healthy images, bounding-box masks for
images with findings).

### Configuration

Edit the config YAML to control preprocessing behavior. Available configs:

- `config/preprocess_config.yaml` — default (grayscale)
- `config/preprocess_config_mame.yaml` — MAM-E / Stable Diffusion (RGB, 512×512)

Use the `image_filter` option to select which images to process:

```yaml
image_filter: null                              # all images
image_filter: ['No Finding']                    # only healthy images
image_filter: ['Mass']                          # only masses
image_filter: ['Mass', 'Suspicious Calcification']  # masses and calcifications
```

### Run preprocessing

```bash
# Locally
FILE_PATH_TO_PREPROCESS_CONFIG=config/preprocess_config_mame.yaml \
    uv run run_data_preprocessing.py

# On the cluster (CPU, via SLURM)
python scripts/submit_hydra_job.py \
    --mode preprocess \
    --config config/preprocess_config_mame.yaml \
    --device cpu

# On the cluster — using the dedicated DICOM conversion script
sbatch scripts/convert_dicom_to_png.sh $PROJECT_DIR $DATA_DIR
```

### Output structure

```
output_dir/{study_id}/{image_id}.png        # preprocessed images
mask_output_dir/{study_id}/{image_id}.png   # masks (all-zero for healthy)
```

## 2. Data Generation

Generate synthetic unhealthy mammograms from healthy preprocessed images.
Three methods are available:

- **Geometric distortion** (`data/generate_masked_distortion.py`) — twirl/spherize
- **Diffusion inpainting** (`data/generate_lesions.py`) — Stable Diffusion inpainting
- **Combined** (`data/generate_combined.py`) — both artifacts in disjoint regions

All produce images, masks, and a `dataset_metadata.jsonl` with per-image
generation parameters.

### Configuration

- `config/generate_masked_distortion.yaml` — distortion parameters
- `config/generate_lesions.yaml` — inpainting parameters
- `config/generate_combined.yaml` — combined generation parameters

The distortion config supports multiple parameter values per run:

```yaml
# Single value
transformation: twirl
twirl_angle: -70

# Multiple values — generates one image per value with a param suffix
transformation: twirl
twirl_angles:
  - -70
  - -237

# Spherize with multiple amounts
transformation: spherize
spherize_amounts:
  - -17
  - -67

# Generate all supported distortions in one run
transformation: all
```

### Run data generation

```bash
# Geometric distortion (locally)
uv run python -m data.generate_masked_distortion --config config/generate_masked_distortion.yaml

# Diffusion inpainting (locally, requires GPU)
uv run python -m data.generate_lesions --config config/generate_lesions.yaml

# Combined lesion + distortion (locally, requires GPU)
uv run python -m data.generate_combined --config config/generate_combined.yaml

# On the cluster (GPU, via SLURM)
python scripts/submit_hydra_job.py \
    --mode generate_masked_distortion \
    --config config/generate_masked_distortion.yaml

python scripts/submit_hydra_job.py \
    --mode generate_lesions \
    --config config/generate_lesions.yaml

python scripts/submit_hydra_job.py \
    --mode generate_combined \
    --config config/generate_combined.yaml
```

### Output structure

```
output_dir/{study_id}/{image_id}.png            # generated images
output_dir/masks/{study_id}/{image_id}.png      # lesion masks
output_dir/dataset_metadata.jsonl               # per-image generation parameters
output_dir/config.yaml                          # copy of generation config
```

### Combined generation

Per source image, `data/generate_combined.py` emits an unhealthy combined image
(lesion + distortion) and a healthy single-modification comparator. The
`classification_target` flag (`lesion` or `distortion`) picks which artifact is the
discriminator; the other appears in **both** classes as a distractor. Optional knobs
(`discriminator_presence_prob`, `distractor_pos_rate`/`distractor_neg_rate`,
`entangle_discriminator`) vary the difficulty and confounding regime.

```
output_dir/combined/{study_id}/*.png                       # label 1
output_dir/{lesion_only|distortion_only}/{study_id}/*.png  # label 0
output_dir/masks/<variant>/...                             # discriminator mask
output_dir/ground_truth/<variant>/...                      # discriminator GT
output_dir/distortion_ground_truth[_rect]/<variant>/...    # distortion GT
output_dir/lesion_ground_truth_rect/<variant>/...          # lesion GT
output_dir/dataset_metadata.jsonl                          # 2 records per source image
```

## 3. Training Split

Combine preprocessed images and (optionally) generated data into a single
`training_split_metadata.jsonl` for training. Labels and splits come from
the VinDr-Mammo annotations. Supports filtering by label.

### Configuration

Edit `config/data_config.yaml`:

```yaml
preprocessed_image_dir: artifacts/data/vindrmammo/mame_preprocessed
preprocessed_mask_dir: artifacts/data/vindrmammo/mame_masks
annotations_path: vindrmammo_data/finding_annotations.csv
output_path: training_split_metadata.jsonl

# Include generated data (uncomment paths as needed)
generated_metadata_paths:
  # - synthetic_twirl/<created>/dataset_metadata.jsonl
  # - synthetic_spherize/<created>/dataset_metadata.jsonl
  # - synthetic_lesions/<created>/dataset_metadata.jsonl
  # - synthetic_combined/twirl/lesion/<created>/dataset_metadata.jsonl

# Filter by label: [0] = healthy only, [1] = unhealthy only, null = all
label_filter: null
```

### Run training split

```bash
# Locally
FILE_PATH_TO_DATA_CONFIG=config/data_config.yaml uv run python -m data.main

# On the cluster (CPU, via SLURM)
python scripts/submit_hydra_job.py \
    --mode data \
    --config config/data_config.yaml \
    --device cpu
```

### Output format

Each line in `training_split_metadata.jsonl`:

```json
{"image_path": "...", "mask_path": "...", "label": 0, "split": "training", "source": "preprocessed"}
{"image_path": "...", "mask_path": "...", "label": 1, "split": "test", "source": "generated", "transformation": "twirl", "twirl_angle": -70}
```

## Experiments

Available modes: `training`, `xai`, `xai_evaluation`, `analyses`, `full`.

### Run experiments

```bash
# Locally (training only)
FILE_PATH_TO_EXPERIMENT_CONFIG=config/experiments_config.yaml \
    uv run run_experiments.py --mode training

# Locally (XAI only via experiment runner)
FILE_PATH_TO_EXPERIMENT_CONFIG=config/experiments_config.yaml \
    uv run run_experiments.py --mode xai

# Locally (XAI script directly)
FILE_PATH_TO_EXPERIMENT_CONFIG=config/experiments_config.yaml \
    uv run python xai/main.py

# On the cluster (GPU, via SLURM)
python scripts/submit_hydra_job.py \
    --mode training \
    --config config/experiments_config.yaml

# On the cluster (CPU, via SLURM)
python scripts/submit_hydra_job.py \
    --mode full \
    --config config/experiments_config.yaml \
    --device cpu
```

### XAI attribution outputs

`xai/main.py` reads `training/training_records.jsonl`, loads each trained
checkpoint, and computes configured attribution methods for each
`(model, dataset_variant)` tuple.

Outputs are written to `{experiment_dir}/xai/`:

- `heatmaps/{tuple_id}/{method}/*.png` (normalized attribution heatmaps)
- `heatmaps/{tuple_id}/{method}/*.npy` (raw attribution maps, optional)
- `xai_records.jsonl` (final attribution index for downstream evaluation)
- `intermediate_xai_records.jsonl` (checkpoint/restart-safe intermediate state)

Each `xai_records.jsonl` line includes model + dataset provenance and per-image
artifact pointers (`model_path`, `samples_path`, `dataset_variant_tag`,
`image_id`, `image_path`, `mask_path`, attribution file paths, and prediction
scores), so evaluation can run without re-discovering inputs.

Supported `xai.methods`:

- Gradient-based — `saliency`, `input_x_gradient`, `integrated_gradient`,
  `guided_backprop`, `deep_lift`, `gradient_shap`, `lrp`
- Perturbation-based — `kernel_shap`, `lime`
- Baselines — `random`, `sobel`, `laplace`

### Cross-evaluation

`xai.cross_evaluation` applies a model trained on one dataset to the test split of
another. Both datasets must have training records in the same experiment, matched by
their derived `dataset_name`; scoring uses the *data* dataset's ground truth. This
measures how a model trained without a distractor attends the discriminator when the
distractor appears at test time. Cross tuples get a distinct `dataset_variant_tag`
and `model_source`, so analyses keep them separate from native runs.

```yaml
xai:
  cross_evaluation:
    - name: lesion_only
      model_dataset_name: synthetic_lesions/<created>
      data_dataset_name: synthetic_combined/twirl/lesion/<created>
```

### XAI evaluation

`xai_evaluation/main.py` reads `xai/xai_records.jsonl` (filtered to `label=1`,
`split=test`), scores each attribution map against the ground-truth masks, and writes
`{experiment_dir}/xai_evaluation/xai_evaluation_records.jsonl` (plus an
`intermediate_evaluation_records.jsonl` for restart safety).

Per-record metrics include `mass_accuracy`, `relative_importance` (attribution density
inside the ground truth vs. inside the breast outside it), and `*_enrichment`. For
combined datasets these are additionally split into `discriminative_*` and
`distractor_*` variants, with `discriminative_preference` and edge-correspondence
scores (`edge_corr_sobel`, `edge_corr_laplace`).

### Analyses

`analyses/main.py` reads `analyses.analyses` from `experiments_config.yaml` and runs
each configured analysis, writing to `{experiment_dir}/analyses/`. Analyses are
grouped by source module:

| Module | Analyses |
|--------|----------|
| `analyses/model.py` | `model_accuracy_plot`, `model_accuracy_all_splits_plot`, `model_accuracy_grouped_bar_plot`, `model_accuracy_heatmap`, `model_loss_plot`, `hyperparameter_summary` |
| `analyses/prediction.py` | `model_transfer_accuracy_summary`, `model_transfer_accuracy_plot` |
| `analyses/xai.py` | `xai_attribution_examples`, `xai_attribution_overlay_examples` |
| `analyses/xai_evaluation.py` | `xai_mass_accuracy_*`, `xai_relative_importance_*` (boxplot / dumbbell / grouped / `_log` / `_mean` variants), `xai_separation_index`, `xai_performance_comparison`, `xai_confidence_vs_mass_accuracy`, `xai_discriminative_vs_distractor_plot`, `xai_saliency_*`, `xai_edge_*`, `xai_summary` |
| `analyses/dataset.py` | `combined_pair_examples` |

`config/experiments_config.yaml` carries the authoritative list in a comment above
`analyses.analyses`; unrecognized names are skipped with a warning that prints every
supported name.

### Training by dataset parameterization

Training can run multiple dataset categories in one command by filtering generated
samples on metadata fields, while keeping non-generated samples in each run.

In `experiments_config.yaml`:

```yaml
data:
  # Root for the data paths below (legacy alias: base_dir)
  data_dir: artifacts/data/vindrmammo
  # training_split_metadata.jsonl produced by data/main.py
  samples_filename: training_split_metadata.jsonl

training:
  dataset_parameterizations:
    dataset_name: all
    twirl_angle: all
    spherize_amount: all
```

Multiple fields produce a **cross-product** of variants. Fields that do not
exist in a dataset subset are automatically skipped for that subset.
`dataset_name` is always evaluated first to partition the data before
checking sub-parameters. The example above produces:

- twirl datasets × each `twirl_angle` value
- spherize datasets × each `spherize_amount` value
- lesion datasets with just `dataset_name` (no sub-parameters)

Values per field can be a scalar (one run), a list (one run per value), or
`"all"` (one run per unique value found in the generated metadata).

`dataset_name` values are derived from the parent directory of each entry in
`data.generated_metadata_paths`. For example,
`synthetic_twirl/2026-03-18-11-35-13-64-128/dataset_metadata.jsonl` produces
`dataset_name` = `synthetic_twirl/2026-03-18-11-35-13-64-128`. To target a
specific dataset, use the derived name directly:

```yaml
training:
  dataset_parameterizations:
    dataset_name: synthetic_spherize/2026-03-18-11-35-13-32-64
    spherize_amount: all
```

`samples_filename` can also be an absolute path.

## Cluster Script Reference

Submit cluster jobs via `scripts/submit_hydra_job.py`:

```bash
python scripts/submit_hydra_job.py --mode <mode> --config <config.yaml> [options]
```

| Option | Description |
|--------|-------------|
| `--mode` | Job mode: `build`, `training`, `xai`, `xai_evaluation`, `analyses`, `full`, `preprocess`, `generate_masked_distortion`, `generate_lesions`, `generate_combined`, `data` |
| `--config` | Path to YAML config file (required unless `--mode build`) |
| `--device` | `gpu` (default), `cpu`, `gpu-sqfs`, `cpu-sqfs`, or the `-test` variants (`gpu-test`, `cpu-test`, `gpu-test-sqfs`, `cpu-test-sqfs`), which target the short `gpu-test`/`cpu-test` SLURM partitions |
| `--project-dir` | Override `project_dir` from YAML config |
| `--data-dir` | Override `data_dir` from YAML config |
| `--sqfs-file` | Sqfs dataset filename (required when `--device gpu-sqfs`) |
| `--mail` | Email for sbatch notifications |
| `--dry-run` | Print the sbatch command without submitting |

The script reads `project_dir` and `data_dir` from the YAML config, resolves
container mount paths automatically, and selects the appropriate SLURM script.
Mode aliases: `evaluation`/`xai-evaluation` → `xai_evaluation`, `preprocessing` →
`preprocess`, `training_split` → `data`.

Example with sqfs dataset:

```bash
python scripts/submit_hydra_job.py \
    --config config/experiments_config.yaml \
    --mode training \
    --device gpu-sqfs \
    --sqfs-file mri-2025-08-25.sqfs
```

## Exploration & Visualization

Standalone helpers outside the main pipeline:

```bash
# Anomaly-generation sweep + low-level distortion metrics, under one timestamped run dir
python exploration/run_exploration.py --config config/exploration.yaml
python exploration/run_exploration.py --config config/exploration.yaml \
    --run-dir artifacts/exploration/run_<timestamp> --steps metrics metrics-viz

# Figure of the diffusion inpainting pipeline (healthy → mask → lesion → diff → GT)
uv run python -m visualizations.main --config config/visualize_pipeline.yaml
```

Related configs: `config/exploration.yaml`, `config/anomaly_distortion_metrics.yaml`,
`config/visualize_{pipeline,lesions,masked_distortions,boxed_samples,anomaly_metrics}.yaml`.
