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

