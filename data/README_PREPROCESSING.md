# VinDr-Mammo Preprocessing for MAM-E

This directory contains preprocessing scripts for the VinDr-Mammo dataset, with support for preparing data for diffusion models like MAM-E.

## Overview

The preprocessing pipeline supports:
- **DICOM to PNG conversion** with VOI LUT application (via vindrmammo-py)
- **Laterality standardization** (flip right images to left)
- **Resizing and center cropping** to 512x512
- **Normalization** to uint8 [0-255]
- **RGB conversion** for diffusion models
- **Lesion mask generation** from bounding box annotations

## Setup

1. Install vindrmammo-py (if not already installed):
```bash
pip install git+https://github.com/rickwg/vindrmammo-py.git
```

2. Set up PhysioNet credentials in `.env`:
```bash
PHYSIONET_USERNAME=your_username
PHYSIONET_PASSWORD=your_password
```

## Usage

### Step 1: Download and Convert DICOM to PNG

```bash
python data/main.py --data-dir ./vindrmammo_data
```

This will:
- Download metadata and DICOM files from PhysioNet
- Convert DICOM to 16-bit PNG with VOI LUT applied
- Preserve manufacturer calibration

Output structure:
```
vindrmammo_data/
├── dicom/
│   └── {study_id}/
│       └── {image_id}.dicom
├── png/
│   └── {study_id}/
│       └── {image_id}.png (16-bit)
├── finding_annotations.csv
├── breast-level_annotations.csv
└── metadata.csv
```

### Step 2: Preprocess for MAM-E

For standard classification tasks (grayscale):
```bash
export PREPROCESS_CONFIG_FILE_PATH=config/preprocess_config.yaml
python data/preprocess_vindrmammo.py
```

For MAM-E diffusion models (RGB + masks):
```bash
export PREPROCESS_CONFIG_FILE_PATH=config/preprocess_config_mame.yaml
python data/preprocess_vindrmammo.py
```

Output structure:
```
artifacts/data/vindrmammo/
├── mame_preprocessed/
│   ├── {study_id}/
│   │   └── {image_id}.png (512x512 RGB uint8)
│   └── preprocess_config.yaml
└── mame_masks/
    └── {study_id}/
        ├── {image_id}.png (binary mask for first lesion)
        └── {image_id}_1.png (binary mask for second lesion, if exists)
```

## Configuration Options

### preprocess_config.yaml

```yaml
input_dir: vindrmammo_data          # Input directory from step 1
output_dir: artifacts/data/...      # Output directory for processed images
target_size: [512, 512]             # Target image dimensions
max_value: 3500.0                   # Normalization max value (Montoya et al. 2024)
n_jobs: -1                          # Parallel jobs (-1 = all cores)
to_rgb: false                       # Convert to RGB (false for grayscale)
manufacturer_filter: null           # Scanner manufacturer filter (null = all)
source_max_value_filter: null       # Source PNG max filter (null = all ranges)
generate_masks: false               # Generate lesion masks (false = no masks)
mask_output_dir: null               # Mask output directory (auto if null)
lesion_types: null                  # Lesion type filter (null = all types)
```

### preprocess_config_mame.yaml (for diffusion models)

```yaml
to_rgb: true                        # RGB for Stable Diffusion
manufacturer_filter: ['SIEMENS']    # Match MAM-E siemens15k assumption
source_max_value_filter: 4095       # Keep siemens15k-like source intensity range
generate_masks: true                # Generate masks for inpainting
lesion_types: ['Mass', 'Suspicious Calcification']  # Filter lesion types
```

## Lesion Types

Available lesion types in finding_annotations.csv:
- `Mass` - Mass lesions
- `Suspicious Calcification` - Calcifications
- Combinations like `['Mass', 'Suspicious Calcification']`

Set `lesion_types: null` to include all lesion types.

## Integration with MAM-E

After preprocessing, use the outputs for MAM-E training:

### For base mammogram generation (DreamBooth):
```yaml
# experiments/sd2/config_files/fusion_sd-2-1.yaml
instance_data_dir: /path/to/artifacts/data/vindrmammo/mame_preprocessed
instance_prompt: a mammogram
```

### For lesion inpainting:
```yaml
# experiments/sd2/config_files/inpainting_sd-2.yaml
instance_data_dir: /path/to/artifacts/data/vindrmammo/mame_preprocessed
instance_prompt: a mammogram with a lesion
val_input_image_path: /path/to/healthy/mammogram.png
val_mask_image_path: /path/to/artifacts/data/vindrmammo/mame_masks/{study_id}/{image_id}.png
```

## Processing Pipeline Details

1. **Load PNG** - Reads 16-bit PNG from vindrmammo-py output
2. **Read Laterality** - Extracts laterality from DICOM metadata
3. **Flip Right Images** - Standardizes all images to left orientation
4. **Resize** - Scales to fit target size maintaining aspect ratio
5. **Center Crop** - Crops to exact target size (512x512)
6. **Normalize** - Clips to max_value and normalizes to [0, 255]
7. **Convert to RGB** - Stacks grayscale to 3 channels (if to_rgb=true)
8. **Generate Masks** - Creates binary masks from bounding boxes (if generate_masks=true)

## Mask Generation Details

Masks are binary images where:
- **White (255)** = Lesion area
- **Black (0)** = Background

Bounding boxes are:
- Adjusted for right laterality flipping
- Scaled to match resizing
- Adjusted for center cropping
- Clipped to image boundaries

Multiple lesions in one image create multiple mask files:
- `{image_id}.png` - First lesion
- `{image_id}_1.png` - Second lesion
- `{image_id}_2.png` - Third lesion, etc.

## Notes

- **VOI LUT**: Applied automatically by vindrmammo-py during DICOM→PNG conversion
- **Manufacturer calibration**: Preserved in 16-bit PNG, normalized in preprocessing
- **Laterality standardization**: Critical for consistent model training
- **Parallel processing**: Use `n_jobs=-1` for faster processing on multi-core systems
- **Memory usage**: RGB images use 3x more memory than grayscale

## References

- Montoya-del-Angel et al. (2024): MAM-E: Mammographic Synthetic Image Generation with Diffusion Models
- VinDr-Mammo Dataset: https://physionet.org/content/vindr-mammo/1.0.0/
