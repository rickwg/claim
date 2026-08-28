# Anomaly Distortion Metrics – Overview

## Purpose

This document describes the metrics used to quantify low-level image property
changes caused by two types of synthetic anomaly generation on mammograms:

- **Masked geometric distortions** (twirl, spherize) from `generate_masked_distortion.py`
- **Inpainted lesions** (diffusion-based) from `generate_lesions.py`

The goal is to characterise *how much* and *in what way* each manipulation
alters the image, and how visible the manipulation boundary is.

---

## Data Basis

All metrics are computed **per image** by comparing:

| Image           | Description |
|-----------------|-------------|
| **Original**    | Preprocessed mammogram before any manipulation. |
| **Manipulated** | The same mammogram after the distortion / lesion has been applied. |
| **Mask**        | Binary mask indicating the manipulated area (circular for geometric distortions, rectangular for inpainting). |

The mask is used in three ways:

1. **Interior** – all pixels where mask > 0 (within-mask metrics).
2. **Inner border** – mask pixels whose immediate neighbour is background (obtained by morphological erosion).
3. **Outer border** – background pixels whose immediate neighbour is inside the mask (obtained by morphological dilation).

Source script: `exploration/anomaly_distortion_metrics.py`
Configuration: `config/anomaly_distortion_metrics.yaml`

---

## Metric Definitions

### 1. Pixel-Value Statistics (within the mask)

These metrics measure direct intensity changes in the manipulated region.

| Metric | Definition | Quantifies |
|--------|-----------|------------|
| `pixel_mae` | Mean of \|manipulated − original\| over all mask pixels | Average magnitude of pixel change |
| `pixel_rmse` | √(mean of (manipulated − original)²) over mask pixels | Average change, penalising large deviations more |
| `pixel_max_diff` | Max \|manipulated − original\| in the mask | Worst-case single-pixel change |
| `pixel_mean_orig` | Mean intensity of the original in the mask | Baseline brightness of the region |
| `pixel_mean_manip` | Mean intensity of the manipulated image in the mask | Post-manipulation brightness |
| `pixel_std_orig` | Std of original intensities in the mask | Baseline texture complexity |
| `pixel_std_manip` | Std of manipulated intensities in the mask | Post-manipulation texture complexity |

### 2. Gradient / Edge Metrics (within the mask)

These metrics measure changes in local image structure by applying differential
operators *before* comparing original and manipulated images.

| Metric | Operator | Definition | Quantifies |
|--------|----------|-----------|------------|
| `sobel_mean_orig` | Sobel | Mean Sobel gradient magnitude in mask (original) | Baseline edge energy |
| `sobel_mean_manip` | Sobel | Mean Sobel gradient magnitude in mask (manipulated) | Post-manipulation edge energy |
| `sobel_mae` | Sobel | Mean \|Sobel(manipulated) − Sobel(original)\| in mask | Change in local edge structure |
| `laplacian_abs_mean_orig` | Laplacian | Mean \|Laplacian\| in mask (original) | Baseline second-order curvature |
| `laplacian_abs_mean_manip` | Laplacian | Mean \|Laplacian\| in mask (manipulated) | Post-manipulation curvature |
| `laplacian_mae` | Laplacian | Mean \|Laplacian(manip) − Laplacian(orig)\| in mask | Change in second-order structure |
| `grad_dx_mae` | Finite diff (horizontal) | Mean \|dx(manip) − dx(orig)\| in mask | Change in horizontal gradients |
| `grad_dy_mae` | Finite diff (vertical) | Mean \|dy(manip) − dy(orig)\| in mask | Change in vertical gradients |
| `grad_mag_mean_orig` | Finite diff magnitude | Mean √(dx² + dy²) in mask (original) | Baseline gradient strength |
| `grad_mag_mean_manip` | Finite diff magnitude | Mean √(dx² + dy²) in mask (manipulated) | Post-manipulation gradient strength |

**Operator details:**

- **Sobel** (3×3 kernel): approximates the first spatial derivative; the magnitude √(Gx² + Gy²) highlights edges.
- **Laplacian** (3×3 kernel): second spatial derivative; highlights rapid intensity changes (blobs, edges).
- **First-order finite differences**: simple dx = I(x+1,y) − I(x,y), dy = I(x,y+1) − I(x,y); the most direct gradient measure.

### 3. Border Metrics (along the mask boundary)

These metrics evaluate how visible the transition is at the edge of the
manipulated region — i.e., how seamlessly the manipulation blends into the
surrounding tissue.

| Metric | Definition | Quantifies |
|--------|-----------|------------|
| `border_intensity_step_manip` | \|mean(inner border) − mean(outer border)\| in manipulated image | Intensity discontinuity at the boundary (manipulated) |
| `border_intensity_step_orig` | Same, measured on the original image | Baseline intensity step (for comparison) |
| `border_grad_mag_manip` | Mean Sobel magnitude on the combined inner + outer border (manipulated) | Edge energy at the boundary |
| `border_grad_mag_orig` | Same, on the original image | Baseline edge energy at the boundary |
| `border_grad_mag_diff` | Mean \|Sobel(manip) − Sobel(orig)\| on the border | Added edge energy from the manipulation |
| `border_pixel_mae_manip` | For each inner-border pixel, \|value − nearest outer-border neighbour\| averaged (manipulated) | Per-pixel cross-boundary intensity jump |
| `border_pixel_mae_orig` | Same, on the original image | Baseline cross-boundary jump |

**Interpretation:** If `border_*_manip ≈ border_*_orig`, the manipulation
boundary is invisible. If `border_grad_mag_diff` is large, the manipulation
has introduced a visible edge at the mask boundary.

### 3a. Border Tracing – How Inner and Outer Borders Are Derived

The border is not a single line but two **one-pixel-wide rings** extracted from
the binary mask via morphological operations:

```
        outer border (○)       inner border (●)
        ○ ○ ○ ○ ○              . . . . .
      ○ ○ . . . ○ ○          . . ● ● ● . .
      ○ . . . . . ○          . ● . . . ● .
      ○ . . . . . ○          . ● . . . ● .
      ○ ○ . . . ○ ○          . . ● ● ● . .
        ○ ○ ○ ○ ○              . . . . .
```

**Step 1 – Dilation.** The binary mask is dilated by a 3×3 rectangular
structuring element (1 pixel in each direction). Every background pixel that
becomes foreground after dilation is an **outer-border** pixel — it sits
directly outside the mask.

```python
kernel   = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilated  = cv.dilate(mask, kernel, iterations=1)
outer_border = (dilated > 0) & (mask == 0)
```

**Step 2 – Erosion.** The same mask is eroded by the same kernel. Every mask
pixel that is removed by erosion is an **inner-border** pixel — it sits on the
inside edge of the mask, directly adjacent to the background.

```python
eroded       = cv.erode(mask, kernel, iterations=1)
inner_border = (mask > 0) & (eroded == 0)
```

The resulting two rings are exactly **1 pixel apart** everywhere along the
boundary. Together they form a 2-pixel-wide band straddling the mask edge,
which is the region where a visible seam would appear if the manipulation
blends poorly.

### 3b. Border Metrics – Detailed Computation

#### `border_intensity_step` (global)

Computes the **mean intensity** over all inner-border pixels and the mean
over all outer-border pixels, then takes the absolute difference:

```
border_intensity_step = |mean(I[inner_border]) − mean(I[outer_border])|
```

This is computed on both the manipulated image (`_manip`) and the original
(`_orig`). The original value serves as a baseline: even an unmanipulated
image has some intensity difference across the ring because of natural tissue
gradients. A manipulation is detectable when the manipulated value deviates
significantly from the original.

#### `border_grad_mag` (Sobel at the border)

The Sobel gradient magnitude (√(Gx² + Gy²), 3×3 kernel) is computed over the
full image. The metric then averages this magnitude over the **union** of
inner and outer border pixels:

```
border_grad_mag = mean(Sobel_magnitude[inner_border ∪ outer_border])
```

A high value means strong edges exist at the mask boundary. Again, both
`_manip` and `_orig` variants are reported.

#### `border_grad_mag_diff` (added edge energy)

The per-pixel absolute difference in Sobel magnitude between the manipulated
and original image, averaged over the border band:

```
border_grad_mag_diff = mean(|Sobel(manip) − Sobel(orig)|  on  inner ∪ outer)
```

This isolates the **new** edge energy introduced solely by the manipulation.
A value near zero means the manipulation did not create any new edge at the
boundary; a large value means a visible seam.

#### `border_pixel_mae` (cross-boundary pixel pairing)

This is the most fine-grained border metric. For **each** inner-border pixel,
the algorithm finds its nearest outer-border neighbour by scanning the 8
immediate neighbours (N, S, E, W, and 4 diagonals). It then computes:

```
border_pixel_mae = mean(|I[inner_pixel] − I[nearest_outer_pixel]|)
```

over all inner-border pixels that have at least one outer-border neighbour.

Because the inner and outer rings are exactly 1 pixel apart, these paired
pixels sit on opposite sides of the mask edge. Their intensity difference
directly measures the **local step** a viewer would perceive when crossing the
boundary. The `_orig` variant provides the natural baseline step, and the
`_manip` variant shows whether the manipulation has increased or decreased it.

---

## Visualizations

Source script: `exploration/visualize_anomaly_metrics.py`
Configuration: `config/visualize_anomaly_metrics.yaml`

### 1. Grouped Bar Chart (`bar_chart.png`)

Shows the **mean ± standard deviation** of each change metric (the MAE / RMSE /
max-diff family), grouped by transformation type (twirl vs spherize). Provides a
quick overview of which transformation produces larger changes and in which
metric category.

### 2. Box Plots (`box_plots.png`)

Displays the **full distribution** (median, quartiles, range) of each change
metric per transformation. Metrics are normalized to [0, 1] so they share a
common y-axis, making it easy to compare spread and outliers across metrics
and transformations.

### 3. Radar Chart (`radar_chart.png`)

Plots the **normalized mean** of each change metric on a polar axis, with one
profile line per transformation. Each spoke represents one metric, scaled by the
global maximum. This gives a compact multi-dimensional "fingerprint" of each
transformation's effect: a larger area means a stronger overall distortion.

### 4. Correlation Heatmap (`correlation_heatmap.png`)

Pairwise **Pearson correlation** between all 24 metrics across images. Reveals
which metrics move together (redundant) and which capture independent aspects
of the distortion. For example, `pixel_mae` and `pixel_rmse` are expected to
be highly correlated, while border metrics may be largely independent of
within-mask gradient metrics.

### 5. Spatial Heatmap Overlays (`spatial_heatmaps/`)

Per-image **4-column plots** showing the manipulated region cropped and zoomed:

| Column | Content | Colour map |
|--------|---------|-----------|
| Original | Original mammogram crop | gray |
| \|Pixel Diff\| | Per-pixel absolute intensity difference | hot |
| Sobel Δ | \|Sobel(manipulated) − Sobel(original)\| | hot |
| Laplacian Δ | \|Laplacian(manipulated) − Laplacian(original)\| | hot |

A cyan dashed circle marks the mask boundary. These plots show *where* the
distortion is strongest and how it distributes spatially — complementing the
scalar aggregate numbers from the other plots.

Images are selected by highest `pixel_mae` per transformation to showcase the
most visually impactful examples.

---

## Current Results (38 images: 19 × twirl + 19 × spherize)

| Metric | Overall Mean | Twirl Mean | Spherize Mean |
|--------|-------------|-----------|--------------|
| pixel_mae | 12.3 | 20.2 | 4.3 |
| pixel_rmse | 18.1 | 28.3 | 7.9 |
| pixel_max_diff | 88.1 | 117.5 | 58.7 |
| sobel_mae | 29.0 | 42.6 | 15.5 |
| laplacian_mae | 47.1 | 65.9 | 28.2 |
| border_grad_mag_diff | 1.7 | 3.4 | 0.08 |
| border_intensity_step_manip | 1.4 | 1.4 | 1.4 |

**Key observations:**

- Twirl produces roughly **4–5× larger pixel-level changes** than spherize
  (pixel_mae 20.2 vs 4.3), consistent with a rotation-based distortion
  redistributing intensities over larger angular distances.
- Gradient and Laplacian changes are also substantially higher for twirl,
  indicating that it introduces more new edges and curvature within the mask.
- Border metrics are nearly identical between manipulated and original for both
  transformations (`border_intensity_step_manip ≈ border_intensity_step_orig`),
  confirming that the cosine fade-out blending produces a seamless boundary.
- Spherize's very low `border_grad_mag_diff` (0.08) indicates the radial
  displacement fade-out effectively hides the manipulation edge.
