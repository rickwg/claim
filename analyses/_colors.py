from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

LESION_ARTIFACT = "lesion"
TWIRL_ARTIFACT = "twirl"
SPHERIZE_ARTIFACT = "spherize"
DISTORTION_ARTIFACT = "distortion"

LESION_COLOR = "#1b9e77"
DISTORTION_COLOR = "#d95f02"
TWIRL_COLOR = "#9c3a06"
SPHERIZE_COLOR = "#e17c05"
UNKNOWN_ARTIFACT_COLOR = "#6e6e6e"

ARTIFACT_COLORS = {
    LESION_ARTIFACT: LESION_COLOR,
    DISTORTION_ARTIFACT: DISTORTION_COLOR,
    TWIRL_ARTIFACT: TWIRL_COLOR,
    SPHERIZE_ARTIFACT: SPHERIZE_COLOR,
}

DISTORTION_FAMILY_ARTIFACTS = (SPHERIZE_ARTIFACT, TWIRL_ARTIFACT)

ATTRIBUTION_METHOD_COLORMAP = plt.cm.Blues
EDGE_FILTER_METHOD_COLORMAP = plt.cm.Purples
BASELINE_METHOD_COLOR = "#8a8a8a"

MAGNITUDE_COLORMAP = "Purples"
MAGNITUDE_TEXT_LIGHT_THRESHOLD = 0.62

DISTRACTOR_HATCH = "//"
NATIVE_MODEL_LINE_STYLE = "solid"
CROSS_MODEL_LINE_STYLE = (0, (3.5, 2.0))
PLAIN_EDGE_COLOR = "#3f3f3f"
NEUTRAL_INK = "#1a1a1a"
PATCH_ALPHA = 0.85
ACCENT_SPINE_WIDTH = 2.2
HATCH_LINE_WIDTH = 0.7


def apply_figure_style() -> None:
    plt.rcParams["boxplot.medianprops.color"] = NEUTRAL_INK
    plt.rcParams["boxplot.medianprops.linewidth"] = 1.4
    plt.rcParams["hatch.linewidth"] = HATCH_LINE_WIDTH


def artifact_color(artifact: str | None) -> str:
    return ARTIFACT_COLORS.get(str(artifact), UNKNOWN_ARTIFACT_COLOR)


def distortion_family_from_name(dataset_name: str) -> str:
    return SPHERIZE_ARTIFACT if SPHERIZE_ARTIFACT in str(dataset_name) else TWIRL_ARTIFACT


def dataset_artifacts_from_name(dataset_name: str) -> tuple[str | None, str | None]:
    name = str(dataset_name)
    if "combined" not in name:
        if LESION_ARTIFACT in name:
            return LESION_ARTIFACT, None
        if any(family in name for family in DISTORTION_FAMILY_ARTIFACTS):
            return distortion_family_from_name(name), None
        return None, None
    distortion = distortion_family_from_name(name)
    if f"/{LESION_ARTIFACT}" in name:
        return LESION_ARTIFACT, distortion
    return distortion, LESION_ARTIFACT


def dataset_artifacts(row) -> tuple[str | None, str | None]:
    dataset_name = str(row.get("dataset_name", ""))
    classification_target = str(row.get("classification_target", ""))
    if "combined" in dataset_name and classification_target in ARTIFACT_COLORS:
        distortion = distortion_family_from_name(dataset_name)
        if classification_target == LESION_ARTIFACT:
            return LESION_ARTIFACT, distortion
        return distortion, LESION_ARTIFACT
    return dataset_artifacts_from_name(dataset_name)


def is_cross_model(row) -> bool:
    return str(row.get("model_source", "native")) != "native"


@dataclass(frozen=True)
class DatasetStyle:
    face_color: str
    edge_color: str
    hatch: str | None
    line_style: object


def dataset_style(row) -> DatasetStyle:
    discriminator, distractor = dataset_artifacts(row)
    return DatasetStyle(
        face_color=artifact_color(discriminator),
        edge_color=artifact_color(distractor) if distractor else PLAIN_EDGE_COLOR,
        hatch=DISTRACTOR_HATCH if distractor else None,
        line_style=CROSS_MODEL_LINE_STYLE if is_cross_model(row) else NATIVE_MODEL_LINE_STYLE,
    )


def dataset_accent_color(row) -> str:
    discriminator, _ = dataset_artifacts(row)
    return artifact_color(discriminator)


def apply_dataset_style(patch, style: DatasetStyle) -> None:
    patch.set_facecolor(style.face_color)
    patch.set_edgecolor(style.edge_color)
    patch.set_linewidth(1.6 if style.hatch else 1.0)
    patch.set_linestyle(style.line_style)
    patch.set_alpha(PATCH_ALPHA)
    if style.hatch:
        patch.set_hatch(style.hatch)


def dataset_legend_handle(label: str, style: DatasetStyle) -> mpatches.Patch:
    return mpatches.Patch(
        facecolor=style.face_color,
        edgecolor=style.edge_color,
        hatch=style.hatch or "",
        linestyle=style.line_style,
        linewidth=1.6 if style.hatch else 1.0,
        alpha=PATCH_ALPHA,
        label=label,
    )


def accent_axis(axis, row) -> None:
    accent = dataset_accent_color(row)
    for spine in axis.spines.values():
        spine.set_color(accent)
        spine.set_linewidth(ACCENT_SPINE_WIDTH)


def magnitude_text_color(value: float, *, vmin: float = 0.0, vmax: float = 1.0) -> str:
    span = vmax - vmin
    normalized = 0.0 if span <= 0 else (float(value) - vmin) / span
    return "white" if normalized >= MAGNITUDE_TEXT_LIGHT_THRESHOLD else NEUTRAL_INK
