"""Shared helpers for synthetic data generation scripts."""

import cv2 as cv
import numpy as np
import pandas as pd


def load_healthy_image_ids(annotations_file: str) -> set[str]:
    """Load image IDs that have only 'No Finding' annotations."""
    df = pd.read_csv(annotations_file)
    healthy_ids = set(
        df[df["finding_categories"] == "['No Finding']"]["image_id"].unique()
    )
    abnormal_ids = set(
        df[df["finding_categories"] != "['No Finding']"]["image_id"].unique()
    )
    return healthy_ids - abnormal_ids


def get_breast_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Extract bounding box of the breast region from a mammogram."""
    _, mask = cv.threshold(image, 10, 255, cv.THRESH_BINARY)
    nb_components, output, stats, _ = cv.connectedComponentsWithStats(
        mask, connectivity=4
    )
    if nb_components < 2:
        return None

    sizes = stats[1:, cv.CC_STAT_AREA]
    max_label = 1 + np.argmax(sizes)
    breast_mask = np.zeros(output.shape, dtype=np.uint8)
    breast_mask[output == max_label] = 255

    contours, _ = cv.findContours(breast_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if not contours:
        return None

    x, y, w, h = cv.boundingRect(contours[0])
    return x, y, x + w, y + h