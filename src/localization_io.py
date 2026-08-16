"""Input validation and grayscale image loading for localization."""

from pathlib import Path
import cv2
import numpy as np


def validate_image_path(label: str, path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} image not found at:\n{path}")


def load_grayscale(path: str, label: str) -> np.ndarray:
    validate_image_path(label, path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {label.lower()} image:\n{path}")
    return image.astype(np.float32)


def trim_reference(reference: np.ndarray, border_trim: int = 8) -> np.ndarray:
    if border_trim <= 0:
        return reference
    h, w = reference.shape
    if 2 * border_trim >= min(h, w):
        return reference
    return reference[border_trim:h-border_trim, border_trim:w-border_trim]
