"""Scale/rotation-aware normalized cross-correlation matching."""

import cv2
import numpy as np

from .localization_config import (
    SCALE_START, SCALE_END, SCALE_STEP,
    ANGLE_START, ANGLE_END, ANGLE_STEP, MAX_PEAKS_PER_TRANSFORM,
)


def rotate_template(template: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-12:
        return template
    h, w = template.shape
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        template,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _local_maxima(result: np.ndarray, template_size: int):
    win = max(3, template_size // 2)
    kernel = np.ones((win, win), dtype=np.uint8)
    local_max = cv2.dilate(result, kernel)
    return np.where(result >= local_max - 1e-7)


def multi_scale_ncc_candidates(
    reference: np.ndarray,
    wide: np.ndarray,
    scale_start: float = SCALE_START,
    scale_end: float = SCALE_END,
    scale_step: float = SCALE_STEP,
    angle_start: float = ANGLE_START,
    angle_end: float = ANGLE_END,
    angle_step: float = ANGLE_STEP,
):
    """Return (score, scale, angle, x, y, template_size) candidates."""
    all_points = []
    scales = np.arange(scale_start, scale_end + scale_step / 2.0, scale_step)
    angles = np.arange(angle_start, angle_end + angle_step / 2.0, angle_step)

    for scale in scales:
        tmpl_size = int(round(reference.shape[0] / float(scale)))
        if tmpl_size < 8 or tmpl_size >= wide.shape[0] or tmpl_size >= wide.shape[1]:
            continue

        template = cv2.resize(
            reference,
            (tmpl_size, tmpl_size),
            interpolation=cv2.INTER_AREA,
        )

        for angle in angles:
            rotated = rotate_template(template, float(angle))
            result = cv2.matchTemplate(wide, rotated, cv2.TM_CCOEFF_NORMED)
            ys, xs = _local_maxima(result, tmpl_size)
            if len(xs) > MAX_PEAKS_PER_TRANSFORM:
                scores = result[ys, xs]
                keep = np.argpartition(scores, -MAX_PEAKS_PER_TRANSFORM)[-MAX_PEAKS_PER_TRANSFORM:]
                ys, xs = ys[keep], xs[keep]

            for x, y in zip(xs, ys):
                all_points.append(
                    (
                        float(result[y, x]),
                        float(scale),
                        float(angle),
                        int(x),
                        int(y),
                        int(tmpl_size),
                    )
                )

    if not all_points:
        raise RuntimeError("No template-matching candidates were found.")
    return all_points
