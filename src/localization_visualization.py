"""Visualization of candidate and selected localization results."""

import os
import cv2
import numpy as np

from .localization_selection import candidate_center


def draw_matches(wide: np.ndarray, distinct, selected):
    marked = cv2.cvtColor(wide.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colors = [
        (0, 0, 255), (0, 128, 255), (0, 255, 255),
        (0, 255, 128), (255, 128, 0), (255, 0, 255),
    ]

    for rank, candidate in enumerate(distinct, start=1):
        score, scale, angle, x, y, size = candidate
        color = colors[min(rank - 1, len(colors) - 1)]
        cv2.rectangle(marked, (x, y), (x + size, y + size), color, 2)
        cv2.putText(
            marked, f"#{rank} {score * 100:.1f}%",
            (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, color, 1, cv2.LINE_AA,
        )

    _, _, _, selected_x, selected_y, selected_size = selected
    cv2.rectangle(
        marked,
        (selected_x, selected_y),
        (selected_x + selected_size, selected_y + selected_size),
        (0, 0, 255), 3,
    )
    selected_cx, selected_cy = candidate_center(selected)
    cv2.drawMarker(
        marked,
        (int(round(selected_cx)), int(round(selected_cy))),
        (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA,
    )
    cv2.putText(
        marked,
        "SELECTED",
        (selected_x, min(marked.shape[0] - 5, selected_y + selected_size + 16)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
    )
    return marked


def save_match_visualization(marked, wide_path: str) -> str:
    wide_directory = os.path.dirname(wide_path) or "."
    output_directory = os.path.join(wide_directory, "opt")
    os.makedirs(output_directory, exist_ok=True)
    out_path = os.path.join(
        output_directory,
        "wide_search_matches_within_10pct_of_best.png",
    )
    if not cv2.imwrite(out_path, marked):
        raise RuntimeError(f"Could not save output image:\n{out_path}")
    return out_path
