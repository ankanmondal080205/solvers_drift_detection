"""End-to-end reusable localization API used by the CLI entry point."""

import time

from .localization_config import BORDER_TRIM
from .localization_io import load_grayscale, trim_reference
from .localization_matching import multi_scale_ncc_candidates
from .localization_selection import filter_and_deduplicate, select_closest_to_search_center
from .localization_visualization import draw_matches, save_match_visualization


def localize_pair(
    reference_path: str,
    wide_path: str,
    *,
    border_trim: int = BORDER_TRIM,
    save_visualization: bool = True,
):
    """Localize a 100x reference inside a 10x search image.

    Returns a dictionary containing the selected centre, all retained
    candidates, NCC scores, selected scale/angle, runtime, and output path.
    """
    reference = load_grayscale(reference_path, "Reference")
    wide = load_grayscale(wide_path, "Wide search")
    reference = trim_reference(reference, border_trim)

    runtime_start = time.perf_counter()
    all_points = multi_scale_ncc_candidates(reference, wide)
    distinct, best_score, score_threshold = filter_and_deduplicate(all_points)
    selected, search_center, selected_center, selected_distance = select_closest_to_search_center(
        distinct, wide.shape
    )
    runtime_seconds = time.perf_counter() - runtime_start

    output_path = None
    if save_visualization:
        marked = draw_matches(wide, distinct, selected)
        output_path = save_match_visualization(marked, wide_path)

    return {
        "reference_shape": (int(reference.shape[1]), int(reference.shape[0])),
        "wide_shape": (int(wide.shape[1]), int(wide.shape[0])),
        "best_score": float(best_score),
        "score_threshold": float(score_threshold),
        "candidates": distinct,
        "selected": selected,
        "search_center": search_center,
        "selected_center": selected_center,
        "distance_to_search_center": selected_distance,
        "runtime_seconds": runtime_seconds,
        "output_path": output_path,
    }
