"""Candidate filtering, spatial deduplication, and centre-based selection."""

import numpy as np

from .localization_config import SCORE_WINDOW, NMS_RADIUS_FRAC, MAX_MATCHES


def candidate_center(candidate):
    _, _, _, x, y, size = candidate
    return x + size / 2.0, y + size / 2.0


def filter_and_deduplicate(all_points, score_window=SCORE_WINDOW, nms_radius_frac=NMS_RADIUS_FRAC, max_matches=MAX_MATCHES):
    all_points = sorted(all_points, key=lambda p: p[0], reverse=True)
    best_score = all_points[0][0]
    threshold = best_score - score_window
    qualified = [p for p in all_points if p[0] >= threshold]

    distinct = []
    for candidate in qualified:
        cx, cy = candidate_center(candidate)
        _, _, _, _, _, size = candidate
        duplicate = False
        for accepted in distinct:
            acx, acy = candidate_center(accepted)
            accepted_size = accepted[5]
            radius = nms_radius_frac * max(size, accepted_size)
            if np.hypot(cx - acx, cy - acy) <= radius:
                duplicate = True
                break
        if not duplicate:
            distinct.append(candidate)

    return distinct[:max_matches], best_score, threshold


def select_closest_to_search_center(distinct, wide_shape):
    if not distinct:
        raise RuntimeError("No distinct matches survived the score window.")

    search_h, search_w = wide_shape
    center_x = search_w / 2.0
    center_y = search_h / 2.0

    def distance_sq(candidate):
        cx, cy = candidate_center(candidate)
        return (cx - center_x) ** 2 + (cy - center_y) ** 2

    selected = min(distinct, key=distance_sq)
    selected_cx, selected_cy = candidate_center(selected)
    selected_distance = float(np.hypot(selected_cx - center_x, selected_cy - center_y))
    return selected, (center_x, center_y), (selected_cx, selected_cy), selected_distance
