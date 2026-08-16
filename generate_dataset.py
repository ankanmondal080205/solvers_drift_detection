"""
Continuous DRAM-only semiconductor navigation dataset generator.

Key improvements:
- Continuous global DRAM routing fabric
- Correlated local pitches/intensities
- Soft bank transitions instead of pasted-looking blocks
- Integrated square, roundabout, and multi-way routing intersections
- Intersections snapped to the DRAM routing grid
- Intersections can be connected by routing paths
- More visible defects, including clustered pixel removal
- Preserves the 10:1 reference/search scale relationship
- Saves ground-truth metadata
"""

import json
import random
import secrets
import cv2
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# ============================================================
# SETTINGS
# ============================================================

SIZE = 1000
PHYSICAL_SCALE = 10
BIG = SIZE * PHYSICAL_SCALE

OUTPUT_ROOT = Path("navigation_dataset")

DEFECTS = [
    "blur",
    "angle_tilt",
    "repetition_increase",
    "edge_stress",
    "scale_mismatch",
    "black_tint",
    "white_tint",
    "scanline_noise",
    "line_dropouts",
    "brightness_drift",
    "none",
]


# ============================================================
# BASIC IMAGE HELPERS
# ============================================================

def u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def save_gray(a, path):
    Image.fromarray(u8(a), "L").save(path)


# ============================================================
# BASIC DRAWING PRIMITIVES
# ============================================================

def rect(a, x0, y0, x1, y1, value):
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(BIG, int(x1))
    y1 = min(BIG, int(y1))

    if x1 <= x0 or y1 <= y0:
        return

    a[y0:y1, x0:x1] = np.maximum(
        a[y0:y1, x0:x1],
        value
    )


def hline(a, y, x0, x1, width, value):
    rect(
        a,
        x0,
        y - width // 2,
        x1,
        y + width // 2 + 1,
        value
    )


def vline(a, x, y0, y1, width, value):
    rect(
        a,
        x - width // 2,
        y0,
        x + width // 2 + 1,
        y1,
        value
    )


def disk(a, cx, cy, radius, value):
    x0 = max(0, int(cx - radius))
    x1 = min(BIG, int(cx + radius + 1))
    y0 = max(0, int(cy - radius))
    y1 = min(BIG, int(cy + radius + 1))

    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]

    mask = (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        <= radius ** 2
    )

    region = a[y0:y1, x0:x1]
    region[mask] = np.maximum(region[mask], value)


def ring(a, cx, cy, radius, thickness, value):
    x0 = max(0, int(cx - radius - thickness))
    x1 = min(BIG, int(cx + radius + thickness + 1))
    y0 = max(0, int(cy - radius - thickness))
    y1 = min(BIG, int(cy + radius + thickness + 1))

    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]

    d2 = (
        (xx - cx) ** 2
        + (yy - cy) ** 2
    )

    outer = (radius + thickness / 2) ** 2
    inner = max(0, radius - thickness / 2) ** 2

    mask = (d2 <= outer) & (d2 >= inner)

    region = a[y0:y1, x0:x1]
    region[mask] = np.maximum(region[mask], value)


# ============================================================
# BACKGROUND
# ============================================================

def make_background(rng):
    yy, xx = np.mgrid[:BIG, :BIG]

    p1 = float(rng.uniform(1800, 5200))
    p2 = float(rng.uniform(1800, 5200))

    base = float(rng.uniform(52, 75))

    field = (
        base
        + float(rng.uniform(1.0, 4.0))
        * np.sin(
            2 * np.pi * xx / p1
            + float(rng.uniform(0, 2 * np.pi))
        )
        + float(rng.uniform(1.0, 4.0))
        * np.cos(
            2 * np.pi * yy / p2
            + float(rng.uniform(0, 2 * np.pi))
        )
    )

    return field.astype(np.float32)


# ============================================================
# GLOBAL DRAM FABRIC
# ============================================================

def draw_global_fabric(image, rng):
    """
    Create a weak continuous routing fabric across the
    entire physical DRAM layout.
    """

    global_bit_pitch = int(rng.integers(90, 150))
    global_word_pitch = int(rng.integers(105, 165))

    bit_offset = int(
        rng.integers(20, global_bit_pitch)
    )

    word_offset = int(
        rng.integers(20, global_word_pitch)
    )

    base_bit_value = int(rng.integers(78, 105))
    base_word_value = int(rng.integers(78, 105))

    for x in range(bit_offset, BIG, global_bit_pitch):
        vline(
            image,
            x,
            0,
            BIG,
            int(rng.integers(3, 7)),
            base_bit_value + int(rng.integers(-7, 8))
        )

    for y in range(word_offset, BIG, global_word_pitch):
        hline(
            image,
            y,
            0,
            BIG,
            int(rng.integers(3, 7)),
            base_word_value + int(rng.integers(-7, 8))
        )

    return (
        global_bit_pitch,
        global_word_pitch,
        bit_offset,
        word_offset
    )


# ============================================================
# CORRELATED PARAMETERS
# ============================================================

def correlated_pitch(
    base_pitch,
    rng,
    minimum_variation=0.92,
    maximum_variation=1.08
):
    return max(
        20,
        int(
            base_pitch
            * rng.uniform(
                minimum_variation,
                maximum_variation
            )
        )
    )


def correlated_value(base_value, rng, variation=10):
    return int(
        np.clip(
            base_value
            + rng.integers(-variation, variation + 1),
            0,
            255
        )
    )


# ============================================================
# STANDARD DRAM SUBARRAY
# ============================================================

def draw_standard_subarray(
    a,
    x0,
    y0,
    x1,
    y1,
    rng,
    base_bit_pitch,
    base_word_pitch,
    style_values
):
    bit_pitch = correlated_pitch(
        base_bit_pitch,
        rng
    )

    word_pitch = correlated_pitch(
        base_word_pitch,
        rng
    )

    bit_offset = x0 + int(
        rng.integers(
            15,
            max(16, bit_pitch)
        )
    )

    word_offset = y0 + int(
        rng.integers(
            15,
            max(16, word_pitch)
        )
    )

    bit_width = int(rng.integers(6, 14))
    word_width = int(rng.integers(6, 14))

    for x in range(bit_offset, x1, bit_pitch):
        vline(
            a,
            x,
            y0,
            y1,
            bit_width,
            correlated_value(
                style_values["line"],
                rng
            )
        )

    for y in range(word_offset, y1, word_pitch):
        hline(
            a,
            y,
            x0,
            x1,
            word_width,
            correlated_value(
                style_values["line"],
                rng
            )
        )

    for x in range(bit_offset, x1, bit_pitch):
        for y in range(word_offset, y1, word_pitch):
            style = int(rng.integers(3))

            if style == 0:
                s = int(rng.integers(6, 13))

                rect(
                    a,
                    x - s,
                    y - s,
                    x + s,
                    y + s,
                    correlated_value(
                        style_values["cell"],
                        rng,
                        8
                    )
                )

                disk(
                    a,
                    x,
                    y,
                    int(rng.integers(2, 6)),
                    correlated_value(
                        style_values["contact"],
                        rng,
                        6
                    )
                )

            elif style == 1:
                ring(
                    a,
                    x,
                    y,
                    int(rng.integers(9, 17)),
                    int(rng.integers(3, 7)),
                    correlated_value(
                        style_values["cell"],
                        rng,
                        8
                    )
                )

            else:
                disk(
                    a,
                    x,
                    y,
                    int(rng.integers(4, 9)),
                    correlated_value(
                        style_values["cell"],
                        rng,
                        8
                    )
                )


# ============================================================
# STAGGERED DRAM SUBARRAY
# ============================================================

def draw_staggered_subarray(
    a,
    x0,
    y0,
    x1,
    y1,
    rng,
    base_bit_pitch,
    base_word_pitch,
    style_values
):
    bit_pitch = correlated_pitch(
        base_bit_pitch,
        rng
    )

    word_pitch = correlated_pitch(
        base_word_pitch,
        rng
    )

    bit_offset = x0 + int(
        rng.integers(
            20,
            min(100, max(21, bit_pitch))
        )
    )

    word_offset = y0 + int(
        rng.integers(
            20,
            min(100, max(21, word_pitch))
        )
    )

    for x in range(bit_offset, x1, bit_pitch):
        vline(
            a,
            x,
            y0,
            y1,
            int(rng.integers(5, 12)),
            correlated_value(
                style_values["line"],
                rng
            )
        )

    row_number = 0

    for y in range(word_offset, y1, word_pitch):
        hline(
            a,
            y,
            x0,
            x1,
            int(rng.integers(6, 14)),
            correlated_value(
                style_values["line"],
                rng
            )
        )

        row_shift = (
            bit_pitch // 2
            if row_number % 2
            else 0
        )

        for x in range(
            bit_offset + row_shift,
            x1,
            bit_pitch
        ):
            if rng.random() < 0.90:
                ring(
                    a,
                    x,
                    y,
                    int(rng.integers(8, 16)),
                    int(rng.integers(3, 7)),
                    correlated_value(
                        style_values["cell"],
                        rng,
                        8
                    )
                )

                disk(
                    a,
                    x,
                    y,
                    int(rng.integers(3, 6)),
                    correlated_value(
                        style_values["contact"],
                        rng,
                        6
                    )
                )

        row_number += 1


# ============================================================
# DENSE COLUMNAR SUBARRAY
# ============================================================

def draw_dense_columnar_subarray(
    a,
    x0,
    y0,
    x1,
    y1,
    rng,
    base_bit_pitch,
    base_word_pitch,
    style_values
):
    bit_pitch = correlated_pitch(
        base_bit_pitch * 0.72,
        rng
    )

    word_pitch = correlated_pitch(
        base_word_pitch,
        rng
    )

    offset_x = x0 + int(
        rng.integers(
            10,
            min(80, max(11, bit_pitch))
        )
    )

    offset_y = y0 + int(
        rng.integers(
            20,
            min(100, max(21, word_pitch))
        )
    )

    for x in range(offset_x, x1, bit_pitch):
        vline(
            a,
            x,
            y0,
            y1,
            int(rng.integers(4, 9)),
            correlated_value(
                style_values["line"],
                rng
            )
        )

    for y in range(offset_y, y1, word_pitch):
        hline(
            a,
            y,
            x0,
            x1,
            int(rng.integers(10, 22)),
            correlated_value(
                style_values["line"],
                rng
            )
        )

    for x in range(offset_x, x1, bit_pitch):
        for y in range(offset_y, y1, word_pitch):
            disk(
                a,
                x,
                y,
                int(rng.integers(3, 7)),
                correlated_value(
                    style_values["cell"],
                    rng,
                    8
                )
            )


# ============================================================
# SPARSE DRAM BANK
# ============================================================

def draw_sparse_bank(
    a,
    x0,
    y0,
    x1,
    y1,
    rng,
    base_bit_pitch,
    base_word_pitch,
    style_values
):
    bit_pitch = correlated_pitch(
        base_bit_pitch * 1.55,
        rng
    )

    word_pitch = correlated_pitch(
        base_word_pitch * 1.55,
        rng
    )

    offset_x = x0 + int(
        rng.integers(
            30,
            min(150, max(31, bit_pitch))
        )
    )

    offset_y = y0 + int(
        rng.integers(
            30,
            min(150, max(31, word_pitch))
        )
    )

    for x in range(offset_x, x1, bit_pitch):
        vline(
            a,
            x,
            y0,
            y1,
            int(rng.integers(7, 16)),
            correlated_value(
                style_values["line"],
                rng
            )
        )

    for y in range(offset_y, y1, word_pitch):
        hline(
            a,
            y,
            x0,
            x1,
            int(rng.integers(7, 18)),
            correlated_value(
                style_values["line"],
                rng
            )
        )

    for x in range(offset_x, x1, bit_pitch):
        for y in range(offset_y, y1, word_pitch):
            if rng.random() < 0.85:
                s = int(rng.integers(9, 19))

                rect(
                    a,
                    x - s,
                    y - s,
                    x + s,
                    y + s,
                    correlated_value(
                        style_values["cell"],
                        rng,
                        8
                    )
                )

                disk(
                    a,
                    x,
                    y,
                    int(rng.integers(4, 8)),
                    correlated_value(
                        style_values["contact"],
                        rng,
                        6
                    )
                )


# ============================================================
# SENSE AMPLIFIER BANK
# ============================================================

def draw_sense_amp_bank(
    a,
    x0,
    y0,
    x1,
    y1,
    rng,
    base_bit_pitch,
    base_word_pitch,
    style_values
):
    draw_standard_subarray(
        a,
        x0,
        y0,
        x1,
        y1,
        rng,
        base_bit_pitch,
        base_word_pitch,
        style_values
    )

    if rng.random() < 0.5:
        sx = int(
            rng.uniform(
                x0 + 0.30 * (x1 - x0),
                x0 + 0.70 * (x1 - x0)
            )
        )

        vline(
            a,
            sx,
            y0,
            y1,
            int(rng.integers(18, 32)),
            correlated_value(
                style_values["boundary"],
                rng,
                8
            )
        )

        local_pitch = correlated_pitch(
            base_word_pitch,
            rng
        )

        for y in range(
            y0 + local_pitch,
            y1,
            local_pitch
        ):
            hline(
                a,
                y,
                sx - 140,
                sx + 140,
                int(rng.integers(8, 18)),
                correlated_value(
                    style_values["cell"],
                    rng,
                    8
                )
            )

            disk(
                a,
                sx,
                y,
                int(rng.integers(7, 13)),
                correlated_value(
                    style_values["contact"],
                    rng,
                    6
                )
            )

    else:
        sy = int(
            rng.uniform(
                y0 + 0.30 * (y1 - y0),
                y0 + 0.70 * (y1 - y0)
            )
        )

        hline(
            a,
            sy,
            x0,
            x1,
            int(rng.integers(18, 32)),
            correlated_value(
                style_values["boundary"],
                rng,
                8
            )
        )


# ============================================================
# ROUTING INTERSECTIONS
# ============================================================

def draw_square_intersection(
    image,
    cx,
    cy,
    pitch,
    rng,
    style_values
):
    half = int(
        pitch * rng.uniform(1.2, 2.0)
    )

    road_width = int(
        rng.integers(8, 18)
    )

    line_value = correlated_value(
        style_values["line"],
        rng,
        8
    )

    junction_value = correlated_value(
        style_values["cell"],
        rng,
        8
    )

    hline(
        image,
        cy,
        cx - half,
        cx + half,
        road_width,
        line_value
    )

    vline(
        image,
        cx,
        cy - half,
        cy + half,
        road_width,
        line_value
    )

    square_size = int(
        pitch * rng.uniform(0.55, 0.85)
    )

    square_width = int(
        rng.integers(6, 13)
    )

    hline(
        image,
        cy - square_size,
        cx - square_size,
        cx + square_size,
        square_width,
        junction_value
    )

    hline(
        image,
        cy + square_size,
        cx - square_size,
        cx + square_size,
        square_width,
        junction_value
    )

    vline(
        image,
        cx - square_size,
        cy - square_size,
        cy + square_size,
        square_width,
        junction_value
    )

    vline(
        image,
        cx + square_size,
        cy - square_size,
        cy + square_size,
        square_width,
        junction_value
    )

    for dx in [-square_size, square_size]:
        for dy in [-square_size, square_size]:
            disk(
                image,
                cx + dx,
                cy + dy,
                int(rng.integers(4, 8)),
                correlated_value(
                    style_values["contact"],
                    rng,
                    6
                )
            )


def draw_roundabout_intersection(
    image,
    cx,
    cy,
    pitch,
    rng,
    style_values
):
    radius = int(
        pitch * rng.uniform(1.0, 1.7)
    )

    road_width = int(
        rng.integers(8, 18)
    )

    line_value = correlated_value(
        style_values["line"],
        rng,
        8
    )

    ring_value = correlated_value(
        style_values["cell"],
        rng,
        8
    )

    hline(
        image,
        cy,
        cx - radius * 2,
        cx + radius * 2,
        road_width,
        line_value
    )

    vline(
        image,
        cx,
        cy - radius * 2,
        cy + radius * 2,
        road_width,
        line_value
    )

    ring(
        image,
        cx,
        cy,
        radius,
        int(rng.integers(7, 14)),
        ring_value
    )

    disk(
        image,
        cx,
        cy,
        int(rng.integers(5, 10)),
        correlated_value(
            style_values["contact"],
            rng,
            6
        )
    )

    for angle in [
        0,
        np.pi / 2,
        np.pi,
        3 * np.pi / 2
    ]:
        px = int(
            cx + radius * np.cos(angle)
        )
        py = int(
            cy + radius * np.sin(angle)
        )

        disk(
            image,
            px,
            py,
            int(rng.integers(4, 8)),
            correlated_value(
                style_values["contact"],
                rng,
                6
            )
        )


def draw_multiway_intersection(
    image,
    cx,
    cy,
    pitch,
    rng,
    style_values
):
    line_value = correlated_value(
        style_values["line"],
        rng,
        8
    )

    contact_value = correlated_value(
        style_values["contact"],
        rng,
        6
    )

    road_width = int(
        rng.integers(7, 15)
    )

    directions = int(
        rng.integers(3, 6)
    )

    radius = int(
        pitch * rng.uniform(1.2, 2.3)
    )

    hline(
        image,
        cy,
        cx - radius,
        cx + radius,
        road_width,
        line_value
    )

    vline(
        image,
        cx,
        cy - radius,
        cy + radius,
        road_width,
        line_value
    )

    for i in range(max(0, directions - 2)):
        angle = float(
            rng.uniform(0, 2 * np.pi)
        )

        steps = max(
            5,
            int(radius / 15)
        )

        for s in range(steps):
            t0 = s / steps
            t1 = (s + 1) / steps

            xa = int(
                cx + radius * t0 * np.cos(angle)
            )
            ya = int(
                cy + radius * t0 * np.sin(angle)
            )

            xb = int(
                cx + radius * t1 * np.cos(angle)
            )
            yb = int(
                cy + radius * t1 * np.sin(angle)
            )

            dx = xb - xa
            dy = yb - ya

            length = max(
                1,
                int(np.sqrt(dx * dx + dy * dy))
            )

            for p in range(
                0,
                length,
                max(2, road_width)
            ):
                tt = p / length

                px = int(xa + dx * tt)
                py = int(ya + dy * tt)

                disk(
                    image,
                    px,
                    py,
                    max(2, road_width // 3),
                    line_value
                )

    disk(
        image,
        cx,
        cy,
        int(rng.integers(6, 12)),
        contact_value
    )


def draw_integrated_intersection(
    image,
    cx,
    cy,
    base_bit_pitch,
    base_word_pitch,
    rng,
    style_values
):
    local_pitch = correlated_pitch(
        (base_bit_pitch + base_word_pitch) / 2,
        rng,
        0.85,
        1.15
    )

    intersection_type = int(
        rng.integers(0, 3)
    )

    if intersection_type == 0:
        draw_square_intersection(
            image,
            cx,
            cy,
            local_pitch,
            rng,
            style_values
        )

    elif intersection_type == 1:
        draw_roundabout_intersection(
            image,
            cx,
            cy,
            local_pitch,
            rng,
            style_values
        )

    else:
        draw_multiway_intersection(
            image,
            cx,
            cy,
            local_pitch,
            rng,
            style_values
        )


def connect_intersections(
    image,
    positions,
    rng,
    style_values
):
    """
    Connect nearby intersections using orthogonal routing.

    This makes the junctions part of one continuous network.
    """

    if len(positions) < 2:
        return

    # Connect each junction to its nearest unconnected
    # neighbor rather than simply drawing a random network.
    connected = set()

    for i, (x1, y1) in enumerate(positions):
        distances = []

        for j, (x2, y2) in enumerate(positions):
            if i == j:
                continue

            distance = (
                (x1 - x2) ** 2
                + (y1 - y2) ** 2
            )

            distances.append(
                (distance, j)
            )

        if not distances:
            continue

        distances.sort()

        _, j = distances[0]

        pair = tuple(
            sorted([i, j])
        )

        if pair in connected:
            continue

        connected.add(pair)

        x2, y2 = positions[j]

        route_value = correlated_value(
            style_values["line"],
            rng,
            10
        )

        route_width = int(
            rng.integers(5, 11)
        )

        # L-shaped routing path.
        hline(
            image,
            y1,
            min(x1, x2),
            max(x1, x2),
            route_width,
            route_value
        )

        vline(
            image,
            x2,
            min(y1, y2),
            max(y1, y2),
            route_width,
            route_value
        )


# ============================================================
# CONTINUOUS DRAM LAYOUT
# ============================================================

def generate_dram_layout(seed):
    rng = np.random.default_rng(seed)

    image = make_background(rng)

    (
        global_bit_pitch,
        global_word_pitch,
        global_bit_offset,
        global_word_offset
    ) = draw_global_fabric(
        image,
        rng
    )

    style_values = {
        "line": int(rng.integers(125, 155)),
        "cell": int(rng.integers(205, 225)),
        "contact": int(rng.integers(230, 245)),
        "boundary": int(rng.integers(100, 125)),
    }

    architecture = int(
        rng.integers(0, 6)
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    state_file = (
        OUTPUT_ROOT
        / ".last_dram_architecture"
    )

    try:
        previous = int(
            state_file.read_text().strip()
        )
    except (FileNotFoundError, ValueError):
        previous = None

    if (
        previous is not None
        and architecture == previous
    ):
        candidates = [
            x for x in range(6)
            if x != previous
        ]

        architecture = int(
            rng.choice(candidates)
        )

    state_file.write_text(
        str(architecture),
        encoding="utf-8"
    )

    # ========================================================
    # ARCHITECTURE 0
    # ========================================================

    if architecture == 0:
        cols = int(rng.integers(3, 6))
        rows = int(rng.integers(3, 6))

        bw = BIG // cols
        bh = BIG // rows

        overlap = int(rng.integers(60, 120))

        for row in range(rows):
            for col in range(cols):
                x0 = col * bw
                y0 = row * bh

                x1 = (
                    BIG
                    if col == cols - 1
                    else (col + 1) * bw
                )

                y1 = (
                    BIG
                    if row == rows - 1
                    else (row + 1) * bh
                )

                draw_standard_subarray(
                    image,
                    max(0, x0 - overlap),
                    max(0, y0 - overlap),
                    min(BIG, x1 + overlap),
                    min(BIG, y1 + overlap),
                    rng,
                    global_bit_pitch,
                    global_word_pitch,
                    style_values
                )

        for x in range(bw, BIG, bw):
            vline(
                image,
                x,
                0,
                BIG,
                int(rng.integers(8, 16)),
                correlated_value(
                    style_values["boundary"],
                    rng,
                    8
                )
            )

        for y in range(bh, BIG, bh):
            hline(
                image,
                y,
                0,
                BIG,
                int(rng.integers(8, 16)),
                correlated_value(
                    style_values["boundary"],
                    rng,
                    8
                )
            )

    # ========================================================
    # ARCHITECTURE 1
    # ========================================================

    elif architecture == 1:
        rows = int(rng.integers(4, 7))
        bh = BIG // rows
        overlap = int(rng.integers(60, 110))

        for row in range(rows):
            y0 = row * bh
            y1 = (
                BIG
                if row == rows - 1
                else (row + 1) * bh
            )

            if row % 2 == 0:
                split = int(
                    rng.uniform(0.30, 0.48) * BIG
                )
            else:
                split = int(
                    rng.uniform(0.52, 0.70) * BIG
                )

            draw_staggered_subarray(
                image,
                30,
                max(0, y0 - overlap),
                min(BIG, split + overlap),
                min(BIG, y1 + overlap),
                rng,
                global_bit_pitch,
                global_word_pitch,
                style_values
            )

            draw_standard_subarray(
                image,
                max(0, split - overlap),
                max(0, y0 - overlap),
                BIG - 30,
                min(BIG, y1 + overlap),
                rng,
                global_bit_pitch,
                global_word_pitch,
                style_values
            )

            hline(
                image,
                y0,
                0,
                BIG,
                int(rng.integers(7, 14)),
                correlated_value(
                    style_values["boundary"],
                    rng,
                    8
                )
            )

    # ========================================================
    # ARCHITECTURE 2
    # ========================================================

    elif architecture == 2:
        bank_count = int(rng.integers(4, 8))

        for col in range(bank_count):
            nominal_bw = BIG / bank_count

            x0 = int(col * nominal_bw)

            x1 = min(
                BIG,
                int(
                    (col + 1) * nominal_bw
                    + rng.integers(-150, 250)
                )
            )

            draw_dense_columnar_subarray(
                image,
                max(0, x0 - 60),
                int(rng.integers(0, 350)),
                min(BIG, x1 + 60),
                BIG - int(rng.integers(0, 350)),
                rng,
                global_bit_pitch,
                global_word_pitch,
                style_values
            )

        for _ in range(int(rng.integers(7, 15))):
            y = int(rng.integers(80, BIG - 80))

            hline(
                image,
                y,
                0,
                BIG,
                int(rng.integers(12, 24)),
                correlated_value(
                    style_values["cell"],
                    rng,
                    10
                )
            )

    # ========================================================
    # ARCHITECTURE 3
    # ========================================================

    elif architecture == 3:
        cols = int(rng.integers(3, 5))
        rows = int(rng.integers(3, 5))

        bw = BIG // cols
        bh = BIG // rows

        overlap = int(rng.integers(70, 130))

        for row in range(rows):
            for col in range(cols):
                x0 = col * bw
                y0 = row * bh

                x1 = (
                    BIG
                    if col == cols - 1
                    else (col + 1) * bw
                )

                y1 = (
                    BIG
                    if row == rows - 1
                    else (row + 1) * bh
                )

                draw_sparse_bank(
                    image,
                    max(0, x0 - overlap),
                    max(0, y0 - overlap),
                    min(BIG, x1 + overlap),
                    min(BIG, y1 + overlap),
                    rng,
                    global_bit_pitch,
                    global_word_pitch,
                    style_values
                )

        for _ in range(int(rng.integers(5, 11))):
            x = int(rng.integers(100, BIG - 100))

            vline(
                image,
                x,
                0,
                BIG,
                int(rng.integers(12, 24)),
                correlated_value(
                    style_values["boundary"],
                    rng,
                    8
                )
            )

        for _ in range(int(rng.integers(5, 11))):
            y = int(rng.integers(100, BIG - 100))

            hline(
                image,
                y,
                0,
                BIG,
                int(rng.integers(12, 24)),
                correlated_value(
                    style_values["boundary"],
                    rng,
                    8
                )
            )

    # ========================================================
    # ARCHITECTURE 4
    # ========================================================

    elif architecture == 4:
        cols = int(rng.integers(3, 5))
        rows = int(rng.integers(3, 6))

        bw = BIG // cols
        bh = BIG // rows

        overlap = int(rng.integers(60, 110))

        for row in range(rows):
            for col in range(cols):
                x0 = col * bw
                y0 = row * bh

                x1 = (
                    BIG
                    if col == cols - 1
                    else (col + 1) * bw
                )

                y1 = (
                    BIG
                    if row == rows - 1
                    else (row + 1) * bh
                )

                draw_sense_amp_bank(
                    image,
                    max(0, x0 - overlap),
                    max(0, y0 - overlap),
                    min(BIG, x1 + overlap),
                    min(BIG, y1 + overlap),
                    rng,
                    global_bit_pitch,
                    global_word_pitch,
                    style_values
                )

        for _ in range(int(rng.integers(4, 8))):
            x = int(rng.integers(300, BIG - 300))

            vline(
                image,
                x,
                0,
                BIG,
                int(rng.integers(18, 30)),
                correlated_value(
                    style_values["boundary"],
                    rng,
                    8
                )
            )

            local_pitch = correlated_pitch(
                global_word_pitch,
                rng
            )

            for y in range(150, BIG, local_pitch):
                disk(
                    image,
                    x,
                    y,
                    int(rng.integers(7, 14)),
                    correlated_value(
                        style_values["contact"],
                        rng,
                        8
                    )
                )

    # ========================================================
    # ARCHITECTURE 5
    # ========================================================

    else:
        cols = int(rng.integers(4, 7))
        rows = int(rng.integers(4, 7))

        bw = BIG // cols
        bh = BIG // rows

        bank_types = [
            draw_standard_subarray,
            draw_staggered_subarray,
            draw_dense_columnar_subarray,
            draw_sparse_bank,
            draw_sense_amp_bank,
        ]

        overlap = int(rng.integers(50, 100))

        for row in range(rows):
            for col in range(cols):
                x0 = col * bw
                y0 = row * bh

                x1 = (
                    BIG
                    if col == cols - 1
                    else (col + 1) * bw
                )

                y1 = (
                    BIG
                    if row == rows - 1
                    else (row + 1) * bh
                )

                fn = bank_types[
                    int(rng.integers(len(bank_types)))
                ]

                fn(
                    image,
                    max(0, x0 - overlap),
                    max(0, y0 - overlap),
                    min(BIG, x1 + overlap),
                    min(BIG, y1 + overlap),
                    rng,
                    global_bit_pitch,
                    global_word_pitch,
                    style_values
                )

                if (row + col) % 2 == 0:
                    vline(
                        image,
                        x0,
                        y0,
                        y1,
                        int(rng.integers(5, 12)),
                        correlated_value(
                            style_values["boundary"],
                            rng,
                            8
                        )
                    )
                else:
                    hline(
                        image,
                        y0,
                        x0,
                        x1,
                        int(rng.integers(5, 12)),
                        correlated_value(
                            style_values["boundary"],
                            rng,
                            8
                        )
                    )

    # ========================================================
    # HIERARCHICAL LARGE -> MEDIUM -> SMALL ROUTING
    # ========================================================

    hierarchical_network = (
        draw_clean_hierarchical_network(
            image,
            rng,
            global_bit_pitch,
            global_word_pitch,
            global_bit_offset,
            global_word_offset,
            style_values
        )
    )

    # ========================================================
    # INTEGRATED INTERSECTIONS
    # ========================================================

    intersection_count = int(
        rng.integers(5, 11)
    )

    intersection_positions = []

    for _ in range(intersection_count):
        grid_x = int(
            rng.integers(
                5,
                max(
                    6,
                    BIG // global_bit_pitch - 5
                )
            )
        )

        grid_y = int(
            rng.integers(
                5,
                max(
                    6,
                    BIG // global_word_pitch - 5
                )
            )
        )

        cx = (
            global_bit_offset
            + grid_x * global_bit_pitch
        )

        cy = (
            global_word_offset
            + grid_y * global_word_pitch
        )

        cx = int(
            np.clip(
                cx,
                500,
                BIG - 500
            )
        )

        cy = int(
            np.clip(
                cy,
                500,
                BIG - 500
            )
        )

        too_close = False

        for px, py in intersection_positions:
            if (
                (cx - px) ** 2
                + (cy - py) ** 2
                < 800 ** 2
            ):
                too_close = True
                break

        if too_close:
            continue

        intersection_positions.append(
            (cx, cy)
        )

        draw_integrated_intersection(
            image,
            cx,
            cy,
            global_bit_pitch,
            global_word_pitch,
            rng,
            style_values
        )

    # Connect intersections.
    connect_intersections(
        image,
        intersection_positions,
        rng,
        style_values
    )

    # ========================================================
    # SMALL ROUTING INTERRUPTIONS
    # ========================================================

    for _ in range(int(rng.integers(12, 30))):
        if rng.random() < 0.5:
            y = int(rng.integers(100, BIG - 100))
            x0 = int(rng.integers(0, BIG - 300))

            rect(
                image,
                x0,
                y - 3,
                min(
                    BIG,
                    x0 + int(rng.integers(80, 450))
                ),
                y + 3,
                int(rng.integers(55, 85))
            )

        else:
            x = int(rng.integers(100, BIG - 100))
            y0 = int(rng.integers(0, BIG - 300))

            rect(
                image,
                x - 3,
                y0,
                x + 3,
                min(
                    BIG,
                    y0 + int(rng.integers(80, 450))
                ),
                int(rng.integers(55, 85))
            )

    # ========================================================
    # FINAL OPTICAL RESPONSE
    # ========================================================

    image = np.asarray(
        Image.fromarray(
            u8(image)
        ).filter(
            ImageFilter.GaussianBlur(
                float(rng.uniform(0.10, 0.25))
            )
        ),
        dtype=np.float32
    )

    return (
        image,
        architecture,
        global_bit_pitch,
        global_word_pitch,
        intersection_positions,
        hierarchical_network
    )


# ============================================================
# REFERENCE + WIDE SEARCH PAIR
# ============================================================

def make_pair(seed, selected_defects=None):
    """
    Generation order:

      1. Generate the clean/native reference.
      2. Scale the reference to wide-search scale.
      3. Generate the normal wide search image.
      4. If repetition is selected, add extra copies of the already-scaled
         reference to that wide image.
      5. All other defects are applied later.
    """

    if selected_defects is None:
        selected_defects = ["none"]

    rng = np.random.default_rng(seed)

    (
        physical,
        architecture,
        global_bit_pitch,
        global_word_pitch,
        intersection_positions,
        hierarchical_network
    ) = generate_dram_layout(seed)

    # 1. Generate reference.
    ref_x = int(
        rng.integers(
            300,
            BIG - SIZE - 300
        )
    )

    ref_y = int(
        rng.integers(
            300,
            BIG - SIZE - 300
        )
    )

    reference = physical[
        ref_y:ref_y + SIZE,
        ref_x:ref_x + SIZE
    ].copy()

    # 2. Scale reference down to the normal wide-image scale.
    scaled_size = max(
        8,
        int(round(SIZE / PHYSICAL_SCALE))
    )

    reference_scaled = np.asarray(
        Image.fromarray(
            u8(reference)
        ).resize(
            (scaled_size, scaled_size),
            Image.Resampling.LANCZOS
        ),
        dtype=np.float32
    )

    # 3. Generate the normal wide search image exactly as before.
    wide = np.asarray(
        Image.fromarray(
            u8(physical)
        ).resize(
            (SIZE, SIZE),
            Image.Resampling.LANCZOS
        ),
        dtype=np.float32
    )

    # The normal wide image already contains the original reference
    # location naturally.
    repetition_boxes = [{
        "x": float(ref_x / PHYSICAL_SCALE),
        "y": float(ref_y / PHYSICAL_SCALE),
        "width": float(scaled_size),
        "height": float(scaled_size),
        "center_x": float(
            ref_x / PHYSICAL_SCALE + scaled_size / 2.0
        ),
        "center_y": float(
            ref_y / PHYSICAL_SCALE + scaled_size / 2.0
        ),
        "is_original": True
    }]

    # 4. Only repetition is handled during wide-image generation.
    if "repetition_increase" in selected_defects:
        wide, extra_boxes = add_reference_repetitions(
            wide,
            reference_scaled,
            rng
        )
        repetition_boxes.extend(extra_boxes)

    return (
        reference,
        wide,
        ref_x,
        ref_y,
        architecture,
        global_bit_pitch,
        global_word_pitch,
        intersection_positions,
        hierarchical_network,
        repetition_boxes
    )



def defect_blur(image, rng):
    radius = float(
        rng.uniform(1.5, 3.0)
    )

    return np.asarray(
        Image.fromarray(
            u8(image)
        ).filter(
            ImageFilter.GaussianBlur(radius)
        ),
        dtype=np.float32
    )


def generate_tilt_angle(rng):
    """
    Generate a small realistic angular defect.

    Magnitude is strictly between 0 and 2 degrees.
    The sign is random so the image may tilt clockwise
    or counter-clockwise.
    """
    magnitude = float(
        rng.uniform(0.05, 2.0)
    )

    sign = -1.0 if rng.random() < 0.5 else 1.0

    return sign * magnitude


def defect_angle_tilt(image, rng):
    angle = generate_tilt_angle(rng)

    result = np.asarray(
        Image.fromarray(
            u8(image)
        ).rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=int(np.median(image))
        ),
        dtype=np.float32
    )

    return result, angle


def add_reference_repetitions(
    wide,
    reference_scaled,
    rng,
    min_extra=1,
    max_extra=4
):
    """Add extra copies of the already-scaled reference to the wide image."""

    image = wide.copy()

    template = np.asarray(
        reference_scaled,
        dtype=np.float32
    )

    th, tw = template.shape[:2]

    # At least one extra occurrence, so repetition always means
    # more than one reference location in total.
    extra_count = int(
        rng.integers(
            min_extra,
            max_extra + 1
        )
    )

    boxes = []
    centres = []

    min_separation = max(
        18,
        int(max(th, tw) * 1.25)
    )

    attempts = 0

    while len(boxes) < extra_count and attempts < 2000:
        attempts += 1

        x = int(
            rng.integers(
                0,
                SIZE - tw + 1
            )
        )
        y = int(
            rng.integers(
                0,
                SIZE - th + 1
            )
        )

        cx = x + tw / 2.0
        cy = y + th / 2.0

        if all(
            (cx - px) ** 2 + (cy - py) ** 2
            >= min_separation ** 2
            for px, py in centres
        ):
            target = image[
                y:y + th,
                x:x + tw
            ].copy()

            ref_mean = float(np.mean(template))
            ref_std = max(float(np.std(template)), 1.0)

            dst_mean = float(np.mean(target))
            dst_std = max(float(np.std(target)), 1.0)

            patch = (
                (template - ref_mean)
                / ref_std
                * (0.72 * dst_std + 0.28 * ref_std)
                + dst_mean
            )

            # Soft boundary so the repeated reference blends into
            # the existing wide-search image.
            yy, xx = np.mgrid[:th, :tw]

            cx0 = (tw - 1) / 2.0
            cy0 = (th - 1) / 2.0

            rx = max(1.0, tw * 0.46)
            ry = max(1.0, th * 0.46)

            distance = np.sqrt(
                ((xx - cx0) / rx) ** 2
                + ((yy - cy0) / ry) ** 2
            )

            mask = np.clip(
                (1.0 - distance) / 0.18,
                0.0,
                1.0
            )

            mask = np.asarray(
                Image.fromarray(
                    u8(mask * 255),
                    "L"
                ).filter(
                    ImageFilter.GaussianBlur(
                        max(0.8, max(th, tw) * 0.04)
                    )
                ),
                dtype=np.float32
            ) / 255.0

            alpha = 0.88 * mask

            image[
                y:y + th,
                x:x + tw
            ] = np.clip(
                alpha * patch
                + (1.0 - alpha) * target,
                0,
                255
            )

            centres.append((cx, cy))

            boxes.append({
                "x": int(x),
                "y": int(y),
                "width": int(tw),
                "height": int(th),
                "center_x": float(cx),
                "center_y": float(cy),
                "is_original": False
            })

    return image, boxes



def defect_repetition_increase(image, rng):
    image = image.copy()

    for _ in range(int(rng.integers(8, 18))):
        n = int(
            rng.integers(100, 240)
        )

        sx = int(
            rng.integers(0, SIZE - n)
        )

        sy = int(
            rng.integers(0, SIZE - n)
        )

        patch = image[
            sy:sy + n,
            sx:sx + n
        ].copy()

        if rng.random() < 0.5:
            patch = np.fliplr(patch)

        if rng.random() < 0.4:
            patch = np.flipud(patch)

        dx = int(
            rng.integers(0, SIZE - n)
        )

        dy = int(
            rng.integers(0, SIZE - n)
        )

        target = image[
            dy:dy + n,
            dx:dx + n
        ]

        strength = float(
            rng.uniform(0.70, 0.92)
        )

        image[
            dy:dy + n,
            dx:dx + n
        ] = (
            strength * patch
            + (1 - strength) * target
        )

    return image


def defect_edge_stress(image, rng):
    image = image.copy()

    side = str(
        rng.choice(
            ["left", "right", "top", "bottom"]
        )
    )

    width = int(
        rng.integers(60, 160)
    )

    strength = float(
        rng.uniform(35, 60)
    )

    if side == "left":
        profile = np.linspace(
            1.0,
            0.0,
            width
        )[None, :]

        image[:, :width] += (
            strength * profile
        )

    elif side == "right":
        profile = np.linspace(
            0.0,
            1.0,
            width
        )[None, :]

        image[:, -width:] += (
            strength * profile
        )

    elif side == "top":
        profile = np.linspace(
            1.0,
            0.0,
            width
        )[:, None]

        image[:width, :] += (
            strength * profile
        )

    else:
        profile = np.linspace(
            0.0,
            1.0,
            width
        )[:, None]

        image[-width:, :] += (
            strength * profile
        )

    return np.clip(
        image,
        0,
        255
    )


def defect_scale_mismatch(image, rng):
    ratio = float(
        rng.uniform(9, 11)
    )

    relative = 10.0 / ratio

    new_size = max(
        20,
        int(
            round(
                SIZE * relative
            )
        )
    )

    resized = Image.fromarray(
        u8(image)
    ).resize(
        (new_size, new_size),
        Image.Resampling.BICUBIC
    )

    if new_size >= SIZE:
        crop = (
            new_size - SIZE
        ) // 2

        resized = resized.crop(
            (
                crop,
                crop,
                crop + SIZE,
                crop + SIZE
            )
        )

    else:
        canvas = Image.new(
            "L",
            (SIZE, SIZE),
            int(np.median(image))
        )

        offset = (
            SIZE - new_size
        ) // 2

        canvas.paste(
            resized,
            (offset, offset)
        )

        resized = canvas

    return (
        np.asarray(
            resized,
            dtype=np.float32
        ),
        ratio
    )


def defect_tint(image, white, rng):
    yy, xx = np.mgrid[:SIZE, :SIZE]

    field = (
        0.55 * np.sin(
            2 * np.pi * xx / 1400
        )
        + 0.45 * np.cos(
            2 * np.pi * yy / 1200
        )
    )

    strength = float(
        rng.uniform(25, 50)
    )

    amount = (
        strength
        * (0.65 + 0.35 * field)
    )

    if white:
        image = image + amount
    else:
        image = image - amount

    return np.clip(
        image,
        0,
        255
    )


# ============================================================
# APPLY DEFECTS
# ============================================================

def defect_scanline_noise(image, rng):
    """
    SEM scanline-noise artifact.

    Produces coherent horizontal/vertical line-like acquisition artifacts
    associated with raster scanning. This is structured line variation,
    not independent pixel noise.
    """
    image = image.astype(np.float32).copy()
    h, w = image.shape

    orientation = (
        "horizontal"
        if rng.random() < 0.75
        else "vertical"
    )

    # A small number of irregular scanline bands.
    n_bands = int(rng.integers(3, 9))

    if orientation == "horizontal":
        profile = np.zeros(h, dtype=np.float32)

        # Slow baseline fluctuation across neighboring scan lines.
        coarse_n = max(2, h // 80)
        coarse = rng.normal(
            0.0,
            float(rng.uniform(1.5, 4.0)),
            coarse_n
        ).astype(np.float32)

        profile = np.interp(
            np.arange(h),
            np.linspace(0, h - 1, coarse_n),
            coarse
        ).astype(np.float32)

        for _ in range(n_bands):
            center = int(rng.integers(0, h))
            half_width = int(rng.integers(0, 2))
            amplitude = float(
                rng.uniform(-14.0, 14.0)
            )

            y0 = max(0, center - half_width)
            y1 = min(h, center + half_width + 1)

            profile[y0:y1] += amplitude

        # Slight within-line modulation so the artifact is not a
        # perfectly uniform stripe.
        x = np.linspace(0, 2 * np.pi, w, dtype=np.float32)
        for y in np.flatnonzero(np.abs(profile) > 0.5):
            modulation = (
                1.0
                + 0.15
                * np.sin(
                    x * float(rng.uniform(0.5, 2.0))
                    + float(rng.uniform(0, 2 * np.pi))
                )
            )
            image[y, :] += profile[y] * modulation

    else:
        profile = np.zeros(w, dtype=np.float32)

        coarse_n = max(2, w // 80)
        coarse = rng.normal(
            0.0,
            float(rng.uniform(1.5, 4.0)),
            coarse_n
        ).astype(np.float32)

        profile = np.interp(
            np.arange(w),
            np.linspace(0, w - 1, coarse_n),
            coarse
        ).astype(np.float32)

        for _ in range(n_bands):
            center = int(rng.integers(0, w))
            half_width = int(rng.integers(0, 2))
            amplitude = float(
                rng.uniform(-14.0, 14.0)
            )

            x0 = max(0, center - half_width)
            x1 = min(w, center + half_width + 1)

            profile[x0:x1] += amplitude

        y = np.linspace(0, 2 * np.pi, h, dtype=np.float32)

        for x in np.flatnonzero(np.abs(profile) > 0.5):
            modulation = (
                1.0
                + 0.15
                * np.sin(
                    y * float(rng.uniform(0.5, 2.0))
                    + float(rng.uniform(0, 2 * np.pi))
                )
            )
            image[:, x] += profile[x] * modulation

    return np.clip(image, 0, 255)

def defect_line_dropouts(image, rng):
    """
    SEM line-dropout artifact.

    Creates missing/incomplete/corrupted horizontal raster scan lines.
    A dropout is a scanline acquisition failure, not a collection of
    small rectangular defects.
    """
    image = image.astype(np.float32).copy()
    h, w = image.shape

    n_dropouts = int(rng.integers(2, 6))

    for _ in range(n_dropouts):
        y = int(rng.integers(0, h))

        # Most are single-line failures; occasionally two adjacent
        # scan lines are affected.
        thickness = 1
        if rng.random() < 0.18:
            thickness = 2

        y1 = min(h, y + thickness)

        # Full-line or partial-line dropout.
        partial = rng.random() < 0.45

        if partial:
            x0 = int(
                rng.integers(
                    0,
                    max(1, w // 5)
                )
            )
            x1 = int(
                rng.integers(
                    max(x0 + 1, 4 * w // 5),
                    w + 1
                )
            )
        else:
            x0 = 0
            x1 = w

        # Estimate the local intensity from neighboring valid lines.
        neighbors = []

        if y > 0:
            neighbors.append(
                image[y - 1, x0:x1]
            )

        if y1 < h:
            neighbors.append(
                image[y1, x0:x1]
            )

        if neighbors:
            background = np.mean(
                np.stack(neighbors, axis=0),
                axis=0
            )
        else:
            background = np.full(
                x1 - x0,
                float(np.mean(image))
            )

        # A true dropout is strongly corrupted, but not necessarily
        # pure black or white.
        corruption = float(
            rng.uniform(0.75, 1.0)
        )

        image[
            y:y1,
            x0:x1
        ] = (
            (1.0 - corruption)
            * image[y:y1, x0:x1]
            +
            corruption
            * background[None, :]
        )

        # Occasionally leave a weak residual trace, representing an
        # incomplete rather than completely lost raster line.
        if rng.random() < 0.35:
            residual = float(
                rng.uniform(0.15, 0.45)
            )
            image[
                y:y1,
                x0:x1
            ] = (
                residual * image[y:y1, x0:x1]
                +
                (1.0 - residual)
                * background[None, :]
            )

    return np.clip(image, 0, 255)

def defect_brightness_drift(image, rng):
    """
    SEM brightness-drift artifact.

    Models unintended intensity changes together with gradual spatial
    shifting over acquisition time. The effect is low-frequency and
    coherent rather than a simple global brightness/tint operation.
    """
    image = image.astype(np.float32).copy()
    h, w = image.shape

    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    # Acquisition-time coordinate. We combine row progression with a
    # small horizontal component so drift is not restricted to a
    # simple vertical brightness gradient.
    t = (
        0.72 * yy / max(1, h - 1)
        +
        0.28 * xx / max(1, w - 1)
    )

    # Smooth brightness drift: low-frequency intensity change.
    amplitude = float(
        rng.uniform(10.0, 28.0)
    )

    phase = float(
        rng.uniform(0.0, 2.0 * np.pi)
    )

    cycles = float(
        rng.uniform(0.45, 1.2)
    )

    brightness = (
        amplitude
        * np.sin(
            2.0 * np.pi * cycles * t
            + phase
        )
    )

    # Add a slow monotonic beam-current/illumination component.
    slope = float(
        rng.uniform(-12.0, 12.0)
    )

    brightness += (
        slope
        * (t - 0.5)
    )

    # Spatial drift: progressively shift image content as acquisition
    # proceeds. Use a smooth displacement field and remap.
    max_dx = float(
        rng.uniform(-4.0, 4.0)
    )
    max_dy = float(
        rng.uniform(-4.0, 4.0)
    )

    # A smooth, time-dependent displacement.
    dx = (
        max_dx
        * (
            t
            - 0.5
        )
    )

    dy = (
        max_dy
        * (
            t
            - 0.5
        )
    )

    map_x = (
        xx
        - dx
    ).astype(np.float32)

    map_y = (
        yy
        - dy
    ).astype(np.float32)

    drifted = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    result = (
        drifted
        + brightness
    )

    return np.clip(
        result,
        0,
        255
    )

def apply_defects(image, selected, rng):
    image = image.copy()
    parameters = {}

    for defect in selected:

        if defect == "blur":
            image = defect_blur(
                image,
                rng
            )
            parameters["blur"] = "applied"

        elif defect == "angle_tilt":
            image, angle = defect_angle_tilt(
                image,
                rng
            )
            parameters["angle_tilt_deg"] = round(
                angle,
                4
            )

        elif defect == "repetition_increase":
            # Already constructed in make_pair(), where the clean
            # reference crop is available.
            parameters["repetition_increase"] = (
                "reference_repetitions_blended_into_wide_search"
            )

        elif defect == "edge_stress":
            image = defect_edge_stress(
                image,
                rng
            )
            parameters["edge_stress"] = "applied"

        elif defect == "scale_mismatch":
            image, ratio = defect_scale_mismatch(
                image,
                rng
            )
            parameters["scale_ratio"] = round(
                ratio,
                4
            )

        elif defect == "black_tint":
            image = defect_tint(
                image,
                False,
                rng
            )
            parameters["black_tint"] = "applied"

        elif defect == "white_tint":
            image = defect_tint(
                image,
                True,
                rng
            )
            parameters["white_tint"] = "applied"

        elif defect == "scanline_noise":
            image = defect_scanline_noise(
                image,
                rng
            )
            parameters["scanline_noise"] = "applied"

        elif defect == "line_dropouts":
            image = defect_line_dropouts(
                image,
                rng
            )
            parameters["line_dropouts"] = "applied"

        elif defect == "brightness_drift":
            image = defect_brightness_drift(
                image,
                rng
            )
            parameters["brightness_drift"] = "applied"

        elif defect == "none":
            pass

    return (
        np.clip(
            image,
            0,
            255
        ),
        parameters
    )


# ============================================================
# INPUT
# ============================================================

def parse_defects(text):
    """
    Numbering:
      0 = none
      1..N = defects
      N+1 = all_combination
    """
    text = text.strip().lower()

    if text in ("none", "no", "clean", "0"):
        return ["none"]

    if text in ("all", "all_combination", "all_combinations"):
        defect_names = [
            d for d in DEFECTS if d != "none"
        ]
        selected = [
            d for d in defect_names
            if d not in ("black_tint", "white_tint")
        ]
        selected.append(
            random.choice(["black_tint", "white_tint"])
        )
        return selected

    try:
        numbers = [
            int(x.strip())
            for x in text.split(",")
            if x.strip()
        ]
    except ValueError:
        raise ValueError(
            "Enter serial number(s), separated by commas."
        )

    defect_names = [
        d for d in DEFECTS
        if d != "none"
    ]

    all_option = len(defect_names) + 1

    if 0 in numbers:
        if len(numbers) != 1:
            raise ValueError(
                "0 means none and cannot be combined with another number."
            )
        return ["none"]

    if all_option in numbers:
        if len(numbers) != 1:
            raise ValueError(
                f"{all_option} (all_combination) cannot be combined "
                "with another number."
            )

        selected = [
            d for d in defect_names
            if d not in ("black_tint", "white_tint")
        ]
        selected.append(
            random.choice(["black_tint", "white_tint"])
        )
        return selected

    invalid = [
        n for n in numbers
        if n < 1 or n > len(defect_names)
    ]

    if invalid:
        raise ValueError(
            "Invalid serial number(s): "
            + ", ".join(map(str, invalid))
            + f". Use 0 for none, 1-{len(defect_names)} for defects, "
              f"or {all_option} for all_combination."
        )

    selected = [
        defect_names[n - 1]
        for n in dict.fromkeys(numbers)
    ]

    if "black_tint" in selected and "white_tint" in selected:
        raise ValueError(
            "Black tint and white tint cannot be selected together."
        )

    if "blur" in selected and "smudge" in selected:
        raise ValueError(
            "Blur and smudge cannot be selected together."
        )

    return selected




# ============================================================
# CLEAN HIERARCHICAL DRAM / CITY-LIKE ROUTING
# ============================================================

def draw_clean_arterial(
    image,
    x1,
    y1,
    x2,
    y2,
    width,
    rng,
    style_values
):
    """
    Clean major DRAM routing corridor.

    The visual idea is inspired by clean overhead urban layouts:
    broad arterials connect districts, while narrower streets
    branch from them. Unlike a literal road map, the corridors
    are composed from semiconductor routing lines.
    """

    value = correlated_value(
        style_values["boundary"],
        rng,
        6
    )

    # Keep all major roads axis-aligned or nearly so.
    if abs(x2 - x1) >= abs(y2 - y1):
        hline(
            image,
            y1,
            min(x1, x2),
            max(x1, x2),
            width,
            value
        )
    else:
        vline(
            image,
            x1,
            min(y1, y2),
            max(y1, y2),
            width,
            value
        )


def draw_clean_orthogonal_connection(
    image,
    x1,
    y1,
    x2,
    y2,
    width,
    rng,
    style_values,
    horizontal_first=True
):
    """
    Clean Manhattan-style connection.

    Every connection uses two perpendicular segments rather
    than noisy sampled curves. This produces clear blocks and
    recognizable intersections.
    """

    value = correlated_value(
        style_values["line"],
        rng,
        6
    )

    if horizontal_first:
        hline(
            image,
            y1,
            min(x1, x2),
            max(x1, x2),
            width,
            value
        )

        vline(
            image,
            x2,
            min(y1, y2),
            max(y1, y2),
            width,
            value
        )
    else:
        vline(
            image,
            x1,
            min(y1, y2),
            max(y1, y2),
            width,
            value
        )

        hline(
            image,
            y2,
            min(x1, x2),
            max(x1, x2),
            width,
            value
        )


def draw_parallel_road(
    image,
    x1,
    y1,
    x2,
    y2,
    width,
    value
):
    """
    Draw a clean, narrow feeder route between two points.

    The route is sampled along the segment and rendered as
    overlapping disks, which supports the small non-axis-aligned
    feeder routes used by the hierarchical network.
    """
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    width = max(1, int(width))
    radius = max(1.0, width / 2.0)

    dx = x2 - x1
    dy = y2 - y1

    length = max(1.0, float(np.hypot(dx, dy)))
    steps = max(1, int(np.ceil(length / max(1.0, radius))))

    for i in range(steps + 1):
        t = i / steps
        cx = x1 + dx * t
        cy = y1 + dy * t
        disk(
            image,
            cx,
            cy,
            radius,
            value
        )


def draw_clean_roundabout(
    image,
    cx,
    cy,
    radius,
    rng,
    style_values
):
    """
    Clean large roundabout.

    Four clean arterial approaches feed a circular DRAM
    routing ring. The central island is kept simple.
    """

    arterial_width = max(
        16,
        int(radius * 0.22)
    )

    ring_width = max(
        9,
        int(radius * 0.14)
    )

    approach = int(
        radius * 2.4
    )

    value = correlated_value(
        style_values["boundary"],
        rng,
        5
    )

    hline(
        image,
        cy,
        cx - approach,
        cx - radius,
        arterial_width,
        value
    )

    hline(
        image,
        cy,
        cx + radius,
        cx + approach,
        arterial_width,
        value
    )

    vline(
        image,
        cx,
        cy - approach,
        cy - radius,
        arterial_width,
        value
    )

    vline(
        image,
        cx,
        cy + radius,
        cy + approach,
        arterial_width,
        value
    )

    ring(
        image,
        cx,
        cy,
        radius,
        ring_width,
        correlated_value(
            style_values["cell"],
            rng,
            6
        )
    )

    # Simple inner DRAM island.
    disk(
        image,
        cx,
        cy,
        max(
            8,
            int(radius * 0.25)
        ),
        correlated_value(
            style_values["contact"],
            rng,
            5
        )
    )


def draw_clean_square_intersection(
    image,
    cx,
    cy,
    half_size,
    rng,
    style_values
):
    """
    Clean rectangular junction.

    The square is deliberately geometric, similar to the
    strongly ordered blocks visible in overhead city grids.
    """

    arterial_width = max(
        16,
        int(half_size * 0.16)
    )

    local_width = max(
        7,
        int(half_size * 0.065)
    )

    value = correlated_value(
        style_values["boundary"],
        rng,
        5
    )

    cell_value = correlated_value(
        style_values["cell"],
        rng,
        6
    )

    # Four broad approaches.
    hline(
        image,
        cy,
        cx - half_size * 2,
        cx - half_size,
        arterial_width,
        value
    )

    hline(
        image,
        cy,
        cx + half_size,
        cx + half_size * 2,
        arterial_width,
        value
    )

    vline(
        image,
        cx,
        cy - half_size * 2,
        cy - half_size,
        arterial_width,
        value
    )

    vline(
        image,
        cx,
        cy + half_size,
        cy + half_size * 2,
        arterial_width,
        value
    )

    # Clean rectangular central routing loop.
    hline(
        image,
        cy - half_size,
        cx - half_size,
        cx + half_size,
        local_width,
        cell_value
    )

    hline(
        image,
        cy + half_size,
        cx - half_size,
        cx + half_size,
        local_width,
        cell_value
    )

    vline(
        image,
        cx - half_size,
        cy - half_size,
        cy + half_size,
        local_width,
        cell_value
    )

    vline(
        image,
        cx + half_size,
        cy - half_size,
        cy + half_size,
        local_width,
        cell_value
    )

    # Four DRAM contacts.
    for dx in (-half_size, half_size):
        for dy in (-half_size, half_size):
            disk(
                image,
                cx + dx,
                cy + dy,
                max(
                    5,
                    local_width // 2
                ),
                correlated_value(
                    style_values["contact"],
                    rng,
                    5
                )
            )


def draw_clean_local_district(
    image,
    x0,
    y0,
    x1,
    y1,
    rng,
    global_bit_pitch,
    global_word_pitch,
    style_values
):
    """
    Clean local DRAM district.

    A district is a rectangular group of smaller arrays.
    Its local routing is dense, regular, and subordinate to
    the wider arterial network.
    """

    bit_pitch = correlated_pitch(
        global_bit_pitch,
        rng,
        0.94,
        1.06
    )

    word_pitch = correlated_pitch(
        global_word_pitch,
        rng,
        0.94,
        1.06
    )

    # Keep a clean rectangular local grid.
    local_x0 = int(x0 + bit_pitch * 1.5)
    local_y0 = int(y0 + word_pitch * 1.5)

    local_x1 = int(x1 - bit_pitch * 1.5)
    local_y1 = int(y1 - word_pitch * 1.5)

    if local_x1 <= local_x0 or local_y1 <= local_y0:
        return

    line_width = int(
        rng.integers(4, 8)
    )

    for x in range(
        local_x0,
        local_x1,
        bit_pitch
    ):
        vline(
            image,
            x,
            local_y0,
            local_y1,
            line_width,
            correlated_value(
                style_values["line"],
                rng,
                6
            )
        )

    for y in range(
        local_y0,
        local_y1,
        word_pitch
    ):
        hline(
            image,
            y,
            local_x0,
            local_x1,
            line_width,
            correlated_value(
                style_values["line"],
                rng,
                6
            )
        )

    # Periodic DRAM cells, aligned to the local grid.
    for x in range(
        local_x0,
        local_x1,
        bit_pitch
    ):
        for y in range(
            local_y0,
            local_y1,
            word_pitch
        ):
            if rng.random() < 0.88:
                disk(
                    image,
                    x,
                    y,
                    int(
                        rng.integers(
                            3,
                            7
                        )
                    ),
                    correlated_value(
                        style_values["cell"],
                        rng,
                        7
                    )
                )

                if rng.random() < 0.40:
                    disk(
                        image,
                        x,
                        y,
                        int(
                            rng.integers(
                                2,
                                4
                            )
                        ),
                        correlated_value(
                            style_values["contact"],
                            rng,
                            5
                        )
                    )


def draw_clean_hierarchical_network(
    image,
    rng,
    global_bit_pitch,
    global_word_pitch,
    global_bit_offset,
    global_word_offset,
    style_values
):
    """
    Generate a clean, top-down, city-like hierarchy while
    preserving DRAM geometry.

    Layout hierarchy:

        LARGE ARTERIAL GRID
             |
        MAJOR JUNCTIONS
             |
        MEDIUM STREETS
             |
        RECTANGULAR DRAM DISTRICTS
             |
        SMALL DRAM CELL GRID

    The result intentionally resembles the visual logic of
    orderly urban plans such as Manhattan while allowing
    small pockets of Boston-like variation.
    """

    # --------------------------------------------------------
    # 1. LARGE ARTERIAL GRID
    # --------------------------------------------------------

    arterial_spacing_x = int(
        BIG / rng.integers(4, 6)
    )

    arterial_spacing_y = int(
        BIG / rng.integers(4, 6)
    )

    arterial_x = [
        int(
            global_bit_offset
            + i * arterial_spacing_x
        )
        for i in range(1, 5)
    ]

    arterial_y = [
        int(
            global_word_offset
            + i * arterial_spacing_y
        )
        for i in range(1, 5)
    ]

    arterial_x = [
        x for x in arterial_x
        if 250 < x < BIG - 250
    ]

    arterial_y = [
        y for y in arterial_y
        if 250 < y < BIG - 250
    ]

    major_width = int(
        rng.integers(28, 46)
    )

    for x in arterial_x:
        vline(
            image,
            x,
            0,
            BIG,
            major_width,
            correlated_value(
                style_values["boundary"],
                rng,
                5
            )
        )

    for y in arterial_y:
        hline(
            image,
            y,
            0,
            BIG,
            major_width,
            correlated_value(
                style_values["boundary"],
                rng,
                5
            )
        )

    # --------------------------------------------------------
    # 2. MAJOR INTERSECTIONS
    # --------------------------------------------------------

    large_nodes = []

    for x in arterial_x:
        for y in arterial_y:
            # Not every intersection needs a landmark.
            # This prevents a repetitive "plus sign" pattern.
            if rng.random() < 0.46:
                large_nodes.append(
                    (x, y)
                )

                if rng.random() < 0.48:
                    draw_clean_roundabout(
                        image,
                        x,
                        y,
                        int(
                            rng.uniform(
                                65,
                                115
                            )
                        ),
                        rng,
                        style_values
                    )
                else:
                    draw_clean_square_intersection(
                        image,
                        x,
                        y,
                        int(
                            rng.uniform(
                                60,
                                105
                            )
                        ),
                        rng,
                        style_values
                    )

    # --------------------------------------------------------
    # 3. MEDIUM ROADS INSIDE CITY BLOCKS
    # --------------------------------------------------------

    medium_nodes = []

    # Store actual (x, y) coordinates of medium-road
    # intersections so metadata remains consistent.
    medium_node_points = []

    for ix in range(
        len(arterial_x) - 1
    ):
        left = arterial_x[ix]
        right = arterial_x[ix + 1]

        # Two or three clean internal vertical streets.
        count = int(
            rng.integers(1, 3)
        )

        for _ in range(count):
            x = int(
                rng.uniform(
                    left + 0.22 * (right - left),
                    left + 0.78 * (right - left)
                )
            )

            width = int(
                rng.integers(11, 19)
            )

            vline(
                image,
                x,
                0,
                BIG,
                width,
                correlated_value(
                    style_values["line"],
                    rng,
                    5
                )
            )

            medium_nodes.append(
                ("vertical", x)
            )

            # Representative points along this medium road.
            for y_ref in arterial_y:
                medium_node_points.append(
                    (x, y_ref)
                )

    for iy in range(
        len(arterial_y) - 1
    ):
        top = arterial_y[iy]
        bottom = arterial_y[iy + 1]

        count = int(
            rng.integers(1, 3)
        )

        for _ in range(count):
            y = int(
                rng.uniform(
                    top + 0.22 * (bottom - top),
                    top + 0.78 * (bottom - top)
                )
            )

            width = int(
                rng.integers(11, 19)
            )

            hline(
                image,
                y,
                0,
                BIG,
                width,
                correlated_value(
                    style_values["line"],
                    rng,
                    5
                )
            )

            medium_nodes.append(
                ("horizontal", y)
            )

            # Representative points along this medium road.
            for x_ref in arterial_x:
                medium_node_points.append(
                    (x_ref, y)
                )

    # --------------------------------------------------------
    # 4. CLEAN "CITY BLOCKS" CONTAINING DRAM ARRAYS
    # --------------------------------------------------------

    districts = []

    x_edges = [0] + arterial_x + [BIG]
    y_edges = [0] + arterial_y + [BIG]

    for ix in range(
        len(x_edges) - 1
    ):
        for iy in range(
            len(y_edges) - 1
        ):
            x0 = x_edges[ix]
            x1 = x_edges[ix + 1]
            y0 = y_edges[iy]
            y1 = y_edges[iy + 1]

            # Leave a clear corridor around major roads.
            margin = max(
                35,
                major_width
            )

            district = (
                x0 + margin,
                y0 + margin,
                x1 - margin,
                y1 - margin
            )

            if (
                district[2] - district[0]
                > global_bit_pitch * 2
                and
                district[3] - district[1]
                > global_word_pitch * 2
            ):
                districts.append(
                    district
                )

                draw_clean_local_district(
                    image,
                    *district,
                    rng,
                    global_bit_pitch,
                    global_word_pitch,
                    style_values
                )

    # --------------------------------------------------------
    # 5. SELECTED SMALL INTERSECTIONS
    # --------------------------------------------------------

    small_nodes = []

    for district in districts:
        x0, y0, x1, y1 = district

        if rng.random() < 0.72:
            x = int(
                rng.uniform(
                    x0 + 0.20 * (x1 - x0),
                    x0 + 0.80 * (x1 - x0)
                )
            )

            y = int(
                rng.uniform(
                    y0 + 0.20 * (y1 - y0),
                    y0 + 0.80 * (y1 - y0)
                )
            )

            small_nodes.append(
                (x, y)
            )

            # Small junctions stay much smaller than the
            # major intersections.
            draw_integrated_intersection(
                image,
                x,
                y,
                global_bit_pitch,
                global_word_pitch,
                rng,
                style_values
            )

    # --------------------------------------------------------
    # 6. SMALL LOCAL CONNECTIONS
    # --------------------------------------------------------

    for i in range(
        len(small_nodes) - 1
    ):
        if rng.random() < 0.32:
            x1, y1 = small_nodes[i]
            x2, y2 = small_nodes[i + 1]

            # Only connect nearby nodes so the network remains
            # clean and does not create long random diagonals.
            distance = float(
                np.hypot(
                    x2 - x1,
                    y2 - y1
                )
            )

            if distance < 900:
                draw_clean_orthogonal_connection(
                    image,
                    x1,
                    y1,
                    x2,
                    y2,
                    int(rng.integers(4, 8)),
                    rng,
                    style_values,
                    horizontal_first=(
                        rng.random() < 0.5
                    )
                )

    # --------------------------------------------------------
    # 7. A SMALL NUMBER OF BOSTON-LIKE LOCAL GRID ROTATIONS
    # --------------------------------------------------------
    #
    # Boston contains pockets of grids with different
    # orientations rather than one globally perfect grid.
    # We keep this subtle: only a few local districts receive
    # a secondary routing direction.
    # --------------------------------------------------------

    if len(districts) >= 3:
        rotated_count = int(
            rng.integers(1, 3)
        )

        for district in rng.choice(
            len(districts),
            size=min(
                rotated_count,
                len(districts)
            ),
            replace=False
        ):
            x0, y0, x1, y1 = districts[
                int(district)
            ]

            # Use a very small angular deviation to preserve
            # clean semiconductor geometry.
            theta = float(
                rng.uniform(
                    np.deg2rad(0.5),
                    np.deg2rad(2.0)
                )
            )

            local_pitch = correlated_pitch(
                (
                    global_bit_pitch
                    + global_word_pitch
                ) / 2,
                rng,
                0.96,
                1.04
            )

            # Sparse diagonal "feeder" routes.
            for k in range(
                2,
                5
            ):
                px = x0 + k * local_pitch
                if px >= x1:
                    break

                length = min(
                    y1 - y0,
                    int(
                        (x1 - x0)
                        * 0.55
                    )
                )

                # Tiny angle only; these are feeder routes,
                # not dominant city streets.
                dx = int(
                    length * np.sin(theta)
                )

                draw_parallel_road(
                    image,
                    px,
                    y0,
                    min(x1, px + dx),
                    min(y1, y0 + length),
                    max(3, local_pitch // 14),
                    correlated_value(
                        style_values["line"],
                        rng,
                        5
                    )
                )

    return {
        "large_nodes": [
            [int(x), int(y)]
            for x, y in large_nodes
        ],
        "medium_nodes": [
            [int(x), int(y)]
            for x, y in medium_node_points
        ],
        "districts": [
            [
                int(x0),
                int(y0),
                int(x1),
                int(y1)
            ]
            for x0, y0, x1, y1 in districts
        ],
        "small_nodes": [
            [int(x), int(y)]
            for x, y in small_nodes
        ]
    }


# ============================================================
# GROUND-TRUTH LOCATION IMAGE
# ============================================================

# ============================================================


def create_multiple_reference_locations_image(
    wide_search,
    repetition_boxes
):
    """Mark all reference occurrences in wide-search-image coordinates."""

    image = Image.fromarray(
        u8(wide_search),
        "L"
    ).convert("RGB")

    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    boxes_wide = []

    for index, box in enumerate(
        repetition_boxes,
        start=1
    ):
        x0 = max(0, min(SIZE - 1, int(round(box["x"]))))
        y0 = max(0, min(SIZE - 1, int(round(box["y"]))))

        x1 = max(
            0,
            min(
                SIZE - 1,
                int(round(
                    box["x"] + box["width"] - 1
                ))
            )
        )
        y1 = max(
            0,
            min(
                SIZE - 1,
                int(round(
                    box["y"] + box["height"] - 1
                ))
            )
        )

        boxes_wide.append([x0, y0, x1, y1])

        draw.rectangle(
            [x0, y0, x1, y1],
            outline=(0, 0, 0),
            width=4
        )
        draw.rectangle(
            [x0, y0, x1, y1],
            outline=(255, 255, 255),
            width=2
        )
        draw.text(
            (x0 + 4, y0 + 4),
            str(index),
            fill=(255, 255, 255)
        )

    return image, boxes_wide



def create_reference_location_image(
    wide_search,
    ref_x,
    ref_y,
    reference_size=SIZE,
    physical_scale=PHYSICAL_SCALE
):
    """
    Create a clean visual verification image.

    The reference is a SIZE x SIZE crop from the physical
    BIG x BIG layout. After the 10:1 downsampling used for
    the search image, its expected location is:

        x = ref_x / physical_scale
        y = ref_y / physical_scale
        w = reference_size / physical_scale
        h = reference_size / physical_scale

    A bright rectangle is drawn around that exact ground-truth
    region. Crosshairs and coordinate text are intentionally
    omitted so the image remains useful for visual comparison
    with model predictions.
    """

    image = Image.fromarray(
        u8(wide_search),
        "L"
    ).convert("RGB")

    # Expected reference location in the 1000x1000 search image.
    x0 = int(round(ref_x / physical_scale))
    y0 = int(round(ref_y / physical_scale))

    box_size = int(
        round(reference_size / physical_scale)
    )

    x1 = x0 + box_size - 1
    y1 = y0 + box_size - 1

    # Clamp to image boundaries.
    x0 = max(0, min(SIZE - 1, x0))
    y0 = max(0, min(SIZE - 1, y0))
    x1 = max(0, min(SIZE - 1, x1))
    y1 = max(0, min(SIZE - 1, y1))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)

    # Use a highly visible outline. The underlying image is
    # grayscale, so alternating bright/dark borders work well.
    outline_width = max(2, SIZE // 250)

    # Outer dark border.
    draw.rectangle(
        [x0, y0, x1, y1],
        outline=(0, 0, 0),
        width=outline_width + 2
    )

    # Inner bright border.
    draw.rectangle(
        [x0, y0, x1, y1],
        outline=(255, 255, 255),
        width=outline_width
    )

    # Add four small corner markers to make the ground-truth
    # region immediately distinguishable without covering it.
    marker_len = max(
        10,
        min(35, box_size // 4)
    )

    marker_width = max(
        2,
        SIZE // 250
    )

    corners = [
        (x0, y0, 1, 1),
        (x1, y0, -1, 1),
        (x0, y1, 1, -1),
        (x1, y1, -1, -1),
    ]

    for cx, cy, sx, sy in corners:
        draw.line(
            [
                (cx, cy),
                (cx + sx * marker_len, cy)
            ],
            fill=(255, 255, 255),
            width=marker_width
        )

        draw.line(
            [
                (cx, cy),
                (cx, cy + sy * marker_len)
            ],
            fill=(255, 255, 255),
            width=marker_width
        )

    return image, (x0, y0, x1, y1)


# ============================================================
# MAIN
# ============================================================

def main():

    seed = secrets.randbits(128)
    random.seed(seed)

    print()
    print("=" * 72)
    print("CONTINUOUS DRAM DATASET GENERATOR")
    print("=" * 72)

    # Ask for defects BEFORE generating any images.
    print("\nAvailable defects:")

    for defect in DEFECTS:
        print(f"  - {defect}")

    print(
        "\nEnter one defect, or multiple defects "
        "separated by commas."
    )

    print("\nDefect serial numbers:")
    print("  0. none")

    defect_names_for_menu = [
        d for d in DEFECTS
        if d != "none"
    ]

    for i, defect in enumerate(
        defect_names_for_menu,
        start=1
    ):
        print(f"  {i}. {defect}")

    all_option = len(defect_names_for_menu) + 1
    print(f"  {all_option}. all_combination")

    print(
        "\nEnter one serial number, or multiple numbers "
        "separated by commas."
    )
    print("Examples:")
    print("  0       -> no defect")
    print("  1       -> first defect")
    print("  1,3,9   -> combination")
    print(
        f"  {all_option}       -> all defects "
        "(one tint only)"
    )

    selected = parse_defects(
        input("\nEnter defect serial number(s): ")
    )


    print(
        "\nGenerating a NEW continuous DRAM "
        "physical layout..."
    )

    (
        reference,
        wide,
        ref_x,
        ref_y,
        architecture,
        global_bit_pitch,
        global_word_pitch,
        intersection_positions,
        hierarchical_network,
        repetition_boxes
    ) = make_pair(
        seed,
        selected
    )

    architecture_names = [
        "large regular DRAM sub-arrays",
        "staggered DRAM banks",
        "dense vertical-bit-line DRAM",
        "sparse large-pitch DRAM banks",
        "sense-amplifier-heavy hierarchical DRAM",
        "mixed DRAM bank topology",
    ]

    print(
        "\nDRAM architecture:"
        f" {architecture_names[architecture]}"
    )

    print(
        "\nGlobal bit-line pitch:"
        f" {global_bit_pitch}"
    )

    print(
        "Global word-line pitch:"
        f" {global_word_pitch}"
    )

    print(
        "\nIntegrated routing intersections:"
        f" {len(intersection_positions)}"
    )

    print(
        "\nLarge routing districts:"
        f" {len(hierarchical_network['large_nodes'])}"
    )

    print(
        "Medium routing areas:"
        f" {len(hierarchical_network['medium_nodes'])}"
    )

    defect_rng = np.random.default_rng(
        seed ^ 0xD7A53C91
    )

    # --------------------------------------------------------------
    # DEFECT ORDER
    #
    # repetition_increase is special:
    #
    #   1. It is already used to construct the wide image.
    #   2. That repetition-only wide image becomes the base image.
    #   3. Every OTHER selected defect is then applied to that base.
    #
    # Therefore:
    #
    #   repetition
    #   repetition + blur
    #   repetition + blur + tint
    #
    # all start from the SAME repetition-generated wide image.
    # --------------------------------------------------------------

    post_generation_defects = [
        defect
        for defect in selected
        if defect != "repetition_increase"
        and defect != "none"
    ]

    if post_generation_defects:
        defective_wide, parameters = apply_defects(
            wide,
            post_generation_defects,
            defect_rng
        )
    else:
        # No other defect: the repetition-generated wide image is
        # already the final search image.
        defective_wide = wide.copy()
        parameters = {}

    if "repetition_increase" in selected:
        parameters["repetition_increase"] = {
            "generated_during_wide_image_creation": True,
            "reference_count": len(repetition_boxes),
            "locations": repetition_boxes
        }

    if "repetition_increase" in selected:
        parameters["repetition_increase"] = {
            "generated_during_wide_image_creation": True,
            "reference_count": len(repetition_boxes),
            "locations": repetition_boxes
        }

    run_dir = (
        OUTPUT_ROOT
        / f"run_{seed:032x}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    reference_path = (
        run_dir / "reference_clean.png"
    )

    wide_clean_path = (
        run_dir / "wide_search_clean.png"
    )

    wide_defective_path = (
        run_dir / "wide_search_defective.png"
    )

    repetition_integrated = (
        "repetition_increase" in selected
    )

    # Ground-truth visualization:
    # shows exactly where the reference crop exists in the
    # 1000x1000 wide-search image.
    reference_location_path = (
        run_dir / "reference_location_ground_truth.png"
    )

    metadata_path = (
        run_dir / "metadata.json"
    )

    if "repetition_increase" in selected:
        reference_location_image, ground_truth_box = (
            create_multiple_reference_locations_image(
                wide,
                repetition_boxes
            )
        )
    else:
        reference_location_image, ground_truth_box = (
            create_reference_location_image(
                wide,
                ref_x,
                ref_y
            )
        )

    save_gray(
        reference,
        reference_path
    )

    save_gray(
        wide,
        wide_clean_path
    )

    save_gray(
        defective_wide,
        wide_defective_path
    )

    reference_location_image.save(
        reference_location_path
    )

    metadata = {
        "seed": str(seed),
        "technology": "DRAM",
        "architecture": architecture_names[architecture],
        "architecture_id": architecture,

        "reference_size": [
            SIZE,
            SIZE
        ],

        "wide_search_size": [
            SIZE,
            SIZE
        ],

        "physical_layout_size": [
            BIG,
            BIG
        ],

        "physical_scale_ratio": "10:1",

        "reference_crop_in_physical_layout": [
            ref_x,
            ref_y,
            SIZE,
            SIZE
        ],

        "reference_expected_location_in_wide_search": [
            ref_x / PHYSICAL_SCALE,
            ref_y / PHYSICAL_SCALE,
            SIZE / PHYSICAL_SCALE,
            SIZE / PHYSICAL_SCALE
        ],

        # Legacy single-box field: keep the first/original reference
        # location so existing downstream code remains compatible.
        "reference_ground_truth_box_xyxy": [
            int(ground_truth_box[0][0])
            if isinstance(ground_truth_box[0], (list, tuple))
            else int(ground_truth_box[0]),

            int(ground_truth_box[0][1])
            if isinstance(ground_truth_box[0], (list, tuple))
            else int(ground_truth_box[1]),

            int(ground_truth_box[0][2])
            if isinstance(ground_truth_box[0], (list, tuple))
            else int(ground_truth_box[2]),

            int(ground_truth_box[0][3])
            if isinstance(ground_truth_box[0], (list, tuple))
            else int(ground_truth_box[3])
        ],

        # In repetition mode this contains EVERY known reference
        # occurrence in the wide-search image.
        "reference_locations_ground_truth_count": (
            len(ground_truth_box)
            if (
                len(ground_truth_box) > 0
                and isinstance(
                    ground_truth_box[0],
                    (list, tuple)
                )
            )
            else 1
        ),

        "reference_locations_ground_truth_xyxy": [
            [
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3])
            ]
            for box in (
                ground_truth_box
                if (
                    len(ground_truth_box) > 0
                    and isinstance(
                        ground_truth_box[0],
                        (list, tuple)
                    )
                )
                else [ground_truth_box]
            )
        ],

        "repetition_increase": {
            "generated_during_wide_image_creation": repetition_integrated,
            "reference_count": len(repetition_boxes),
            "locations_wide_xywh": [
                [
                    float(box["x"]),
                    float(box["y"]),
                    float(box["width"]),
                    float(box["height"])
                ]
                for box in repetition_boxes
            ]
        },

        "ground_truth_visualization": {
            "description": (
                (
                    "All known reference locations are outlined and "
                    "numbered in reference_location_ground_truth.png."
                )
                if "repetition_increase" in selected
                else
                (
                    "The reference location is outlined in "
                    "reference_location_ground_truth.png."
                )
            ),
            "box_format": "xyxy",
            # ground_truth_box is a single [x0,y0,x1,y1] box in the
            # normal case, but a list of boxes in repetition mode.
            # Keep this legacy field pointing to the first occurrence.
            "box": (
                [
                    int(ground_truth_box[0][0]),
                    int(ground_truth_box[0][1]),
                    int(ground_truth_box[0][2]),
                    int(ground_truth_box[0][3])
                ]
                if (
                    len(ground_truth_box) > 0
                    and isinstance(
                        ground_truth_box[0],
                        (list, tuple)
                    )
                )
                else [
                    int(ground_truth_box[0]),
                    int(ground_truth_box[1]),
                    int(ground_truth_box[2]),
                    int(ground_truth_box[3])
                ]
            )
        },

        "global_bit_pitch": global_bit_pitch,
        "global_word_pitch": global_word_pitch,

        "intersection_count": len(
            intersection_positions
        ),

        "intersection_positions_physical": [
            [int(x), int(y)]
            for x, y in intersection_positions
        ],

        "intersection_positions_wide": [
            [
                x / PHYSICAL_SCALE,
                y / PHYSICAL_SCALE
            ]
            for x, y in intersection_positions
        ],

        "hierarchical_network": {
            "large_nodes_physical": (
                hierarchical_network["large_nodes"]
            ),
            "large_nodes_wide": [
                [
                    x / PHYSICAL_SCALE,
                    y / PHYSICAL_SCALE
                ]
                for x, y
                in hierarchical_network["large_nodes"]
            ],
            "medium_nodes_physical": (
                hierarchical_network["medium_nodes"]
            ),
            "medium_nodes_wide": [
                [
                    x / PHYSICAL_SCALE,
                    y / PHYSICAL_SCALE
                ]
                for x, y
                in hierarchical_network["medium_nodes"]
            ]
        },

        "selected_defects": selected,
        "defect_parameters": parameters,

        "files": {
            "reference_clean": str(
                reference_path
            ),
            "wide_search_clean": str(
                wide_clean_path
            ),
            "wide_search_defective": str(
                wide_defective_path
            ),
            "reference_location_ground_truth": str(
                reference_location_path
            ),
        }
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print("=" * 72)
    print("GENERATED")
    print("=" * 72)

    print(
        "\nReference:"
        f"\n  {reference_path}"
    )

    print(
        "\nClean wide search:"
        f"\n  {wide_clean_path}"
    )

    print(
        "\nDefective wide search:"
        f"\n  {wide_defective_path}"
    )

    print(
        "\nReference-location ground truth:"
        f"\n  {reference_location_path}"
    )

    print(
        "\nMetadata:"
        f"\n  {metadata_path}"
    )

    print(
        f"\nReference size: "
        f"{SIZE} x {SIZE}"
    )

    print(
        f"Wide search size: "
        f"{SIZE} x {SIZE}"
    )

    print(
        f"Physical layout: "
        f"{BIG} x {BIG}"
    )

    print("\nPhysical scale ratio: 10:1")

    print(
        "\nEvery execution generates:"
        "\n  - a new continuous DRAM layout"
        "\n  - a new reference crop"
        "\n  - large clean DRAM routing districts"
        "\n  - medium clean connected routing areas"
        "\n  - small DRAM routing intersections"
        "\n  - connected hierarchical routing paths"
        "\n  - the final wide-search image"
        "\n  - reference-location ground-truth image"
        "\n  - ground-truth metadata"
    )

    if repetition_integrated:
        print(
            "\nRepetition mode:"
            f" {len(repetition_boxes)} reference locations "
            "integrated into the physical layout"
        )


if __name__ == "__main__":
    main()