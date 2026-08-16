"""Integrated routing-intersection primitives."""
import numpy as np
from .image_ops import hline, vline, disk, ring
from .dram_patterns import correlated_pitch, correlated_value

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
