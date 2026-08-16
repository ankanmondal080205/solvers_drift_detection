"""Hierarchical clean routing network used by the DRAM layout."""

import numpy as np
from .image_ops import hline, vline, disk, ring
from .config import BIG
from .dram_patterns import correlated_pitch, correlated_value
from .routing import draw_integrated_intersection

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