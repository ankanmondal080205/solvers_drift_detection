"""Physical DRAM layout and paired reference/search-image generation."""
import numpy as np
from PIL import Image, ImageFilter
from .config import SIZE, PHYSICAL_SCALE, BIG, OUTPUT_ROOT
from .image_ops import u8, rect, hline, vline, disk
from .dram_patterns import (
    make_background, draw_global_fabric, correlated_pitch, correlated_value,
    draw_standard_subarray, draw_staggered_subarray, draw_dense_columnar_subarray,
    draw_sparse_bank, draw_sense_amp_bank,
)
from .routing import draw_integrated_intersection, connect_intersections
from .clean_network import draw_clean_hierarchical_network
from .repetitions import add_reference_repetitions

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
