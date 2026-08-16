"""DRAM structural primitives used by the synthetic layout generator."""
import numpy as np
from .image_ops import rect, hline, vline, disk, ring
from .config import BIG

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
