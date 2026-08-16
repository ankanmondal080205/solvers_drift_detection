"""Ground-truth location overlays for generated pairs."""
import numpy as np
from PIL import Image
from PIL import ImageDraw
from .config import SIZE, PHYSICAL_SCALE
from .image_ops import u8

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
