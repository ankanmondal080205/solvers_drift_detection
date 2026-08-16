"""Repeated target occurrences used to test ambiguity handling."""
import numpy as np
from PIL import Image, ImageFilter
from .config import SIZE, PHYSICAL_SCALE
from .image_ops import u8

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
