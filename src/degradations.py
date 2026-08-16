"""SEM-style synthetic degradations applied to the wide search image."""
import numpy as np
import cv2
from PIL import Image, ImageFilter
from .config import SIZE
from .image_ops import u8

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
