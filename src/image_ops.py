"""Low-level grayscale drawing and image helpers."""
import numpy as np
from PIL import Image
from .config import BIG

def u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def save_gray(a, path):
    Image.fromarray(u8(a), "L").save(path)


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
