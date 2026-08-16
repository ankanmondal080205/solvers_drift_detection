"""Shared generator configuration for Drift-Sense synthetic data."""
from pathlib import Path

SIZE = 1000
PHYSICAL_SCALE = 10
BIG = SIZE * PHYSICAL_SCALE
OUTPUT_ROOT = Path("navigation_dataset")

DEFECTS = [
    "blur", "angle_tilt", "repetition_increase", "edge_stress",
    "scale_mismatch", "black_tint", "white_tint",
    "scanline_noise", "line_dropouts", "brightness_drift", "none",
]
