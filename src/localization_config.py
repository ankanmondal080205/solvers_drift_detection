"""Configuration for cross-magnification localization."""

BORDER_TRIM = 8
SCALE_START = 9.0
SCALE_END = 11.0
SCALE_STEP = 0.2

# The PDF asks for small rotation changes of about 1-2 degrees.
ANGLE_START = -2.0
ANGLE_END = 2.0
ANGLE_STEP = 1.0

# Candidates within this NCC distance from the best score are retained.
SCORE_WINDOW = 0.05

# Fraction of the larger candidate template used for spatial NMS.
NMS_RADIUS_FRAC = 0.35
MAX_MATCHES = 5
MAX_PEAKS_PER_TRANSFORM = 100
