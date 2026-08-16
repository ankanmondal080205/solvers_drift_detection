import os
import time
import numpy as np
import cv2
from scipy.ndimage import maximum_filter


# ==============================================================
# USER SETTINGS
# ==============================================================

# Example:
# Reference = 1000x1000 high-resolution capture
# Wide      = 1000x1000 coarse-resolution capture covering 10x
# physical field in each dimension.

BORDER_TRIM = 8

# Nominal scale difference.
# Search a range around 10x rather than assuming it is exact.
SCALE_START = 9
SCALE_END = 11
SCALE_STEP = 0.2

# Keep every match whose NCC score is within 10 percentage
# POINTS of the best match.
#
# Example:
# best = 0.95
# threshold = 0.85
#
SCORE_WINDOW = 0.05

# Suppress duplicate detections caused by nearby scales/positions.
NMS_RADIUS_FRAC = 0.5


# ==============================================================
# STEP 1 — ASK FOR IMAGES
# ==============================================================

REF_PATH = input("Path to reference image: ").strip().strip('"')
WIDE_PATH = input("Path to wide search image: ").strip().strip('"')


# ==============================================================
# STEP 2 — CHECK FILES
# ==============================================================

for label, path in [
    ("Reference", REF_PATH),
    ("Wide search", WIDE_PATH)
]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} image not found at:\n{path}"
        )


# ==============================================================
# STEP 3 — LOAD IMAGES
# ==============================================================

reference = cv2.imread(
    REF_PATH,
    cv2.IMREAD_GRAYSCALE
)

wide = cv2.imread(
    WIDE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if reference is None:
    raise ValueError(
        f"Could not read reference image:\n{REF_PATH}"
    )

if wide is None:
    raise ValueError(
        f"Could not read wide search image:\n{WIDE_PATH}"
    )


# Convert to float32 for template matching
reference = reference.astype(np.float32)
wide = wide.astype(np.float32)


print("\nLoaded images:")
print(
    f"  Reference: {reference.shape[1]} x {reference.shape[0]}"
)
print(
    f"  Wide:      {wide.shape[1]} x {wide.shape[0]}"
)


# ==============================================================
# STEP 4 — TRIM ARTIFICIAL REFERENCE BORDER
# ==============================================================

# The outermost pixels of the reference are a crop boundary.
# Removing them prevents the artificial crop edge / SEM edge
# brightening from becoming a matching feature.

if BORDER_TRIM > 0:

    h, w = reference.shape

    if 2 * BORDER_TRIM < min(h, w):

        reference = reference[
            BORDER_TRIM:h - BORDER_TRIM,
            BORDER_TRIM:w - BORDER_TRIM
        ]

        print(
            f"\nTrimmed {BORDER_TRIM}px border from reference."
        )

        print(
            f"New reference size: "
            f"{reference.shape[1]} x {reference.shape[0]}"
        )

    else:

        print(
            "\nBORDER_TRIM is too large; "
            "skipping reference trimming."
        )


# ==============================================================
# STEP 5 — MULTI-SCALE TEMPLATE MATCHING
# ==============================================================

print("\nRunning multi-scale NCC template matching...")

runtime_start = time.perf_counter()

all_points = []

scales = np.arange(
    SCALE_START,
    SCALE_END + SCALE_STEP / 2,
    SCALE_STEP
)

for scale in scales:

    # ----------------------------------------------------------
    # Convert the high-resolution reference into wide-image
    # pixel scale.
    #
    # Example:
    #
    # reference = 1000 px
    # scale    = 10
    #
    # template = 100 px
    # ----------------------------------------------------------

    tmpl_size = int(
        round(reference.shape[0] / scale)
    )

    # Ignore impossible template sizes
    if tmpl_size < 8:
        continue

    if tmpl_size >= wide.shape[0]:
        continue

    if tmpl_size >= wide.shape[1]:
        continue


    # ----------------------------------------------------------
    # Downsample reference to current candidate scale
    # ----------------------------------------------------------

    template = cv2.resize(
        reference,
        (tmpl_size, tmpl_size),
        interpolation=cv2.INTER_AREA
    )


    # ----------------------------------------------------------
    # NORMALIZED CROSS CORRELATION
    # ----------------------------------------------------------

    result = cv2.matchTemplate(
        wide,
        template,
        cv2.TM_CCOEFF_NORMED
    )


    # ----------------------------------------------------------
    # Find local maxima.
    #
    # We don't want only the global maximum at this stage.
    # We want all potentially interesting locations.
    # ----------------------------------------------------------

    win = max(
        3,
        tmpl_size // 2
    )

    local_max = maximum_filter(
        result,
        size=win,
        mode="reflect"
    )

    peak_mask = (
        result == local_max
    )

    ys, xs = np.where(
        peak_mask
    )


    # ----------------------------------------------------------
    # Store every local candidate
    # ----------------------------------------------------------

    for x, y in zip(xs, ys):

        score = float(
            result[y, x]
        )

        all_points.append(
            (
                score,
                float(scale),
                int(x),
                int(y),
                int(tmpl_size)
            )
        )


    print(
        f"  scale={scale:.2f}x "
        f"template={tmpl_size}x{tmpl_size} "
        f"candidates={len(xs)}"
    )


# ==============================================================
# STEP 6 — MAKE SURE WE FOUND SOMETHING
# ==============================================================

if not all_points:

    raise RuntimeError(
        "No template-matching candidates were found."
    )


# ==============================================================
# STEP 7 — FIND THE SINGLE GLOBAL BEST MATCH
# ==============================================================

all_points.sort(
    key=lambda p: p[0],
    reverse=True
)

(
    best_score,
    best_scale,
    best_x,
    best_y,
    best_template_size
) = all_points[0]


# ==============================================================
# STEP 8 — CREATE THE 10-PERCENTAGE-POINT WINDOW
# ==============================================================

score_threshold = (
    best_score - SCORE_WINDOW
)


print("\n")
print("=" * 70)
print("GLOBAL BEST MATCH")
print("=" * 70)

print(
    f"Best NCC score : "
    f"{best_score * 100:.2f}%"
)

print(
    f"Best scale     : "
    f"{best_scale:.2f}x"
)

print(
    f"Best position  : "
    f"({best_x}, {best_y})"
)

print(
    f"Template size  : "
    f"{best_template_size} x {best_template_size}px"
)

print("\n")
print("=" * 70)
print("ACCEPTANCE WINDOW")
print("=" * 70)

print(
    f"Best score     : "
    f"{best_score * 100:.2f}%"
)

print(
    f"10-point lower : "
    f"{score_threshold * 100:.2f}%"
)

print(
    f"\nKeeping ALL candidates with NCC between "
    f"{score_threshold * 100:.2f}% and "
    f"{best_score * 100:.2f}%"
)


# ==============================================================
# STEP 9 — KEEP ALL STRONG, SPATIALLY DISTINCT MATCHES
# ==============================================================

# First keep every candidate whose NCC is inside the requested
# 10-percentage-point window from the global best.
score_threshold = best_score - SCORE_WINDOW

qualified = [
    candidate
    for candidate in all_points
    if candidate[0] >= score_threshold
]

# Sort strongest first.
qualified.sort(
    key=lambda p: p[0],
    reverse=True
)

print(
    f"\nCandidates inside score window: "
    f"{len(qualified)}"
)


# ==============================================================
# STEP 10 — COLLAPSE ONLY DUPLICATES OF THE SAME LOCATION
# ==============================================================

# The same physical occurrence appears at many nearby scales.
# Therefore, keep the best candidate for each spatial occurrence.
#
# IMPORTANT:
# Do NOT collapse different repetition locations merely because
# their NCC scores are similar. Spatial separation determines whether
# they are the same occurrence.

distinct = []

for candidate in qualified:

    (
        score,
        scale,
        x,
        y,
        template_size
    ) = candidate

    cx = (
        x + template_size / 2.0
    )
    cy = (
        y + template_size / 2.0
    )

    duplicate = False

    for accepted in distinct:

        (
            accepted_score,
            accepted_scale,
            accepted_x,
            accepted_y,
            accepted_size
        ) = accepted

        accepted_cx = (
            accepted_x
            + accepted_size / 2.0
        )

        accepted_cy = (
            accepted_y
            + accepted_size / 2.0
        )

        # Only merge candidates that are genuinely the same
        # spatial occurrence. Use a conservative radius.
        suppression_radius = (
            0.35
            * max(
                template_size,
                accepted_size
            )
        )

        distance = np.sqrt(
            (cx - accepted_cx) ** 2
            +
            (cy - accepted_cy) ** 2
        )

        if distance <= suppression_radius:
            duplicate = True
            break

    if not duplicate:
        distinct.append(candidate)


# ==============================================================
# STEP 11 — KEEP TOP 5 DISTINCT LOCATIONS
# ==============================================================

# If more than five genuine candidate locations are inside the
# score window, keep the five strongest.
MAX_MATCHES = 5

distinct.sort(
    key=lambda p: p[0],
    reverse=True
)

if len(distinct) > MAX_MATCHES:
    distinct = distinct[:MAX_MATCHES]


print(
    f"Distinct valid pattern locations: "
    f"{len(distinct)}"
)

for rank, candidate in enumerate(
    distinct,
    start=1
):
    (
        score,
        scale,
        x,
        y,
        template_size
    ) = candidate

    print(
        f"  Location #{rank}: "
        f"NCC={score * 100:.2f}% "
        f"center=("
        f"{x + template_size / 2.0:.2f},"
        f"{y + template_size / 2.0:.2f})"
    )


# ==============================================================
# SELECT VALID MATCH CLOSEST TO SEARCH-IMAGE CENTRE
# ==============================================================

search_h, search_w = wide.shape
search_center_x = search_w / 2.0
search_center_y = search_h / 2.0

if not distinct:
    raise RuntimeError(
        "No distinct matches survived the score window."
    )

def centre_distance_squared(candidate):
    score, scale, x, y, template_size = candidate
    cx = x + template_size / 2.0
    cy = y + template_size / 2.0

    return (
        (cx - search_center_x) ** 2
        + (cy - search_center_y) ** 2
    )

selected_match = min(
    distinct,
    key=centre_distance_squared
)

(
    selected_score,
    selected_scale,
    selected_x,
    selected_y,
    selected_template_size
) = selected_match

selected_center_x = (
    selected_x + selected_template_size / 2.0
)

selected_center_y = (
    selected_y + selected_template_size / 2.0
)

selected_distance = float(
    np.sqrt(
        centre_distance_squared(selected_match)
    )
)



# ==============================================================
# PRINT FINAL MATCHES
# ==============================================================

print("\n")
print("=" * 70)
print("FINAL MATCHES")
print("=" * 70)

print(
    f"Number of final matches: {len(distinct)}"
)

for rank, candidate in enumerate(
    distinct,
    start=1
):

    (
        score,
        scale,
        x,
        y,
        template_size
    ) = candidate

    print(
        f"#{rank}  "
        f"NCC={score * 100:.2f}%  "
        f"scale={scale:.2f}x  "
        f"position=({x},{y})  "
        f"center=({x + template_size / 2.0:.2f},"
        f"{y + template_size / 2.0:.2f})  "
        f"size={template_size}px"
    )


# ==============================================================
# STEP 13 — CREATE VISUALIZATION
# ==============================================================

marked = cv2.cvtColor(
    wide.astype(np.uint8),
    cv2.COLOR_GRAY2BGR
)


# Colors for visualization
colors = [
    (0, 0, 255),       # red
    (0, 128, 255),     # orange
    (0, 255, 255),     # yellow
    (0, 255, 128),     # green
    (255, 128, 0),     # blue
    (255, 0, 255),     # magenta
]


for rank, candidate in enumerate(
    distinct,
    start=1
):

    (
        score,
        scale,
        x,
        y,
        template_size
    ) = candidate


    color = colors[
        min(
            rank - 1,
            len(colors) - 1
        )
    ]


    # ----------------------------------------------------------
    # Draw match rectangle
    # ----------------------------------------------------------

    cv2.rectangle(
        marked,
        (x, y),
        (
            x + template_size,
            y + template_size
        ),
        color,
        2
    )


    # ----------------------------------------------------------
    # Label
    # ----------------------------------------------------------

    label = (
        f"#{rank} "
        f"{score * 100:.1f}%"
    )


    cv2.putText(
        marked,
        label,
        (
            x,
            max(
                15,
                y - 6
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA
    )


# ==============================================================
# STEP 14 — MARK THE SELECTED MATCH
# ==============================================================

# If several valid matches exist, choose the one whose centre is
# closest to the centre of the search image.

cv2.rectangle(
    marked,
    (selected_x, selected_y),
    (
        selected_x + selected_template_size,
        selected_y + selected_template_size
    ),
    (0, 0, 255),
    3
)

cv2.drawMarker(
    marked,
    (
        int(round(selected_center_x)),
        int(round(selected_center_y))
    ),
    (0, 0, 255),
    markerType=cv2.MARKER_CROSS,
    markerSize=18,
    thickness=2,
    line_type=cv2.LINE_AA
)

cv2.putText(
    marked,
    "SELECTED",
    (
        selected_x,
        min(
            wide.shape[0] - 5,
            selected_y + selected_template_size + 16
        )
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 255),
    2,
    cv2.LINE_AA
)


# STEP 15 — SAVE RESULT IN "opt" FOLDER
# ==============================================================

# Get the folder containing the wide search image
wide_directory = (
    os.path.dirname(WIDE_PATH)
    or "."
)

# Create/use the "opt" folder inside that directory
output_directory = os.path.join(
    wide_directory,
    "opt"
)

os.makedirs(
    output_directory,
    exist_ok=True
)

# Output filename
out_path = os.path.join(
    output_directory,
    "wide_search_matches_within_10pct_of_best.png"
)


success = cv2.imwrite(
    out_path,
    marked
)

if not success:
    raise RuntimeError(
        f"Could not save output image:\n{out_path}"
    )

# ==============================================================
# STEP 16 — FINAL SUMMARY
# ==============================================================

runtime_seconds = time.perf_counter() - runtime_start

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Best NCC match:   {best_score * 100:.2f}%"
)

print(
    f"Accepted minimum: {score_threshold * 100:.2f}%"
)

print(
    f"Valid matches:    {len(distinct)}"
)

print(
    f"Selected NCC:     {selected_score * 100:.2f}%"
)

print(
    f"Search centre:    "
    f"({search_center_x:.2f}, {search_center_y:.2f})"
)

print(
    f"Marked location:  "
    f"({selected_center_x:.2f}, {selected_center_y:.2f})"
)

print(
    f"Distance to centre: "
    f"{selected_distance:.2f}px"
)

print()
print("=" * 70)
print("PREDICTED TARGET CENTRES — ALL VALID MATCHES")
print("=" * 70)

for rank, candidate in enumerate(
    distinct,
    start=1
):
    (
        score,
        scale,
        x,
        y,
        template_size
    ) = candidate

    center_x = (
        x + template_size / 2.0
    )
    center_y = (
        y + template_size / 2.0
    )

    distance = float(
        np.sqrt(
            (
                center_x - search_center_x
            ) ** 2
            +
            (
                center_y - search_center_y
            ) ** 2
        )
    )

    primary = "  <-- PRIMARY (closest to search centre)"         if candidate == selected_match         else ""

    print(
        f"#{rank}: "
        f"center=({center_x:.2f}, {center_y:.2f}), "
        f"NCC={score * 100:.2f}%, "
        f"scale={scale:.2f}x, "
        f"distance_to_search_centre={distance:.2f}px"
        f"{primary}"
    )

print(
    f"Output image:     {out_path}"
)

print()
print("=" * 70)
print("PREDICTED OUTPUT")
print("=" * 70)

print(
    "Predicted target-centre coordinates (x, y) "
    "in search-image pixels."
)

print(
    "Origin (0, 0) is top-left; "
    "x increases right and y increases downward."
)

print()
print(
    f"Number of predicted target locations: "
    f"{len(distinct)}"
)

for rank, candidate in enumerate(
    distinct,
    start=1
):
    (
        score,
        scale,
        x,
        y,
        template_size
    ) = candidate

    center_x = (
        x + template_size / 2.0
    )
    center_y = (
        y + template_size / 2.0
    )

    print(
        f"Prediction #{rank}: "
        f"({center_x:.2f}, {center_y:.2f}) "
        f"NCC={score * 100:.2f}%"
    )

print()
print(
    f"Primary/selected target centre: "
    f"({selected_center_x:.2f}, {selected_center_y:.2f})"
)

print(
    f"Marked location coordinates: "
    f"({selected_center_x:.2f}, {selected_center_y:.2f})"
)

print(
    f"Runtime: {runtime_seconds:.4f} seconds"
)

print("=" * 70)