# Drift-Sense
## AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

This repository contains the synthetic-data generation and image-localization solution developed for the Applied Materials Drift-Sense problem.

The solution is designed for the cross-magnification localization task:

- **Reference image:** 1000 × 1000 grayscale image representing the 100× close-up view.
- **Search image:** 1000 × 1000 grayscale image representing the wider 10× view.
- **Nominal scale relationship:** 10:1, with robustness testing around 9:1–11:1.
- **Rotation variation:** approximately 1–2°.
- **Output:** predicted target-centre coordinates `(x, y)` in search-image pixels.
- **Coordinate convention:** origin `(0, 0)` is at the top-left; `x` increases to the right and `y` increases downward.

The supplied problem statement requires the solution to explicitly account for the scale difference, handle repeated patterns, generate reproducible synthetic data, and select the valid repeated match closest to the centre of the search image.

---

## Repository Layout

```text
submission/
│
├── solution_presentation.pptx
│   └── Solution presentation covering the problem, approach,
│       synthetic-data generation, localization, experiments,
│       results, runtime and failure analysis.
│
├── README.md
│   └── Repository documentation, setup and execution instructions.
│
├── requirements.txt
│   └── Python dependencies required to run the solution.
│
├── generate_dataset.py
│   └── Entry point for generating synthetic DRAM-style image pairs.
│
├── localize.py
│   └── Entry point for reference-to-search-image localization.
│
├── configs/
│   └── Configuration files and experiment settings.
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── image_ops.py
│   ├── dram_patterns.py
│   ├── routing.py
│   ├── clean_network.py
│   ├── layout.py
│   ├── degradations.py
│   ├── repetitions.py
│   ├── ground_truth.py
│   ├── cli_utils.py
│   ├── localization_config.py
│   ├── localization_io.py
│   ├── localization_matching.py
│   ├── localization_selection.py
│   ├── localization_visualization.py
│   └── localization_pipeline.py
│
├── model/
│   └── Optional model files/weights if a learned model is used.
│
├── results/
│   └── Generated evaluation outputs, metrics, plots,
│       overlays and failure-case visualizations.
│
└── references/
    └── Public references supporting DRAM structures,
        SEM image formation, degradation modelling and
        synthetic-data choices.
```

> `model/` is optional and is not required for the current classical computer-vision solution.

---

# 1. Problem Overview

The Drift-Sense task is a visual navigation-recovery problem for semiconductor wafer inspection.

A high-resolution **100× reference image** identifies the structure previously inspected. A lower-resolution **10× search image** covers a larger physical area and contains the same structure somewhere inside it.

The objective is to locate the reference structure inside the search image and return its centre coordinates.

Because semiconductor layouts contain highly repetitive structures, several locations may produce valid visual matches. When multiple valid matches exist, the selected location is the one whose centre is closest to the centre of the search image.

---

# 2. Solution Overview

The repository contains two connected components.

### A. Synthetic Dataset Generation

`generate_dataset.py` generates reproducible DRAM-style semiconductor layouts and creates:

1. 1000 × 1000 reference images.
2. 1000 × 1000 clean search images.
3. Defective/degraded search images.
4. Ground-truth target-location information.
5. Per-pair metadata.
6. Random seed and transformation/degradation parameters.

The generator models a continuous DRAM-style routing fabric and includes different structural regions, routing intersections and repeated target occurrences.

### B. Reference Localization

`localize.py` receives:

- a high-resolution reference image, and
- a lower-resolution search image.

The localization pipeline:

1. Loads and validates both images.
2. Removes the artificial reference-image border where required.
3. Explicitly searches across the 9:1–11:1 scale range.
4. Downsamples the reference to candidate search-image scales.
5. Performs normalized cross-correlation (NCC).
6. Detects multiple local maxima rather than using only one global maximum.
7. Removes duplicate detections belonging to the same spatial occurrence.
8. Keeps strong, spatially distinct candidate locations.
9. Selects the candidate closest to the search-image centre.
10. Reports the predicted target centre, score, scale and runtime.
11. Saves a visualization of the detected locations.

---

# 3. Synthetic Data Generation

The generator is based on DRAM-style structural knowledge and participant-created synthetic data.

The generated physical layout is constructed at a higher internal resolution and then converted into the 1000 × 1000 search-image representation.

The reference crop is extracted from the same physical layout so that its ground-truth location is known exactly.

The generator preserves the required nominal 10:1 physical relationship.

## Synthetic structures

The generator includes DRAM-style elements such as:

- continuous routing lines,
- word-line and bit-line structures,
- repeated memory-cell patterns,
- standard subarrays,
- staggered subarrays,
- dense columnar regions,
- sparse banks,
- sense-amplifier regions,
- routing intersections,
- connected routing paths,
- hierarchical routing regions.

The structural parameters are randomized while maintaining correlated local characteristics so that each generated case is not simply a pasted copy of another case.

---

# 4. Image Degradations

The search image can be affected by several synthetic SEM-style degradations.

Implemented examples include:

- blur,
- small angular tilt,
- scale mismatch,
- brightness/contrast tint,
- scanline noise,
- line dropouts,
- brightness drift,
- repeated-pattern increase,
- edge-related stress,
- other configured defects.

The degradation parameters are stored in the generated metadata so that experiments remain reproducible.

The target repetition mode can introduce additional occurrences of the reference pattern into the search image. This is important because the problem specifically tests ambiguity caused by repeated semiconductor structures.

---

# 5. Localization Method

The localization system uses a scale-aware classical computer-vision approach.

## Multi-scale matching

The reference image cannot be matched directly against the search image because the two images represent different magnifications.

For every candidate scale:

```text
reference image
       │
       ▼
downsample to candidate scale
       │
       ▼
template matching against search image
       │
       ▼
NCC response map
       │
       ▼
local candidate peaks
```

The current search range covers approximately:

```text
9.0x → 11.0x
```

This explicitly handles the nominal 10:1 magnification relationship while allowing scale mismatch.

## Rotation handling

Small rotation differences are also considered within the approximately ±2° range expected by the problem.

## Candidate selection

The algorithm does not blindly select the first or highest response.

It:

1. Finds the strongest NCC response.
2. Keeps candidates within the configured score window of the best response.
3. Suppresses duplicate detections belonging to the same physical location.
4. Retains multiple distinct candidate locations.
5. Computes each candidate's distance from the centre of the search image.
6. Selects the valid candidate with minimum centre distance.

For a 1000 × 1000 search image, the search-image centre is:

```text
(500, 500)
```

The final output is the selected centre in search-image pixel coordinates.

---

# 6. Coordinate Convention

All coordinates follow the problem statement convention:

```text
(0, 0) ───────────────► x
  │
  │
  │
  ▼
  y
```

Therefore:

- `(0, 0)` = top-left.
- `x` increases from left to right.
- `y` increases from top to bottom.
- The returned coordinate represents the centre of the predicted target region.

---

# 7. Installation

Create a clean Python environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 8. Generate a Dataset Pair

Run:

```bash
python generate_dataset.py
```

The generator creates a new randomized synthetic case.

A generated run contains files similar to:

```text
navigation_dataset/
└── run_<seed>/
    ├── reference_clean.png
    ├── wide_search_clean.png
    ├── wide_search_defective.png
    ├── reference_location_ground_truth.png
    └── metadata.json
```

The metadata records the generation seed, DRAM architecture information, scale-related information, selected defects, defect parameters, target locations and generated file paths.

---

# 9. Run Localization

Run:

```bash
python localize.py
```

The program asks for:

```text
Path to reference image:
Path to wide search image:
```

Provide the corresponding image paths.

Example:

```text
Path to reference image: navigation_dataset/run_xxx/reference_clean.png
Path to wide search image: navigation_dataset/run_xxx/wide_search_defective.png
```

The program reports information including:

```text
Best NCC match
Selected scale
Valid candidate locations
Selected target centre
Distance from search-image centre
Runtime
```

A visualization is also generated showing the candidate locations and the selected match.

---

# 10. Reproducibility

Each generated dataset run stores a random seed and generation metadata.

The metadata is intended to record:

- random seed,
- technology,
- DRAM architecture,
- physical layout parameters,
- routing information,
- target location,
- scale,
- rotation,
- selected defects,
- defect parameters,
- generated image paths,
- ground-truth information.

This allows generated cases to be traced back to their generation configuration.

---

# 11. Validation and Evaluation

The problem statement specifies validation on at least **30 varied, independently generated pairs**.

Evaluation should include:

### Localization error

```text
error = sqrt(
    (x_pred - x_true)^2 +
    (y_pred - y_true)^2
)
```

### Pass-rate thresholds

Report the percentage of cases satisfying:

```text
error ≤ 5 pixels
error ≤ 4 pixels
error ≤ 2 pixels
error ≤ 1 pixel
```

Sub-pixel performance can also be reported where supported.

### Error statistics

Report:

- mean error,
- median error,
- worst-case error.

### Runtime

Report:

- runtime per image pair,
- Python version,
- hardware,
- timing method.

### Robustness

Evaluate cases with variation in:

- noise/degradation level,
- target position,
- scale,
- rotation,
- repeated patterns,
- edge positions.

### Failure analysis

At least one genuine failure case should be visualized and explained, including the likely root cause.

No evaluation values should be fabricated. Results in `results/` should correspond to actual executions of the submitted code.

---

# 12. Results Directory

Use the `results/` directory for generated evaluation artifacts.

Recommended contents:

```text
results/
├── metrics.csv
├── evaluation_summary.csv
├── localization_overlays/
├── robustness/
├── failures/
└── runtime/
```

The exact filenames can be chosen according to the team's experiment scripts.

---

# 13. References Directory

Use `references/` for public material supporting:

- DRAM structural assumptions,
- semiconductor layouts,
- SEM imaging,
- image degradation/noise modelling,
- blur and downsampling,
- synthetic-data design choices.

The final presentation and documentation should cite at least 2–3 credible public sources, as required by the problem statement.

---

# 14. Configuration

Experiment-specific settings should be placed under:

```text
configs/
```

This keeps configuration separate from the reusable implementation in `src/`.

Typical configurable values include:

- image size,
- physical scale,
- localization scale range,
- scale step,
- score window,
- candidate suppression radius,
- number of generated cases,
- degradation selection,
- random seed settings.

---

# 15. Source Code Organization

The `src/` directory contains the reusable implementation.

### Dataset generation

```text
config.py
image_ops.py
dram_patterns.py
routing.py
clean_network.py
layout.py
degradations.py
repetitions.py
ground_truth.py
cli_utils.py
```

### Localization

```text
localization_config.py
localization_io.py
localization_matching.py
localization_selection.py
localization_visualization.py
localization_pipeline.py
```

The top-level scripts should remain lightweight entry points, while the implementation remains inside `src/`.

---

# 16. Design Rationale

The solution deliberately uses a classical computer-vision approach rather than requiring a deep-learning model.

Advantages include:

- no external model weights,
- no training dependency,
- transparent scale handling,
- interpretable NCC scores,
- easy visualization of candidate matches,
- reproducible execution,
- straightforward deployment in a clean Python environment.

The main challenge is not simply finding the highest correlation. Repeated semiconductor structures can produce several strong matches. Therefore, candidate generation, spatial de-duplication and the search-centre decision rule are treated as explicit parts of the localization pipeline.

---

# 17. Limitations

The current solution is a synthetic-data and classical-computer-vision approach.

Potential limitations include:

- very severe degradation can destroy discriminative features;
- highly repetitive regions can produce ambiguous NCC peaks;
- extreme rotation or scale changes outside the tested range may reduce accuracy;
- synthetic DRAM structures cannot reproduce every physical characteristic of real SEM imagery;
- runtime depends on the number of scales and rotation candidates evaluated.

These limitations should be quantified using actual validation experiments rather than assumed performance values.

---

# 18. Submission Checklist

Before creating the final GitHub submission, verify:

- [ ] `solution_presentation.pptx` is included.
- [ ] `README.md` is included.
- [ ] `requirements.txt` is included.
- [ ] `generate_dataset.py` is runnable.
- [ ] `localize.py` is runnable.
- [ ] `configs/` is included where required.
- [ ] `src/` contains the reusable implementation.
- [ ] `model/` is included only if a model/weights are used.
- [ ] `results/` contains actual evaluation outputs.
- [ ] `references/` contains the public supporting sources.
- [ ] 1000 × 1000 grayscale reference/search images are used.
- [ ] The 10:1 relationship is explicitly implemented.
- [ ] 9:1–11:1 scale robustness is tested.
- [ ] Approximately 1–2° rotation variation is tested.
- [ ] Top-left coordinate convention is documented.
- [ ] Closest-to-search-centre selection is implemented.
- [ ] At least 30 varied cases are evaluated.
- [ ] 5-, 4-, 2- and 1-pixel pass rates are reported.
- [ ] Mean, median and worst-case errors are reported.
- [ ] Runtime, hardware and timing method are documented.
- [ ] At least one genuine failure case is analyzed.
- [ ] CSV/manifest information is preserved for generated cases.
- [ ] At least 2–3 credible public references are included.
- [ ] No confidential/proprietary fab data is included.
- [ ] No hard-coded local machine paths remain.
- [ ] The repository has been tested in a clean environment.

---

# 19. Quick Start

From the repository root:

```bash
pip install -r requirements.txt
python generate_dataset.py
python localize.py
```

The generated dataset and localization outputs can then be used to populate `results/` and the quantitative evaluation reported in the solution presentation.

---

## Problem-Statement Alignment

This repository is organized around the supplied Applied Materials Drift-Sense requirements:

- synthetic DRAM/FinFET-style data generation,
- 100× reference to 10× search localization,
- explicit scale handling,
- robustness to scale and rotation variation,
- repeated-pattern handling,
- centre-based candidate selection,
- reproducible metadata,
- separate generator and localization programs,
- measurable validation and failure analysis,
- GitHub-ready source organization.
