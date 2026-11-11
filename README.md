# TCD Pool Vision

Pool-ball localisation and identity recognition using C++17 and OpenCV. The complete application is contained in `myApplication.cpp`, including its `main()` entry point. No development wrapper, custom project headers or diagnostic runner is required.

The program calibrates shape and appearance from `AllBalls.jpeg`, processes the supplied table images, and saves final recognition overlays, localisation evaluation overlays, detected-ball records and performance metrics.

## Requirements

- A C++17 compiler.
- OpenCV 4.12.0 with `core`, `imgproc`, `imgcodecs` and `highgui`.
- CMake 3.20 or later when using the CMake build.
- The original images and supplied ground-truth CSV.

The C++ source is self-contained, but OpenCV and the dataset are still external requirements. `CMakeLists.txt` is a local build convenience rather than an additional source dependency.

## Project layout

```text
TCD-Pool-Vision/
  myApplication.cpp
  CMakeLists.txt
  README.md
  data/
    images/
      AllBalls.jpeg
      Table1.jpeg
      Table2.jpeg
      Table3.jpeg
      Table4.jpeg
      Table5.jpeg
      Table6.jpeg
    ground_truth/
      ground_truth.csv
  results/                  Generated final outputs
  build_submission/         Generated CMake build files
```

Historical diagnostic folders, scripts and helper files may still exist in the repository, but they are not used by this version. No automatic cleanup or deletion of historical material is performed.

## Build on macOS

Run these commands from the repository root. The OpenCV path below uses the installation previously configured on the development machine:

```sh
cmake -S . -B build_submission \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$HOME/Libraries/opencv-4.12.0-install/lib/cmake/opencv4"

cmake --build build_submission --parallel
```

On another machine, replace `OpenCV_DIR` with the directory containing the appropriate `OpenCVConfig.cmake`. The CMake configuration requires OpenCV 4.12.0 exactly.

The executable target must contain only the application source:

```cmake
add_executable(PoolBallVision myApplication.cpp)
```

Do not include `dev_main.cpp`: the application already defines `main()`, and compiling both files would introduce a duplicate entry point. No supplied support `.cpp` files or `PoolDiagnostics.hpp` are needed by the single-file application.

### Build without CMake

If OpenCV is configured through `pkg-config`, the same file can be built directly:

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  myApplication.cpp -o poolvision \
  $(pkg-config --cflags --libs opencv4)

./poolvision ./data ./results
```

This alternative requires `pkg-config` to locate the intended OpenCV installation; it does not use the CMake version check.

## Run

### Process all supplied images

```sh
./build_submission/PoolBallVision ./data ./results
```

The arguments are optional:

```sh
./build_submission/PoolBallVision
```

Defaults are `data` for input and `results` for output, relative to the current working directory. Run from the repository root, or provide explicit paths:

```sh
./build_submission/PoolBallVision /path/to/data /path/to/results
```

By default, the program processes AllBalls and Table1–Table6. AllBalls is always needed to train the geometry and appearance models, even when only one evaluation image is selected. Its 16 labelled annotations are required for appearance training. Missing evaluation annotations cause evaluation to be skipped for that scene.

### Process one image

```sh
POOL_IMAGE=Table5 ./build_submission/PoolBallVision ./data ./results
```

Supported selections are `AllBalls` and `Table1` through `Table6`. A filename such as `Table5.jpeg` is also accepted.

### Display the recognition overlay

```sh
POOL_GUI=1 POOL_IMAGE=Table5 \
  ./build_submission/PoolBallVision ./data ./results
```

The default is non-interactive. With `POOL_GUI=1`, the program displays each processed scene's final recognition overlay and waits for a key press before continuing.

The old diagnostic environment settings, including `POOL_RUN` and `POOL_ARCHIVE`, are not used. Do not use `run_diagnostics.sh` to run this version.

## Final outputs

The application writes a flat output directory rather than a diagnostic-stage hierarchy:

```text
results/
  metrics.csv
  AllBalls_recognised.png
  AllBalls_evaluation.png
  AllBalls_balls.csv
  Table1_recognised.png
  Table1_evaluation.png
  Table1_balls.csv
  ... corresponding files for Table2–Table6
```

| File | Contents |
|---|---|
| `<image>_recognised.png` | Accepted ellipses and predicted ball identities. |
| `<image>_evaluation.png` | Localisation matches, unmatched detections and visible missed annotations; produced when annotations are available. |
| `<image>_balls.csv` | Candidate ID, centre, ellipse axes, orientation, score, predicted and alternative identities, appearance distance and distance margin. |
| `metrics.csv` | Per-image localisation and identity metrics, plus a pooled `TestTotal` row when evaluation annotations are processed. |

The alternative-identity margin is a distance difference, not a calibrated probability. CSV centres are apparent silhouette centres in rectified pixels, not corrected table-contact coordinates.

**Output files are overwritten when their names are reused.** Use a separate output directory to preserve a comparison run:

```sh
./build_submission/PoolBallVision ./data ./results_submission_check
```

`metrics.csv` is recreated on each invocation. When running one image, it contains only that invocation's evaluated scenes. The `TestTotal` row pools only processed evaluation images and always excludes AllBalls. Use a full run for the report's six-table totals. Unprocessed image files from an earlier invocation are not automatically removed.

## Processing pipeline

1. Rectify the table to 1434 × 683 pixels using fixed corner correspondences.
2. Generate foreground evidence using HSV cloth thresholding and morphological opening.
3. Find distance-transform maxima, treating interior and border-connected regions separately.
4. Fit contour ellipses and recover whole-ball hypotheses from calibrated templates.
5. Validate geometry, visible boundary support and foreground contrast.
6. Reject persistent pocket structure using AllBalls reference evidence.
7. Suppress duplicate hypotheses, including clipped candidates at image boundaries.
8. Assign identities using hue, brightness, white/dark fractions and nearest-prototype comparison.
9. Evaluate localisation with one-to-one matching and evaluate identity on those matches.

The independent diagnostic Hough branch, intermediate masks, contact sheets, per-candidate audit exports, settings manifests and source snapshots have been removed. The retained outputs are final application results and evaluation records.

## Evaluation and limitations

- AllBalls supplies shape and appearance training and is excluded from pooled test metrics.
- Table1–Table6 annotations are used downstream for evaluation, not candidate generation.
- The supplied CSV uses row/column coordinates; the application converts them to image x/y coordinates.
- Localisation matching uses a fixed 21.5-pixel tolerance, maximising valid pair count before minimising total centre distance.
- Matched identity accuracy measures naming among matched detections. End-to-end identity metrics additionally account for unmatched detections and annotations.
- Out-of-image annotation entries remain in primary denominators and cannot match. Blank coordinate pairs are skipped.
- Five AllBalls training-label reassignments from the existing implementation are retained; supplied evaluation labels remain unchanged.
- Parameter development involved inspecting evaluation failures. Report comparisons as retrospective development on this collection, rather than independent validation on unseen scenes.
- The method assumes a fixed camera and table arrangement. Dense clusters, occlusion, lighting changes and solid/stripe appearance variation remain limitations.

The refactor is intended to preserve detection and recognition behaviour. Core numerical routines were checked against the original source, but compilation and full-dataset output equivalence were not verified in the editing environment because OpenCV C++ headers and the original dataset directory were unavailable. Run the final source locally and compare its metrics with the report before submission. Historical results should not be presented as a newly verified run of the reduced source.

## Source map

| Symbol | Responsibility |
|---|---|
| `prep::Settings` | Preprocessing, segmentation and seed settings. |
| `prep::Ellipses` | Contour ellipse proposals in the original traversal order. |
| `ballvision::LearnShape` | AllBalls outline calibration. |
| `ballvision::Select` | Recovery, validation, pocket rejection and duplicate suppression. |
| `ballid::Train` / `ballid::Recognise` | Appearance prototypes and identity predictions. |
| `BuildForegroundMask` / `FindSeeds` | Foreground segmentation and distance maxima. |
| `LoadGroundTruth` / `MatchLocalisation` | Annotation loading and one-to-one assignment. |
| `Evaluate` / `WriteMetrics` | Evaluation overlay and summary metrics. |
| `Run` / `main` | Inputs, stage sequence, outputs and error handling. |
