# TCD Pool Vision

C++17/OpenCV pool-ball localisation project. All assignment-specific processing, diagnostics and evaluation are in **myApplication.cpp** and run through **MyApplication()**. The small dev_main.cpp file is only the local executable wrapper.

## Start here

1. Edit myApplication.cpp. The source uses four spaces, Allman braces and descriptive stage comments.
2. Use **AllBalls only** for training and parameter selection. Table1–Table6 are evaluation images, not tuning examples.
3. Run `zsh run_diagnostics.sh AllBalls` from this directory.
4. Read `diagnostics/latest.txt` to find the last completed run, then open its `INDEX.md` (clickable stage links) or `run_summary.txt`.
5. Inspect `AllBalls/05_detections/validated_detections.png` and `AllBalls/07_evaluation/localisation_evaluation.png`.

For a quick return to the project, see [current state and next steps](docs/HANDOVER.md). The diagnostic field definitions are in [DIAGNOSTICS.md](docs/DIAGNOSTICS.md).

The cleanup reorganises saving and readability only. Detector settings, matching and known false positives are unchanged. **Pocket rejection and ball identity recognition remain unfinished.**

## Build and run

The current build configuration expects OpenCV 4.12.0. For the local installation:

```sh
cmake -S . -B build -DOpenCV_DIR="$HOME/Libraries/opencv-4.12.0-install/lib/cmake/opencv4"
cmake --build build --parallel 4
zsh run_diagnostics.sh AllBalls
```

After freezing settings, run the evaluation set with the same executable/settings:

```sh
zsh run_diagnostics.sh all
```

`all` processes AllBalls and Table1–Table6. Keep the AllBalls training diagnostics separate from test results when reporting metrics.

Without the optional script:

```sh
POOL_GUI=0 POOL_IMAGE=AllBalls ./build/PoolBallVision
```

No environment settings are required by MyApplication(); by default it processes all seven images. `POOL_GUI=1` enables the original-image preview. `POOL_RUN=descriptive_name` chooses a unique run name. Existing runs are not overwritten. Run from the repository root so relative input paths resolve correctly.

## Repository layout

```text
myApplication.cpp           Assignment implementation: edit here
CMakeLists.txt              Local build configuration
dev_main.cpp                Local executable wrapper
.clang-format               Four-space / Allman formatting
run_diagnostics.sh          Optional build, log and source-snapshot runner
data/images/                Original supplied JPEGs: do not move or alter
data/ground_truth/          Supplied reference annotations: do not alter
docs/                       Diagnostic guide, handover and cleanup history
tools/compare_runs.py       Optional behaviour-preservation check (Python 3)
diagnostics/runs/           Active run folders, one folder per run
diagnostics/latest.txt      Path to the last successfully completed run
diagnostics/exports/        Optional ZIP exports, disabled by default
archive/                    Preserved historical results and obsolete helper
backups/                    Before-edit source/configuration snapshots
build/                      Generated build files
```

Utilities.cpp/.h, Geometric.cpp, Histograms.cpp and Region.cpp are supplied support files; leave them in place. All task-specific additions stay in myApplication.cpp.

The obsolete PoolDiagnostics.hpp has been archived, not deleted. It is no longer included by the application: editing it will not affect the current program.

## Generated output

```text
diagnostics/runs/<run>/
  INDEX.md                  Clickable links to each image's stage index
  run_summary.txt
  run.log                   Build and program log when using the runner
  source/                   Pre-run source/configuration snapshot and environment.txt
  AllBalls/                 Or Table1 ... Table6
    INDEX.md                Direct links to overlays, measurements and masks
    00_settings/            Parameter manifests and proposal summary
    01_preprocessing/       Rectified RGB, grayscale, Canny edges, hsv/
    02_segmentation/        Cloth/foreground masks and component evidence
    03_seeds/               Peaks, border seeds, distance fields and selection audit
    04_candidates/          Ellipse/Hough evidence, rejected proposals, ellipse_crops/
    05_detections/          Accepted overlay, selection.csv, detections.csv, summary
    06_ball_crops/          crop_quality.csv and <candidate_id>/ images/masks
    07_evaluation/          Ground truth, matching audit, metrics and evaluation overlay
```

New runs no longer also regenerate the duplicate output/<image>/ tree. Its previous contents are preserved in archive/legacy_output/. Historical runs are not silently deleted. ZIP creation is opt-in: `POOL_ARCHIVE=1 zsh run_diagnostics.sh AllBalls`.

The runner captures source before building and keeps failed logs/snapshots in `diagnostics/logs/`. A successful run moves them into its own `source/` and `run.log`; `diagnostics/latest.txt` only advances when the application completes. Do not edit source while a build/run is active.

For an organisation-only edit, compare an old and a new **image folder**:

```sh
python3 tools/compare_runs.py diagnostics/runs/<before>/AllBalls diagnostics/runs/<after>/AllBalls
```

The checker compares all evidence (including settings, masks, crops, detections and metrics), ignores `INDEX.md`, and decompresses gzip files before comparison. It does not train or change any parameters.

## Source map

Search for these names rather than scrolling through the entire file:

| Section / symbol | Responsibility |
|---|---|
| pooldiag::Settings | Shared preprocessing, mask and seed settings |
| pooldiag::OutputPath | Central filename-to-stage-folder routing |
| pooldiag::Write | Diagnostic measurements, independent of acceptance |
| ballvision::SelectorSettings | Shared candidate-validation settings |
| ballvision::LearnShape | AllBalls-only outline calibration |
| ballvision::Select | Boundary evidence, seed recovery and duplicate suppression |
| ballvision::ExportCrops | Geometric crop/appearance-mask estimates |
| LoadGroundTruth / EvaluateLocalisation | Reference loading and downstream one-to-one matching |
| BuildForegroundMask / DrawDistanceMarkers | Binary segmentation and distance-peak generation |
| MyApplication | Inputs, shared rectification, stage sequence and outputs |

Both settings structures are in myApplication.cpp. Do not introduce per-test-image constants or filter candidates using annotation coordinates.

## Submission boundaries

Final assignment files are the raw myApplication.cpp and a PDF report of at most eight pages, submitted separately. The report must explain processing without code/library-call references and automatically computed metrics for detection and recognition. The runner, support files, archives and diagnostics are development aids, not substitutes for the required submission.

Some reference annotations are reportedly intentionally incorrect. Primary metrics must retain the supplied reference; record evidenced discrepancies separately. Nearest-distance diagnostic checks are not the same as the one-to-one evaluator.

The repository already tracked some build outputs before this cleanup. .gitignore prevents new generated files being added, but does not untrack old files. This cleanup does not stage changes, modify the Git index, commit, or discard existing user work.
