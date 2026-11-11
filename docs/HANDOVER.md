# Project handover

## Resume here

- Edit `myApplication.cpp`: all assignment-specific processing stays in this one file.
- Keep the existing four-space indentation, Allman braces and descriptive stage comments.
- Run `zsh run_diagnostics.sh AllBalls` from the repository root.
- Read `diagnostics/latest.txt`, then open that run's `INDEX.md`.
- For field definitions and interpretation, see [DIAGNOSTICS.md](DIAGNOSTICS.md).

## Verified cleanup state

Latest complete run: `diagnostics/runs/cleanup_all_verified_20261030_1515/`.

The program built and completed AllBalls and Table1--Table6 with unchanged detector settings. This was a regression check, not parameter selection from test images.

- AllBalls: TP 16, FP 3, FN 0 at the existing 21.5-pixel matching tolerance.
- Precision 0.84211, recall 1.00000, F1 0.91429; mean matched-centre error 2.84373 pixels.
- All 153 AllBalls evidence files match the immediately pre-edit baseline after decompressing gzip files. Only navigation indexes were added.
- For every one of the seven images, all existing PNG/CSV evidence matches the earlier complete run byte-for-byte. Historical summary wording differs; no detector output changed.
- Original images and supplied ground truth were hashed before/after and are unchanged.
- Invalid image selections and invalid/duplicate run names fail without changing the latest completed-run pointer.
- No Git staging, commits, resets or index changes were performed.

## What was cleaned up

The repository already contained a staged-output cleanup when folder access became available. It was retained, not overwritten or rolled back. This pass added clickable run/image indexes, clearer parameter-group comments, early image validation, pre-run source/environment snapshots, build logs and an optional regression checker.

Required code remains in `myApplication.cpp`, with no `PoolDiagnostics.hpp` dependency. Future runs retain the numbered stage folders. Detailed crops stay under `06_ball_crops/<candidate_id>/`.

Only the verified complete run remains in `diagnostics/runs/`. Four superseded runs were moved intact into `archive/diagnostics/cleanup_20261030_1518/`. Nothing was permanently deleted. Original input/support files and the build setup were left in place.

Before-edit files and audit records: `backups/access_cleanup_20261030_150431/`.

- `verification_audit.json`: input hashes, seven-image regression and guard checks.
- `regression_check.txt`: immediate AllBalls evidence comparison.
- `archived_run_moves.csv`: reversible source/destination map.
- `archived_run_sha256.json`: archived run file hashes.
- `working_tree_before.patch` and `git_status_before.txt`: existing user changes before this pass.

Some generated build/output files were already tracked by Git. They remain tracked intentionally: `.gitignore` does not untrack existing files, and this cleanup did not alter the index. Handle that separately if desired.

## Pocket problem: not fixed yet

Confirmed AllBalls pocket proposals are `RB3`, `RB25` and `RB26`. Their visible boundary support is approximately 1.000, 0.909 and 0.764 respectively. `RB3` has the minimum passing visible fraction, 0.59375, but simply raising that cutoff can also remove clipped real balls. Pocket netting supplies substantial internal edge evidence, so perimeter proximity alone is not sufficient to establish a ball boundary.

Next algorithm revision should measure outside-to-inside boundary contrast, edge orientation/continuity and local appearance or fixed-camera pocket evidence using AllBalls only. Preserve near-cushion candidates; do not use a wide blanket border crop or force 16 detections. Compare against this retained baseline, then freeze settings before evaluation.

## Recognition: not implemented yet

Normalised crops, geometric ownership masks and estimated appearance masks already exist. They are not identity predictions. Next: learn colour and solid/stripe descriptors from AllBalls, retain uncertainty when evidence is weak, and add automatically computed matched-ball and end-to-end recognition metrics.

Do not train or tune on Table1--Table6. Keep supplied reference annotations unchanged for primary metrics; separately document evidenced annotation discrepancies.

## Quick checks

```sh
cmake --build build --parallel 4
zsh run_diagnostics.sh AllBalls
```

Organisation-only regression example using the archived pre-edit baseline:

```sh
python3 tools/compare_runs.py \
  archive/diagnostics/cleanup_20261030_1518/cleanup_before_20261030_150431/AllBalls \
  diagnostics/runs/cleanup_all_verified_20261030_1515/AllBalls
```

The Python helper is optional tooling, not an application or submission dependency. Final submission remains the raw `myApplication.cpp` plus the PDF report, separately, with an eight-page limit including figures/references.
