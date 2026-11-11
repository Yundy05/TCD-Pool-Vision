# Diagnostic field guide

Run `zsh run_diagnostics.sh AllBalls` from the project root. Use `diagnostics/latest.txt` to locate the latest completed run and open its `INDEX.md`, then the image's `INDEX.md`. A direct call to MyApplication() creates the same stage folders and indexes; the optional runner additionally captures a build/program log, pre-run source snapshot and environment/checksum record. Failed runner logs remain in `diagnostics/logs/`.

For the verified baseline, return instructions and unfinished work, see [HANDOVER.md](HANDOVER.md). For behaviour-preserving edits, compare image-result folders using `python3 tools/compare_runs.py <before>/AllBalls <after>/AllBalls`.

## Where to look

| Question | Stage and files |
|---|---|
| What was kept? | 05_detections/validated_detections.png, detections.csv |
| Why was a proposal kept/rejected? | 05_detections/selection.csv; 04_candidates/rejected_candidates.png |
| Is the mask removing a ball? | 02_segmentation/foreground_raw.png and foreground_opened.png; component CSVs |
| Is the interior absent or the seed suppressed? | 03_seeds/peak_selection.csv, seeds.csv, local_maxima_*.png |
| Is an ellipse fitted to a number patch? | 04_candidates/ellipses.csv, ellipses_numbered.png, ellipse_crops/ |
| Is a pocket being mistaken for a ball? | Compare its selection.csv row, candidate crop and perimeter evidence |
| What can recognition use? | 06_ball_crops/crop_quality.csv and <id>/rgb.png, appearance.png, masks |
| Are predictions matched correctly? | 07_evaluation/localisation_matches.csv, localisation_evaluation.png |
| What are the headline metrics? | 07_evaluation/localisation_metrics.csv; run_summary.txt |
| Which parameters generated the run? | 00_settings/settings.yml and selector_settings.yml |

Do not confuse 00_settings/summary.txt (proposal measurements) with 05_detections/selector_summary.txt (accepted candidates). A green validated outline is not a recognised identity.

## Seeds and proposal IDs

- P/B labels in the original marker views are strength-ordered.
- SI/SB seed IDs in the diagnostic tables are scan-ordered.
- E IDs describe contour ellipse proposals.
- RI/RB IDs describe candidates recovered from interior/border seeds.
- Border-connected means the entire foreground component touches an image edge. It does not prove that the local candidate is a rail, nor that the ball itself touches the image edge.

## Measurement cautions

Perimeter support measures proximity to any Canny edge. Pocket netting can produce very high support, so it is not by itself proof of a ball silhouette. Support fractions are relative to visible samples; also inspect visible_fraction and angular coverage. Foreground occupancy is soft evidence, not a mandatory complete mask.

Crops use geometric ownership estimates. They do not reconstruct occluded surfaces. Appearance fractions describe the surviving pixels, not a proven full-ball segmentation or ball identity.

Distances are in rectified pixels, approximately the brief's millimetre scale. Apparent ellipse centres have not been corrected to physical ball/table contact points.

## Evaluation

The evaluator matches positions without using identities: maximise valid one-to-one matches within 21.5 pixels, then minimise the summed distance. It computes TP/FP/FN, precision, recall, F1 and matched-centre error. Out-of-image reference annotations remain in the denominator and are separately flagged. A missing reference file is not treated as a negative example.

The ground_truth_checks.csv file contains independent nearest-candidate diagnostics, not the matching evaluator. Supplied references are unchanged. Investigate suspected annotation errors with evidence and keep corrected-reference experiments secondary and fully disclosed.

## Current settings locations

Edit pooldiag::Settings and ballvision::SelectorSettings in myApplication.cpp. Original preprocessing values remain: median 5; Canny 50/100; cloth H 35..95, S 25..95, V >=25; opening 3x3 once; peak neighbourhood 21; raw distance >12; seed spacing 24. Selector starting settings remain: 96 perimeter samples, 12 sectors, edge tolerance 2 pixels, minimum support 0.50, minimum visible fraction 0.55, duplicate IoU 0.50.

Use AllBalls and the supplied physical dimensions for development. Do not choose settings from Table1–Table6 results. Freeze settings before evaluation; report failures rather than adding per-image exceptions.

## Immediate work remaining

1. Address repeated pocket false positives using general evidence learned only from AllBalls/the fixed camera geometry, without a wide blind border crop.
2. Add AllBalls-trained identity descriptors and recognition outputs.
3. Add recognition and end-to-end identification metrics.
4. Preserve original-reference results and document suspected annotation defects separately.

The cleanup itself intentionally implements none of these algorithm changes; its regression test checks unchanged detections and metrics.
