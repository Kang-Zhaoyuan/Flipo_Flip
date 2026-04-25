# Blue Video Analysis Workflow Feasibility Report

## Conclusion
The workflow is operational on this screenshot set: the blue-body segmentation, centroid recovery, and ellipse fitting stages are stable enough to build a usable contact sheet, while the white-marker step remains the most sensitive and should still be checked by eye.

## Summary
- Screens analyzed: 8
- Blue body recovered: 7 / 8
- Ellipse fit recovered: 7 / 8
- White marker recovered: 7 / 8
- Average blue-mask area: 75754.8 pixels
- Average final-body area: 32017.9 pixels
- Average white-mask area: 8458.2 pixels
- Contact sheet: blue_workflow_contact_sheet.png

## Frame-by-Frame Results
| Frame | Body | Ellipse | Marker | Ground Y | Blue Area | Final Area | White Area |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | No | No | No | 1343 | 45444 | 0 | 0 |
| 02 | Yes | Yes | Yes | 645 | 57327 | 37588 | 7849 |
| 03 | Yes | Yes | Yes | 578 | 67595 | 43819 | 13021 |
| 04 | Yes | Yes | Yes | 802 | 72093 | 44973 | 13631 |
| 05 | Yes | Yes | Yes | 716 | 66292 | 38382 | 12409 |
| 06 | Yes | Yes | Yes | 1038 | 107771 | 29357 | 6731 |
| 07 | Yes | Yes | Yes | 1037 | 101989 | 32932 | 8472 |
| 08 | Yes | Yes | Yes | 1029 | 87527 | 29092 | 5553 |

## Interpretation
- The blue-disc body was not isolated on every screenshot, so the threshold or morphology will need tuning for some frames.
- Ellipse fitting failed on at least one screenshot, which means the final contour is occasionally too sparse or fragmented.
- Marker-line recovery is usable but still the least stable stage, so manual confirmation is still recommended.

## Notes
- The ground-line estimate is derived automatically from edge density in the lower part of each screenshot.
- The thickness-fix step is preserved because it is the main defense against fused ground-strip artifacts.
- English-only labels are used in the generated figure and report to match the requested output format.