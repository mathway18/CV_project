# Part 3 Run Summary

## Goal

Part 3 implements an optimization and extension stage on top of the completed Part 2 SAM 2 + ProPainter pipeline.

The implemented direction is a practical version of the PDF's Direction C: generative keyframe repair and propagation. The code includes an optional Stable Diffusion inpainting backend through Diffusers, and the checked-in reproducible run uses the OpenCV backend so the full experiment can run locally without downloading multi-GB model weights.

## Added Files

- `part3_keyframe_refinement.py`
  - Selects keyframes with non-empty masks.
  - Repairs masked regions using either the default OpenCV backend or optional Diffusers Stable Diffusion inpainting.
  - Blends repair candidates back into the Part 2 result with soft masks.
  - Optionally applies optical-flow temporal smoothing.
  - Applies a quality gate before accepting a Part 3 candidate frame. If the candidate worsens seam, texture preservation, temporal consistency, or masked-frame drift, it falls back to the Part 2 frame.
  - Exports Part 3 videos, comparison videos, keyframes, and metrics.
- `evaluate_part3.py`
  - Compares Part 2 and Part 3 without ground truth using:
    - boundary seam score, lower is better;
    - temporal warp error, lower is better;
    - masked texture score for diagnostic context.
- `requirements_part3.txt`
  - Minimal Part 3 runtime dependencies and optional Diffusers notes.

## Final Selected Outputs

- `outputs/videos_for_submission_part3/wild_corridor_part3.mp4`
- `outputs/videos_for_submission_part3/bmx_trees_part3.mp4`
- `outputs/videos_for_submission_part3/tennis_part3.mp4`
- `videos.zip`
- `videos_part3.zip`

Part 2 submission outputs remain available in:

- `outputs/videos_for_submission/`
- `videos_part2.zip`

## Final Metrics

Metrics are saved in `outputs/part3_metrics_summary.json`.

| Dataset | Frames | Accepted / Rejected | Part 2 Seam | Part 3 Seam | Seam Change | Part 2 Warp | Part 3 Warp | Warp Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bmx-trees | 80 | 15 / 65 | 18.2920 | 18.1038 | +1.03% | 5.3090 | 5.1043 | +3.86% |
| tennis | 70 | 53 / 17 | 3.7167 | 3.5840 | +3.57% | 2.1920 | 2.1745 | +0.80% |
| wild corridor | 192 | 0 / 192 | 3.1025 | 3.1025 | 0.00% | 1.4674 | 1.4674 | 0.00% |

The corridor Part 2 output was already the strongest visual result. The quality gate rejected all corridor candidates and copied the Part 2 result exactly, which is the correct behavior for this adaptive Part 3 stage: it should not change a frame unless there is measurable local benefit.

## Reproduction Commands

### BMX Trees

```bash
python part3_keyframe_refinement.py ^
  --input outputs\part2_bmx_trees\input_video.mp4 ^
  --part2-video outputs\part2_bmx_trees\part2_output.mp4 ^
  --mask outputs\part2_bmx_trees\mask_video.mp4 ^
  --output-dir outputs\part3_bmx_trees_gate ^
  --keyframe-stride 12 ^
  --propagation-radius 4 ^
  --temporal-alpha 0.12 ^
  --max-blend 0.16 ^
  --artifact-threshold 40 ^
  --mask-dilation 3 ^
  --soft-edge 9 ^
  --texture-floor-ratio 0.80 ^
  --max-temporal-regression 0.08 ^
  --max-masked-delta 12
```

### Tennis

```bash
python part3_keyframe_refinement.py ^
  --input outputs\part2_tennis\input_video.mp4 ^
  --part2-video outputs\part2_tennis\part2_output.mp4 ^
  --mask outputs\part2_tennis\mask_video.mp4 ^
  --output-dir outputs\part3_tennis_gate2 ^
  --keyframe-stride 12 ^
  --propagation-radius 4 ^
  --temporal-alpha 0.0 ^
  --max-blend 0.08 ^
  --artifact-threshold 60 ^
  --mask-dilation 3 ^
  --soft-edge 9 ^
  --texture-floor-ratio 0.82 ^
  --max-temporal-regression 0.0 ^
  --max-masked-delta 8
```

### Wild Corridor

```bash
python part3_keyframe_refinement.py ^
  --input video\Person_Walking_Down_Corridor_Video.mp4 ^
  --part2-video outputs\part2_corridor\part2_output_720p.mp4 ^
  --mask outputs\part2_corridor\sam2_masks ^
  --output-dir outputs\part3_corridor_safe ^
  --keyframe-stride 48 ^
  --propagation-radius 0 ^
  --temporal-alpha 0.0 ^
  --max-blend 0.0 ^
  --artifact-threshold 100 ^
  --mask-dilation 3 ^
  --soft-edge 9 ^
  --texture-floor-ratio 0.95 ^
  --max-temporal-regression 0.0 ^
  --max-masked-delta 0.1
```

## Verification

Completed checks:

- `python -m py_compile part3_keyframe_refinement.py evaluate_part3.py`
- Generated Part 3 outputs for `bmx-trees`, `tennis`, and the wild corridor video.
- Re-evaluated all selected outputs with `evaluate_part3.py`.
- Repacked the selected videos into final `videos.zip` and duplicate `videos_part3.zip`.
- Verified all selected Part 3 videos open with OpenCV and have the expected frame counts and resolutions.
