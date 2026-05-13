# Part 2 Run Summary

## What Was Completed

- Installed and verified the official SAM 2 repository under `external/sam2`.
- Installed and verified the official ProPainter repository under `external/ProPainter`.
- Downloaded the SAM 2.1 tiny checkpoint:
  - `external/sam2/checkpoints/sam2.1_hiera_tiny.pt`
- Downloaded ProPainter weights automatically during inference:
  - `external/ProPainter/weights/raft-things.pth`
  - `external/ProPainter/weights/recurrent_flow_completion.pth`
  - `external/ProPainter/weights/ProPainter.pth`
- Generated Part 2 outputs for the wild corridor video using YOLO detection, SAM 2 mask refinement, and ProPainter restoration.
- Generated ProPainter outputs for the mandatory sample scenes `bmx-trees` and `tennis` using the sample frame-wise masks included with the ProPainter repository.
- Also generated extra SAM2-mask variants for `bmx-trees` and `tennis` for method comparison. These are kept separately because the provided sample masks are cleaner for final video submission.
- Created the Part 2 submission package. It is now preserved as `videos_part2.zip` because the final `videos.zip` package contains the quality-gated Part 3 outputs.

## Main Outputs

- Wild video:
  - `outputs/part2_corridor/sam2_mask_video.mp4`
  - `outputs/part2_corridor/sam2_overlay.mp4`
  - `outputs/part2_corridor/part2_output.mp4`
  - `outputs/part2_corridor/part2_output_720p.mp4`
  - `outputs/part2_corridor/comparison_demo.mp4`
  - `outputs/part2_corridor/ppt_frames/`
- Sample data:
  - `outputs/part2_bmx_trees/part2_output.mp4`
  - `outputs/part2_bmx_trees/comparison_demo.mp4`
  - `outputs/part2_bmx_trees/ppt_frames/`
  - `outputs/part2_tennis/part2_output.mp4`
  - `outputs/part2_tennis/comparison_demo.mp4`
  - `outputs/part2_tennis/ppt_frames/`
- Extra SAM2 sample variants:
  - `outputs/part2_bmx_trees_sam2/sam2_mask_video.mp4`
  - `outputs/part2_bmx_trees_sam2/part2_output.mp4`
  - `outputs/part2_bmx_trees_sam2/comparison_demo.mp4`
  - `outputs/part2_tennis_sam2/sam2_mask_video.mp4`
  - `outputs/part2_tennis_sam2/part2_output.mp4`
  - `outputs/part2_tennis_sam2/comparison_demo.mp4`
- Submission package:
  - `videos_part2.zip`
  - `outputs/videos_for_submission/wild_corridor_part2.mp4`
  - `outputs/videos_for_submission/bmx_trees_part2.mp4`
  - `outputs/videos_for_submission/tennis_part2.mp4`

## Notes

- The corridor ProPainter inference was run at `656x368` because higher resolutions such as `768x432` and `960x544` exceeded the 8 GB GPU memory limit. A `1280x720` display copy was exported from the successful result.
- SAM 2.1 tiny was used to fit local runtime and memory constraints. The code can use `sam2.1_hiera_large.pt` by changing the config and checkpoint arguments.
- `sam2_mask_extraction.py` now constrains SAM 2 predictions with YOLOv8-Seg masks. This prevents SAM 2 box prompts from leaking into large background areas.
- Temporal mask stabilization is now disabled by default because it propagated false-positive mask regions in the corridor video. It can be enabled with `--enable-temporal-stabilization`.

## Useful Commands

Run the full Part 2 corridor pipeline:

```bash
python video_object_removal_part2.py ^
  --input "video\Person_Walking_Down_Corridor_Video.mp4" ^
  --output-dir "outputs\part2_corridor_full" ^
  --sam2-config "configs/sam2.1/sam2.1_hiera_t.yaml" ^
  --sam2-checkpoint "external\sam2\checkpoints\sam2.1_hiera_tiny.pt" ^
  --propainter-repo "external\ProPainter" ^
  --detector-model "yolov8s-seg.pt" ^
  --classes 0 ^
  --conf-threshold 0.35 ^
  --disable-motion-filter ^
  --width 656 ^
  --height 368 ^
  --fp16 ^
  --subvideo-length 30 ^
  --neighbor-length 6 ^
  --debug
```

Evaluate metrics when ground truth is available:

```bash
python evaluate_metrics.py ^
  --pred-mask-dir "outputs\part2_corridor\sam2_masks" ^
  --gt-mask-dir "<ground_truth_mask_dir>" ^
  --pred-video "outputs\part2_corridor\part2_output_720p.mp4" ^
  --gt-video "<ground_truth_clean_video>" ^
  --output-json "outputs\part2_corridor\metrics.json"
```
