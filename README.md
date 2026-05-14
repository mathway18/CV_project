# CV Project: Video Object Removal & Background Restoration

This public repository contains the full code for our CV project, including:
- **Part 1**: hand-crafted baseline
- **Part 2**: YOLO + SAM2 + ProPainter pipeline
- **Part 3**: quality-gated keyframe refinement on top of Part 2

---

## 1) Repository Contents

### Core scripts
- `video_object_removal.py` - Part 1 end-to-end pipeline
- `mask_extraction.py` - Part 1 dynamic object mask extraction
- `background_restoration.py` - Part 1 restoration module
- `video_object_removal_part2.py` - Part 2 end-to-end pipeline
- `sam2_mask_extraction.py` - YOLO-prompted SAM2 mask generation
- `propainter_inpainting.py` - ProPainter inference wrapper
- `part3_keyframe_refinement.py` - Part 3 refinement pipeline
- `evaluate_part3.py` - Part 3 evaluation script

### Data / outputs
- Input videos: `video/`
- Submission videos (Part 2): `outputs/videos_for_submission/`
- Submission videos (Part 3): `outputs/videos_for_submission_part3/`
- Final metrics summary: `outputs/part3_metrics_summary.json`

---

## 2) Environment & Dependencies

Use Python 3.10+ (recommended for SAM2/ProPainter compatibility).

### Part 1 dependencies
```bash
pip install -r requirements.txt
```

### Part 2 dependencies
```bash
pip install -r requirements_part2.txt
```

### Part 3 dependencies (OpenCV backend)
```bash
pip install -r requirements_part3.txt
```

> Optional diffusion backend for Part 3 is documented in `requirements_part3.txt` comments.

---

## 3) Model Weights and External Repositories

### Included in this repository
- `yolov8s-seg.pt` (YOLOv8 segmentation checkpoint)

### Required for Part 2 / Part 3
1. Clone external repositories:
```bash
git clone https://github.com/facebookresearch/sam2 external/sam2
git clone https://github.com/sczhou/ProPainter external/ProPainter
```
2. Download and place checkpoints:
- **SAM2 checkpoint** (example):
  - `external/sam2/checkpoints/sam2.1_hiera_large.pt`
- **ProPainter pretrained weights**:
  - place all required files under `external/ProPainter/weights/` according to the official ProPainter instructions.

Official references:
- SAM2: https://github.com/facebookresearch/sam2
- ProPainter: https://github.com/sczhou/ProPainter

---

## 4) How to Run

### Part 1 (baseline)

```bash
python video_object_removal.py
```

Or run modules separately:

```bash
python mask_extraction.py
python background_restoration.py
```

### Part 2 (YOLO + SAM2 + ProPainter)

```bash
python video_object_removal_part2.py \
  --input video/bmx_trees.mp4 \
  --output-dir outputs/part2_bmx_trees \
  --sam2-config external/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint external/sam2/checkpoints/sam2.1_hiera_large.pt \
  --propainter-repo external/ProPainter \
  --debug
```

### Part 3 (quality-gated refinement)

```bash
python part3_keyframe_refinement.py \
  --input outputs/part2_bmx_trees/input_video.mp4 \
  --part2-video outputs/part2_bmx_trees/part2_output.mp4 \
  --mask outputs/part2_bmx_trees/mask_video.mp4 \
  --output-dir outputs/part3_bmx_trees_gate \
  --keyframe-stride 12 \
  --propagation-radius 4 \
  --temporal-alpha 0.12 \
  --max-blend 0.16 \
  --artifact-threshold 40 \
  --mask-dilation 3 \
  --soft-edge 9 \
  --texture-floor-ratio 0.80 \
  --max-temporal-regression 0.08 \
  --max-masked-delta 12
```

### Evaluate Part 3 vs Part 2

```bash
python evaluate_part3.py \
  --input outputs/part2_bmx_trees/input_video.mp4 \
  --part2-video outputs/part2_bmx_trees/part2_output.mp4 \
  --part3-video outputs/part3_bmx_trees_gate/part3_output.mp4 \
  --mask outputs/part2_bmx_trees/mask_video.mp4 \
  --output-json outputs/part3_bmx_trees_gate/part3_eval.json
```

---

## 5) Visual Results (Required)

### Input videos
- `video/Person_Walking_Down_Corridor_Video.mp4`
- `video/bmx_trees.mp4`
- `video/tennis.mp4`

### Final output videos
- **Part 2**
  - `outputs/videos_for_submission/wild_corridor_part2.mp4`
  - `outputs/videos_for_submission/bmx_trees_part2.mp4`
  - `outputs/videos_for_submission/tennis_part2.mp4`
- **Part 3**
  - `outputs/videos_for_submission_part3/wild_corridor_part3.mp4`
  - `outputs/videos_for_submission_part3/bmx_trees_part3.mp4`
  - `outputs/videos_for_submission_part3/tennis_part3.mp4`

### Packaged submissions
- `videos_part2.zip`
- `videos_part3.zip`
- `videos.zip`

---

## 6) Notes
- Part 2 and Part 3 quality strongly depends on correct external model setup.
- Final Part 3 metric summary is available in `outputs/part3_metrics_summary.json`.
- Detailed run notes are in `PART2_RUN_SUMMARY.md` and `PART3_RUN_SUMMARY.md`.
