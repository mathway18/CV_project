# Video Object Removal and Background Restoration

This repository now contains the PDF-required baseline (`Part 1`), the AI-driven reproduction pipeline (`Part 2`), and the optimization/extension stage (`Part 3`) for video object removal and background restoration.

## Part 1: Hand-crafted Baseline

### Pipeline Overview
1. **Semantic Extraction**: Detect and segment objects using YOLOv8-Seg.
2. **Dynamic Judgment**: Filter static objects using Lucas-Kanade optical flow.
3. **Mask Optimization**: Apply dilation to cover motion blur.
4. **Background Restoration**: Use temporal borrowing plus spatial inpainting.

### Scripts

#### Mask Extraction (`mask_extraction.py`)
Generates a binary mask video showing areas to be removed.

```python
from mask_extraction import MaskExtractor

extractor = MaskExtractor(classes=[15])  # Only cats
extractor.process_video("input_video.mp4", "mask_video.mp4")
```

#### Background Restoration (`background_restoration.py`)
Applies the handcrafted restoration pipeline using the original video and mask video.

```python
from background_restoration import BackgroundRestorer

restorer = BackgroundRestorer()
restorer.process_video("input_video.mp4", "mask_video.mp4", "output_video.mp4", debug=True)
```

#### Combined Baseline (`video_object_removal.py`)
Runs Part 1 end-to-end with debug visualization support.

```python
from video_object_removal import VideoObjectRemoval

remover = VideoObjectRemoval()
remover.process_video("input_video.mp4", "output_video.mp4")
```

## Part 2: AI-driven Pipeline

This implementation follows the project PDF's second part with:
1. **Dynamic Mask Extraction**: YOLO auto-detection + SAM 2 mask refinement.
2. **Video Inpainting**: ProPainter integration for temporally consistent restoration.

### Mandatory Input Videos
- `video/Person_Walking_Down_Corridor_Video.mp4`: wild video.
- `video/bmx_trees.mp4`: mandatory `bmx-trees` sample video.
- `video/tennis.mp4`: mandatory `tennis` sample video.

### Added Files
- `sam2_mask_extraction.py`: automatic dynamic-object mask extraction using YOLO prompts and SAM 2.
- `propainter_inpainting.py`: wrapper around the official `inference_propainter.py`.
- `video_object_removal_part2.py`: full Part 2 pipeline entry point.
- `part2_utils.py`: shared video and mask utilities.
- `requirements_part2.txt`: direct Python dependencies for the wrapper code.

### Part 2 Environment Notes
- This workspace includes the wrapper pipeline plus local upstream checkouts under `external/sam2` and `external/ProPainter`. If these folders are missing after cloning from GitHub, follow the setup commands below.
- According to the official SAM 2 repository, installation requires `python>=3.10` and Meta strongly recommends using `WSL` on Windows.
- According to the official ProPainter repository, the recommended environment is a dedicated Python environment with PyTorch and CUDA support.
- In practice, Part 2 should be run in a separate environment prepared for these upstream projects instead of reusing the lightweight Part 1 environment.

### Part 2 Setup
Install the wrapper dependencies:

```bash
pip install -r requirements_part2.txt
```

If the upstream folders are missing, clone and install the official projects:

```bash
git clone https://github.com/facebookresearch/sam2 external/sam2
git clone https://github.com/sczhou/ProPainter external/ProPainter
```

Prepare SAM 2 and ProPainter following their official instructions:
- SAM 2: install the repo and download a checkpoint such as `sam2.1_hiera_large.pt`.
- ProPainter: install the repo and place the pretrained weights in its `weights/` directory.

### Part 2 Usage

```bash
python video_object_removal_part2.py ^
  --input input_video.mp4 ^
  --output-dir outputs/part2 ^
  --sam2-config external/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml ^
  --sam2-checkpoint external/sam2/checkpoints/sam2.1_hiera_large.pt ^
  --propainter-repo external/ProPainter ^
  --debug
```

### Part 2 Outputs
- `sam2_mask_video.mp4`: video-form mask visualization.
- `sam2_masks/`: frame-wise masks for ProPainter.
- `part2_output.mp4`: restored video copied from ProPainter results.
- `sam2_overlay.mp4`: optional debug overlay.

## Expected Results

### Part 1
- Works for relatively simple scenes with static backgrounds.
- May blur details in large or textured holes.

### Part 2
- Produces cleaner masks than the Part 1 baseline.
- Handles large occlusions and temporal consistency better than OpenCV inpainting.
- Depends heavily on correct SAM 2 and ProPainter setup.

## Part 3: Optimization and Extension

Part 3 adds a quality-gated keyframe-repair refinement stage on top of the Part 2 output. A candidate Part 3 frame is accepted only when local no-reference checks indicate that it improves the result; otherwise the pipeline falls back to the Part 2 frame.

Implemented files:
- `part3_keyframe_refinement.py`: repairs selected masked keyframes, softly blends repair candidates into the Part 2 result, applies quality-gated fallback, and exports comparison videos/keyframes.
- `evaluate_part3.py`: evaluates Part 2 vs Part 3 using no-reference boundary seam and temporal warp metrics.
- `requirements_part3.txt`: lightweight dependencies plus optional Diffusers notes.

The script supports two repair backends:
- default `opencv`: fully reproducible local backend used for the generated outputs;
- optional `diffusers`: Stable Diffusion inpainting backend for Direction C experiments when model weights are available.

Final Part 3 outputs:
- `outputs/videos_for_submission_part3/wild_corridor_part3.mp4`
- `outputs/videos_for_submission_part3/bmx_trees_part3.mp4`
- `outputs/videos_for_submission_part3/tennis_part3.mp4`
- `videos.zip` for final Canvas video submission
- `videos_part3.zip` as an explicit Part 3 copy

The Part 2-only package is preserved as `videos_part2.zip`.

Final Part 3 metrics are summarized in `outputs/part3_metrics_summary.json` and documented in `PART3_RUN_SUMMARY.md`.
The final selected run improves `bmx-trees` and `tennis` while safely preserving the already strong corridor Part 2 result.

Example:

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

Evaluate:

```bash
python evaluate_part3.py ^
  --input outputs\part2_bmx_trees\input_video.mp4 ^
  --part2-video outputs\part2_bmx_trees\part2_output.mp4 ^
  --part3-video outputs\part3_bmx_trees_gate\part3_output.mp4 ^
  --mask outputs\part2_bmx_trees\mask_video.mp4 ^
  --output-json outputs\part3_bmx_trees_gate\part3_eval.json
```

## Dependencies
- OpenCV
- NumPy
- SciPy
- Ultralytics
- PyTorch / Torchvision
- SAM 2 (official repo)
- ProPainter (official repo)
- Optional for Part 3 diffusion backend: Diffusers, Accelerate, Safetensors

## References Used For Part 2 Integration
- SAM 2 official repository: https://github.com/facebookresearch/sam2
- ProPainter official repository: https://github.com/sczhou/ProPainter
