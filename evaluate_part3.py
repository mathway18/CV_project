from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from part2_utils import normalize_mask
from part3_keyframe_refinement import (
    boundary_seam_score,
    load_mask_sequence,
    read_video_frames,
    resize_like,
    texture_score,
)


def warp_previous(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    # Backward flow maps current-frame coordinates to the previous frame.
    flow = cv2.calcOpticalFlowFarneback(
        curr_gray,
        prev_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    height, width = curr_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(previous, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def temporal_warp_error(frames: list[np.ndarray], masks: list[np.ndarray]) -> float:
    errors: list[float] = []
    for index in range(1, len(frames)):
        mask = normalize_mask(cv2.bitwise_or(masks[index - 1], masks[index])).astype(bool)
        if not np.any(mask):
            continue
        warped = warp_previous(frames[index - 1], frames[index])
        diff = np.mean(np.abs(warped.astype(np.float32) - frames[index].astype(np.float32)), axis=2)
        errors.append(float(diff[mask].mean()))
    return float(np.mean(errors)) if errors else 0.0


def mean_boundary_seam(frames: list[np.ndarray], masks: list[np.ndarray]) -> float:
    values = [boundary_seam_score(frame, mask) for frame, mask in zip(frames, masks) if np.any(mask)]
    return float(np.mean(values)) if values else 0.0


def mean_texture(frames: list[np.ndarray], masks: list[np.ndarray]) -> float:
    values = [texture_score(frame, mask) for frame, mask in zip(frames, masks) if np.any(mask)]
    return float(np.mean(values)) if values else 0.0


def masked_delta(reference: list[np.ndarray], candidate: list[np.ndarray], masks: list[np.ndarray]) -> float:
    values: list[float] = []
    for ref, cand, mask in zip(reference, candidate, masks):
        mask_bool = normalize_mask(mask).astype(bool)
        if not np.any(mask_bool):
            continue
        diff = np.mean(np.abs(ref.astype(np.float32) - cand.astype(np.float32)), axis=2)
        values.append(float(diff[mask_bool].mean()))
    return float(np.mean(values)) if values else 0.0


def evaluate_dataset(
    input_video: str,
    part2_video: str,
    part3_video: str,
    mask_path: str,
) -> dict[str, float | int | list[int]]:
    original_frames, fps = read_video_frames(input_video)
    part2_frames, _ = read_video_frames(part2_video)
    part3_frames, _ = read_video_frames(part3_video)
    masks = load_mask_sequence(mask_path, original_frames)
    frame_count = min(len(original_frames), len(part2_frames), len(part3_frames), len(masks))
    if frame_count == 0:
        raise ValueError("No overlapping frames to evaluate.")

    original_frames = original_frames[:frame_count]
    part2_frames = [resize_like(frame, original_frames[index]) for index, frame in enumerate(part2_frames[:frame_count])]
    part3_frames = [resize_like(frame, original_frames[index]) for index, frame in enumerate(part3_frames[:frame_count])]
    masks = masks[:frame_count]

    part2_seam = mean_boundary_seam(part2_frames, masks)
    part3_seam = mean_boundary_seam(part3_frames, masks)
    part2_warp = temporal_warp_error(part2_frames, masks)
    part3_warp = temporal_warp_error(part3_frames, masks)
    part2_texture = mean_texture(part2_frames, masks)
    part3_texture = mean_texture(part3_frames, masks)

    return {
        "frames": frame_count,
        "fps": fps,
        "size": [original_frames[0].shape[1], original_frames[0].shape[0]],
        "mask_mean_pixels": float(np.mean([normalize_mask(mask).sum() for mask in masks])),
        "part2_boundary_seam": part2_seam,
        "part3_boundary_seam": part3_seam,
        "boundary_seam_delta": part3_seam - part2_seam,
        "boundary_seam_improvement_pct": float((part2_seam - part3_seam) / max(part2_seam, 1e-6) * 100.0),
        "part2_temporal_warp_error": part2_warp,
        "part3_temporal_warp_error": part3_warp,
        "temporal_warp_delta": part3_warp - part2_warp,
        "temporal_warp_improvement_pct": float((part2_warp - part3_warp) / max(part2_warp, 1e-6) * 100.0),
        "part2_masked_texture": part2_texture,
        "part3_masked_texture": part3_texture,
        "masked_part2_to_part3_delta": masked_delta(part2_frames, part3_frames, masks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Part 3 refinement against Part 2 without GT.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--part2-video", required=True)
    parser.add_argument("--part3-video", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    results = evaluate_dataset(
        input_video=args.input,
        part2_video=args.part2_video,
        part3_video=args.part3_video,
        mask_path=args.mask,
    )
    text = json.dumps(results, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
