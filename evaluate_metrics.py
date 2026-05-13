from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    files = sorted(file for file in root.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No image files found in {root}")
    return files


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    if shape is not None and mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def evaluate_masks(pred_dir: str | Path, gt_dir: str | Path, recall_threshold: float) -> dict[str, float | int]:
    pred_files = list_images(pred_dir)
    gt_files = list_images(gt_dir)
    count = min(len(pred_files), len(gt_files))
    if count == 0:
        raise ValueError("No overlapping mask frames to evaluate.")

    ious: list[float] = []
    for pred_path, gt_path in zip(pred_files[:count], gt_files[:count]):
        gt = read_mask(gt_path)
        pred = read_mask(pred_path, gt.shape)
        union = np.logical_or(pred, gt).sum()
        intersection = np.logical_and(pred, gt).sum()
        iou = 1.0 if union == 0 else float(intersection / union)
        ious.append(iou)

    return {
        "mask_frames": count,
        "JM_mean_iou": float(np.mean(ious)),
        "JR_iou_recall": float(np.mean(np.array(ious) >= recall_threshold)),
        "recall_threshold": recall_threshold,
    }


def read_video_frames(video_path: str | Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    return frames, fps


def frame_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)
    mse = float(np.mean((pred_f - gt_f) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def evaluate_videos(pred_video: str | Path, gt_video: str | Path) -> dict[str, float | int]:
    pred_frames, pred_fps = read_video_frames(pred_video)
    gt_frames, gt_fps = read_video_frames(gt_video)
    count = min(len(pred_frames), len(gt_frames))
    if count == 0:
        raise ValueError("No overlapping video frames to evaluate.")

    psnrs: list[float] = []
    ssims: list[float] = []
    for pred, gt in zip(pred_frames[:count], gt_frames[:count]):
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        psnrs.append(frame_psnr(pred, gt))
        ssims.append(
            float(
                structural_similarity(
                    cv2.cvtColor(pred, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(gt, cv2.COLOR_BGR2RGB),
                    channel_axis=2,
                    data_range=255,
                )
            )
        )

    finite_psnrs = [value for value in psnrs if np.isfinite(value)]
    return {
        "video_frames": count,
        "pred_fps": pred_fps,
        "gt_fps": gt_fps,
        "PSNR": float(np.mean(finite_psnrs)) if finite_psnrs else float("inf"),
        "SSIM": float(np.mean(ssims)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mask IoU/JR and video PSNR/SSIM.")
    parser.add_argument("--pred-mask-dir", default=None)
    parser.add_argument("--gt-mask-dir", default=None)
    parser.add_argument("--pred-video", default=None)
    parser.add_argument("--gt-video", default=None)
    parser.add_argument("--recall-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    results: dict[str, dict[str, float | int]] = {}
    if args.pred_mask_dir and args.gt_mask_dir:
        results["mask"] = evaluate_masks(args.pred_mask_dir, args.gt_mask_dir, args.recall_threshold)
    if args.pred_video and args.gt_video:
        results["video"] = evaluate_videos(args.pred_video, args.gt_video)
    if not results:
        raise SystemExit("Provide --pred-mask-dir/--gt-mask-dir and/or --pred-video/--gt-video.")

    text = json.dumps(results, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
