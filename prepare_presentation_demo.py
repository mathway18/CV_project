from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from background_restoration import BackgroundRestorer
from mask_extraction import MaskExtractor
from part2_utils import ensure_directory, overlay_mask


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (260, 42), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def resize_like(frame: np.ndarray, width: int, height: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def build_side_by_side_video(
    input_video: str,
    mask_video: str,
    restored_video: str,
    output_video: str,
) -> None:
    cap_input = cv2.VideoCapture(input_video)
    cap_mask = cv2.VideoCapture(mask_video)
    cap_restored = cv2.VideoCapture(restored_video)

    if not (cap_input.isOpened() and cap_mask.isOpened() and cap_restored.isOpened()):
        raise FileNotFoundError("Unable to open one or more videos for side-by-side export.")

    fps = float(cap_input.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width * 3, height))

    while cap_input.isOpened() and cap_mask.isOpened() and cap_restored.isOpened():
        ok_input, frame_input = cap_input.read()
        ok_mask, frame_mask = cap_mask.read()
        ok_restored, frame_restored = cap_restored.read()
        if not (ok_input and ok_mask and ok_restored):
            break

        frame_mask = resize_like(frame_mask, width, height, cv2.INTER_NEAREST)
        frame_restored = resize_like(frame_restored, width, height, cv2.INTER_CUBIC)
        mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask_gray > 127).astype(np.uint8)
        overlay = overlay_mask(frame_input, mask_binary)

        panel = np.hstack(
            [
                add_label(frame_input, "Original"),
                add_label(overlay, "Mask Overlay"),
                add_label(frame_restored, "Restored"),
            ]
        )
        writer.write(panel)

    cap_input.release()
    cap_mask.release()
    cap_restored.release()
    writer.release()


def export_keyframes(
    input_video: str,
    mask_video: str,
    restored_video: str,
    output_dir: str,
    frame_step: int = 30,
    max_samples: int = 4,
) -> None:
    export_dir = ensure_directory(output_dir)
    cap_input = cv2.VideoCapture(input_video)
    cap_mask = cv2.VideoCapture(mask_video)
    cap_restored = cv2.VideoCapture(restored_video)

    if not (cap_input.isOpened() and cap_mask.isOpened() and cap_restored.isOpened()):
        raise FileNotFoundError("Unable to open one or more videos for keyframe export.")

    frame_index = 0
    sample_index = 0

    while cap_input.isOpened() and cap_mask.isOpened() and cap_restored.isOpened():
        ok_input, frame_input = cap_input.read()
        ok_mask, frame_mask = cap_mask.read()
        ok_restored, frame_restored = cap_restored.read()
        if not (ok_input and ok_mask and ok_restored):
            break

        if frame_index % frame_step == 0:
            frame_mask = resize_like(frame_mask, frame_input.shape[1], frame_input.shape[0], cv2.INTER_NEAREST)
            frame_restored = resize_like(frame_restored, frame_input.shape[1], frame_input.shape[0], cv2.INTER_CUBIC)
            mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
            mask_binary = (mask_gray > 127).astype(np.uint8)
            overlay = overlay_mask(frame_input, mask_binary)

            cv2.imwrite(str(export_dir / f"{sample_index:02d}_original.png"), frame_input)
            cv2.imwrite(str(export_dir / f"{sample_index:02d}_overlay.png"), overlay)
            cv2.imwrite(str(export_dir / f"{sample_index:02d}_restored.png"), frame_restored)

            sample_index += 1
            if sample_index >= max_samples:
                break

        frame_index += 1

    cap_input.release()
    cap_mask.release()
    cap_restored.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare oral-presentation demo assets for Part 1.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Directory for all generated demo assets.")
    parser.add_argument("--classes", nargs="+", type=int, default=[0, 1, 15], help="Dynamic classes for YOLOv8-Seg.")
    parser.add_argument("--debug-interval", type=int, default=30, help="Frame interval for debug snapshots.")
    parser.add_argument("--keyframe-step", type=int, default=30, help="Frame interval for exported PPT images.")
    parser.add_argument("--keyframe-count", type=int, default=4, help="Maximum number of PPT sample groups.")
    args = parser.parse_args()

    output_dir = ensure_directory(args.output_dir)
    mask_video = output_dir / "mask_video.mp4"
    overlay_video = output_dir / "mask_overlay.mp4"
    restored_video = output_dir / "restored_video.mp4"
    comparison_video = output_dir / "comparison_demo.mp4"
    debug_dir = output_dir / "debug_frames"
    ppt_frames_dir = output_dir / "ppt_frames"

    extractor = MaskExtractor(classes=args.classes)
    extractor.process_video(
        input_path=args.input,
        output_path=str(mask_video),
        debug_overlay=True,
        debug_overlay_path=str(overlay_video),
    )

    restorer = BackgroundRestorer()
    restorer.process_video(
        input_path=args.input,
        mask_path=str(mask_video),
        output_path=str(restored_video),
        debug=True,
        debug_dir=str(debug_dir),
        debug_interval=args.debug_interval,
    )

    build_side_by_side_video(
        input_video=args.input,
        mask_video=str(mask_video),
        restored_video=str(restored_video),
        output_video=str(comparison_video),
    )
    export_keyframes(
        input_video=args.input,
        mask_video=str(mask_video),
        restored_video=str(restored_video),
        output_dir=str(ppt_frames_dir),
        frame_step=args.keyframe_step,
        max_samples=args.keyframe_count,
    )

    print(f"Mask video: {mask_video}")
    print(f"Mask overlay video: {overlay_video}")
    print(f"Restored video: {restored_video}")
    print(f"Comparison demo: {comparison_video}")
    print(f"Debug frames: {debug_dir}")
    print(f"PPT keyframes: {ppt_frames_dir}")


if __name__ == "__main__":
    main()
