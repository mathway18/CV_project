from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from part2_utils import ensure_directory
from propainter_inpainting import ProPainterInpainter
from sam2_mask_extraction import SAM2DynamicMaskExtractor


@dataclass
class Part2Outputs:
    mask_video: Path
    mask_frames_dir: Path
    restored_video: Path
    debug_overlay: Path | None


class AIDrivenVideoObjectRemoval:
    def __init__(
        self,
        mask_extractor: SAM2DynamicMaskExtractor,
        inpainter: ProPainterInpainter,
    ) -> None:
        self.mask_extractor = mask_extractor
        self.inpainter = inpainter

    def process_video(
        self,
        input_path: str,
        output_dir: str,
        final_output_name: str = "part2_output.mp4",
        mask_video_name: str = "sam2_mask_video.mp4",
        mask_dir_name: str = "sam2_masks",
        debug: bool = False,
        width: int | None = None,
        height: int | None = None,
    ) -> Part2Outputs:
        output_root = ensure_directory(output_dir)
        mask_frames_dir = ensure_directory(output_root / mask_dir_name)
        mask_video_path = output_root / mask_video_name
        debug_overlay_path = output_root / "sam2_overlay.mp4" if debug else None

        self.mask_extractor.process_video(
            input_path=input_path,
            mask_video_path=str(mask_video_path),
            mask_frames_dir=str(mask_frames_dir),
            debug_overlay_path=str(debug_overlay_path) if debug_overlay_path else None,
        )

        restored_video = self.inpainter.process_video(
            video_path=input_path,
            mask_path=str(mask_frames_dir),
            output_video_path=str(output_root / final_output_name),
            width=width,
            height=height,
        )

        return Part2Outputs(
            mask_video=mask_video_path,
            mask_frames_dir=mask_frames_dir,
            restored_video=restored_video,
            debug_overlay=debug_overlay_path,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part 2 AI-driven pipeline: SAM 2 + ProPainter.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--sam2-config", required=True, help="SAM 2 config path.")
    parser.add_argument("--sam2-checkpoint", required=True, help="SAM 2 checkpoint path.")
    parser.add_argument("--propainter-repo", required=True, help="Local ProPainter repository path.")
    parser.add_argument("--detector-model", default="yolov8s.pt", help="Detector checkpoint.")
    parser.add_argument("--classes", nargs="+", type=int, default=[0, 1, 2, 3, 5, 7, 15])
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--motion-threshold", type=float, default=1.0)
    parser.add_argument("--disable-motion-filter", action="store_true")
    parser.add_argument("--enable-temporal-stabilization", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--resize-ratio", type=float, default=1.0)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--subvideo-length", type=int, default=80)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    extractor = SAM2DynamicMaskExtractor(
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        detector_model=args.detector_model,
        classes=args.classes,
        conf_threshold=args.conf_threshold,
        motion_threshold=args.motion_threshold,
        enable_motion_filter=not args.disable_motion_filter,
        enable_temporal_stabilization=args.enable_temporal_stabilization,
    )
    inpainter = ProPainterInpainter(
        repo_dir=args.propainter_repo,
        resize_ratio=args.resize_ratio,
        neighbor_length=args.neighbor_length,
        ref_stride=args.ref_stride,
        subvideo_length=args.subvideo_length,
        fp16=args.fp16,
    )

    pipeline = AIDrivenVideoObjectRemoval(extractor, inpainter)
    outputs = pipeline.process_video(
        input_path=args.input,
        output_dir=args.output_dir,
        debug=args.debug,
        width=args.width,
        height=args.height,
    )
    print(f"Mask video: {outputs.mask_video}")
    print(f"Mask frames: {outputs.mask_frames_dir}")
    print(f"Restored video: {outputs.restored_video}")
    if outputs.debug_overlay is not None:
        print(f"Debug overlay: {outputs.debug_overlay}")
