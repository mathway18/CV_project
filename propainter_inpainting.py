from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class ProPainterInpainter:
    def __init__(
        self,
        repo_dir: str,
        python_executable: str | None = None,
        resize_ratio: float = 1.0,
        neighbor_length: int = 10,
        ref_stride: int = 10,
        subvideo_length: int = 80,
        fp16: bool = False,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.python_executable = python_executable or sys.executable
        self.resize_ratio = resize_ratio
        self.neighbor_length = neighbor_length
        self.ref_stride = ref_stride
        self.subvideo_length = subvideo_length
        self.fp16 = fp16

    def _validate_repo(self) -> Path:
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"ProPainter repo not found: {self.repo_dir}. "
                "Clone the official repository before running Part 2."
            )

        inference_script = self.repo_dir / "inference_propainter.py"
        if not inference_script.exists():
            raise FileNotFoundError(
                f"Missing ProPainter inference script: {inference_script}"
            )
        return inference_script

    def _build_command(
        self,
        video_path: str,
        mask_path: str,
        output_dir: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> list[str]:
        inference_script = self._validate_repo()
        command = [
            self.python_executable,
            str(inference_script),
            "--video",
            str(video_path),
            "--mask",
            str(mask_path),
        ]

        if output_dir is not None:
            command.extend(["--output", str(output_dir)])
        if width is not None:
            command.extend(["--width", str(width)])
        if height is not None:
            command.extend(["--height", str(height)])
        if self.resize_ratio != 1.0:
            command.extend(["--resize_ratio", str(self.resize_ratio)])
        if self.neighbor_length != 10:
            command.extend(["--neighbor_length", str(self.neighbor_length)])
        if self.ref_stride != 10:
            command.extend(["--ref_stride", str(self.ref_stride)])
        if self.subvideo_length != 80:
            command.extend(["--subvideo_length", str(self.subvideo_length)])
        if self.fp16:
            command.append("--fp16")

        return command

    def _latest_output_video(self, results_dir: Path, previous_files: set[Path]) -> Path:
        preferred = [
            path
            for path in results_dir.rglob("inpaint_out.mp4")
            if path.is_file() and path.resolve() not in previous_files
        ]
        if preferred:
            return max(preferred, key=lambda item: item.stat().st_mtime)

        candidates = [
            path
            for path in results_dir.rglob("*.mp4")
            if path.is_file() and path.resolve() not in previous_files
        ]
        if not candidates:
            candidates = [path for path in results_dir.rglob("*.mp4") if path.is_file()]

        if not candidates:
            raise FileNotFoundError(
                "ProPainter finished without creating an output mp4 in its results directory."
            )

        return max(candidates, key=lambda item: item.stat().st_mtime)

    def process_video(
        self,
        video_path: str,
        mask_path: str,
        output_video_path: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Path:
        video = Path(video_path)
        mask = Path(mask_path)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video}")
        if not mask.exists():
            raise FileNotFoundError(f"Mask input not found: {mask}")

        self._validate_repo()
        output = Path(output_video_path) if output_video_path is not None else None
        results_dir = (
            output.parent / "_propainter_results"
            if output is not None
            else self.repo_dir / "results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        previous_files = {path.resolve() for path in results_dir.rglob("*") if path.is_file()}

        command = self._build_command(
            video_path=str(video.resolve()),
            mask_path=str(mask.resolve()),
            output_dir=str(results_dir.resolve()),
            width=width,
            height=height,
        )
        subprocess.run(command, cwd=str(self.repo_dir), check=True)

        generated_video = self._latest_output_video(results_dir, previous_files)
        if output_video_path is None:
            return generated_video

        output = Path(output_video_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_video, output)
        return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ProPainter on a video and frame-wise masks.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--mask", required=True, help="Mask directory or mask file path.")
    parser.add_argument("--repo", required=True, help="Local ProPainter repository path.")
    parser.add_argument("--output", default=None, help="Optional final output video path.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--resize-ratio", type=float, default=1.0)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--subvideo-length", type=int, default=80)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    inpainter = ProPainterInpainter(
        repo_dir=args.repo,
        resize_ratio=args.resize_ratio,
        neighbor_length=args.neighbor_length,
        ref_stride=args.ref_stride,
        subvideo_length=args.subvideo_length,
        fp16=args.fp16,
    )
    result = inpainter.process_video(
        video_path=args.video,
        mask_path=args.mask,
        output_video_path=args.output,
        width=args.width,
        height=args.height,
    )
    print(f"ProPainter output saved to: {result}")
