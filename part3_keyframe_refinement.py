from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from part2_utils import ensure_directory, normalize_mask, overlay_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Part3Config:
    keyframe_stride: int = 12
    propagation_radius: int = 6
    inpaint_radius: int = 5
    mask_dilation: int = 5
    soft_edge: int = 17
    temporal_alpha: float = 0.12
    max_blend: float = 0.55
    artifact_threshold: float = 8.0
    quality_gate: bool = True
    min_seam_improvement: float = 0.005
    texture_floor_ratio: float = 0.6
    max_temporal_regression: float = 0.2
    max_masked_delta: float = 18.0
    backend: str = "opencv"
    diffusion_model: str | None = None
    prompt: str = "clean empty background, realistic video frame, no foreground object"
    negative_prompt: str = "person, bicycle, tennis player, moving object, blur, artifacts"
    diffusion_steps: int = 24
    guidance_scale: float = 7.0


@dataclass
class Part3Outputs:
    refined_video: Path
    comparison_video: Path
    keyframes_dir: Path
    metrics_json: Path


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


def list_image_files(path: str | Path) -> list[Path]:
    root = Path(path)
    files = sorted(file for file in root.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No mask images found in {root}")
    return files


def resize_like(frame: np.ndarray, reference: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    height, width = reference.shape[:2]
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def load_mask_sequence(mask_path: str | Path, reference_frames: list[np.ndarray]) -> list[np.ndarray]:
    source = Path(mask_path)
    masks: list[np.ndarray] = []
    if source.is_dir():
        for file in list_image_files(source):
            mask = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Unable to read mask: {file}")
            masks.append(mask)
    else:
        mask_frames, _ = read_video_frames(source)
        masks = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in mask_frames]

    if not masks:
        raise ValueError(f"No masks loaded from {source}")

    aligned: list[np.ndarray] = []
    for index, frame in enumerate(reference_frames):
        mask = masks[min(index, len(masks) - 1)]
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        aligned.append(normalize_mask(mask))
    return aligned


def odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return normalize_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_kernel(size), odd_kernel(size)))
    return normalize_mask(cv2.dilate(normalize_mask(mask), kernel, iterations=1))


def soft_mask(mask: np.ndarray, dilation: int, edge: int) -> np.ndarray:
    expanded = dilate_mask(mask, dilation).astype(np.float32)
    blurred = cv2.GaussianBlur(expanded, (odd_kernel(edge), odd_kernel(edge)), 0)
    return np.clip(blurred[..., None], 0.0, 1.0)


def boundary_rings(mask: np.ndarray, ring_size: int = 7) -> tuple[np.ndarray, np.ndarray]:
    mask_u8 = normalize_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_kernel(ring_size), odd_kernel(ring_size)))
    inner = mask_u8 - normalize_mask(cv2.erode(mask_u8, kernel, iterations=1))
    outer = normalize_mask(cv2.dilate(mask_u8, kernel, iterations=1)) - mask_u8
    return inner.astype(bool), outer.astype(bool)


def boundary_seam_score(frame: np.ndarray, mask: np.ndarray) -> float:
    inner, outer = boundary_rings(mask)
    if not np.any(inner) or not np.any(outer):
        return 0.0

    frame_f = frame.astype(np.float32)
    inner_mean = frame_f[inner].mean(axis=0)
    outer_mean = frame_f[outer].mean(axis=0)
    return float(np.linalg.norm(inner_mean - outer_mean))


def texture_score(frame: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = normalize_mask(mask).astype(bool)
    if not np.any(mask_bool):
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(laplacian[mask_bool])))


def write_video(frames: Iterable[np.ndarray], output_path: str | Path, fps: float, size: tuple[int, int]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()
    return output


def warp_previous_to_current(
    previous_original: np.ndarray,
    current_original: np.ndarray,
    previous_frame: np.ndarray,
) -> np.ndarray:
    prev_gray = cv2.cvtColor(previous_original, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_original, cv2.COLOR_BGR2GRAY)
    # Backward flow maps each current-frame pixel to the coordinate that should be
    # sampled from the previous frame.
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
    return cv2.remap(previous_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def masked_temporal_error(
    previous_original: np.ndarray,
    current_original: np.ndarray,
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    mask: np.ndarray,
) -> float:
    mask_bool = normalize_mask(mask).astype(bool)
    if not np.any(mask_bool):
        return 0.0

    warped_previous = warp_previous_to_current(previous_original, current_original, previous_frame)
    diff = np.mean(
        np.abs(warped_previous.astype(np.float32) - current_frame.astype(np.float32)),
        axis=2,
    )
    return float(diff[mask_bool].mean())


def masked_frame_delta(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = normalize_mask(mask).astype(bool)
    if not np.any(mask_bool):
        return 0.0
    diff = np.mean(np.abs(reference.astype(np.float32) - candidate.astype(np.float32)), axis=2)
    return float(diff[mask_bool].mean())


class DiffusionInpaintBackend:
    def __init__(self, config: Part3Config) -> None:
        if not config.diffusion_model:
            raise ValueError("--diffusion-model is required when --backend diffusers is used.")

        try:
            import torch
            from PIL import Image
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as exc:  # pragma: no cover - optional heavy dependency
            raise ImportError(
                "Diffusers backend requires: pip install diffusers accelerate safetensors"
            ) from exc

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.torch = torch
        self.image_cls = Image
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.diffusion_model,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

    def repair(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image = self.image_cls.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask_image = self.image_cls.fromarray((normalize_mask(mask) * 255).astype(np.uint8))
        with self.torch.inference_mode():
            result = self.pipe(
                prompt=self.config.prompt,
                negative_prompt=self.config.negative_prompt,
                image=image,
                mask_image=mask_image,
                num_inference_steps=self.config.diffusion_steps,
                guidance_scale=self.config.guidance_scale,
            ).images[0]
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


class Part3KeyframeRefiner:
    def __init__(self, config: Part3Config | None = None) -> None:
        self.config = config or Part3Config()
        self.diffusion_backend = None
        if self.config.backend == "diffusers":
            self.diffusion_backend = DiffusionInpaintBackend(self.config)
        elif self.config.backend != "opencv":
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _opencv_repair(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        work_mask = dilate_mask(mask, self.config.mask_dilation) * 255
        return cv2.inpaint(frame, work_mask.astype(np.uint8), self.config.inpaint_radius, cv2.INPAINT_TELEA)

    def _repair_keyframe(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.diffusion_backend is not None:
            return self.diffusion_backend.repair(frame, dilate_mask(mask, self.config.mask_dilation))
        return self._opencv_repair(frame, mask)

    def _initial_refinement(
        self,
        original: np.ndarray,
        part2: np.ndarray,
        mask: np.ndarray,
        keyframe_repair: np.ndarray | None,
    ) -> np.ndarray:
        if not np.any(mask):
            return part2.copy()

        local_repair = self._opencv_repair(original, mask)
        if keyframe_repair is not None:
            keyframe_repair = resize_like(keyframe_repair, original)
            local_repair = cv2.addWeighted(local_repair, 0.55, keyframe_repair, 0.45, 0)

        seam_part2 = boundary_seam_score(part2, mask)
        seam_repair = boundary_seam_score(local_repair, mask)
        seam_gain = max(0.0, seam_part2 - seam_repair) / max(seam_part2, 1.0)
        seam_weight = 0.25 + min(self.config.max_blend - 0.25, seam_gain)

        diff = np.mean(np.abs(part2.astype(np.float32) - local_repair.astype(np.float32)), axis=2)
        artifact = np.clip((diff - self.config.artifact_threshold) / 32.0, 0.0, 1.0)
        alpha = soft_mask(mask, self.config.mask_dilation, self.config.soft_edge)
        alpha = alpha * np.maximum(seam_weight, artifact[..., None] * self.config.max_blend)
        alpha = np.clip(alpha, 0.0, self.config.max_blend)

        refined = part2.astype(np.float32) * (1.0 - alpha) + local_repair.astype(np.float32) * alpha
        return np.clip(refined, 0, 255).astype(np.uint8)

    def _temporal_smooth(
        self,
        previous_original: np.ndarray,
        current_original: np.ndarray,
        previous_refined: np.ndarray,
        current_refined: np.ndarray,
        current_mask: np.ndarray,
    ) -> np.ndarray:
        if self.config.temporal_alpha <= 0.0 or not np.any(current_mask):
            return current_refined

        warped_prev = warp_previous_to_current(previous_original, current_original, previous_refined)
        alpha = soft_mask(current_mask, self.config.mask_dilation, self.config.soft_edge)
        alpha *= self.config.temporal_alpha
        smoothed = current_refined.astype(np.float32) * (1.0 - alpha) + warped_prev.astype(np.float32) * alpha
        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def _quality_gate(
        self,
        index: int,
        previous_original: np.ndarray | None,
        current_original: np.ndarray,
        previous_part2: np.ndarray | None,
        current_part2: np.ndarray,
        previous_refined: np.ndarray | None,
        candidate: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, bool, str, dict[str, float]]:
        if not self.config.quality_gate or not np.any(mask):
            return candidate, True, "accepted_gate_disabled", {}

        part2_seam = boundary_seam_score(current_part2, mask)
        candidate_seam = boundary_seam_score(candidate, mask)
        part2_texture = texture_score(current_part2, mask)
        candidate_texture = texture_score(candidate, mask)
        masked_delta = masked_frame_delta(current_part2, candidate, mask)

        gate_metrics = {
            "candidate_boundary_seam": candidate_seam,
            "candidate_texture": candidate_texture,
            "candidate_masked_delta": masked_delta,
        }

        required_seam = part2_seam * (1.0 - self.config.min_seam_improvement)
        if candidate_seam > required_seam:
            return current_part2.copy(), False, "rejected_seam", gate_metrics

        if part2_texture > 1e-6 and candidate_texture < part2_texture * self.config.texture_floor_ratio:
            return current_part2.copy(), False, "rejected_texture_loss", gate_metrics

        if masked_delta > self.config.max_masked_delta:
            return current_part2.copy(), False, "rejected_large_delta", gate_metrics

        if (
            index > 0
            and previous_original is not None
            and previous_part2 is not None
            and previous_refined is not None
        ):
            part2_temporal = masked_temporal_error(
                previous_original,
                current_original,
                previous_part2,
                current_part2,
                mask,
            )
            candidate_temporal = masked_temporal_error(
                previous_original,
                current_original,
                previous_refined,
                candidate,
                mask,
            )
            gate_metrics["part2_temporal_error"] = part2_temporal
            gate_metrics["candidate_temporal_error"] = candidate_temporal
            if candidate_temporal > part2_temporal + self.config.max_temporal_regression:
                return current_part2.copy(), False, "rejected_temporal_regression", gate_metrics

        return candidate, True, "accepted_quality_gate", gate_metrics

    def process(
        self,
        input_video: str,
        part2_video: str,
        mask_path: str,
        output_dir: str,
        output_name: str = "part3_output.mp4",
    ) -> Part3Outputs:
        output_root = ensure_directory(output_dir)
        keyframes_dir = ensure_directory(output_root / "keyframe_repairs")
        frames_dir = ensure_directory(output_root / "ppt_frames")

        original_frames, fps = read_video_frames(input_video)
        part2_frames, _ = read_video_frames(part2_video)
        masks = load_mask_sequence(mask_path, original_frames)
        frame_count = min(len(original_frames), len(part2_frames), len(masks))
        if frame_count == 0:
            raise ValueError("No overlapping frames between input, Part 2 video, and masks.")

        original_frames = original_frames[:frame_count]
        part2_frames = [resize_like(frame, original_frames[index]) for index, frame in enumerate(part2_frames[:frame_count])]
        masks = masks[:frame_count]
        height, width = original_frames[0].shape[:2]

        keyframe_indices = [
            index
            for index, mask in enumerate(masks)
            if index % self.config.keyframe_stride == 0 and np.any(mask)
        ]
        keyframe_repairs: dict[int, np.ndarray] = {}
        for index in keyframe_indices:
            repair = self._repair_keyframe(original_frames[index], masks[index])
            keyframe_repairs[index] = repair
            cv2.imwrite(str(keyframes_dir / f"{index:05d}_repair.png"), repair)
            cv2.imwrite(str(keyframes_dir / f"{index:05d}_overlay.png"), overlay_mask(original_frames[index], masks[index]))

        refined_frames: list[np.ndarray] = []
        per_frame_metrics: list[dict[str, float | int]] = []
        accepted_count = 0
        for index in range(frame_count):
            nearest_keyframe = None
            if keyframe_repairs:
                nearest_index = min(keyframe_repairs, key=lambda item: abs(item - index))
                if abs(nearest_index - index) <= self.config.propagation_radius:
                    nearest_keyframe = keyframe_repairs[nearest_index]

            refined = self._initial_refinement(
                original=original_frames[index],
                part2=part2_frames[index],
                mask=masks[index],
                keyframe_repair=nearest_keyframe,
            )
            if refined_frames:
                refined = self._temporal_smooth(
                    previous_original=original_frames[index - 1],
                    current_original=original_frames[index],
                    previous_refined=refined_frames[-1],
                    current_refined=refined,
                    current_mask=masks[index],
                )

            refined, accepted, gate_reason, gate_metrics = self._quality_gate(
                index=index,
                previous_original=original_frames[index - 1] if index > 0 else None,
                current_original=original_frames[index],
                previous_part2=part2_frames[index - 1] if index > 0 else None,
                current_part2=part2_frames[index],
                previous_refined=refined_frames[-1] if refined_frames else None,
                candidate=refined,
                mask=masks[index],
            )
            if accepted and np.any(masks[index]):
                accepted_count += 1

            refined_frames.append(refined)
            per_frame_metrics.append(
                {
                    "frame": index,
                    "mask_pixels": int(normalize_mask(masks[index]).sum()),
                    "part2_boundary_seam": boundary_seam_score(part2_frames[index], masks[index]),
                    "part3_boundary_seam": boundary_seam_score(refined, masks[index]),
                    "part2_texture": texture_score(part2_frames[index], masks[index]),
                    "part3_texture": texture_score(refined, masks[index]),
                    "accepted": int(accepted),
                    "gate_reason": gate_reason,
                    **gate_metrics,
                }
            )

        refined_video = output_root / output_name
        if accepted_count == 0:
            refined_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(part2_video, refined_video)
        else:
            refined_video = write_video(refined_frames, refined_video, fps, (width, height))
        comparison_video = self._write_comparison(
            original_frames,
            masks,
            part2_frames,
            refined_frames,
            output_root / "part3_comparison.mp4",
            fps,
            (width, height),
        )
        self._export_keyframes(original_frames, masks, part2_frames, refined_frames, frames_dir)

        summary = {
            "config": asdict(self.config),
            "input_video": str(input_video),
            "part2_video": str(part2_video),
            "mask_path": str(mask_path),
            "frames": frame_count,
            "fps": fps,
            "size": [width, height],
            "keyframes": keyframe_indices,
            "accepted_frames": accepted_count,
            "rejected_frames": int(sum(1 for item in per_frame_metrics if item["mask_pixels"] and not item["accepted"])),
            "mean_part2_boundary_seam": float(np.mean([item["part2_boundary_seam"] for item in per_frame_metrics])),
            "mean_part3_boundary_seam": float(np.mean([item["part3_boundary_seam"] for item in per_frame_metrics])),
            "mean_part2_texture": float(np.mean([item["part2_texture"] for item in per_frame_metrics])),
            "mean_part3_texture": float(np.mean([item["part3_texture"] for item in per_frame_metrics])),
            "per_frame": per_frame_metrics,
        }
        metrics_json = output_root / "part3_metrics.json"
        metrics_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        return Part3Outputs(
            refined_video=refined_video,
            comparison_video=comparison_video,
            keyframes_dir=keyframes_dir,
            metrics_json=metrics_json,
        )

    def _write_comparison(
        self,
        originals: list[np.ndarray],
        masks: list[np.ndarray],
        part2_frames: list[np.ndarray],
        part3_frames: list[np.ndarray],
        output_path: Path,
        fps: float,
        size: tuple[int, int],
    ) -> Path:
        width, height = size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 4, height))
        for original, mask, part2, part3 in zip(originals, masks, part2_frames, part3_frames):
            panels = [
                add_label(original, "Original"),
                add_label(overlay_mask(original, mask), "Mask"),
                add_label(part2, "Part 2"),
                add_label(part3, "Part 3"),
            ]
            writer.write(np.hstack(panels))
        writer.release()
        return output_path

    def _export_keyframes(
        self,
        originals: list[np.ndarray],
        masks: list[np.ndarray],
        part2_frames: list[np.ndarray],
        part3_frames: list[np.ndarray],
        output_dir: Path,
        max_samples: int = 4,
    ) -> None:
        candidates = [index for index, mask in enumerate(masks) if np.any(mask)]
        if not candidates:
            return
        if len(candidates) <= max_samples:
            sample_indices = candidates
        else:
            positions = np.linspace(0, len(candidates) - 1, max_samples).round().astype(int)
            sample_indices = [candidates[int(pos)] for pos in positions]

        for sample_index, frame_index in enumerate(sample_indices):
            cv2.imwrite(str(output_dir / f"{sample_index:02d}_original.png"), originals[frame_index])
            cv2.imwrite(str(output_dir / f"{sample_index:02d}_mask_overlay.png"), overlay_mask(originals[frame_index], masks[frame_index]))
            cv2.imwrite(str(output_dir / f"{sample_index:02d}_part2.png"), part2_frames[frame_index])
            cv2.imwrite(str(output_dir / f"{sample_index:02d}_part3.png"), part3_frames[frame_index])


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (190, 38), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 3 extension: keyframe repair, propagation-aware blending, and temporal smoothing."
    )
    parser.add_argument("--input", required=True, help="Original input video.")
    parser.add_argument("--part2-video", required=True, help="Part 2 restored video.")
    parser.add_argument("--mask", required=True, help="Frame-wise mask directory or mask video.")
    parser.add_argument("--output-dir", required=True, help="Directory for Part 3 outputs.")
    parser.add_argument("--output-name", default="part3_output.mp4")
    parser.add_argument("--backend", choices=["opencv", "diffusers"], default="opencv")
    parser.add_argument("--diffusion-model", default=None)
    parser.add_argument("--prompt", default=Part3Config.prompt)
    parser.add_argument("--negative-prompt", default=Part3Config.negative_prompt)
    parser.add_argument("--diffusion-steps", type=int, default=Part3Config.diffusion_steps)
    parser.add_argument("--guidance-scale", type=float, default=Part3Config.guidance_scale)
    parser.add_argument("--keyframe-stride", type=int, default=Part3Config.keyframe_stride)
    parser.add_argument("--propagation-radius", type=int, default=Part3Config.propagation_radius)
    parser.add_argument("--inpaint-radius", type=int, default=Part3Config.inpaint_radius)
    parser.add_argument("--mask-dilation", type=int, default=Part3Config.mask_dilation)
    parser.add_argument("--soft-edge", type=int, default=Part3Config.soft_edge)
    parser.add_argument("--temporal-alpha", type=float, default=Part3Config.temporal_alpha)
    parser.add_argument("--max-blend", type=float, default=Part3Config.max_blend)
    parser.add_argument("--artifact-threshold", type=float, default=Part3Config.artifact_threshold)
    parser.add_argument("--disable-quality-gate", action="store_true")
    parser.add_argument("--min-seam-improvement", type=float, default=Part3Config.min_seam_improvement)
    parser.add_argument("--texture-floor-ratio", type=float, default=Part3Config.texture_floor_ratio)
    parser.add_argument("--max-temporal-regression", type=float, default=Part3Config.max_temporal_regression)
    parser.add_argument("--max-masked-delta", type=float, default=Part3Config.max_masked_delta)
    args = parser.parse_args()

    config = Part3Config(
        keyframe_stride=args.keyframe_stride,
        propagation_radius=args.propagation_radius,
        inpaint_radius=args.inpaint_radius,
        mask_dilation=args.mask_dilation,
        soft_edge=args.soft_edge,
        temporal_alpha=args.temporal_alpha,
        max_blend=args.max_blend,
        artifact_threshold=args.artifact_threshold,
        quality_gate=not args.disable_quality_gate,
        min_seam_improvement=args.min_seam_improvement,
        texture_floor_ratio=args.texture_floor_ratio,
        max_temporal_regression=args.max_temporal_regression,
        max_masked_delta=args.max_masked_delta,
        backend=args.backend,
        diffusion_model=args.diffusion_model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        diffusion_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
    )
    refiner = Part3KeyframeRefiner(config)
    outputs = refiner.process(
        input_video=args.input,
        part2_video=args.part2_video,
        mask_path=args.mask,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
    print(f"Part 3 output: {outputs.refined_video}")
    print(f"Comparison video: {outputs.comparison_video}")
    print(f"Keyframe repairs: {outputs.keyframes_dir}")
    print(f"Metrics: {outputs.metrics_json}")


if __name__ == "__main__":
    main()
