from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from part2_utils import ensure_directory, mask_to_bgr, normalize_mask, overlay_mask

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:  # pragma: no cover - optional dependency
    build_sam2 = None
    SAM2ImagePredictor = None


@dataclass
class Detection:
    box: np.ndarray
    confidence: float
    class_id: int
    prior_mask: np.ndarray | None = None


class SAM2DynamicMaskExtractor:
    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        detector_model: str = "yolov8s.pt",
        classes: Iterable[int] | None = None,
        conf_threshold: float = 0.4,
        motion_threshold: float = 1.0,
        box_expansion_ratio: float = 0.05,
        dilation_kernel_size: int = 7,
        close_kernel_size: int = 5,
        enable_motion_filter: bool = True,
        enable_temporal_stabilization: bool = False,
        device: str | None = None,
    ) -> None:
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.detector_model = detector_model
        self.classes = list(classes) if classes is not None else [0, 1, 2, 3, 5, 7, 15]
        self.conf_threshold = conf_threshold
        self.motion_threshold = motion_threshold
        self.box_expansion_ratio = box_expansion_ratio
        self.dilation_kernel_size = dilation_kernel_size
        self.close_kernel_size = close_kernel_size
        self.enable_motion_filter = enable_motion_filter
        self.enable_temporal_stabilization = enable_temporal_stabilization
        self.device = device or self._default_device()

        self.detector = None
        self.predictor = None
        self.prev_frame = None
        self.prev_mask = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def _default_device(self) -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_dependencies(self) -> None:
        missing = []
        if YOLO is None:
            missing.append("ultralytics")
        if build_sam2 is None or SAM2ImagePredictor is None:
            missing.append("sam2")

        if missing:
            raise ImportError(
                "Missing dependencies for Part 2 mask extraction: "
                + ", ".join(missing)
                + ". Install Ultralytics and the official SAM 2 repository first."
            )

    def _build_detector(self) -> None:
        if self.detector is None:
            self._ensure_dependencies()
            self.detector = YOLO(self.detector_model)

    def _build_predictor(self) -> None:
        if self.predictor is None:
            self._ensure_dependencies()
            model = build_sam2(self.sam2_config, self.sam2_checkpoint)
            if hasattr(model, "to"):
                model = model.to(self.device)
            self.predictor = SAM2ImagePredictor(model)

    @contextmanager
    def _inference_context(self):
        if torch is None:
            yield
            return

        with torch.inference_mode():
            if self.device == "cuda" and torch.cuda.is_available():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield

    def _expand_box(self, box: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = box.astype(np.float32)
        pad_x = (x2 - x1) * self.box_expansion_ratio
        pad_y = (y2 - y1) * self.box_expansion_ratio
        return np.array(
            [
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(width - 1), x2 + pad_x),
                min(float(height - 1), y2 + pad_y),
            ],
            dtype=np.float32,
        )

    def _box_mask(self, box: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_shape[1], x2)
        y2 = min(frame_shape[0], y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        return mask

    def _resize_prior_mask(self, mask: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        resized = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        return normalize_mask(resized)

    def _is_dynamic_box(self, frame: np.ndarray, box: np.ndarray) -> bool:
        if self.prev_frame is None:
            return True

        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_mask = self._box_mask(box, frame.shape)
        points = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=roi_mask,
            maxCorners=100,
            qualityLevel=0.2,
            minDistance=5,
        )

        if points is None or len(points) == 0:
            return False

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            points,
            None,
            **self.lk_params,
        )
        if next_points is None or status is None:
            return False

        good_old = points[status == 1]
        good_new = next_points[status == 1]
        if len(good_old) == 0:
            return False

        motion = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
        return bool(np.mean(motion) > self.motion_threshold)

    def detect_dynamic_objects(self, frame: np.ndarray) -> list[Detection]:
        self._build_detector()
        results = self.detector(
            frame,
            classes=self.classes,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            prior_masks = None
            if result.masks is not None:
                prior_masks = result.masks.data.cpu().numpy()

            for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                if self.enable_motion_filter and not self._is_dynamic_box(frame, box):
                    continue

                prior_mask = None
                if prior_masks is not None and index < len(prior_masks):
                    prior_mask = self._resize_prior_mask(prior_masks[index], frame.shape)

                detections.append(
                    Detection(
                        box=self._expand_box(box, frame.shape),
                        confidence=float(confidence),
                        class_id=int(class_id),
                        prior_mask=prior_mask,
                    )
                )
        return detections

    def _apply_prior_mask(self, sam_mask: np.ndarray, prior_mask: np.ndarray | None) -> np.ndarray:
        if prior_mask is None or not np.any(prior_mask):
            return sam_mask

        kernel_size = max(3, self.dilation_kernel_size)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        prior_region = cv2.dilate(normalize_mask(prior_mask), kernel, iterations=2)
        guided_mask = cv2.bitwise_and(normalize_mask(sam_mask), prior_region)

        prior_pixels = int(normalize_mask(prior_mask).sum())
        guided_pixels = int(guided_mask.sum())
        if prior_pixels > 0 and guided_pixels >= 0.2 * prior_pixels:
            return guided_mask
        return normalize_mask(prior_mask)

    def _predict_mask_from_box(self, frame_shape: tuple[int, int, int], box: np.ndarray) -> np.ndarray:
        self._build_predictor()

        with self._inference_context():
            masks, scores, _ = self.predictor.predict(
                box=box[None, :],
                multimask_output=False,
            )

        if masks is None or len(masks) == 0:
            return np.zeros(frame_shape[:2], dtype=np.uint8)

        best_index = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
        return normalize_mask(masks[best_index])

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = normalize_mask(mask)

        if self.close_kernel_size > 1:
            kernel = np.ones((self.close_kernel_size, self.close_kernel_size), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        if self.dilation_kernel_size > 1:
            kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

        if self.enable_temporal_stabilization and self.prev_mask is not None:
            overlap = cv2.bitwise_and(mask_u8, self.prev_mask)
            if np.any(overlap):
                temporal_kernel = np.ones((3, 3), np.uint8)
                stabilized = cv2.erode(self.prev_mask, temporal_kernel, iterations=1)
                mask_u8 = cv2.bitwise_or(mask_u8, stabilized)

        return normalize_mask(mask_u8)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        detections = self.detect_dynamic_objects(frame)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if detections:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._build_predictor()
            self.predictor.set_image(frame_rgb)
            for detection in detections:
                mask = self._predict_mask_from_box(frame.shape, detection.box)
                mask = self._apply_prior_mask(mask, detection.prior_mask)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        combined_mask = self._postprocess_mask(combined_mask)
        self.prev_frame = frame.copy()
        self.prev_mask = combined_mask.copy()
        return combined_mask, detections

    def process_video(
        self,
        input_path: str,
        mask_video_path: str,
        mask_frames_dir: str | None = None,
        debug_overlay_path: str | None = None,
    ) -> tuple[Path, Path | None]:
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {input_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        mask_video = Path(mask_video_path)
        mask_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        mask_writer = cv2.VideoWriter(str(mask_video), fourcc, fps, (width, height))

        overlay_writer = None
        if debug_overlay_path is not None:
            overlay_output = Path(debug_overlay_path)
            overlay_output.parent.mkdir(parents=True, exist_ok=True)
            overlay_writer = cv2.VideoWriter(str(overlay_output), fourcc, fps, (width, height))

        frames_dir = ensure_directory(mask_frames_dir) if mask_frames_dir is not None else None
        frame_index = 0
        self.prev_frame = None
        self.prev_mask = None

        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            mask, _ = self.process_frame(frame)
            mask_writer.write(mask_to_bgr(mask))

            if frames_dir is not None:
                frame_path = frames_dir / f"{frame_index:05d}.png"
                cv2.imwrite(str(frame_path), normalize_mask(mask) * 255)

            if overlay_writer is not None:
                overlay_writer.write(overlay_mask(frame, mask))

            frame_index += 1

        capture.release()
        mask_writer.release()
        if overlay_writer is not None:
            overlay_writer.release()

        return mask_video, frames_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part 2 mask extraction with YOLO + SAM 2.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--mask-video", required=True, help="Output mask video path.")
    parser.add_argument("--mask-dir", default=None, help="Optional directory for frame-wise masks.")
    parser.add_argument("--debug-overlay", default=None, help="Optional overlay video path.")
    parser.add_argument("--sam2-config", required=True, help="SAM 2 config path.")
    parser.add_argument("--sam2-checkpoint", required=True, help="SAM 2 checkpoint path.")
    parser.add_argument("--detector-model", default="yolov8s.pt", help="Detector checkpoint.")
    parser.add_argument("--classes", nargs="+", type=int, default=[0, 1, 2, 3, 5, 7, 15])
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--motion-threshold", type=float, default=1.0)
    parser.add_argument("--disable-motion-filter", action="store_true")
    parser.add_argument("--enable-temporal-stabilization", action="store_true")
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
    extractor.process_video(
        input_path=args.input,
        mask_video_path=args.mask_video,
        mask_frames_dir=args.mask_dir,
        debug_overlay_path=args.debug_overlay,
    )
