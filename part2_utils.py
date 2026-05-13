from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    metadata = VideoMetadata(
        fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    capture.release()
    return metadata


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    mask_u8 = normalize_mask(mask) * 255
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = frame.copy()
    colored_mask = np.zeros_like(frame)
    colored_mask[normalize_mask(mask) > 0] = color
    return cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0)


def write_binary_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), normalize_mask(mask) * 255)
    return output
