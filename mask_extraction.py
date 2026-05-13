import cv2
import numpy as np
from ultralytics import YOLO
import os

class MaskExtractor:
    def __init__(self, model_path='yolov8s-seg.pt', classes=[0, 1, 15], conf_threshold=0.5, enable_dynamic_filter=True):  # 0: person, 1: bicycle, 15: cat
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.enable_dynamic_filter = enable_dynamic_filter
        self.prev_frame = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def extract_masks(self, frame):
        results = self.model(frame, classes=self.classes, conf=self.conf_threshold)
        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8)
                    masks.append(mask)
        return masks

    def is_dynamic(self, frame, mask, threshold=1.0):
        if self.prev_frame is None:
            return True

        # Find good features in the mask region
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get points inside the mask
        mask_255 = (mask * 255).astype(np.uint8)
        points = cv2.goodFeaturesToTrack(gray_prev, mask=mask_255, maxCorners=100, qualityLevel=0.3, minDistance=7)

        if points is None or len(points) == 0:
            return False

        # Calculate optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, points, None, **self.lk_params)

        # Calculate motion magnitude
        if next_pts is not None and status is not None:
            good_old = points[status == 1]
            good_new = next_pts[status == 1]
            motion = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
            if len(motion) > 0:
                avg_motion = np.mean(motion)
                return avg_motion > threshold
        return False

    def apply_dilation(self, mask, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

    def overlay_mask(self, frame, mask, color=(0, 0, 255), alpha=0.5):
        overlay = frame.copy()
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = color
        cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
        return overlay

    def process_frame(self, frame):
        masks = self.extract_masks(frame)

        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        for mask in masks:
            if not self.enable_dynamic_filter or self.is_dynamic(frame, mask):
                dilated_mask = self.apply_dilation(mask)
                combined_mask = cv2.bitwise_or(combined_mask, dilated_mask)

        self.prev_frame = frame.copy()
        return combined_mask

    def process_video(self, input_path, output_path, debug_overlay=False, debug_overlay_path="mask_overlay.mp4"):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        out_overlay = None
        if debug_overlay:
            out_overlay = cv2.VideoWriter(debug_overlay_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mask_frame = self.process_frame(frame)
            # Convert mask to 3-channel for video writing
            mask_frame_3ch = np.stack([mask_frame * 255] * 3, axis=-1).astype(np.uint8)
            out.write(mask_frame_3ch)

            if debug_overlay:
                overlay_frame = self.overlay_mask(frame, mask_frame)
                out_overlay.write(overlay_frame)

        cap.release()
        out.release()
        if debug_overlay:
            out_overlay.release()
            print(f"Debug overlay video saved: {debug_overlay_path}")
        print(f"Mask extraction completed. Output: {output_path}")

if __name__ == "__main__":
    extractor = MaskExtractor()
    extractor.process_video("input_video.mp4", "mask_video.mp4", debug_overlay=True)
