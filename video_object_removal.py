import cv2
import numpy as np
from ultralytics import YOLO
import os

class VideoObjectRemoval:
    def __init__(self, model_path='yolov8s-seg.pt', classes=[0, 1, 15]):  # 0: person, 1: bicycle, 15: cat
        self.model = YOLO(model_path)
        self.classes = classes
        self.prev_frame = None
        self.prev_mask = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def extract_masks(self, frame):
        results = self.model(frame, classes=self.classes)
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

    def inpaint_frame(self, frame, mask):
        # Advanced Idea: Temporal Background Propagation
        if self.prev_frame is not None and self.prev_mask is not None:
            # Create temporal mask: pixels that are masked in current but clean in previous
            temporal_mask = mask & ~self.prev_mask  # Current masked, previous clean
            if np.any(temporal_mask):
                # Borrow from previous frame
                frame[temporal_mask > 0] = self.prev_frame[temporal_mask > 0]
                # Update mask: remove temporally filled pixels
                mask = mask & ~temporal_mask

        # Fallback: Spatial inpainting for remaining masked regions
        if np.any(mask):
            frame = cv2.inpaint(frame, mask * 255, 3, cv2.INPAINT_TELEA)

        return frame

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
            if self.is_dynamic(frame, mask):
                dilated_mask = self.apply_dilation(mask)
                combined_mask = cv2.bitwise_or(combined_mask, dilated_mask)

        if np.any(combined_mask):
            result = self.inpaint_frame(frame, combined_mask)
        else:
            result = frame.copy()

        self.prev_frame = frame.copy()
        self.prev_mask = combined_mask.copy()
        return result, combined_mask

    def process_video(self, input_path, output_path, debug=False, debug_dir="debug", debug_interval=30):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if debug:
            os.makedirs(debug_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, combined_mask = self.process_frame(frame)
            out.write(processed_frame)

            if debug and frame_index % debug_interval == 0:
                mask_overlay = self.overlay_mask(frame, combined_mask)
                debug_image = np.hstack((frame, mask_overlay, processed_frame))
                debug_path = os.path.join(debug_dir, f"debug_frame_{frame_index:04d}.png")
                cv2.imwrite(debug_path, debug_image)

            frame_index += 1

        cap.release()
        out.release()
        print("Video processing completed")
        if debug:
            print(f"Debug images saved to: {debug_dir}")

if __name__ == "__main__":
    remover = VideoObjectRemoval()
    remover.process_video("input_video.mp4", "output_video.mp4", debug=True, debug_interval=30)
