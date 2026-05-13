import cv2
import numpy as np
import os

class BackgroundRestorer:
    def __init__(self):
        self.prev_frame = None
        self.prev_mask = None

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

    def process_video(self, input_path, mask_path, output_path, debug=False, debug_dir="debug", debug_interval=30):
        cap = cv2.VideoCapture(input_path)
        mask_cap = cv2.VideoCapture(mask_path)

        if not cap.isOpened() or not mask_cap.isOpened():
            print("Error opening video files")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if debug:
            os.makedirs(debug_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_index = 0
        while cap.isOpened() and mask_cap.isOpened():
            ret, frame = cap.read()
            ret_mask, mask_frame = mask_cap.read()
            if not ret or not ret_mask:
                break

            # Convert mask to grayscale
            mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            mask = (mask > 127).astype(np.uint8)

            processed_frame = self.inpaint_frame(frame, mask)
            out.write(processed_frame)

            if debug and frame_index % debug_interval == 0:
                mask_overlay = cv2.addWeighted(frame, 0.7, cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)
                debug_image = np.hstack((frame, mask_overlay, processed_frame))
                debug_path = os.path.join(debug_dir, f"debug_frame_{frame_index:04d}.png")
                cv2.imwrite(debug_path, debug_image)

            self.prev_frame = frame.copy()
            self.prev_mask = mask.copy()
            frame_index += 1

        cap.release()
        mask_cap.release()
        out.release()
        print(f"Background restoration completed. Output: {output_path}")
        if debug:
            print(f"Debug images saved to: {debug_dir}")

if __name__ == "__main__":
    restorer = BackgroundRestorer()
    restorer.process_video("input_video.mp4", "mask_video.mp4", "output_video.mp4", debug=True, debug_interval=30)