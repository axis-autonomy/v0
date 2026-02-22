import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from utils.interface import Detector
    print("✅ Detector successfully imported!")
except ImportError as e:
    print(f"❌ Import still failing. Check if this path exists: {SRC_PATH}")
    print(f"Error detail: {e}")
    sys.exit(1)

VIDEO_PATH = "./data/videos/raw/cow.mp4"
OUTPUT_PATH = "./data/videos/demo_output_tepnet.mp4"


class TEPNetRailDemo:
    def __init__(self, model_name="twinkling-rocket-21", logo_path="logo.png"):
        print(f"Loading TEP-Net model: {model_name}")

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"Using device: {device}")

        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", model_name)

        self.detector = Detector(
            model_path=MODEL_PATH,
            crop_coords=None,
            runtime="pytorch",
            device=device,
        )

        self.logo = None
        if os.path.exists(logo_path):
            self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if self.logo is not None:
                self.logo = cv2.resize(self.logo, (80, 120))

    def process_frame(self, frame):
        orig_h, orig_w = frame.shape[:2]

        # Convert frame once
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # ---- SEGMENTATION TIMING ----
        seg_start = time.time()
        res = self.detector.detect(pil_img)
        seg_end = time.time()

        seg_time = seg_end - seg_start
        seg_fps = 1 / seg_time if seg_time > 0 else 0
        print(f"Segmentation FPS: {seg_fps:.2f}")

        combined = frame.copy()

        # ---- SEGMENTATION OVERLAY ----
        if isinstance(res, Image.Image):
            ego_mask = np.array(res)

            if ego_mask.shape != (orig_h, orig_w):
                ego_mask = cv2.resize(
                    ego_mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            overlay = frame.copy()
            overlay[ego_mask > 0] = [0, 255, 0]
            combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        else:
            try:
                left, right = res[0], res[1]
                if len(left) >= 2 and len(right) >= 2:
                    left_pts = np.array(left, dtype=np.int32)
                    right_pts = np.array(right, dtype=np.int32)
                    polygon = np.vstack([left_pts, right_pts[::-1]])

                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [polygon], (0, 255, 0))
                    combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            except Exception as e:
                print(f"Error drawing ego-path: {e}")

        # ---- LOGO ----
        

        return combined


def run_demo():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
        return

    demo = TEPNetRailDemo(model_name="twinkling-rocket-21")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    print("Running TEP-Net Demo... Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ---- FULL PIPELINE TIMING ----
            loop_start = time.time()

            result = demo.process_frame(frame)

            loop_end = time.time()
            full_fps = 1 / (loop_end - loop_start)
            print(f"Full Pipeline FPS: {full_fps:.2f}")

            cv2.imshow("Axis v0 - TEP-Net Ego-Path Detection", result)
            out.write(result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print("\nDemo complete!")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
