import os
import sys
import cv2
import torch
import numpy as np
import json
import time
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# PATH SETUP
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.interface import Detector
from utils.icon_manager import IconManager

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "./data/videos/raw/cow.mp4"
OUTPUT_PATH = "./data/videos/hazard_output.mp4"

YOLO_MODEL_PATH = "models/yolo26n_rail_final.mlpackage"
TEPNET_MODEL_PATH = "models/twinkling-rocket-21"

CONF_THRESHOLD = 0.15
OVERLAP_THRESHOLD = 100
H_FOV_DEG = 63
COW_HEIGHT_M = 1.5

# All red for hazards
COLORS = {
    'person': (0, 0, 255),
    'car': (0, 0, 255),
    'bus': (0, 0, 255),
    'truck': (0, 0, 255),
    'cow': (0, 0, 255),
    'horse': (0, 0, 255),
    'deer': (0, 0, 255),
    'dog': (0, 0, 255),
    'bear': (0, 0, 255),
}


# -----------------------------
# HAZARD PIPELINE
# -----------------------------
class HazardPipeline:
    def __init__(self, logo_path="logo.png"):
        print("Initializing Hazard Pipeline...")

        # Device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"Using device: {device}")

        # Models
        self.yolo = YOLO(YOLO_MODEL_PATH)

        self.tepnet = Detector(
            model_path=TEPNET_MODEL_PATH,
            crop_coords=None,
            runtime="pytorch",
            device=device
        )

        # Icons
        self.icon_mgr = IconManager()

        # Logo
        self.logo = None
        if os.path.exists(logo_path):
            self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if self.logo is not None:
                self.logo = cv2.resize(self.logo, (80, 120))

        print("Models + UI assets loaded.\n")

    # -----------------------------
    # Distance Estimation
    # -----------------------------
    def estimate_distance(self, pixel_height, image_width):
        if pixel_height <= 0:
            return None

        focal_length_px = (image_width / 2) / np.tan(np.radians(H_FOV_DEG / 2))
        distance_m = (focal_length_px * COW_HEIGHT_M) / pixel_height
        return distance_m

    # -----------------------------
    # Frame Processing
    # -----------------------------
    def process_frame(self, frame):
        orig_h, orig_w = frame.shape[:2]
        combined = frame.copy()

        # -------------------------
        # 1) Run TEP-Net
        # -------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        seg_result = self.tepnet.detect(pil_img)

        ego_mask_resized = None

        if isinstance(seg_result, Image.Image):
            ego_mask = np.array(seg_result)

            if ego_mask.shape != (orig_h, orig_w):
                ego_mask_resized = cv2.resize(
                    ego_mask,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                ego_mask_resized = ego_mask

            # Draw ego-path overlay
            overlay = frame.copy()
            overlay[ego_mask_resized > 0] = [0, 255, 0]
            combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # -------------------------
        # 2) Run YOLO
        # -------------------------
        results = self.yolo(
            frame,
            imgsz=1536,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.yolo.names[cls]

                pixel_height = y2 - y1
                distance_m = self.estimate_distance(pixel_height, orig_w)

                # -------------------------
                # 3) Fusion Logic
                # -------------------------
                hazard = False

                if ego_mask_resized is not None:
                    obj_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    obj_mask[int(y1):int(y2), int(x1):int(x2)] = 1

                    intersection = (ego_mask_resized > 0) & (obj_mask > 0)
                    overlap_pixels = np.sum(intersection)

                    if overlap_pixels > OVERLAP_THRESHOLD:
                        hazard = True

                # -------------------------
                # 4) Draw Using IconManager
                # -------------------------
                color = COLORS.get(class_name, (0, 0, 255))

                if hazard:
                    self.icon_mgr.draw_detection_with_icon(
                        combined,
                        [x1, y1, x2, y2],
                        class_name,
                        conf,
                        color
                    )

                    # Emit JSON event
                    hazard_event = {
                        "timestamp": time.time(),
                        "class": class_name,
                        "confidence": conf,
                        "distance_m": distance_m,
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                        }
                    }

                    print(json.dumps(hazard_event))

                else:
                    # Non-hazard boxes can be drawn lightly if desired
                    cv2.rectangle(
                        combined,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 255, 0),
                        1
                    )

        # -------------------------
        # 5) Logo Overlay
        # -------------------------
        if self.logo is not None:
            l_h, l_w = self.logo.shape[:2]
            margin = 15
            y1_l, y2_l = margin, margin + l_h
            x1_l, x2_l = orig_w - l_w - margin, orig_w - margin

            if y2_l <= orig_h and x2_l <= orig_w:
                roi = combined[y1_l:y2_l, x1_l:x2_l]
                if self.logo.shape[2] == 4:
                    alpha = self.logo[:, :, 3] / 255.0
                    for c in range(3):
                        roi[:, :, c] = (
                            alpha * self.logo[:, :, c]
                            + (1 - alpha) * roi[:, :, c]
                        )
                else:
                    combined[y1_l:y2_l, x1_l:x2_l] = self.logo[:, :, :3]

        return combined


# -----------------------------
# RUN
# -----------------------------
def run():
    if not os.path.exists(VIDEO_PATH):
        print("Video not found.")
        return

    pipeline = HazardPipeline()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        return

    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    print("Running Hazard Detection...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)

        cv2.imshow("Axis v0 - Hazard Detection", result)
        out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
