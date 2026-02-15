import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Now root
PROJECT_ROOT = CURRENT_DIR  # Same as current dir

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# 3. Now the import will work because 'utils' is directly inside TEPNET_SRC
try:
    from utils.interface import Detector
    print("✅ Detector successfully imported!")
except ImportError as e:
    print(f"❌ Import still failing. Check if this path exists: {SRC_PATH}")
    print(f"Error detail: {e}")
    sys.exit(1)

# Configuration
VIDEO_PATH = "./data/videos/raw/cow.mp4"
OUTPUT_PATH = "./data/videos/demo_output_tepnet.mp4"

class TEPNetRailDemo:
    def __init__(self, model_name="twinkling-rocket-21", logo_path="logo.png"):
        """
        Initialize TEP-Net demo
        
        Model options:
        - 'logical-tree-1': ResNet-18, fastest (2.3ms), IoU 0.9695
        - 'chromatic-laughter-5': EfficientNet-B3, most accurate (14.9ms), IoU 0.9753
        """
        print(f"Loading TEP-Net model: {model_name}")
        
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        print(f"Using device: {device}")
        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", model_name)

        # Initialize TEP-Net detector
        self.detector = Detector(
            model_path=MODEL_PATH,
            crop_coords=None, 
            runtime='pytorch',
            device=device
        )
        
        # Load logo
        self.logo = None
        if os.path.exists(logo_path):
            self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if self.logo is not None:
                self.logo = cv2.resize(self.logo, (80, 120))
    
    def process_frame(self, frame): 
        """Optimized frame processing for ResNet-18 Segmentation"""
        # 1. Prepare image for TEP-Net
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 2. Inference
        res = self.detector.detect(pil_img)

        # 3. Fast Overlay for Segmentation (brilliant-horse-15)
        if isinstance(res, Image.Image):
            # Convert PIL mask to numpy once
            ego_mask = np.array(res) 
            
            # Resize mask to match original frame size
            ego_mask_resized = cv2.resize(
                ego_mask, (frame.shape[1], frame.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )

            # Create a boolean mask where the path is detected
            mask_bool = ego_mask_resized > 0
            
            # Blend: 70% original, 30% green tint on the detected path
            combined = frame.copy()
            # This one-liner is faster than cv2.addWeighted for simple color overlays
            combined[mask_bool] = (frame[mask_bool] * 0.7 + np.array([0, 255, 0]) * 0.3).astype(np.uint8)
        else:
            combined = frame # Fallback if model logic fails
    
        orig_h, orig_w = frame.shape[:2]

        # OpenCV (BGR) -> PIL (RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Detector exposes detect(), not predict()
        res = self.detector.detect(pil_img)

        combined = frame.copy()

        # Segmentation: res is PIL.Image mask (0/255)
        if isinstance(res, Image.Image):
            ego_mask = np.array(res)  # (H,W), values 0/255

            if ego_mask.shape != (orig_h, orig_w):
                ego_mask_resized = cv2.resize(
                    ego_mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                ego_mask_resized = ego_mask.astype(np.uint8)

            overlay = frame.copy()
            overlay[ego_mask_resized > 0] = [0, 255, 0]
            combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Classification/Regression: res is rails points list
        else:
            try:
                left, right = res[0], res[1]
                
                # FILL THE AREA between rails instead of just drawing lines
                if len(left) >= 2 and len(right) >= 2:
                    # Combine left rail + reversed right rail to make a closed polygon
                    left_pts = np.array(left, dtype=np.int32)
                    right_pts = np.array(right, dtype=np.int32)
                    
                    # Create closed polygon: left -> right (reversed) -> back to start
                    polygon = np.vstack([left_pts, right_pts[::-1]])
                    
                    # Create overlay with filled polygon
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [polygon], (0, 255, 0))
                    combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                    
            except Exception as e:
                print(f"Error drawing ego-path: {e}")
                pass

        # Logo code stays the same...
        if self.logo is not None:
            l_h, l_w = self.logo.shape[:2]
            margin = 15
            y1, y2 = margin, margin + l_h
            x1, x2 = orig_w - l_w - margin, orig_w - margin

            if y2 <= orig_h and x2 <= orig_w:
                roi = combined[y1:y2, x1:x2]
                if self.logo.shape[2] == 4:
                    alpha = self.logo[:, :, 3] / 255.0
                    for c in range(3):
                        roi[:, :, c] = (alpha * self.logo[:, :, c] +
                                        (1 - alpha) * roi[:, :, c])
                else:
                    combined[y1:y2, x1:x2] = self.logo[:, :, :3]

        return combined


def run_demo():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
        return
    
    # Initialize demo with ResNet-18 for speed
    # Change to 'brilliant-horse-15' or'twinkling-rocket-21' for best accuracy
    demo = TEPNetRailDemo(model_name="twinkling-rocket-21")

    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    
    print("Running TEP-Net Demo... Press 'q' to quit.")
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = demo.process_frame(frame)
            
            cv2.imshow("Axis v0 - TEP-Net Ego-Path Detection", result)
            out.write(result)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"\nDemo complete! Processed {frame_count} frames.")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_demo()