import os
import sys
import cv2
from ultralytics import YOLO
import time
from collections import deque

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.icon_manager import IconManager

VIDEO_PATH = "./data/videos/raw/cow.mp4"
OUTPUT_PATH = "./data/videos/detection_output.mp4"

# All red for critical alerts
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

RAIL_CLASSES = [0, 1, 2]  # cow, horse, deer only


class ObjectDetectionDemo:
    def __init__(self, logo_path="logo.png"):
        print("🤖 Initializing Object Detection System...")
        
        print("   Loading YOLOv8m...")
        self.yolo = YOLO('models/yolo26n_rail_final.onnx')

        
        print("   Loading detection icons...")
        self.icon_mgr = IconManager()
        
        self.logo = None
        if os.path.exists(logo_path):
            self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if self.logo is not None:
                self.logo = cv2.resize(self.logo, (80, 120))
        
        print("✅ System ready!\n")
    
    def process_frame(self, frame):
        """Detect objects with tracking for stable detections"""
        orig_h, orig_w = frame.shape[:2]
        combined = frame.copy()
        
        
        results = self.yolo(
            frame,
            imgsz=1536,
            conf=0.15,
            verbose=False
        )

        # results = self.yolo.track(
        #     frame,
        #     imgsz=1536,
        #     conf=0.15,
        #     persist=True,
        #     verbose=False
        # )


        # Draw each detection
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.yolo.names[cls]
                    height = y2 - y1
                    width = x2 - x1
                    # --- Distance Estimation ---
                    IMAGE_WIDTH = orig_w         # use actual frame width
                    H_FOV_DEG = 70               # replace with your real camera FOV
                    COW_HEIGHT_M = 1.5           # approx real cow height

                    import math

                    focal_length_px = (IMAGE_WIDTH / 2) / math.tan(math.radians(H_FOV_DEG / 2))
                    distance_m = (focal_length_px * COW_HEIGHT_M) / height

                    print(
                        f"{class_name} | conf: {conf:.2f} | "
                        f"pixel_h: {height:.1f} | "
                        f"est_distance: {distance_m:.1f} m"
                    )


                    # print(f"Detected: {class_name} ({conf:.2f})")
                    
                    # Get color
                    color = COLORS.get(class_name, (0, 0, 255))
                    
                    # Draw with icon
                    self.icon_mgr.draw_detection_with_icon(
                        combined,
                        [x1, y1, x2, y2],
                        class_name,
                        conf,
                        color
                    )
        
        # Add logo (same as before)
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
    
    demo = ObjectDetectionDemo()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    
    print("Running Object Detection... Press 'q' to quit.")
    frame_count = 0
    
    fps_buffer = deque(maxlen=30)

    try:
        while cap.isOpened():
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            infer_start = time.time()
            result = demo.process_frame(frame)
            infer_end = time.time()

            cv2.imshow("Axis v0 - Object Detection", result)
            out.write(result)

            loop_end = time.time()

            # Calculate FPS
            inference_time = infer_end - infer_start
            loop_time = loop_end - loop_start

            inference_fps = 1 / inference_time if inference_time > 0 else 0
            full_fps = 1 / loop_time if loop_time > 0 else 0

            fps_buffer.append(full_fps)
            avg_fps = sum(fps_buffer) / len(fps_buffer)

            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"\n✅ Complete! Processed {frame_count} frames.")
    print(f"   Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()