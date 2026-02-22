import os
import time
import threading
from collections import deque

import cv2
import numpy as np

# NO eventlet — threading mode works on Python 3.13 with no monkey_patch issues
from flask import Flask, Response, jsonify, send_from_directory
from flask_socketio import SocketIO

import sys
import torch
from PIL import Image
from ultralytics import YOLO

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
YOLO_MODEL_PATH = "models/yolo26n_rail_final.onnx"
TEPNET_MODEL_PATH = "models/twinkling-rocket-21"

CONF_THRESHOLD = 0.15
OVERLAP_THRESHOLD = 100
H_FOV_DEG = 63
COW_HEIGHT_M = 1.5

COLORS = {
    'person': (0, 0, 255), 'car': (0, 0, 255), 'bus': (0, 0, 255),
    'truck': (0, 0, 255), 'cow': (0, 0, 255), 'horse': (0, 0, 255),
    'deer': (0, 0, 255), 'dog': (0, 0, 255), 'bear': (0, 0, 255),
}

# -----------------------------
# Flask + SocketIO in THREADING mode (no eventlet, no monkey_patch)
# -----------------------------
app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

latest_jpeg = None
latest_state = None
lock = threading.Lock()
event_ring = deque(maxlen=80)


# -----------------------------
# HAZARD PIPELINE
# -----------------------------
class HazardPipeline:
    def __init__(self, logo_path="logo.png"):
        print("Initializing Hazard Pipeline...")

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"Using device: {device}")

        self.yolo = YOLO(YOLO_MODEL_PATH)
        self.tepnet = Detector(
            model_path=TEPNET_MODEL_PATH,
            crop_coords=None,
            runtime="pytorch",
            device=device
        )
        ICON_DIR = os.path.join(CURRENT_DIR, "static", "assets", "icons")
        self.icon_mgr = IconManager(icons_path=ICON_DIR)

        self.ego_mask_cache = None
        self.frame_count = 0

        self.logo = None
        if os.path.exists(logo_path):
            self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if self.logo is not None:
                self.logo = cv2.resize(self.logo, (80, 120))

        print("Models + UI assets loaded.\n")

    def estimate_distance(self, pixel_height, image_width):
        if pixel_height <= 0:
            return None
        focal_length_px = (image_width / 2) / np.tan(np.radians(H_FOV_DEG / 2))
        return (focal_length_px * COW_HEIGHT_M) / pixel_height

    def process_frame(self, frame):
        orig_h, orig_w = frame.shape[:2]
        combined = frame.copy()
        hazard_events = []
        self.frame_count += 1

        # TEPNet every 10 frames
        if self.frame_count % 10 == 0 or self.ego_mask_cache is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (512, 512))
            pil_img = Image.fromarray(rgb_small)
            seg_result = self.tepnet.detect(pil_img)
            if isinstance(seg_result, Image.Image):
                ego_mask = np.array(seg_result)
                self.ego_mask_cache = cv2.resize(ego_mask, (orig_w, orig_h),
                                                interpolation=cv2.INTER_NEAREST)

        ego_mask_resized = self.ego_mask_cache

        if ego_mask_resized is not None:
            combined = frame.copy()
            combined[ego_mask_resized > 0] = (
                combined[ego_mask_resized > 0] * 0.7 +
                np.array([0, 255, 0]) * 0.3
            ).astype(np.uint8)

            # DELETE these 4 lines:
            # overlay = frame.copy()
            # overlay[ego_mask_resized > 0] = [0, 255, 0]
            # combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            combined = frame.copy()
            combined[ego_mask_resized > 0] = (
                combined[ego_mask_resized > 0] * 0.7 + 
                np.array([0, 255, 0]) * 0.3
            ).astype(np.uint8)

        results = self.yolo(frame, imgsz=1536, conf=CONF_THRESHOLD, verbose=False)

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

                hazard = False
                if ego_mask_resized is not None:
                    obj_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    obj_mask[int(y1):int(y2), int(x1):int(x2)] = 1
                    intersection = (ego_mask_resized > 0) & (obj_mask > 0)
                    overlap_pixels = int(np.sum(intersection))
                    if overlap_pixels > OVERLAP_THRESHOLD:
                        hazard = True

                color = COLORS.get(class_name, (0, 0, 255))

                if hazard:
                    self.icon_mgr.draw_detection_with_icon(
                        combined, [x1, y1, x2, y2], class_name, conf, color
                    )
                    hazard_events.append({
                        "timestamp": time.time(),
                        "class": class_name,
                        "confidence": conf,
                        "distance_m": distance_m,
                        "bbox": {"x1": float(x1), "y1": float(y1),
                                 "x2": float(x2), "y2": float(y2)}
                    })
                else:
                    cv2.rectangle(combined,
                                  (int(x1), int(y1)), (int(x2), int(y2)),
                                  (255, 255, 0), 1)

        # if self.logo is not None:
        #     l_h, l_w = self.logo.shape[:2]
        #     margin = 15
        #     y1_l, y2_l = margin, margin + l_h
        #     x1_l, x2_l = orig_w - l_w - margin, orig_w - margin
        #     if y2_l <= orig_h and x2_l <= orig_w:
        #         roi = combined[y1_l:y2_l, x1_l:x2_l]
        #         if self.logo.shape[2] == 4:
        #             alpha = self.logo[:, :, 3] / 255.0
        #             for c in range(3):
        #                 roi[:, :, c] = (alpha * self.logo[:, :, c]
        #                                 + (1 - alpha) * roi[:, :, c])
        #         else:
        #             combined[y1_l:y2_l, x1_l:x2_l] = self.logo[:, :, :3]

        return combined, hazard_events


# -----------------------------
# UI STATE BUILDER
# -----------------------------
def to_ui_state(hazard_events, frame_w, frame_h, fps_est):
    hazards = []
    events = []

    for idx, ev in enumerate(
        sorted(hazard_events, key=lambda e: e["timestamp"], reverse=True)
    ):
        x1, y1 = ev["bbox"]["x1"], ev["bbox"]["y1"]
        x2, y2 = ev["bbox"]["x2"], ev["bbox"]["y2"]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        dist = ev["distance_m"]
        dist_int = int(dist) if dist is not None else None

        hazards.append({
            "id": f"hz-{int(ev['timestamp']*1000)}-{idx}",
            "type": ev["class"].title(),
            "distanceM": dist_int if dist_int is not None else 0,
            "bbox": {
                "x": float(x1 / frame_w), "y": float(y1 / frame_h),
                "w": float(w / frame_w),  "h": float(h / frame_h),
            },
            "confidence": float(ev["confidence"]),
            "inCorridor": True,
        })

        ts = time.strftime("%H:%M:%S", time.localtime(ev["timestamp"]))
        events.append({
            "id": f"ev-{int(ev['timestamp']*1000)}",
            "timestamp": ts,
            "type": "DETECTION",
            "message": f"{ev['class']} in corridor — {dist_int if dist_int is not None else '?'} m",
            "severity": "CRITICAL",
        })

    for e in events:
        event_ring.appendleft(e)

    now = time.time()
    ui = {
        "timestamp": (time.strftime("%H:%M:%S", time.localtime(now))
                      + f".{int((now % 1) * 1000):03d}"),
        "connectivity": {"wifi": True, "cellular": False, "gps": True},
        "speedKmh": 0,
        "impactTimeSec": 0,
        "hazards": hazards,
        "systemStatus": {
            "camera": "OK", "detection": "OK", "segmentation": "OK",
            "gps": "Locked", "fps": int(fps_est), "cpuTempC": 0, "recording": True,
        },
        "events": list(event_ring),
    }

    if not ui["events"]:
        ui["events"] = [{
            "id": "ev-boot",
            "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
            "type": "SYSTEM",
            "message": "System initialized — waiting for detections",
            "severity": "INFO",
        }]

    return ui


# -----------------------------
# INFERENCE THREAD
# -----------------------------
def inference_loop():
    global latest_jpeg, latest_state

    pipeline = HazardPipeline()

    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    ))
    picam2.start()
    time.sleep(1)

    last_t = time.time()
    fps_est = 0.0

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        annotated, hazard_events = pipeline.process_frame(frame)

        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            print(f"FPS: {fps_est:.1f} | Frame time: {dt*1000:.1f}ms")
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt) if fps_est else (1.0 / dt)

        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        h, w = frame.shape[:2]
        ui_state = to_ui_state(hazard_events, w, h, fps_est)

        with lock:
            latest_jpeg = buf.tobytes()
            latest_state = ui_state

        socketio.emit("state", ui_state)


# -----------------------------
# SOCKET.IO EVENTS
# -----------------------------
@socketio.on("connect")
def on_connect():
    """Send an immediate state snapshot when a client first connects."""
    with lock:
        state = latest_state
    if state:
        socketio.emit("state", state)


# -----------------------------
# HTTP ROUTES
# -----------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/state")
def api_state():
    with lock:
        if latest_state is None:
            return jsonify({"ok": False, "message": "warming up"})
        return jsonify({"ok": True, "data": latest_state})


def mjpeg_generator():
    boundary = b"--frame"
    while True:
        with lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.02)
            continue
        yield (boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
               + frame + b"\r\n")


@app.route("/stream/primary")
def stream_primary():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    # Start inference in a plain daemon thread — no eventlet needed
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    # socketio.run() replaces app.run() — this is what makes /socket.io respond
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False)