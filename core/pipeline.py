import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from core.config import * 


SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.interface import Detector
from utils.icon_manager import IconManager


class HazardPipeline:
    def __init__(self):
        print("Initializing HazardPipeline...")

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.use_half = self.device != "cpu"
        print(f"  Device: {self.device}")

        if USE_HAILO:
            from hailo_platform import (
                HEF, VDevice, InputVStreamParams, OutputVStreamParams,
                InferVStreams, FormatType,
            )
            self._hef        = HEF(HEF_MODEL_PATH)
            self._vdevice    = VDevice()
            self._ng         = self._vdevice.configure(self._hef)[0]
            self._ng_params  = self._ng.create_params()
            self._in_params  = InputVStreamParams.make(self._ng, format_type=FormatType.UINT8)
            self._out_params = OutputVStreamParams.make(self._ng, format_type=FormatType.FLOAT32)
            self._ng_ctx     = self._ng.activate(self._ng_params).__enter__()
            self._infer_ctx  = InferVStreams(self._ng, self._in_params, self._out_params).__enter__()
            self._input_name = self._hef.get_input_vstream_infos()[0].name
        else:
            self.yolo        = YOLO(YOLO_MODEL_PATH).to(self.device)
            self.yolo_person = YOLO(COCO_MODEL_PATH).to(self.device)
        self.tepnet      = Detector(TEPNET_MODEL_PATH, None, "pytorch", self.device)
        self.icon_mgr    = IconManager(icons_path=ICON_DIR)

        self.ego_mask_cache = None
        self.frame_count    = 0

        print("  HazardPipeline ready.\n")

    # ── Distance estimate ─────────────────────────────────────────────────
    def estimate_distance(self, pixel_height, image_width):
        if pixel_height <= 0:
            return None
        focal_px = (image_width / 2) / np.tan(np.radians(H_FOV_DEG / 2))
        return (focal_px * PERSON_HEIGHT_M) / pixel_height

    # ── Main frame processor ──────────────────────────────────────────────
    def process_frame(self, frame):
        orig_h, orig_w = frame.shape[:2]
        self.frame_count += 1
        hazard_events = []

        # TEPNet — every N frames
        if self.frame_count % TEPNET_INTERVAL == 0 or self.ego_mask_cache is None:
            rgb_small  = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    (TEPNET_SIZE, TEPNET_SIZE))
            seg_result = self.tepnet.detect(Image.fromarray(rgb_small))
            if isinstance(seg_result, Image.Image):
                self.ego_mask_cache = cv2.resize(
                    np.array(seg_result), (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                )

        combined = frame.copy()
        if self.ego_mask_cache is not None:
            mask = self.ego_mask_cache > 0
            combined[mask] = (combined[mask] * 0.7 + np.array([0, 255, 0]) * 0.3).astype(np.uint8)

        # ── Inference ─────────────────────────────────────────────────────
        if USE_HAILO:
            resized    = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
            raw_output = self._infer_ctx.infer({self._input_name: np.expand_dims(resized, 0)})
            boxes_xywh, scores, cls_ids = decode_hailo_output(raw_output, orig_w, orig_h)
            detections = list(zip(boxes_xywh, scores, cls_ids))
        else:
            results_rail   = self.yolo(frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD,
                                       verbose=False, half=self.use_half)
            results_person = self.yolo_person(frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD,
                                              verbose=False, classes=[0], half=self.use_half)
            detections = []
            for result in list(results_rail) + list(results_person):
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), int(box.cls[0])))

        # ── Per-detection logic ────────────────────────────────────────────
        for bbox, conf, cls in detections:
            x1, y1      = int(bbox[0]), int(bbox[1])
            x2, y2      = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            class_name  = CLASS_NAMES.get(cls, 'person')

            if class_name == 'crossing':
                is_hazard = True
            elif self.ego_mask_cache is not None:
                cx1 = max(0, x1); cy1 = max(0, y1)
                cx2 = min(orig_w, x2); cy2 = min(orig_h, y2)
                overlap   = int(np.count_nonzero(self.ego_mask_cache[cy1:cy2, cx1:cx2]))
                is_hazard = overlap > 0
            else:
                is_hazard = False

            if is_hazard:
                color = COLORS.get(class_name, (0, 0, 255))
                self.icon_mgr.draw_detection_with_icon(
                    combined, [x1, y1, x2, y2], class_name, conf, color
                )
                hazard_events.append({
                    "timestamp":  time.time(),
                    "class":      class_name,
                    "confidence": conf,
                    "distance_m": self.estimate_distance(y2 - y1, orig_w),
                    "bbox": {"x1": float(x1), "y1": float(y1),
                             "x2": float(x2), "y2": float(y2)},
                })

        return combined, hazard_events