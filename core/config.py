import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH   = os.path.join(PROJECT_ROOT, "models/best.pt")
COCO_MODEL_PATH   = os.path.join(PROJECT_ROOT, "yolo26n.pt")
TEPNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/twinkling-rocket-21")
ICON_DIR          = os.path.join(PROJECT_ROOT, "static/assets/icons")
USE_PICAMERA = False 
# ---------------------------------------------------------------------------
# Video source — swap between webcam (0) and file path
# ---------------------------------------------------------------------------
VIDEO_SOURCE = "/Users/jeffreyiyamah/Downloads/generated-video-1 (2).mp4"   # 0 = webcam | "/path/to/file.mp4" for file

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
CONF_THRESHOLD    = 0.25
OVERLAP_THRESHOLD = 100
INFER_SIZE        = 1280
TEPNET_SIZE       = 512
TEPNET_INTERVAL   = 10    # run TEPNet every N frames

H_FOV_DEG       = 63
PERSON_HEIGHT_M  = 1.7

CLASS_NAMES = {0: 'person', 1: 'vehicle', 2: 'railcar', 3: 'crossing'}
COLORS = {
    'person':   (0, 0, 255),    # red
    'vehicle':  (0, 0, 255),    # red
    'railcar':  (0, 0, 255),    # red
    'crossing': (0, 165, 255),  # orange
}