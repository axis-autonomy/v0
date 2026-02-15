# web_integration.py
import os
import sys
import cv2
import numpy as np
import json
from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading
import time
from ultralytics import YOLO
import torch
from PIL import Image

# Add src to path for TEPNet imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from utils.interface import Detector
    print("✅ TEPNet Detector successfully imported!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
camera_active = False
video_capture = None
current_detections = []
track_coverage = 100.0
current_speed = 0
ego_path_mask = None  # Store the latest track segmentation mask

# Load models
print("🔧 Loading models...")

# 1. Load YOLO for object detection
print("Loading YOLO model...")
yolo_model = YOLO('models/twinkling-rocket-21/best.pt')  # Your custom model
print("✅ YOLO loaded")

# 2. Load TEPNet for track segmentation
print("Loading TEPNet model...")
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_PATH = os.path.join(CURRENT_DIR, "models", "twinkling-rocket-21")
tepnet_detector = Detector(
    model_path=MODEL_PATH,
    crop_coords=None,
    runtime='pytorch',
    device=device
)
print("✅ TEPNet loaded")

# Distance estimation
def estimate_distance(bbox_height, image_height):
    """
    Estimate distance based on bounding box size
    Calibrate these values based on your camera setup
    """
    if bbox_height == 0:
        return 200
    
    # Improved estimation (tune these based on your camera height/angle)
    # Assuming camera mounted ~3m high, looking ~30° down
    reference_height = image_height * 0.4  # Object at 50m takes 40% of frame
    distance = (reference_height / bbox_height) * 50
    return max(5, min(distance, 200))  # Clamp between 5-200m

# Process frame with YOLO
def detect_objects(frame):
    """Run YOLO object detection"""
    results = yolo_model(frame, conf=0.5, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = yolo_model.names[cls]
            
            width = x2 - x1
            height = y2 - y1
            
            # Estimate distance
            distance = estimate_distance(height, frame.shape[0])
            
            detection = {
                'class': class_name,
                'confidence': conf,
                'bbox': {
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(width),
                    'height': int(height)
                },
                'distance': float(distance)
            }
            detections.append(detection)
    
    return detections

# Process track segmentation with TEPNet
def segment_tracks(frame):
    """
    Run TEPNet track segmentation
    Returns: track_coverage percentage (0-100)
    """
    global ego_path_mask
    
    try:
        # Convert BGR to RGB for TEPNet
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Run TEPNet detection
        res = tepnet_detector.detect(pil_img)
        
        # Handle segmentation mask result
        if isinstance(res, Image.Image):
            ego_mask = np.array(res)  # (H,W), values 0/255
            
            # Resize mask to match frame size if needed
            if ego_mask.shape != (frame.shape[0], frame.shape[1]):
                ego_path_mask = cv2.resize(
                    ego_mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                ego_path_mask = ego_mask.astype(np.uint8)
            
            # Calculate track coverage (percentage of frame that is track)
            total_pixels = ego_path_mask.size
            track_pixels = np.sum(ego_path_mask > 0)
            coverage = (track_pixels / total_pixels) * 100
            
            return float(coverage)
        
        # Handle regression result (rail points)
        else:
            try:
                left, right = res[0], res[1]
                if len(left) >= 2 and len(right) >= 2:
                    # Create mask from polygon
                    left_pts = np.array(left, dtype=np.int32)
                    right_pts = np.array(right, dtype=np.int32)
                    polygon = np.vstack([left_pts, right_pts[::-1]])
                    
                    ego_path_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    cv2.fillPoly(ego_path_mask, [polygon], 255)
                    
                    # Calculate coverage
                    total_pixels = ego_path_mask.size
                    track_pixels = np.sum(ego_path_mask > 0)
                    coverage = (track_pixels / total_pixels) * 100
                    
                    return float(coverage)
            except Exception as e:
                print(f"Error processing rail points: {e}")
                return 100.0
        
    except Exception as e:
        print(f"Track segmentation error: {e}")
        ego_path_mask = None
        return 100.0
    
    return 100.0

# Video streaming generator
def generate_frames():
    global current_detections, track_coverage, video_capture, ego_path_mask
    
    while camera_active and video_capture is not None:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Run object detection
        current_detections = detect_objects(frame)
        
        # Run track segmentation
        track_coverage = segment_tracks(frame)
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # 1. Draw track segmentation overlay (green tint)
        if ego_path_mask is not None:
            overlay = vis_frame.copy()
            overlay[ego_path_mask > 0] = [0, 255, 0]  # Green
            vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
        
        # 2. Draw object detection boxes
        for det in current_detections:
            bbox = det['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Red box for detected objects
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Label with class, confidence, and distance
            label = f"{det['class']} {det['confidence']:.2f}"
            dist_label = f"{det['distance']:.0f}m"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x, y-25), (x+tw+10, y), (0, 0, 255), -1)
            cv2.putText(vis_frame, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Distance on the box
            cv2.putText(vis_frame, dist_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 3. Add status overlay
        status_text = f"Objects: {len(current_detections)} | Track: {track_coverage:.1f}% | Speed: {current_speed} km/h"
        cv2.rectangle(vis_frame, (10, 10), (600, 50), (0, 0, 0), -1)
        cv2.putText(vis_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

# Routes
@app.route('/video_feed')
def video_feed():
    """Stream video with overlays"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Get current detection data as JSON"""
    return jsonify({
        'objects': current_detections,
        'track_coverage': track_coverage,
        'speed': current_speed,
        'timestamp': time.time()
    })

@app.route('/start_camera')
def start_camera():
    """Start camera feed"""
    global camera_active, video_capture
    
    if not camera_active:
        # Try webcam first, fallback to test video
        video_capture = cv2.VideoCapture(0)  # 0 = webcam
        
        # If webcam fails, try the test video
        if not video_capture.isOpened():
            print("Webcam not available, using test video...")
            video_capture = cv2.VideoCapture("./data/videos/raw/cow.mp4")
        
        if video_capture.isOpened():
            camera_active = True
            return jsonify({'status': 'success', 'message': 'Camera started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to open camera or video'}), 500
    
    return jsonify({'status': 'info', 'message': 'Camera already running'})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera feed"""
    global camera_active, video_capture, ego_path_mask
    
    if camera_active:
        camera_active = False
        if video_capture:
            video_capture.release()
            video_capture = None
        ego_path_mask = None
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    
    return jsonify({'status': 'info', 'message': 'Camera not running'})

@app.route('/speed/<int:speed_kmh>')
def set_speed(speed_kmh):
    """Update current speed"""
    global current_speed
    current_speed = speed_kmh
    return jsonify({'status': 'success', 'speed': speed_kmh})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'camera_active': camera_active,
        'models_loaded': True,
        'device': device
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚂 Railway Safety Monitor Backend")
    print("=" * 60)
    print(f"✅ YOLO Object Detection: Loaded")
    print(f"✅ TEPNet Track Segmentation: Loaded")
    print(f"🖥️  Device: {device}")
    print(f"🌐 Server: http://localhost:5000")
    print(f"📹 Video Feed: http://localhost:5000/video_feed")
    print(f"📊 Detections API: http://localhost:5000/detections")
    print("=" * 60)
    print("\n🎯 Next steps:")
    print("1. Open locomotive-display.html in your browser")
    print("2. Click 'Start Camera' button")
    print("3. Watch real-time detection and segmentation!")
    print("\n💡 Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)