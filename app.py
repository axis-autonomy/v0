"""
Axis Robotics — Rail Hazard Detection
Run: python app.py
"""

import os
import time
import threading

import cv2
from flask import Flask, Response, jsonify, send_from_directory
from flask_socketio import SocketIO

from core.config import VIDEO_SOURCE, PROJECT_ROOT, USE_PICAMERA
from core.pipeline import HazardPipeline
from core.ui_state import to_ui_state

from src.sensors.gps import GPSReader

gps = GPSReader(port="/dev/ttyAMA0")

app      = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

latest_jpeg  = None
latest_state = None
lock         = threading.Lock()


def inference_loop():
    global latest_jpeg, latest_state

    pipeline = HazardPipeline()

    if USE_PICAMERA:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        ))
        picam2.start()
        time.sleep(1.0)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    last_t  = time.time()
    fps_est = 0.0

    while True:
        if USE_PICAMERA:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                if isinstance(VIDEO_SOURCE, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        annotated, hazard_events = pipeline.process_frame(frame)

        now    = time.time()
        dt     = now - last_t
        last_t = now
        if dt > 0:
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt) if fps_est else (1.0 / dt)
            # print(f"FPS: {fps_est:.1f} | frame: {dt*1000:.1f}ms")

        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        h, w     = frame.shape[:2]
        ui_state = to_ui_state(hazard_events, w, h, fps_est, gps.data)

        with lock:
            latest_jpeg  = buf.tobytes()
            latest_state = ui_state

        socketio.emit("state", ui_state)

# ---------------------------------------------------------------------------
# SOCKET.IO
# ---------------------------------------------------------------------------
@socketio.on("connect")
def on_connect():
    with lock:
        state = latest_state
    if state:
        socketio.emit("state", state)


# ---------------------------------------------------------------------------
# HTTP ROUTES
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    socketio.run(app, host="0.0.0.0", port=8000, debug=False, use_reloader=False)