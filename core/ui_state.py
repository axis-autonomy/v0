import time
from collections import deque

event_ring = deque(maxlen=80)


def to_ui_state(hazard_events, frame_w, frame_h, fps_est, gps_data=None):
    gps_data = gps_data or {}
    hazards = []
    events  = []
    now     = time.time()

    has_fix = gps_data.get("fix", 0) > 0

    for idx, ev in enumerate(
        sorted(hazard_events, key=lambda e: e["timestamp"], reverse=True)
    ):
        x1, y1   = ev["bbox"]["x1"], ev["bbox"]["y1"]
        x2, y2   = ev["bbox"]["x2"], ev["bbox"]["y2"]
        w        = max(1.0, x2 - x1)
        h        = max(1.0, y2 - y1)
        dist     = ev["distance_m"]
        dist_int = int(dist) if dist is not None else None

        hazards.append({
            "id":        f"hz-{int(ev['timestamp']*1000)}-{idx}",
            "type":      ev["class"].title(),
            "distanceM": dist_int if dist_int is not None else 0,
            "bbox": {
                "x": float(x1 / frame_w), "y": float(y1 / frame_h),
                "w": float(w  / frame_w), "h": float(h  / frame_h),
            },
            "confidence": float(ev["confidence"]),
            "inCorridor": True,
        })

        ts = time.strftime("%H:%M:%S", time.localtime(ev["timestamp"]))
        events.append({
            "id":        f"ev-{int(ev['timestamp']*1000)}",
            "timestamp": ts,
            "type":      "DETECTION",
            "message":   f"{ev['class']} in corridor — {dist_int if dist_int is not None else '?'} m",
            "severity":  "CRITICAL",
        })

    for e in events:
        event_ring.appendleft(e)

    # All clear — override events directly, don't pollute history
    if not hazard_events:
        ui_events = [{
            "id":        f"ev-clear-{int(now*1000)}",
            "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
            "type":      "CLEAR",
            "message":   "All clear — corridor empty",
            "severity":  "CLEAR",
        }]
    else:
        ui_events = list(event_ring)

    return {
        "timestamp": (time.strftime("%H:%M:%S", time.localtime(now))
                      + f".{int((now % 1) * 1000):03d}"),
        "connectivity": {
            "wifi":     True,
            "cellular": False,
            "gps":      has_fix,
        },
        "speedKmh":     gps_data.get("speed_kmh", 0),
        "impactTimeSec": 0,
        "hazards":      hazards,
        "gps": {
            "lat": gps_data.get("lat", 0.0),
            "lon": gps_data.get("lon", 0.0),
        } if has_fix else None,
        "systemStatus": {
            "camera":      "OK",
            "detection":   "OK",
            "segmentation":"OK",
            "gps":         "Locked" if has_fix else "Searching",
            "fps":         int(fps_est),
            "cpuTempC":    0,
            "recording":   True,
        },
        "events": ui_events,
    }