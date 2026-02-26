import serial
import pynmea2
import threading

class GPSReader:
    def __init__(self, port="/dev/ttyAMA0", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.data = {
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "speed_kmh": 0,
            "fix": 0,
            "satellites": 0
        }
        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                while self.running:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                    if not line: continue
                    try:
                        msg = pynmea2.parse(line)
                        
                        if isinstance(msg, pynmea2.types.talker.GGA):
                            self.data["lat"] = msg.latitude
                            self.data["lon"] = msg.longitude
                            self.data["alt"] = msg.altitude
                            self.data["fix"] = msg.gps_qual
                            self.data["satellites"] = int(msg.num_sats)
                        
                        elif isinstance(msg, pynmea2.types.talker.RMC):
                            # msg.speed is in knots, convert to km/h
                            self.data["speed_kmh"] = int(msg.spd_over_grnd * 1.852) if msg.spd_over_grnd else 0
                            
                    except pynmea2.ParseError:
                        continue
        except Exception as e:
            print(f"GPS Serial Error: {e}")

# Usage in your main script:
# gps = GPSReader()
# ... inside to_ui_state ...
# "speedKmh": gps.data["speed_kmh"]