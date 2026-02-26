const ICONS = {
  cow:    "/static/assets/icons/cow.png",
  person: "/static/assets/icons/person.png",
  deer:   "/static/assets/icons/deer.png",
  car:    "/static/assets/icons/car.png",
  truck:  "/static/assets/icons/truck.png",
  hazard: "/static/assets/icons/hazard.png",
};

function iconForType(type) {
  const key = String(type || "").toLowerCase().trim();
  return ICONS[key] || ICONS.hazard;
}

const { useState, useEffect, useRef, useMemo, useCallback } = React;

/* ── Fallback state shown before first WebSocket message ─────────────────── */
const INITIAL_STATE = {
  timestamp: "--:--:--.---",
  connectivity: { wifi: false, cellular: false, gps: false },
  speedKmh: 0,
  impactTimeSec: 0,
  hazards: [],
  systemStatus: {
    camera: "OK", detection: "OK", segmentation: "OK",
    gps: "Searching", fps: 0, cpuTempC: 0, recording: false,
  },
  events: [],
};

/* ── Severity / type maps ─────────────────────────────────────────────────── */
const TYPE_COLOR = {
  CRITICAL: "tc-critical", WARNING: "tc-warning", INFO: "tc-info",
  SYSTEM: "tc-system", CLEAR: "tc-clear", CONFIRMATION: "tc-info", DETECTION: "tc-critical",
};
const SEV_BADGE = {
  CRITICAL: "bc-critical", WARNING: "bc-warning", INFO: "bc-info",
  SYSTEM: "bc-system", CLEAR: "bc-clear",
};

/* ════════════════════════════════════════════════════════════
   ALERT MODE — derived from corridor hazards
════════════════════════════════════════════════════════════ */
function deriveMode(hazards, speedKmh) {
  const inCorridor = hazards.filter(h => h.inCorridor);
  if (!inCorridor.length) return "SAFE";
  const closestM = Math.min(...inCorridor.map(h => h.distanceM));
  return (closestM < 500 && speedKmh >= 0) ? "ALERT" : "SAFE";
}

/* ════════════════════════════════════════════════════════════
   HAZARD ICON
════════════════════════════════════════════════════════════ */
function HazardTypeIcon({ type, size = 28, style }) {
  const src = iconForType(type);
  return (
    <img
      src={src}
      alt={type || "hazard"}
      width={size}
      height={size}
      style={{
        display: "block",
        objectFit: "contain",
        filter: "drop-shadow(0 2px 8px rgba(0,0,0,0.65))",
        ...style
      }}
    />
  );
}

/* ════════════════════════════════════════════════════════════
   TRACK CORRIDOR OVERLAY (SVG polygon on top of video)
════════════════════════════════════════════════════════════ */
function TrackCorridorOverlay({ mode }) {
  return null;
}

/* ════════════════════════════════════════════════════════════
   DETECTION OVERLAY — uses normalised bbox coords from server
════════════════════════════════════════════════════════════ */
function DetectionOverlay({ hazards, compact = false }) {
  return (
    <div className="overlay-layer" style={{ position:"absolute", inset:0, pointerEvents:"none" }}>
      {hazards.map((h, idx) => (
        <div key={h.id} style={{
          position: "absolute",
          left:   `${h.bbox.x * 100}%`,
          top:    `${h.bbox.y * 100}%`,
          width:  `${h.bbox.w * 100}%`,
          height: `${h.bbox.h * 100}%`,
        }}>
          {!compact && idx === 0 && (
            <div className="det-marker-wrap">
              <div style={{ display:"flex", alignItems:"flex-end" }}>
                <HazardMarkerIcon size={32}/>
                {hazards.length > 1 && <HazardMarkerIcon size={32} style={{ marginLeft:"-10px" }}/>}
              </div>
            </div>
          )}
          <div className="bbox-rect"/>
        </div>
      ))}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   TOP STATUS BAR
════════════════════════════════════════════════════════════ */
function TopStatusBar({ timestamp, connectivity, connStatus }) {
  const badgeLabel = connStatus === "live" ? "LIVE" : connStatus === "connecting" ? "CONNECTING" : "OFFLINE";
  return (
    <div className="status-bar">
      <div className="status-logo">
        <img className="brand-logo" src="/static/assets/icons/logo.png" />
        <span className="brand-text">AXIS ROBOTICS</span>
      </div>
      <div className="status-right">
        <span className={`conn-badge ${connStatus}`}>{badgeLabel}</span>
        <span className={`conn-icon ${connectivity.wifi     ? "active":"inactive"}`} title="WiFi">
          <svg width="17" height="13" viewBox="0 0 24 18" fill="none">
            <path d="M12 16a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3zm0-4c1.7 0 3.2.68 4.3 1.8l-1.4 1.4A4.5 4.5 0 0 0 12 14a4.5 4.5 0 0 0-2.9 1.2l-1.4-1.4A6.4 6.4 0 0 1 12 12zm0-4c2.6 0 5 1.05 6.7 2.75l-1.4 1.4A7.5 7.5 0 0 0 12 10a7.5 7.5 0 0 0-5.3 2.15l-1.4-1.4A9.5 9.5 0 0 1 12 8zm0-4c3.55 0 6.8 1.4 9.2 3.65l-1.4 1.4A11.5 11.5 0 0 0 12 6a11.5 11.5 0 0 0-7.8 2.95l-1.4-1.4A13.5 13.5 0 0 1 12 4z" fill="currentColor"/>
          </svg>
        </span>
        <span className={`conn-icon ${connectivity.cellular ? "active":"inactive"}`} title="Cellular">
          <svg width="15" height="13" viewBox="0 0 20 18" fill="none">
            <rect x="1"  y="12" width="3" height="6" rx="1" fill="currentColor"/>
            <rect x="5"  y="8"  width="3" height="10" rx="1" fill="currentColor"/>
            <rect x="9"  y="4"  width="3" height="14" rx="1" fill="currentColor"/>
            <rect x="13" y="0"  width="3" height="18" rx="1" fill="currentColor"/>
          </svg>
        </span>
        <span className={`conn-icon ${connectivity.gps     ? "active":"inactive"}`} title="GPS">
          <svg width="12" height="13" viewBox="0 0 14 18" fill="none">
            <path d="M7 0a6 6 0 0 0-6 6c0 4.5 6 12 6 12s6-7.5 6-12A6 6 0 0 0 7 0zm0 8a2 2 0 1 1 0-4 2 2 0 0 1 0 4z" fill="currentColor"/>
          </svg>
        </span>
        <span className="status-time">{timestamp}</span>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   PRIMARY VIDEO PANEL — MJPEG stream from /stream/primary
════════════════════════════════════════════════════════════ */
function PrimaryVideoPanel({ mode, hazards, streamReady }) {
  return (
    <div className="video-panel primary-video">
      {streamReady ? (
        <img
          src="/stream/primary"
          className="video-element"
          alt="Primary forward-facing camera feed"
          style={{ display:"block", width:"100%", height:"100%", objectFit:"cover" }}
        />
      ) : (
        <div className="video-placeholder">
          <span className="video-label">Primary Feed</span>
          <span className="video-sublabel">Waiting for stream…</span>
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   SECONDARY VIDEO PANEL
════════════════════════════════════════════════════════════ */
function SecondaryVideoPanel({ mode, hazards, streamReady }) {
  return (
    <div className="video-panel secondary-video">
      {streamReady ? (
        <img
          src="/stream/primary"
          className="video-element"
          alt="Secondary / zoom camera feed"
          style={{ display:"block", width:"100%", height:"100%", objectFit:"cover" }}
        />
      ) : (
        <div className="video-placeholder" style={{ minHeight:"60px" }}>
          <span className="video-label">Zoom / Thermal</span>
          <span className="video-sublabel">Waiting for stream…</span>
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   ALERT CARD
════════════════════════════════════════════════════════════ */
function AlertCard({ hazard }) {
  return (
    <div className="alert-card">
      <div className="alert-icon-circle">
        <HazardTypeIcon type={hazard.type} size={26} />
      </div>
      <div className="alert-text">
        <span className="alert-label">Hazard Nearby</span>
        <div className="alert-distance">
          <span className="alert-number">{hazard.distanceM}</span>
          <span className="alert-unit">m</span>
        </div>
        <span className="alert-class">{hazard.type}</span>
      </div>
    </div>
  );
}

function AllClearCard() {
  return (
    <div className="alert-card">
      <div className="alert-icon-circle" style={{ background: "var(--safe)", animation: "none" }}>
        <img src="/static/assets/icons/check.png" width={26} height={26}
             style={{ objectFit:"contain", filter:"drop-shadow(0 2px 8px rgba(0,0,0,0.65))" }}/>
      </div>
      <div className="alert-text">
        <span className="alert-label">Status</span>
        <div className="alert-distance">
          <span className="alert-number" style={{ color:"var(--safe)", fontSize:"28px" }}>All Clear</span>
        </div>
        <span className="alert-class">Corridor Empty</span>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   METRIC TILE + SPEED GAUGE
════════════════════════════════════════════════════════════ */
function MetricTile({ label, value }) {
  return (
    <div className="metric-tile">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

function SpeedGaugeTile({ speedValue, unit, maxSpeed = 200 }) {
  const SIZE = 108, CX = 54, CY = 54, R = 38, SW = 5;
  const START = 215, SWEEP = 110;
  const active = Math.min(speedValue / maxSpeed, 1) * SWEEP;
  const toRad = d => d * Math.PI / 180;
  const arc = (s, sw) => {
    const e = s + sw;
    const sx = CX + R * Math.cos(toRad(s)), sy = CY + R * Math.sin(toRad(s));
    const ex = CX + R * Math.cos(toRad(e)), ey = CY + R * Math.sin(toRad(e));
    return `M ${sx.toFixed(2)} ${sy.toFixed(2)} A ${R} ${R} 0 ${sw>180?1:0} 1 ${ex.toFixed(2)} ${ey.toFixed(2)}`;
  };
  return (
    <div className="gauge-tile">
      <div className="gauge-wrap">
        <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
          <path d={arc(START,SWEEP)} stroke="rgba(255,255,255,0.08)" strokeWidth={SW} fill="none" strokeLinecap="round"/>
          {active > 0 && <path d={arc(START,active)} stroke="var(--gauge-green)" strokeWidth={SW} fill="none" strokeLinecap="round"/>}
        </svg>
        <div className="gauge-center">
          <span className="gauge-value">{speedValue}</span>
          <span className="gauge-unit">{unit}</span>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   SYSTEM STATUS STRIP
════════════════════════════════════════════════════════════ */
function SystemStatusStrip({ status }) {
  const classify = (key, val) => {
    if (["camera","detection","segmentation"].includes(key))
      return val === "OK" ? "ok" : val === "Degraded" ? "warn" : "err";
    if (key === "gps") return val === "Locked" ? "ok" : val === "Searching" ? "warn" : "err";
    if (key === "fps") return val >= 20 ? "neutral" : "warn";
    if (key === "cpuTempC") return val < 75 ? "neutral" : val < 90 ? "warn" : "err";
    if (key === "recording") return val ? "ok" : "off";
    return "neutral";
  };
  const dotClass = s => ({ok:"d-ok",warn:"d-warn",err:"d-err",off:"d-off",neutral:"d-off"}[s]||"d-off");
  const valClass = s => ({ok:"s-ok",warn:"s-warn",err:"s-err",off:"s-neutral",neutral:"s-neutral"}[s]||"s-neutral");

  const items = [
    { key:"camera",       label:"CAM",    val:status.camera },
    { key:"detection",    label:"DETECT", val:status.detection },
    { key:"segmentation", label:"SEG",    val:status.segmentation },
    { key:"gps",          label:"GPS",    val:status.gps },
    { key:"fps",          label:"FPS",    val:`${status.fps} fps` },
    { key:"cpuTempC",     label:"CPU",    val:`${status.cpuTempC}°C` },
    { key:"recording",    label:"REC",    val:status.recording ? "ON" : "OFF", isRec:true },
  ];

  return (
    <div className="sysstat-strip">
      {items.map(it => {
        const state = classify(it.key, status[it.key] ?? it.val);
        return (
          <div key={it.key} className="sysstat-item">
            <span className="ss-key">{it.label}</span>
            <span className={`ss-dot ${dotClass(state)} ${it.isRec && status.recording ? "rec-blink":""}`}/>
            <span className={`ss-val ${valClass(state)}`}>{it.val}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   EVENT LOG ROW
════════════════════════════════════════════════════════════ */
function EventLogRow({ event }) {
  return (
    <div className="log-row">
      <span className="log-ts">{event.timestamp}</span>
      <span className={`log-type ${TYPE_COLOR[event.type] || "tc-system"}`}>{event.type}</span>
      <span className="log-msg">{event.message}</span>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   STATUS & EVENT LOG PANEL — always visible, no dropdown
════════════════════════════════════════════════════════════ */
function StatusEventLogPanel({ events, systemStatus }) {
  const scrollRef = useRef(null);
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [events.length]);

  return (
    <div className="log-panel log-panel-fixed">
      <div className="log-header-fixed">
        <span className="log-section-label">Status / Event Log</span>
      </div>
      <SystemStatusStrip status={systemStatus}/>
      <div className="log-scroll" ref={scrollRef}>
        {events.map(ev => <EventLogRow key={ev.id} event={ev}/>)}
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   MAP CARD — Leaflet, updates marker on new GPS coords
════════════════════════════════════════════════════════════ */
function MapCard({ gps }) {
  const mapRef    = useRef(null);  // DOM node
  const leafletRef = useRef(null); // L.map instance
  const markerRef  = useRef(null); // L.marker instance

  // Init map once
  useEffect(() => {
    if (leafletRef.current) return;
    const defaultPos = gps ? [gps.lat, gps.lon] : [42.4534, -76.4735]; // fallback: Ithaca, NY
    const map = L.map(mapRef.current, {
      center: defaultPos,
      zoom: 15,
      zoomControl: false,
      attributionControl: false,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);

    const icon = L.divIcon({
      className: "",
      html: `<div style="
        width:12px;height:12px;border-radius:50%;
        background:var(--danger);
        border:2px solid #fff;
        box-shadow:0 0 8px rgba(223,46,40,0.8);
      "></div>`,
      iconSize: [12, 12],
      iconAnchor: [6, 6],
    });

    markerRef.current = L.marker(defaultPos, { icon }).addTo(map);
    leafletRef.current = map;
  }, []);

  // Update marker + pan when GPS changes
  useEffect(() => {
    if (!gps || !leafletRef.current || !markerRef.current) return;
    const pos = [gps.lat, gps.lon];
    markerRef.current.setLatLng(pos);
    leafletRef.current.panTo(pos);
  }, [gps]);

  return (
    <div className="map-card">
      <div className="map-label-row">
        <span className="video-label">Location</span>
        {gps
          ? <span className="map-coords">{gps.lat.toFixed(5)}, {gps.lon.toFixed(5)}</span>
          : <span className="map-coords map-coords-dim">No GPS fix</span>
        }
      </div>
      <div ref={mapRef} className="map-container"/>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   ROOT APP — WebSocket integration
════════════════════════════════════════════════════════════ */
function App() {
  const [data, setData]             = useState(INITIAL_STATE);
  const [connStatus, setConnStatus] = useState("connecting");
  const [streamReady, setStreamReady] = useState(false);
  const socketRef = useRef(null);

  /* ── WebSocket connection ─────────────────────────────── */
  useEffect(() => {
    const socket = io({ transports: ["websocket", "polling"] });
    socketRef.current = socket;

    socket.on("connect", () => {
      setConnStatus("live");
      console.log("[Socket] connected:", socket.id);
    });
    socket.on("disconnect", () => {
      setConnStatus("disconnected");
      console.warn("[Socket] disconnected");
    });
    socket.on("connect_error", () => {
      setConnStatus("disconnected");
    });
    socket.on("state", (incoming) => {
      setData(incoming);
      setStreamReady(true);
    });

    fetch("/api/state")
      .then(r => r.json())
      .then(s => { if (s && s.hazards) { setData(s); setStreamReady(true); } })
      .catch(() => {});

    return () => socket.disconnect();
  }, []);

  /* ── Live clock ───────────────────────────────────────── */
  const [localTs, setLocalTs] = useState("");
  useEffect(() => {
    const id = setInterval(() => {
      const n = new Date();
      setLocalTs(n.toTimeString().slice(0,8) + "." + String(n.getMilliseconds()).padStart(3,"0"));
    }, 100);
    return () => clearInterval(id);
  }, []);

  const displayTs = data.timestamp !== INITIAL_STATE.timestamp ? data.timestamp : localTs;

  /* ── Derived UI state ─────────────────────────────────── */
  const mode = useMemo(() => deriveMode(data.hazards, data.speedKmh), [data.hazards, data.speedKmh]);
  const activeHazard = useMemo(() => {
    if (mode !== "ALERT") return null;
    const inC = data.hazards.filter(h => h.inCorridor);
    if (!inC.length) return null;
    return inC.reduce((best, h) => (h.distanceM < best.distanceM ? h : best), inC[0]);
  }, [mode, data.hazards]);

  const formatTime = sec => {
    const m = String(Math.floor(sec/60)).padStart(2,"0");
    const s = String(sec%60).padStart(2,"0");
    return `${m}:${s}`;
  };

  return (
    <div className="stage-shell">
      <div className="stage">
        <TopStatusBar timestamp={displayTs} connectivity={data.connectivity} connStatus={connStatus}/>
        <div className="content-area">
          <div className="primary-col">
            <PrimaryVideoPanel mode={mode} hazards={data.hazards} streamReady={streamReady}/>
          </div>
          <div className="right-col">
            <SecondaryVideoPanel mode={mode} hazards={data.hazards} streamReady={streamReady}/>
            <MapCard gps={data.gps || null}/>
            <div className="metrics-row">
              {mode === "ALERT" && activeHazard && <AlertCard hazard={activeHazard}/>}
              {mode === "SAFE" && <AllClearCard />}
              <SpeedGaugeTile speedValue={data.speedKmh} unit="km/h"/>
            </div>
            <StatusEventLogPanel events={data.events} systemStatus={data.systemStatus}/>
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);