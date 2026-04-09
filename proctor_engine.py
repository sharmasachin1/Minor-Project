"""
=============================================================================
 REAL-TIME AI PROCTORING SYSTEM — Parallel Multi-Threaded Architecture
=============================================================================
 Architecture Overview:
   ┌─────────────────────────────────────────────────────────────┐
   │  Main Thread  →  Capture frames → Shared Buffer             │
   │  Thread-1     →  YuNet Face Detection                       │
   │  Thread-2     →  MediaPipe Head Pose Estimation             │
   │  Thread-3     →  YOLOv8 Object Detection (every N frames)   │
   │  Main Thread  →  Aggregate results → Render UI              │
   └─────────────────────────────────────────────────────────────┘
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS & DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import threading
import time
import csv
import os
import warnings
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import mediapipe as mp
from ultralytics import YOLO

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # Model paths / URLs
    YUNET_URL   = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    YUNET_PATH  = "yunet.onnx"
    YOLO_MODEL  = "yolov8n.pt"              # Use nano for best CPU speed

    # Detection targets
    TARGET_OBJECTS = {"cell phone", "laptop"}

    # Thresholds
    FACE_CONFIDENCE   = 0.6
    YOLO_CONFIDENCE   = 0.35
    YAW_THRESHOLD     = 20.0               # degrees — "looking away"
    SMOOTHING_FRAMES  = 4                  # consecutive frames to confirm event
    YOLO_EVERY_N      = 5                  # run YOLO every N frames

    # Cooldown (seconds) per event type before re-logging
    COOLDOWN = {
        "No Face Detected":   5.0,
        "Multiple Faces":     5.0,
        "Looking Away":       4.0,
        "Object Detected":    6.0,
    }

    # UI
    TIMELINE_MAX_EVENTS = 10
    LOG_CSV_PATH        = "proctor_log.csv"

    # Severity colours (BGR)
    COLOR_HIGH   = (0,  50, 220)   # Red
    COLOR_MEDIUM = (0, 200, 220)   # Yellow
    COLOR_OK     = (50, 200,  50)  # Green
    COLOR_INFO   = (200, 200, 200) # White


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VulnerabilityEvent:
    event_type:  str
    timestamp:   datetime
    frame_number: int
    duration:    float = 0.0
    detail:      str   = ""


@dataclass
class DetectionResult:
    """Shared result bus — written by worker threads, read by main thread."""
    # Face
    num_faces:   int   = 0
    face_boxes:  list  = field(default_factory=list)

    # Head pose
    yaw:         float = 0.0
    pitch:       float = 0.0
    roll:        float = 0.0
    pose_valid:  bool  = False

    # Objects
    object_boxes:  list = field(default_factory=list)   # [(label, x1,y1,x2,y2)]

    # Frame metadata
    frame_number: int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. THREAD-SAFE FRAME BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class FrameBuffer:
    """
    Single-slot buffer:  producer (capture) overwrites, consumers grab latest.
    Uses a Condition so workers sleep until a new frame arrives.
    """
    def __init__(self):
        self._frame        = None
        self._frame_number = 0
        self._lock         = threading.Lock()
        self._cond         = threading.Condition(self._lock)

    def put(self, frame: np.ndarray):
        with self._cond:
            self._frame        = frame.copy()
            self._frame_number += 1
            self._cond.notify_all()

    def get(self, timeout: float = 0.1):
        """Block until a new frame is available, return (frame, frame_number)."""
        with self._cond:
            self._cond.wait(timeout=timeout)
            if self._frame is None:
                return None, 0
            return self._frame.copy(), self._frame_number


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHARED RESULT BUS  (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
class ResultBus:
    def __init__(self):
        self._result = DetectionResult()
        self._lock   = threading.Lock()

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._result, k, v)

    def snapshot(self) -> DetectionResult:
        with self._lock:
            import copy
            return copy.copy(self._result)


# ─────────────────────────────────────────────────────────────────────────────
# 6. VULNERABILITY LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class VulnerabilityLogger:
    def __init__(self, csv_path: str = Config.LOG_CSV_PATH):
        self._events: List[VulnerabilityEvent] = []
        self._lock            = threading.Lock()
        self._cooldown_clock  = {}              # event_type → last_log_time
        self._csv_path        = csv_path
        self._event_start     = {}              # event_type → start_time

        # Prepare CSV
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "frame_number",
                              "duration_sec", "detail"])

    def try_log(self, event_type: str, frame_number: int,
                detail: str = "") -> Optional[VulnerabilityEvent]:
        """
        Log an event if cooldown has elapsed.
        Returns the VulnerabilityEvent on success, else None.
        """
        now = time.time()
        cooldown = Config.COOLDOWN.get(event_type, 5.0)

        with self._lock:
            last = self._cooldown_clock.get(event_type, 0)
            if now - last < cooldown:
                return None                     # still in cooldown

            # Track duration
            start  = self._event_start.get(event_type, now)
            duration = now - start

            evt = VulnerabilityEvent(
                event_type   = event_type,
                timestamp    = datetime.now(),
                frame_number = frame_number,
                duration     = round(duration, 2),
                detail       = detail,
            )
            self._events.append(evt)
            self._cooldown_clock[event_type] = now
            self._event_start[event_type]    = now  # reset for next occurrence

            # Write CSV
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([evt.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                  evt.event_type, evt.frame_number,
                                  evt.duration, evt.detail])
            return evt

    def mark_started(self, event_type: str):
        """Call when an event begins to track duration."""
        with self._lock:
            if event_type not in self._event_start:
                self._event_start[event_type] = time.time()

    def mark_ended(self, event_type: str):
        """Call when an event ends to reset start tracking."""
        with self._lock:
            self._event_start.pop(event_type, None)

    def last_n(self, n: int = 10) -> List[VulnerabilityEvent]:
        with self._lock:
            return list(self._events[-n:])

    @property
    def total(self) -> int:
        with self._lock:
            return len(self._events)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SMOOTHING HELPER
# ─────────────────────────────────────────────────────────────────────────────
class FrameSmoother:
    """
    Confirms a binary event only after it appears in K consecutive frames.
    Also confirms 'cleared' only after K consecutive clear frames.
    """
    def __init__(self, k: int = Config.SMOOTHING_FRAMES):
        self._k       = k
        self._pos_buf = {}   # event_type → deque of booleans
        self._confirmed = {}

    def update(self, event_type: str, detected: bool) -> bool:
        buf = self._pos_buf.setdefault(event_type, deque(maxlen=self._k))
        buf.append(detected)
        if len(buf) < self._k:
            return self._confirmed.get(event_type, False)
        result = all(buf)
        self._confirmed[event_type] = result
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. WORKER THREAD — FACE DETECTION (YuNet)
# ─────────────────────────────────────────────────────────────────────────────
class FaceDetectionThread(threading.Thread):
    """
    Continuously pulls frames from the buffer, runs YuNet, writes to ResultBus.
    """
    def __init__(self, buffer: FrameBuffer, bus: ResultBus,
                 logger: VulnerabilityLogger, smoother: FrameSmoother):
        super().__init__(name="FaceDetectionThread", daemon=True)
        self._buffer   = buffer
        self._bus      = bus
        self._logger   = logger
        self._smoother = smoother
        self._detector = None
        self._stopped  = threading.Event()

    # ── lazy init inside thread ──
    def _init_detector(self, w: int, h: int):
        self._detector = cv2.FaceDetectorYN.create(
            Config.YUNET_PATH, "", (w, h),
            score_threshold=Config.FACE_CONFIDENCE,
            nms_threshold=0.3, top_k=5
        )

    def stop(self):
        self._stopped.set()

    def run(self):
        prev_size = (0, 0)
        while not self._stopped.is_set():
            frame, fnum = self._buffer.get(timeout=0.05)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            if (w, h) != prev_size:
                self._init_detector(w, h)
                prev_size = (w, h)
            else:
                self._detector.setInputSize((w, h))

            _, faces = self._detector.detect(frame)
            num_faces  = len(faces) if faces is not None else 0
            face_boxes = []

            if faces is not None:
                for f in faces:
                    x, y, fw, fh = map(int, f[:4])
                    face_boxes.append((x, y, x + fw, y + fh))

            self._bus.update(num_faces=num_faces, face_boxes=face_boxes,
                             frame_number=fnum)

            # Smoothed events
            no_face  = self._smoother.update("No Face Detected",  num_faces == 0)
            multi    = self._smoother.update("Multiple Faces",    num_faces > 1)

            if no_face:
                self._logger.mark_started("No Face Detected")
                self._logger.try_log("No Face Detected", fnum)
            else:
                self._logger.mark_ended("No Face Detected")

            if multi:
                self._logger.mark_started("Multiple Faces")
                self._logger.try_log("Multiple Faces", fnum,
                                      detail=f"{num_faces} faces")
            else:
                self._logger.mark_ended("Multiple Faces")


# ─────────────────────────────────────────────────────────────────────────────
# 9. WORKER THREAD — HEAD POSE (MediaPipe)
# ─────────────────────────────────────────────────────────────────────────────
class HeadPoseThread(threading.Thread):
    MESH_INDICES = [1, 152, 263, 33, 287, 57]   # 6-point set for solvePnP
    MODEL_POINTS = np.array([
        (0.0,    0.0,    0.0),
        (0.0, -330.0,  -65.0),
        (-225.0, 170.0, -135.0),
        (225.0,  170.0, -135.0),
        (-150.0,-150.0, -125.0),
        (150.0, -150.0, -125.0),
    ], dtype=np.float64)

    def __init__(self, buffer: FrameBuffer, bus: ResultBus,
                 logger: VulnerabilityLogger, smoother: FrameSmoother):
        super().__init__(name="HeadPoseThread", daemon=True)
        self._buffer   = buffer
        self._bus      = bus
        self._logger   = logger
        self._smoother = smoother
        self._stopped  = threading.Event()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def stop(self):
        self._stopped.set()

    def run(self):
        while not self._stopped.is_set():
            frame, fnum = self._buffer.get(timeout=0.05)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                self._bus.update(pose_valid=False)
                self._smoother.update("Looking Away", False)
                self._logger.mark_ended("Looking Away")
                continue

            lm = results.multi_face_landmarks[0].landmark
            lm2d = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

            img_pts = np.array([lm2d[i] for i in self.MESH_INDICES], dtype=np.float64)
            fl      = w
            cam_mat = np.array([[fl, 0, w/2],[0, fl, h/2],[0,0,1]], dtype=np.float64)

            ok, rvec, _ = cv2.solvePnP(
                self.MODEL_POINTS, img_pts, cam_mat,
                np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue

            rmat, _       = cv2.Rodrigues(rvec)
            angles, *_    = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = angles

            self._bus.update(yaw=yaw, pitch=pitch, roll=roll, pose_valid=True)

            looking_away = abs(yaw) > Config.YAW_THRESHOLD
            confirmed    = self._smoother.update("Looking Away", looking_away)

            if confirmed:
                self._logger.mark_started("Looking Away")
                self._logger.try_log("Looking Away", fnum,
                                      detail=f"yaw={yaw:.1f}°")
            else:
                self._logger.mark_ended("Looking Away")


# ─────────────────────────────────────────────────────────────────────────────
# 10. WORKER THREAD — OBJECT DETECTION (YOLOv8)
# ─────────────────────────────────────────────────────────────────────────────
class ObjectDetectionThread(threading.Thread):
    def __init__(self, buffer: FrameBuffer, bus: ResultBus,
                 logger: VulnerabilityLogger, smoother: FrameSmoother):
        super().__init__(name="ObjectDetectionThread", daemon=True)
        self._buffer   = buffer
        self._bus      = bus
        self._logger   = logger
        self._smoother = smoother
        self._stopped  = threading.Event()
        self._model    = YOLO(Config.YOLO_MODEL)
        self._last_fnum = -1

    def stop(self):
        self._stopped.set()

    def run(self):
        while not self._stopped.is_set():
            frame, fnum = self._buffer.get(timeout=0.05)
            if frame is None or fnum == self._last_fnum:
                continue

            # Run only every N frames
            if fnum % Config.YOLO_EVERY_N != 0:
                continue

            self._last_fnum = fnum
            results  = self._model(frame, conf=Config.YOLO_CONFIDENCE,
                                   verbose=False)[0]
            obj_boxes = []
            found_labels = set()

            for box in results.boxes:
                label = self._model.names[int(box.cls[0])]
                if label in Config.TARGET_OBJECTS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    obj_boxes.append((label, x1, y1, x2, y2, conf))
                    found_labels.add(label)

            self._bus.update(object_boxes=obj_boxes)

            detected = len(found_labels) > 0
            confirmed = self._smoother.update("Object Detected", detected)

            if confirmed and found_labels:
                self._logger.mark_started("Object Detected")
                self._logger.try_log("Object Detected", fnum,
                                      detail=", ".join(found_labels))
            else:
                self._logger.mark_ended("Object Detected")


# ─────────────────────────────────────────────────────────────────────────────
# 11. MODEL DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────
def ensure_yunet():
    if not os.path.exists(Config.YUNET_PATH):
        print("[INIT] Downloading YuNet model...")
        import urllib.request
        urllib.request.urlretrieve(Config.YUNET_URL, Config.YUNET_PATH)
        print("[INIT] YuNet downloaded.")


# ─────────────────────────────────────────────────────────────────────────────
# 12. UI RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY = {
    "No Face Detected":  ("HIGH",   Config.COLOR_HIGH),
    "Multiple Faces":    ("HIGH",   Config.COLOR_HIGH),
    "Looking Away":      ("MEDIUM", Config.COLOR_MEDIUM),
    "Object Detected":   ("HIGH",   Config.COLOR_HIGH),
}


def draw_status_bar(frame: np.ndarray, result: DetectionResult,
                    fps: float, total_events: int):
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Face status
    if result.num_faces == 0:
        face_txt, face_col = "NO FACE", Config.COLOR_HIGH
    elif result.num_faces > 1:
        face_txt, face_col = f"MULTI-FACE ({result.num_faces})", Config.COLOR_HIGH
    else:
        face_txt, face_col = "FACE OK", Config.COLOR_OK
    cv2.putText(frame, face_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, face_col, 1, cv2.LINE_AA)

    # Head pose
    if result.pose_valid:
        away = abs(result.yaw) > Config.YAW_THRESHOLD
        pose_txt = f"YAW:{result.yaw:+.1f} PITCH:{result.pitch:+.1f}"
        pose_col = Config.COLOR_HIGH if away else Config.COLOR_OK
    else:
        pose_txt, pose_col = "POSE N/A", Config.COLOR_MEDIUM
    cv2.putText(frame, pose_txt, (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, pose_col, 1, cv2.LINE_AA)

    # FPS  + event counter
    cv2.putText(frame, f"FPS:{fps:.1f}", (w - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, Config.COLOR_INFO, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Events:{total_events}", (w - 115, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, Config.COLOR_MEDIUM, 1, cv2.LINE_AA)


def draw_bounding_boxes(frame: np.ndarray, result: DetectionResult):
    # Face boxes (green)
    for (x1, y1, x2, y2) in result.face_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), Config.COLOR_OK, 2)

    # Object boxes (red) with label
    for item in result.object_boxes:
        label, x1, y1, x2, y2, conf = item
        cv2.rectangle(frame, (x1, y1), (x2, y2), Config.COLOR_HIGH, 2)
        tag = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), Config.COLOR_HIGH, -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_timeline_panel(frame: np.ndarray,
                        events: List[VulnerabilityEvent]):
    """
    Draws a translucent panel on the right side listing the last N events.
    """
    h, w = frame.shape[:2]
    panel_w   = 310
    row_h     = 28
    panel_h   = row_h * (len(events) + 1) + 12
    x0        = w - panel_w - 5
    y0        = 70

    if not events:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4),
                  (x0 + panel_w, y0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Header
    cv2.putText(frame, "── TIMELINE ──", (x0, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_INFO, 1, cv2.LINE_AA)

    for i, evt in enumerate(reversed(events)):
        y = y0 + row_h * (i + 1) + 12
        _, col = SEVERITY.get(evt.event_type, ("MEDIUM", Config.COLOR_MEDIUM))
        ts  = evt.timestamp.strftime("%H:%M:%S")
        txt = f"{ts} → {evt.event_type}"
        cv2.putText(frame, txt, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN APPLICATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_proctor():
    ensure_yunet()

    # ── Shared infrastructure ──
    buffer   = FrameBuffer()
    bus      = ResultBus()
    logger   = VulnerabilityLogger(Config.LOG_CSV_PATH)
    smoother = FrameSmoother(k=Config.SMOOTHING_FRAMES)

    # ── Worker threads ──
    threads = [
        FaceDetectionThread(buffer, bus, logger, smoother),
        HeadPoseThread(buffer, bus, logger, smoother),
        ObjectDetectionThread(buffer, bus, logger, smoother),
    ]
    for t in threads:
        t.start()

    cap         = cv2.VideoCapture(0)
    fps_deque   = deque(maxlen=30)
    prev_time   = time.time()
    frame_count = 0

    print("=" * 60)
    print("  AI PROCTOR STARTED  —  press 'q' to quit")
    print(f"  Logs → {Config.LOG_CSV_PATH}")
    print("=" * 60)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            buffer.put(frame)

            # FPS calculation
            now          = time.time()
            fps_deque.append(1.0 / max(now - prev_time, 1e-6))
            prev_time    = now
            fps          = sum(fps_deque) / len(fps_deque)

            # Snapshot latest results
            result              = bus.snapshot()
            result.frame_number = frame_count
            timeline_events     = logger.last_n(Config.TIMELINE_MAX_EVENTS)

            # ── Draw ──
            draw_bounding_boxes(frame, result)
            draw_timeline_panel(frame, timeline_events)
            draw_status_bar(frame, result, fps, logger.total)

            cv2.imshow("AI Proctor — Real-Time", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\n[SHUTDOWN] Stopping worker threads…")
        for t in threads:
            t.stop()
        for t in threads:
            t.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        print(f"[SHUTDOWN] Done. {logger.total} events logged to '{Config.LOG_CSV_PATH}'.")
# ADD THIS FUNCTION AT END

def start_engine():
    ensure_yunet()

    buffer   = FrameBuffer()
    bus      = ResultBus()
    logger   = VulnerabilityLogger(Config.LOG_CSV_PATH)
    smoother = FrameSmoother()

    threads = [
        FaceDetectionThread(buffer, bus, logger, smoother),
        HeadPoseThread(buffer, bus, logger, smoother),
        ObjectDetectionThread(buffer, bus, logger, smoother),
    ]

    for t in threads:
        t.start()

    return buffer, bus, threads