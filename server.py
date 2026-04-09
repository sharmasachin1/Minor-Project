from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import numpy as np
import cv2
import os

# ================= VIDEO ENGINE =================
from proctor_engine import start_engine, Config

# ================= AUDIO ENGINE =================
from audio_engine import RealtimeAnalyser, load_audio_model
from faster_whisper import WhisperModel

app = FastAPI()

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(BASE_DIR, "frontend")

print("Frontend path:", frontend_path)

# -------------------------------
# SERVE FRONTEND
# -------------------------------
@app.get("/")
def serve_home():
    return FileResponse(os.path.join(frontend_path, "index.html"))

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= START VIDEO AI =================
buffer, bus, threads = start_engine()
print("✅ Video AI Backend Running")

# ================= START AUDIO AI =================
print("🎤 Starting Audio Engine...")

audio_model = load_audio_model("outputs/best_model.pth")  # if missing → test mode

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

audio_analyser = RealtimeAnalyser(audio_model, whisper_model)
audio_analyser.start()

print("✅ Audio AI Running")

# -------------------------------
# VIDEO ANALYZE API
# -------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    buffer.put(frame)
    result = bus.snapshot()

    return {
        "faces": result.num_faces,
        "yaw": round(result.yaw, 2),
        "objects": [obj[0] for obj in result.object_boxes],
        "alerts": {
            "no_face": result.num_faces == 0,
            "multiple_faces": result.num_faces > 1,
            "looking_away": abs(result.yaw) > Config.YAW_THRESHOLD,
            "object_detected": len(result.object_boxes) > 0
        }
    }

# -------------------------------
# 🔥 AUDIO STATUS API
# -------------------------------
@app.get("/audio-status")
def get_audio_status():
    return audio_analyser.latest_result

# -------------------------------
# 🔥 COMBINED STATUS (BEST API)
# -------------------------------
@app.get("/full-status")
def full_status():
    video_result = bus.snapshot()

    return {
        "video": {
            "faces": video_result.num_faces,
            "yaw": round(video_result.yaw, 2),
            "objects": [obj[0] for obj in video_result.object_boxes],
            "alerts": {
                "no_face": video_result.num_faces == 0,
                "multiple_faces": video_result.num_faces > 1,
                "looking_away": abs(video_result.yaw) > Config.YAW_THRESHOLD,
                "object_detected": len(video_result.object_boxes) > 0
            }
        },
        "audio": audio_analyser.latest_result
    }

# -------------------------------
# STATUS API
# -------------------------------
@app.get("/api/status")
def status():
    return {"message": "AI Proctor Running (Audio + Video)"}

# -------------------------------
# CLEAN SHUTDOWN
# -------------------------------
@app.on_event("shutdown")
def shutdown():
    for t in threads:
        t.stop()

    audio_analyser.stop()
    print("🛑 All AI systems stopped")