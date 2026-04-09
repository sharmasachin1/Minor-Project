import string
import torch
import sounddevice as sd
import numpy as np
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
from faster_whisper import WhisperModel

# ================= CONFIG =================
KEYWORDS = ["alexa", "siri", "google", "chatgpt", "gemini"]

SAMPLE_RATE = 16000
WINDOW_SEC = 6
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)

LABELS = ["normal", "whisper"]

SILENCE_THRESHOLD = 0.001
WHISPER_THRESHOLD = 0.48
LOW_VOLUME_THRESHOLD = 0.01
CHEAT_COOLDOWN = 3  # seconds

# ================= DATA STRUCTURE =================
@dataclass
class SegmentResult:
    segment_index: int
    label: str
    transcript: Optional[str] = None
    keywords_found: List[str] = field(default_factory=list)

# ================= HELPERS =================
def find_keywords(text):
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return [w for w in words if w in KEYWORDS]

def is_silent(chunk):
    return chunk.abs().mean().item() < SILENCE_THRESHOLD

def get_volume(chunk):
    return chunk.abs().mean().item()

def zero_crossing_rate(chunk):
    return ((chunk[:-1] * chunk[1:]) < 0).float().mean().item()

# ================= MODEL LOADER =================
def load_audio_model(path):
    try:
        from pytorch.models import Cnn14

        model = Cnn14(
            sample_rate=16000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=7600,
            classes_num=527,
        )

        model.fc_audioset = torch.nn.Linear(model.fc_audioset.in_features, 2)

        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])

        print("✅ Audio model loaded")
        return model.eval()

    except Exception as e:
        print("⚠ Audio model NOT loaded:", e)
        return None

def classify_segment(model, chunk):
    if model is None:
        return "normal", [0.9, 0.1]

    with torch.no_grad():
        out = model(chunk.unsqueeze(0))
        logits = out["clipwise_output"] if isinstance(out, dict) else out
        probs = torch.softmax(logits, dim=1)[0]

    return LABELS[torch.argmax(probs).item()], probs.tolist()

# ================= TRANSCRIPTION =================
def transcribe_segment(wmodel, chunk):
    audio = chunk.numpy().astype("float32")

    segments, _ = wmodel.transcribe(
        audio,
        beam_size=5,
        language="en"
    )

    text = ""
    for seg in segments:
        text += seg.text

    return text.strip()

# ================= REAL-TIME ENGINE =================
class RealtimeAnalyser:

    def __init__(self, audio_model, whisper_model):
        self.audio_model = audio_model
        self.whisper_model = whisper_model

        self._ring = deque(maxlen=WINDOW_SAMPLES)
        self._queue = queue.Queue()

        self._running = False
        self._seg_index = 0

        self.last_cheat_time = 0

        self.latest_result = {
            "cheating": False,
            "keywords": [],
            "transcript": ""
        }

    # 🎤 MIC CALLBACK
    def _mic_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0].astype(np.float32)

        self._ring.extend(samples)

        if len(self._ring) >= WINDOW_SAMPLES:
            print(f"📦 {WINDOW_SEC} sec audio captured")

            chunk = torch.tensor(np.array(self._ring, dtype=np.float32))
            self._queue.put(chunk.clone())

            self._ring.clear()

    # 🧠 ANALYSIS THREAD
    def _analyse_worker(self):
        print("🧠 Analysis thread started")

        while self._running:
            try:
                chunk = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            print("\n⚡ Processing audio...")

            # 🔇 Silence check
            if is_silent(chunk):
                print("🔇 Silence detected — skipping")
                continue

            # 📊 Metrics
            volume = get_volume(chunk)
            zcr = zero_crossing_rate(chunk)

            # 🧠 Classification
            label, probs = classify_segment(self.audio_model, chunk)
            whisper_prob = probs[1]

            # 🤫 Whisper logic (improved)
            is_whisper = (
                whisper_prob > WHISPER_THRESHOLD and
                volume < LOW_VOLUME_THRESHOLD and
                zcr > 0.05
            )

            transcript = ""
            keywords = []

            # 🗣 ONLY transcribe if NOT whisper
            if not is_whisper:
                transcript = transcribe_segment(self.whisper_model, chunk)
                keywords = find_keywords(transcript)

                print("🗣 Transcript:", transcript)
                print("🔍 Keywords:", keywords)
            else:
                print(f"🤫 Whisper detected — skipping transcription")

            print(f"🔊 Volume: {volume:.5f} | 🤫 Whisper prob: {whisper_prob:.2f} | ZCR: {zcr:.3f}")

            # 🚨 Cheating logic
            current_time = time.time()
            cheating = False

            if (is_whisper and keywords) or len(keywords) > 0:
                if current_time - self.last_cheat_time > CHEAT_COOLDOWN:
                    cheating = True
                    self.last_cheat_time = current_time
                    print("🚨 CHEATING DETECTED 🚨")
                else:
                    print("⏳ Cooldown active")
            else:
                print("✅ Normal Behavior")

            # 💾 Store result
            self.latest_result = {
                "cheating": cheating,
                "keywords": keywords,
                "transcript": transcript
            }

    # ▶ START
    def start(self):
        self._running = True

        threading.Thread(target=self._analyse_worker, daemon=True).start()

        print("\n🎧 Available Audio Devices:")
        print(sd.query_devices())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=None,
            blocksize=1024,
            callback=self._mic_callback,
        )

        self._stream.start()
        print(f"\n🎙 Speak something ({WINDOW_SEC} sec window)...\n")

    # ⛔ STOP
    def stop(self):
        self._running = False
        self._stream.stop()
        self._stream.close()
        print("🛑 Stopped")

# ================= MAIN =================
if __name__ == "__main__":
    print("Loading models...")

    audio_model = load_audio_model("outputs/best_model.pth")

    whisper_model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    analyser = RealtimeAnalyser(audio_model, whisper_model)
    analyser.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        analyser.stop()