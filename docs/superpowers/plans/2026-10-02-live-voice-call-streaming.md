# Live Voice-Call Streaming POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working proof of concept that simulates a live voice-call leg where a Web UI streams Linear PCM WAV audio progressively over WebSocket to a FastAPI backend running the Qwen3-ASR ONNX engine at 1x real-time pace, receiving live recognition deltas and call telemetry.

**Architecture:** A FastAPI WebSocket server (`main.py`) manages a `LiveCallSession` instance that converts incoming 16kHz 16-bit PCM bytes to float32 arrays, buffers streaming audio, invokes `ONNXQwen3ASR.transcribe_stream`, and returns real-time JSON deltas and performance metrics back to a single-page HTML5/JS Web Audio API interface.

**Tech Stack:** Python (FastAPI, Uvicorn, NumPy, ONNX Runtime, Pytest), JavaScript (Web Audio API, WebSockets, HTML5 Canvas).

**Spec:** `docs/superpowers/specs/2026-10-02-live-voice-call-streaming-design.md`

## Global Constraints
- Target Audio Format: 16kHz 16-bit Linear PCM Mono WAV.
- Streaming Pacing: 500ms audio chunks (~16,000 bytes per binary packet).
- Real-time communication protocol: WebSockets (`/ws/call-stream`).
- High-aesthetic glassmorphism dark theme UI with Inter typography.

---

### Task 1: Live Call Session Manager

**Files:**
- Create: `src/engines/live_call_session.py`
- Test: `tests/test_live_call_session.py`

**Interfaces:**
- Consumes: `ONNXQwen3ASR` from `src.engines.qwen3_onnx_engine`
- Produces: `LiveCallSession` class with methods `process_pcm_bytes(raw_bytes: bytes)`, `finish()`, and properties `buffered_seconds`, `call_metrics`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_call_session.py
import numpy as np
import pytest
from src.engines.live_call_session import LiveCallSession

def test_live_call_session_pcm_conversion():
    session = LiveCallSession(sample_rate=16000)
    # 0.5s of 16kHz 16-bit int16 mono audio (8000 samples = 16000 bytes)
    raw_int16 = (np.sin(np.linspace(0, 100, 8000)) * 16384).astype(np.int16)
    raw_bytes = raw_int16.tobytes()

    result = session.process_pcm_bytes(raw_bytes)
    assert result["buffered_seconds"] == pytest.approx(0.5, abs=0.05)
    assert len(session.audio_buffer) == 8000
    assert session.audio_buffer.dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_live_call_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.engines.live_call_session'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/engines/live_call_session.py
import time
import numpy as np
from typing import Dict, Any, Optional

class LiveCallSession:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_buffer = np.array([], dtype=np.float32)
        self.start_time = time.time()
        self.total_bytes = 0
        self.chunks_received = 0

    def process_pcm_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Convert int16 PCM bytes to float32 normalized array and add to buffer."""
        self.chunks_received += 1
        self.total_bytes += len(raw_bytes)
        
        # Convert int16 bytes to numpy float32 [-1.0, 1.0]
        int16_samples = np.frombuffer(raw_bytes, dtype=np.int16)
        float32_samples = int16_samples.astype(np.float32) / 32768.0
        
        self.audio_buffer = np.concatenate([self.audio_buffer, float32_samples])
        buffered_seconds = len(self.audio_buffer) / self.sample_rate

        return {
            "chunks_received": self.chunks_received,
            "total_bytes": self.total_bytes,
            "buffered_seconds": buffered_seconds,
            "new_samples_count": len(float32_samples),
        }

    def get_metrics(self, processing_duration_s: float) -> Dict[str, Any]:
        audio_dur_s = len(self.audio_buffer) / self.sample_rate
        rtf = processing_duration_s / audio_dur_s if audio_dur_s > 0 else 0.0
        return {
            "audio_duration_s": round(audio_dur_s, 2),
            "processing_time_s": round(processing_duration_s, 3),
            "rtf": round(rtf, 3),
            "chunks_received": self.chunks_received,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_live_call_session.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/engines/live_call_session.py tests/test_live_call_session.py
git commit -m "feat: add LiveCallSession manager for PCM audio buffering"
```

---

### Task 2: FastAPI Web Server & WebSocket Streaming Endpoint

**Files:**
- Create: `main.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `LiveCallSession` from `src.engines.live_call_session`, `ONNXQwen3ASR` from `src.engines.qwen3_onnx_engine`
- Produces: FastAPI app with `/`, `/api/samples`, `/api/samples/{filename}`, `/ws/call-stream`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0

def test_websocket_connect():
    with client.websocket_connect("/ws/call-stream") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "connected"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write minimal implementation**

```python
# main.py
import json
import time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.engines.qwen3_onnx_engine import ONNXQwen3ASR
from src.engines.live_call_session import LiveCallSession

app = FastAPI(title="RT-MASR Live Voice-Call Simulation")

# Serve static files if directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Lazy-loaded engine singleton
_engine: Optional[ONNXQwen3ASR] = None

def get_engine() -> ONNXQwen3ASR:
    global _engine
    if _engine is None:
        _engine = ONNXQwen3ASR()
    return _engine

@app.get("/")
def get_ui():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "RT-MASR Web UI Server Ready"})

@app.get("/api/samples")
def list_samples():
    test_audio_dir = Path("test_audio")
    samples = []
    if test_audio_dir.exists():
        for wav_file in test_audio_dir.glob("*.wav"):
            samples.append({
                "name": wav_file.name,
                "size_bytes": wav_file.stat().st_size
            })
    return {"samples": samples}

@app.get("/api/samples/{filename}")
def get_sample_file(filename: str):
    file_path = Path("test_audio") / filename
    if file_path.exists() and file_path.suffix == ".wav":
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.websocket("/ws/call-stream")
async def websocket_call_stream(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type": "connected",
        "message": "Connected to RT-MASR Live Call Stream"
    })
    
    session = LiveCallSession()
    cumulative_text = ""
    start_call_time = time.time()
    
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                pcm_data = message["bytes"]
                stats = session.process_pcm_bytes(pcm_data)
                
                # Send chunk acknowledgment
                await websocket.send_json({
                    "type": "chunk_ack",
                    "buffered_seconds": stats["buffered_seconds"],
                    "chunks_received": stats["chunks_received"]
                })
                
                # Perform streaming transcription if sufficient audio
                if len(session.audio_buffer) >= 8000 and stats["chunks_received"] % 2 == 0:
                    engine = get_engine()
                    t0 = time.time()
                    deltas = list(engine.transcribe_stream(session.audio_buffer))
                    proc_time = time.time() - t0
                    
                    new_text = "".join(deltas)
                    if new_text != cumulative_text:
                        cumulative_text = new_text
                        metrics = session.get_metrics(proc_time)
                        await websocket.send_json({
                            "type": "transcript_delta",
                            "full_text": cumulative_text,
                            "metrics": metrics
                        })
            
            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("type") == "end_call":
                    # Final recognition pass
                    engine = get_engine()
                    t0 = time.time()
                    deltas = list(engine.transcribe_stream(session.audio_buffer))
                    proc_time = time.time() - t0
                    cumulative_text = "".join(deltas)
                    total_call_time = time.time() - start_call_time
                    metrics = session.get_metrics(total_call_time)
                    
                    await websocket.send_json({
                        "type": "call_ended",
                        "final_text": cumulative_text,
                        "metrics": metrics
                    })
                    break
    except WebSocketDisconnect:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_server.py
git commit -m "feat: implement FastAPI server with WebSocket call streaming"
```

---

### Task 3: Interactive Web UI Frontend

**Files:**
- Create: `static/index.html`
- Create: `static/style.css`
- Create: `static/app.js`

**Interfaces:**
- Consumes: WebSocket endpoint `/ws/call-stream`, REST endpoints `/api/samples`
- Produces: Responsive Web Audio API call leg streaming UI with dark mode theme, dynamic oscilloscope canvas, live telemetry dashboard, and streaming text reader.

- [ ] **Step 1: Create `static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RT-MASR — Live Voice Call Leg Simulator</title>
  <link rel="stylesheet" href="/static/style.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
  <div class="app-container">
    <header class="header">
      <div class="logo-group">
        <span class="pulse-dot" id="status-dot"></span>
        <h1>RT-MASR Live Call Simulator</h1>
      </div>
      <div class="status-badge" id="status-text">DISCONNECTED</div>
    </header>

    <main class="grid-layout">
      <!-- Left Column: Controls & Audio Inputs -->
      <section class="card input-card">
        <h2>1. Select Audio Call Leg</h2>
        <div class="sample-buttons" id="sample-buttons">
          <p class="loading-text">Loading audio samples...</p>
        </div>
        
        <div class="divider">OR UPLOAD WAV</div>
        <div class="file-dropzone" id="dropzone">
          <input type="file" id="audio-file-input" accept=".wav" />
          <p>Drag & drop Linear PCM .wav file or click to browse</p>
          <span class="file-info" id="file-info">No file selected</span>
        </div>

        <div class="controls-group">
          <button id="start-btn" class="btn btn-primary" disabled>Start Call Leg</button>
          <button id="hangup-btn" class="btn btn-danger" disabled>Hang Up</button>
        </div>
      </section>

      <!-- Right Column: Visualizer, Telemetry & Live Text -->
      <section class="main-display">
        <!-- Telemetry Cards -->
        <div class="telemetry-grid">
          <div class="metric-card">
            <span class="label">Call Duration</span>
            <span class="value" id="metric-duration">00:00</span>
          </div>
          <div class="metric-card">
            <span class="label">Latency</span>
            <span class="value" id="metric-latency">0 ms</span>
          </div>
          <div class="metric-card">
            <span class="label">Real-Time Factor (RTF)</span>
            <span class="value" id="metric-rtf">0.00x</span>
          </div>
          <div class="metric-card">
            <span class="label">Chunks Processed</span>
            <span class="value" id="metric-chunks">0</span>
          </div>
        </div>

        <!-- Audio Oscilloscope Canvas -->
        <div class="card visualizer-card">
          <h2>Live Audio Stream Waveform</h2>
          <canvas id="waveform-canvas" height="100"></canvas>
        </div>

        <!-- Streaming Transcript Terminal -->
        <div class="card transcript-card">
          <h2>Live ASR Recognition Stream</h2>
          <div class="transcript-box" id="transcript-box">
            <span class="placeholder">Recognition output will stream live here during the call leg...</span>
          </div>
        </div>
      </section>
    </main>
  </div>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `static/style.css`**

```css
:root {
  --bg-color: #0b0f19;
  --card-bg: rgba(22, 30, 46, 0.7);
  --border-color: rgba(255, 255, 255, 0.1);
  --primary-color: #3b82f6;
  --primary-hover: #2563eb;
  --danger-color: #ef4444;
  --text-main: #f3f4f6;
  --text-muted: #9ca3af;
  --accent-glow: #10b981;
}

* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }

body {
  background: var(--bg-color);
  color: var(--text-main);
  min-height: 100vh;
  padding: 20px;
}

.app-container { max-width: 1200px; margin: 0 auto; }

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-color);
}

.logo-group { display: flex; align-items: center; gap: 12px; }

.pulse-dot {
  width: 12px; height: 12px; border-radius: 50%; background: #6b7280;
  transition: background 0.3s;
}
.pulse-dot.active { background: var(--accent-glow); box-shadow: 0 0 10px var(--accent-glow); }

.status-badge {
  background: rgba(255, 255, 255, 0.05); padding: 6px 14px; border-radius: 20px;
  font-size: 0.85rem; font-weight: 600; border: 1px solid var(--border-color);
}

.grid-layout { display: grid; grid-template-columns: 340px 1fr; gap: 20px; }

.card {
  background: var(--card-bg);
  backdrop-filter: blur(10px);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
}

.input-card h2, .visualizer-card h2, .transcript-card h2 {
  font-size: 1rem; font-weight: 600; margin-bottom: 16px; color: var(--text-muted);
}

.sample-buttons { display: flex; flex-direction: column; gap: 8px; }
.sample-btn {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 10px; border-radius: 8px; cursor: pointer; text-align: left;
  transition: all 0.2s;
}
.sample-btn:hover { background: rgba(59, 130, 246, 0.2); border-color: var(--primary-color); }

.divider { text-align: center; margin: 16px 0; font-size: 0.75rem; color: var(--text-muted); }

.file-dropzone {
  border: 2px dashed var(--border-color); border-radius: 8px; padding: 20px;
  text-align: center; cursor: pointer; font-size: 0.85rem; color: var(--text-muted);
  position: relative;
}
.file-dropzone input { opacity: 0; position: absolute; inset: 0; width: 100%; height: 100%; cursor: pointer; }
.file-info { display: block; margin-top: 8px; font-weight: 500; color: var(--primary-color); }

.controls-group { display: flex; gap: 10px; margin-top: 20px; }
.btn {
  flex: 1; padding: 12px; border: none; border-radius: 8px; font-weight: 600;
  cursor: pointer; transition: all 0.2s;
}
.btn-primary { background: var(--primary-color); color: white; }
.btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
.btn-danger { background: var(--danger-color); color: white; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }

.telemetry-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
.metric-card {
  background: var(--card-bg); border: 1px solid var(--border-color);
  border-radius: 10px; padding: 14px; display: flex; flex-direction: column; gap: 6px;
}
.metric-card .label { font-size: 0.75rem; color: var(--text-muted); }
.metric-card .value { font-size: 1.25rem; font-weight: 700; color: var(--text-main); }

canvas { width: 100%; border-radius: 6px; background: rgba(0, 0, 0, 0.3); }

.transcript-box {
  background: rgba(0, 0, 0, 0.4); border-radius: 8px; padding: 16px;
  min-height: 160px; font-size: 1.1rem; line-height: 1.6; color: #6ee7b7;
}
.transcript-box .placeholder { color: var(--text-muted); font-size: 0.95rem; }
```

- [ ] **Step 3: Create `static/app.js`**

```javascript
document.addEventListener("DOMContentLoaded", () => {
  const sampleButtonsContainer = document.getElementById("sample-buttons");
  const audioFileInput = document.getElementById("audio-file-input");
  const fileInfo = document.getElementById("file-info");
  const startBtn = document.getElementById("start-btn");
  const hangupBtn = document.getElementById("hangup-btn");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  
  const metricDuration = document.getElementById("metric-duration");
  const metricLatency = document.getElementById("metric-latency");
  const metricRtf = document.getElementById("metric-rtf");
  const metricChunks = document.getElementById("metric-chunks");
  const transcriptBox = document.getElementById("transcript-box");
  const canvas = document.getElementById("waveform-canvas");
  const canvasCtx = canvas.getContext("2d");

  let selectedAudioArrayBuffer = null;
  let websocket = null;
  let audioContext = null;
  let timerInterval = null;
  let callStartTime = 0;

  // Load available sample audio files from backend
  fetch("/api/samples")
    .then(res => res.json())
    .then(data => {
      sampleButtonsContainer.innerHTML = "";
      if (data.samples && data.samples.length > 0) {
        data.samples.forEach(sample => {
          const btn = document.createElement("button");
          btn.className = "sample-btn";
          btn.textContent = `🎵 ${sample.name} (${(sample.size_bytes / 1024).toFixed(0)} KB)`;
          btn.onclick = () => loadSampleAudio(sample.name);
          sampleButtonsContainer.appendChild(btn);
        });
      } else {
        sampleButtonsContainer.innerHTML = "<p class='loading-text'>No samples found.</p>";
      }
    });

  function loadSampleAudio(filename) {
    fileInfo.textContent = `Loading ${filename}...`;
    fetch(`/api/samples/${filename}`)
      .then(res => res.arrayBuffer())
      .then(buffer => {
        selectedAudioArrayBuffer = buffer;
        fileInfo.textContent = `Selected: ${filename}`;
        startBtn.disabled = false;
      });
  }

  audioFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      fileInfo.textContent = `Selected: ${file.name}`;
      const reader = new FileReader();
      reader.onload = (evt) => {
        selectedAudioArrayBuffer = evt.target.result;
        startBtn.disabled = false;
      };
      reader.readAsArrayBuffer(file);
    }
  });

  startBtn.onclick = startCallLeg;
  hangupBtn.onclick = endCallLeg;

  function startCallLeg() {
    if (!selectedAudioArrayBuffer) return;

    startBtn.disabled = true;
    hangupBtn.disabled = false;
    statusDot.classList.add("active");
    statusText.textContent = "CALL IN PROGRESS";
    transcriptBox.innerHTML = "<span class='placeholder'>Call connected. Streaming audio...</span>";
    
    callStartTime = Date.now();
    timerInterval = setInterval(updateCallTimer, 1000);

    // Setup Web Audio Context for PCM conversion and streaming
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    
    // Connect WebSocket
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    websocket = new WebSocket(`${protocol}//${location.host}/ws/call-stream`);
    websocket.binaryType = "arraybuffer";

    websocket.onopen = () => {
      audioContext.decodeAudioData(selectedAudioArrayBuffer.slice(0), (audioBuffer) => {
        streamAudioBuffer(audioBuffer);
      });
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "chunk_ack") {
        metricChunks.textContent = data.chunks_received;
      } else if (data.type === "transcript_delta") {
        transcriptBox.textContent = data.full_text;
        if (data.metrics) {
          metricLatency.textContent = `${(data.metrics.processing_time_s * 1000).toFixed(0)} ms`;
          metricRtf.textContent = `${data.metrics.rtf}x`;
        }
      } else if (data.type === "call_ended") {
        endCallLeg();
        transcriptBox.textContent = data.final_text || "Call completed.";
      }
    };
  }

  function streamAudioBuffer(audioBuffer) {
    const channelData = audioBuffer.getChannelData(0); // 16kHz float32
    const chunkSize = 8000; // 0.5s chunk at 16kHz
    let offset = 0;

    const streamInterval = setInterval(() => {
      if (offset >= channelData.length || !websocket || websocket.readyState !== WebSocket.OPEN) {
        clearInterval(streamInterval);
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(JSON.stringify({ type: "end_call" }));
        }
        return;
      }

      const chunkFloat32 = channelData.subarray(offset, offset + chunkSize);
      // Convert float32 [-1, 1] to int16 PCM bytes
      const int16Buffer = new Int16Array(chunkFloat32.length);
      for (let i = 0; i < chunkFloat32.length; i++) {
        int16Buffer[i] = Math.max(-1, Math.min(1, chunkFloat32[i])) * 0x7FFF;
      }

      drawWaveform(chunkFloat32);
      websocket.send(int16Buffer.buffer);
      offset += chunkSize;
    }, 500); // 0.5s real-time pacing
  }

  function drawWaveform(samples) {
    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.3)";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "#10b981";
    canvasCtx.beginPath();

    const sliceWidth = canvas.width / samples.length;
    let x = 0;
    for (let i = 0; i < samples.length; i += 10) {
      const v = (samples[i] + 1) / 2;
      const y = v * canvas.height;
      if (i === 0) canvasCtx.moveTo(x, y);
      else canvasCtx.lineTo(x, y);
      x += sliceWidth * 10;
    }
    canvasCtx.stroke();
  }

  function updateCallTimer() {
    const elapsedSec = Math.floor((Date.now() - callStartTime) / 1000);
    const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
    const secs = String(elapsedSec % 60).padStart(2, '0');
    metricDuration.textContent = `${mins}:${secs}`;
  }

  function endCallLeg() {
    clearInterval(timerInterval);
    if (websocket) { websocket.close(); websocket = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    startBtn.disabled = false;
    hangupBtn.disabled = true;
    statusDot.classList.remove("active");
    statusText.textContent = "COMPLETED";
  }
});
```

- [ ] **Step 4: Commit UI files**

```bash
git add static/index.html static/style.css static/app.js
git commit -m "feat: add interactive glassmorphism Web UI for live call streaming"
```

---

### Task 4: Server Launch & Verification

**Files:**
- Test: `main.py`, browser UI

- [ ] **Step 1: Test Python tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Start server in background**

Run: `uvicorn main:app --host 0.0.0.0 --port 8000`

- [ ] **Step 3: Verification commit**

```bash
git add .
git commit -m "chore: finalize live voice call leg proof of concept"
```
