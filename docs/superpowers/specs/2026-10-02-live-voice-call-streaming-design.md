# Design Spec: Live Voice-Call Leg ASR Simulation

## Overview
This specification details the architecture and implementation plan for a working proof of concept (POC) that simulates a live voice-call leg. The application allows a user to select or upload a Linear PCM WAV audio file in a Web UI, streams the PCM audio progressive chunk-by-chunk over a WebSocket connection to a FastAPI backend at approximately 1x real-time pace, processes the streaming audio using the `Qwen3-ASR` ONNX engine, and streams recognition deltas and live call telemetry metrics back to the Web UI in real time.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Browser (UI)                       │
│  - WAV Parser & Web Audio API Playback                       │
│  - Real-time Audio Chunk Scheduler (500ms / 1x Real Time)   │
│  - Oscilloscope Waveform Canvas                             │
│  - Live Telemetry Dashboard & Streaming Transcript Display  │
└──────────────────────────────┬──────────────────────────────┘
                               │ WebSocket (Binary PCM Chunks & JSON Control/Data)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                          │
│  - WebSockets Endpoint (/ws/call-stream)                    │
│  - Audio Buffer Manager (float32 conversion, 16kHz)         │
│  - Streaming ASR Orchestrator                               │
└──────────────────────────────┬──────────────────────────────┘
                               │ numpy audio array
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Qwen3-ASR ONNX Engine                       │
│  - Mel Spectrogram Extraction                               │
│  - ONNX Encoder Conv + Transformer                          │
│  - Greedy Decoder with KV Caching & Delta Text Generator     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Backend Server (`main.py` & `src/engines/live_call_session.py`)
- **Framework**: FastAPI with Uvicorn server and WebSocket support.
- **API Endpoints**:
  - `GET /`: Serves static `index.html` UI.
  - `GET /api/samples`: Lists available sample WAV files in `test_audio/`.
  - `GET /api/samples/{filename}`: Serves WAV file for built-in sample selection.
  - `WS /ws/call-stream`: WebSocket endpoint for the live call session.
- **WebSocket Protocol Messages**:
  - Client -> Server:
    - Text JSON: `{"type": "start_call", "sample_rate": 16000, "channels": 1}`
    - Binary message: Raw 16-bit Linear PCM audio bytes (e.g., 500ms chunks = 16,000 bytes at 16kHz mono).
    - Text JSON: `{"type": "end_call"}`
  - Server -> Client:
    - Text JSON (`type: "connected"`): Connection confirmation.
    - Text JSON (`type: "chunk_ack"`): Acknowledgment of frame received, containing buffered audio duration in seconds.
    - Text JSON (`type: "transcript_delta"`): Incremental text delta and cumulative transcript text.
    - Text JSON (`type: "metrics"`): Audio duration, processing latency, real-time factor (RTF).
    - Text JSON (`type: "call_ended"`): Final transcription, total call duration, total processing time, and overall RTF.

### 2.2 Streaming ASR Integration (`src/engines/qwen3_engine.py` & Session Engine)
- The ONNX engine loads Qwen3-ASR INT8 ONNX models (`encoder_conv.onnx`, `encoder_transformer.onnx`, `decoder_init.int8.onnx`, `decoder_step.int8.onnx`).
- As binary audio frames arrive, they are converted to float32 NumPy arrays normalized to `[-1.0, 1.0]`.
- The session accumulates incoming frames into a rolling stream buffer.
- On each frame update or threshold interval, `transcribe_stream()` performs streaming token generation, yielding incremental text deltas back to the client.

### 2.3 Web UI (`static/index.html`, `static/app.js`, `static/style.css`)
- **Theme**: Modern dark mode with glowing status badges, glassmorphism telemetry cards, dynamic canvas oscilloscope, and live streaming transcript view.
- **Audio Control**:
  - File picker for custom local `.wav` files (enforces/verifies PCM WAV format).
  - One-click sample file selection (`librispeech_0_1089_0.wav`, `harvard.wav`, `jackhammer.wav`).
  - Audio player Web Audio API node syncing local audio playback with WebSocket streaming.
- **Metrics Dashboard**:
  - Live Call Status (Idle, In Call, Completed).
  - Audio Call Duration (MM:SS).
  - Recognition Latency (ms).
  - Real-Time Factor (RTF = processing_time / audio_duration).
  - Chunks Sent / Received.

---

## 3. Verification & Testing Strategy
1. **Unit Testing**:
   - Verify PCM byte decoding and normalization (`int16` -> `float32` 16kHz).
   - Test `ONNXQwen3ASR.transcribe_stream` output on test audio sample files.
2. **WebSocket Integration Testing**:
   - Test `start_call`, streaming binary chunks, receiving `transcript_delta` and `metrics`, and `end_call` messages using `pytest-asyncio` / FastAPI TestClient.
3. **End-to-End Browser Testing**:
   - Run FastAPI server.
   - Load UI in browser, select a sample WAV audio file, click "Start Call Leg".
   - Verify live audio playback, live waveform canvas drawing, real-time metrics updating, and progressive live text recognition.

---

## 4. Dependencies & Files
- Existing: `src/engines/qwen3_engine.py`, `utils/audio_utils.py`, `config/`
- To Create/Update:
  - `main.py` (FastAPI app and WebSocket server)
  - `src/engines/live_call_session.py` (Call leg session manager)
  - `static/index.html` (Web UI HTML)
  - `static/style.css` (Glassmorphism Dark CSS)
  - `static/app.js` (Web Audio API + WebSocket streaming JS)
  - `tests/test_live_call.py` (Tests for streaming live call session)
