/**
 * app.js — RT-MASR Frontend Logic
 *
 * Timestamp / state model
 * ─────────────────────────────────────────────────────────────
 *  clientT0   Date.now() when start_call WS message is sent
 *             (mirrors server T0 as closely as network RTT allows)
 *  callStartTime  alias for clientT0, used for the wall-clock timer
 *
 * Stream state machine
 * ─────────────────────────────────────────────────────────────
 *  IDLE        → no call active, no WS connection
 *  CONNECTING  → WS opened, start_call sent, waiting for call_ready
 *  BUFFERING   → call_ready received, audio chunks streaming, no transcript yet
 *  TRANSCRIBING→ first transcript_delta received
 *  COMPLETED   → call_ended received or hang-up pressed successfully
 *  ERROR       → WebSocket error or unexpected disconnect
 */

document.addEventListener("DOMContentLoaded", () => {

  /* ── DOM refs ─────────────────────────────────────────── */
  const sampleButtonsContainer = document.getElementById("sample-buttons");
  const audioFileInput         = document.getElementById("audio-file-input");
  const fileInfo               = document.getElementById("file-info");
  const startBtn               = document.getElementById("start-btn");
  const hangupBtn              = document.getElementById("hangup-btn");
  const resetBtn               = document.getElementById("reset-btn");
  const muteBtn                = document.getElementById("mute-btn");
  const muteIcon               = document.getElementById("mute-icon");
  const muteLabel              = document.getElementById("mute-label");
  const languageSelect         = document.getElementById("language-select");
  const modelStatusText        = document.getElementById("model-status-text");
  const transcriptBox          = document.getElementById("transcript-box");
  const transcriptMeta         = document.getElementById("transcript-meta");
  const titleDesc              = document.getElementById("title-desc");
  const streamModeLabel        = document.getElementById("stream-mode-label");

  /* State pill */
  const streamStatePill = document.getElementById("stream-state-pill");
  const stateDot        = document.getElementById("state-dot");
  const stateLabel      = document.getElementById("state-label");

  /* Primary metrics */
  const mPipelineLatency = document.getElementById("metric-pipeline-latency");
  const mTtft            = document.getElementById("metric-ttft");
  const mInferLatency    = document.getElementById("metric-infer-latency");
  const mRtf             = document.getElementById("metric-rtf");
  const cardRtf          = document.getElementById("card-rtf");

  /* Secondary metrics */
  const mEncoder         = document.getElementById("metric-encoder");
  const mPrefill         = document.getElementById("metric-prefill");
  const mDecode          = document.getElementById("metric-decode");
  const mThroughput      = document.getElementById("metric-throughput");
  const mDuration        = document.getElementById("metric-duration");
  const mChunks          = document.getElementById("metric-chunks");
  const mTokens          = document.getElementById("metric-tokens");
  const mRssDelta        = document.getElementById("metric-rss-delta");

  /* Process telemetry */
  const procRss     = document.getElementById("proc-rss");
  const procThreads = document.getElementById("proc-threads");
  const procCpu     = document.getElementById("proc-cpu");

  /* Waterfall segments */
  const wfMel     = document.getElementById("wf-mel");
  const wfEncoder = document.getElementById("wf-encoder");
  const wfPrefill = document.getElementById("wf-prefill");
  const wfDecode  = document.getElementById("wf-decode");

  /* ── State ────────────────────────────────────────────── */
  let selectedAudioArrayBuffer = null;
  let websocket        = null;
  let audioContext     = null;
  let audioSourceNode  = null;
  let gainNode         = null;
  let isMuted          = true;   // playback is muted by default; ASR still receives the audio
  let timerInterval    = null;
  let healthInterval   = null;
  let callStartTime    = 0;
  let streamInterval   = null;
  let decodedAudioBuffer = null;
  let currentState     = "idle";

  /* ── State machine ────────────────────────────────────── */
  const STATE_CFG = {
    idle:          { label: "IDLE",          cls: "state-idle" },
    connecting:    { label: "CONNECTING…",   cls: "state-connecting" },
    buffering:     { label: "BUFFERING",     cls: "state-buffering" },
    transcribing:  { label: "TRANSCRIBING",  cls: "state-transcribing" },
    completed:     { label: "COMPLETED",     cls: "state-completed" },
    error:         { label: "ERROR",         cls: "state-error" },
  };

  function setState(name) {
    currentState = name;
    const cfg = STATE_CFG[name] || STATE_CFG.idle;
    stateLabel.textContent = cfg.label;
    streamStatePill.className = `stream-state-pill ${cfg.cls}`;
  }

  /* ── Active model / stream mode ───────────────────────── */
  const STREAM_MODE_LABEL = {
    sliding_window: "sliding window",
    vad_utterance:  "VAD utterances",
  };

  function showActiveModel(data) {
    if (titleDesc && data.active_model && data.active_model !== "(not loaded)") {
      titleDesc.textContent = `${data.active_model} · ${data.backend} · CPU Inference`;
    }
    if (streamModeLabel && data.stream_mode) {
      streamModeLabel.textContent = STREAM_MODE_LABEL[data.stream_mode] || data.stream_mode;
    }
  }

  /* Confirmed text in the normal colour, tentative (may still change) dimmed. */
  function renderTranscript(committed, tentative) {
    transcriptBox.textContent = "";
    if (committed) {
      const c = document.createElement("span");
      c.className = "committed";
      c.textContent = committed;
      transcriptBox.appendChild(c);
    }
    if (tentative) {
      const t = document.createElement("span");
      t.className = "tentative";
      /* CJK text is written without spaces; everything else needs one. */
      const wide = /[\u3000-\u303f\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]/;
      const needSpace = committed && !wide.test(committed.slice(-1)) && !wide.test(tentative[0]);
      t.textContent = (needSpace ? " " : "") + tentative;
      transcriptBox.appendChild(t);
    }
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
  }

  /* Process CPU% uses the per-core scale (like `top`): 100% = ONE core fully busy, so a
     multi-threaded model can exceed 100%. The value stays raw; a compact "≈ N cores" tag
     is added next to it and the full explanation lives in the (i) tooltip. */
  const procCpuSub = document.getElementById("proc-cpu-sub");
  function showCpu(data) {
    if (data.cpu_percent === undefined) return;
    const pct = Math.round(data.cpu_percent);
    procCpu.textContent = `${pct}%`;
    if (procCpuSub) {
      procCpuSub.textContent = pct > 100 ? `≈ ${(data.cpu_percent / 100).toFixed(1)} cores` : "";
    }
    const info = document.getElementById("proc-cpu-info");
    if (info && data.cpu_cores) {
      info.dataset.tooltip = "Per-core scale: 100% = one CPU core fully busy. "
        + "The model runs on several cores at once, so values above 100% are normal "
        + `(max ${data.cpu_cores * 100}% on this ${data.cpu_cores}-core machine).`;
    }
  }

  /* ── Health polling ───────────────────────────────────── */
  function checkHealth() {
    fetch("/api/health")
      .then(r => r.json())
      .then(data => {
        showActiveModel(data);
        if (data.model_ready) {
          modelStatusText.textContent = "MODEL: READY";
          modelStatusText.style.background = "rgba(16,185,129,0.2)";
          modelStatusText.style.color = "#10b981";
          modelStatusText.style.borderColor = "rgba(16,185,129,0.3)";
        } else {
          modelStatusText.textContent = "MODEL: LOADING…";
          setTimeout(checkHealth, 2000);
        }
        /* Process telemetry */
        if (data.rss_mb !== undefined)  procRss.textContent     = `${data.rss_mb} MB`;
        if (data.num_threads !== undefined) procThreads.textContent = data.num_threads;
        showCpu(data);
      })
      .catch(() => setTimeout(checkHealth, 3000));
  }
  checkHealth();

  /* Poll health for process telemetry while call is active */
  function pollTelemetry() {
    fetch("/api/health", { cache: "no-store" }).then(r => r.json()).then(data => {
      if (data.rss_mb !== undefined)  procRss.textContent     = `${data.rss_mb} MB`;
      if (data.num_threads !== undefined) procThreads.textContent = data.num_threads;
      showCpu(data);
    }).catch(() => {});
  }
  function startHealthPolling() {
    if (healthInterval) clearInterval(healthInterval);
    pollTelemetry();                                   // first reading immediately
    healthInterval = setInterval(pollTelemetry, 1000); // then every second
  }
  function stopHealthPolling() {
    if (healthInterval) { clearInterval(healthInterval); healthInterval = null; }
  }

  /* ── Sample files ─────────────────────────────────────── */
  fetch("/api/samples")
    .then(r => r.json())
    .then(data => {
      sampleButtonsContainer.innerHTML = "";
      if (!data.samples || data.samples.length === 0) {
        sampleButtonsContainer.innerHTML = "<p class='loading-text'>No sample files found.</p>";
        return;
      }
      const groups = {};
      data.samples.forEach(s => {
        const g = s.language_name || "Other";
        if (!groups[g]) groups[g] = [];
        groups[g].push(s);
      });
      Object.keys(groups).forEach(groupName => {
        const hdr = document.createElement("div");
        hdr.className = "lang-group-header";
        hdr.textContent = `🌐 ${groupName}`;
        sampleButtonsContainer.appendChild(hdr);
        groups[groupName].forEach(sample => {
          const btn = document.createElement("button");
          btn.className = "sample-btn";
          btn.innerHTML = `<span class="lang-badge">${sample.language_code.toUpperCase()}</span>🎵 ${sample.name} (${(sample.size_bytes / 1024).toFixed(0)} KB)`;
          btn.onclick = () => {
            document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("selected"));
            btn.classList.add("selected");
            if (sample.language_code && sample.language_code !== "auto") {
              languageSelect.value = sample.language_code;
            }
            loadSampleAudio(sample.path, sample.name);
          };
          sampleButtonsContainer.appendChild(btn);
        });
      });
    });

  function loadSampleAudio(path, name) {
    fileInfo.textContent = `Loading ${name}…`;
    fetch(`/api/samples/${path}`)
      .then(r => r.arrayBuffer())
      .then(buf => {
        selectedAudioArrayBuffer = buf;
        fileInfo.textContent = `Selected: ${name}`;
        startBtn.disabled = false;
      });
  }

  audioFileInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (!file) return;
    document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("selected"));
    fileInfo.textContent = `Selected: ${file.name}`;
    const reader = new FileReader();
    reader.onload = evt => {
      selectedAudioArrayBuffer = evt.target.result;
      startBtn.disabled = false;
    };
    reader.readAsArrayBuffer(file);
  });

  /* ── Control handlers ─────────────────────────────────── */
  startBtn.onclick = startCallLeg;
  hangupBtn.onclick = endCallLeg;
  resetBtn.onclick = resetUI;

  /* ── Playback mute (default: muted) ───────────────────── */
  const ICON_MUTED = `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M2 6v4h3l4 3V3L5 6H2z"/><path d="M11 5.5l4 5M15 5.5l-4 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" fill="none"/></svg>`;
  const ICON_ON    = `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M2 6v4h3l4 3V3L5 6H2z"/><path d="M11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.7a6 6 0 0 1 0 8.6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" fill="none"/></svg>`;

  function applyMute() {
    if (gainNode) gainNode.gain.value = isMuted ? 0 : 1;
    muteBtn.classList.toggle("is-muted", isMuted);
    muteBtn.setAttribute("aria-pressed", String(isMuted));
    muteLabel.textContent = isMuted ? "Sound: Off (muted)" : "Sound: On";
    muteIcon.innerHTML = isMuted ? ICON_MUTED : ICON_ON;   // static markup, no user data
  }
  muteBtn.onclick = () => { isMuted = !isMuted; applyMute(); };
  applyMute();

  /* ── Metric helpers ───────────────────────────────────── */
  function setMetric(el, text) {
    if (!el) return;
    el.textContent = text;
    el.classList.remove("flash-update");
    void el.offsetWidth; // reflow to restart animation
    el.classList.add("flash-update");
  }

  function applyRtfClass(rtfVal) {
    if (!cardRtf) return;
    cardRtf.classList.remove("rtf-good", "rtf-warn", "rtf-bad");
    if      (rtfVal < 0.8)  cardRtf.classList.add("rtf-good");
    else if (rtfVal < 1.2)  cardRtf.classList.add("rtf-warn");
    else                    cardRtf.classList.add("rtf-bad");
  }

  function updateWaterfall(mel_ms, encoder_ms, prefill_ms, decode_ms) {
    const total = (mel_ms || 0) + (encoder_ms || 0) + (prefill_ms || 0) + (decode_ms || 0);
    if (total <= 0) return;
    const pct = v => `${((v || 0) / total * 100).toFixed(1)}%`;
    wfMel.style.flex     = `0 0 ${pct(mel_ms)}`;
    wfEncoder.style.flex = `0 0 ${pct(encoder_ms)}`;
    wfPrefill.style.flex = `0 0 ${pct(prefill_ms)}`;
    wfDecode.style.flex  = `0 0 ${pct(decode_ms)}`;
  }

  function applyMetrics(m) {
    if (!m) return;
    if (m.pipeline_latency_ms != null) setMetric(mPipelineLatency, `${m.pipeline_latency_ms.toFixed(0)} ms`);
    if (m.ttft_ms != null)             setMetric(mTtft,            `${m.ttft_ms.toFixed(0)} ms`);
    if (m.infer_latency_ms != null)    setMetric(mInferLatency,    `${m.infer_latency_ms.toFixed(0)} ms`);
    if (m.rtf != null) {
      setMetric(mRtf, `${m.rtf.toFixed(3)}x`);
      applyRtfClass(m.rtf);
    }
    if (m.encoder_ms != null)   setMetric(mEncoder,    `${m.encoder_ms.toFixed(0)} ms`);
    if (m.prefill_ms != null)   setMetric(mPrefill,    `${m.prefill_ms.toFixed(0)} ms`);
    if (m.decode_ms != null)    setMetric(mDecode,     `${m.decode_ms.toFixed(0)} ms`);
    if (m.throughput_tps != null) setMetric(mThroughput, `${m.throughput_tps} tok/s`);
    if (m.chunks_received != null) setMetric(mChunks,  `${m.chunks_received}`);
    if (m.tokens_generated != null) setMetric(mTokens, `${m.tokens_generated}`);
    if (m.rss_delta_mb != null) setMetric(mRssDelta,   `${m.rss_delta_mb} MB`);

    if (m.mel_ms != null || m.encoder_ms != null) {
      updateWaterfall(m.mel_ms, m.encoder_ms, m.prefill_ms, m.decode_ms);
    }

    /* Transcript meta */
    if (m.audio_duration_s != null && m.infer_latency_ms != null) {
      transcriptMeta.textContent =
        `audio: ${m.audio_duration_s.toFixed(2)}s · ` +
        `infer: ${(m.infer_latency_ms/1000).toFixed(2)}s · ` +
        (m.inference_passes != null ? `pass #${m.inference_passes}` : "") +
        (m.window_start_s != null ? ` · window ${m.window_start_s.toFixed(1)}s→${(m.window_start_s + m.window_s).toFixed(1)}s` : "");
    }
  }

  function resetMetrics() {
    [mPipelineLatency, mTtft, mInferLatency].forEach(el => { if (el) el.textContent = "— ms"; });
    if (mRtf)       mRtf.textContent = "—x";
    if (mEncoder)   mEncoder.textContent = "— ms";
    if (mPrefill)   mPrefill.textContent = "— ms";
    if (mDecode)    mDecode.textContent = "— ms";
    if (mThroughput) mThroughput.textContent = "—";
    if (mDuration)  mDuration.textContent = "00:00";
    if (mChunks)    mChunks.textContent = "0";
    if (mTokens)    mTokens.textContent = "—";
    if (mRssDelta)  mRssDelta.textContent = "— MB";
    if (cardRtf)    cardRtf.classList.remove("rtf-good", "rtf-warn", "rtf-bad");
    if (wfMel)      wfMel.style.flex = "0 0 0%";
    if (wfEncoder)  wfEncoder.style.flex = "0 0 0%";
    if (wfPrefill)  wfPrefill.style.flex = "0 0 0%";
    if (wfDecode)   wfDecode.style.flex = "0 0 0%";
    if (transcriptMeta) transcriptMeta.textContent = "";
  }

  /* ── Timer ────────────────────────────────────────────── */
  function updateCallTimer() {
    const elapsedSec = Math.floor((Date.now() - callStartTime) / 1000);
    const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
    const secs = String(elapsedSec % 60).padStart(2, "0");
    mDuration.textContent = `${mins}:${secs}`;
  }

  /* ── Call leg ─────────────────────────────────────────── */
  function startCallLeg() {
    if (!selectedAudioArrayBuffer) return;

    setState("connecting");
    startBtn.disabled  = true;
    hangupBtn.disabled = false;

    transcriptBox.innerHTML = "<span class='placeholder'>Connecting call leg &amp; warming pipeline…</span>";
    resetMetrics();

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    audioContext.decodeAudioData(
      selectedAudioArrayBuffer.slice(0),
      buf => {
        decodedAudioBuffer = buf;
        connectWebSocket();
      },
      err => {
        setState("error");
        transcriptBox.innerHTML =
          "<span class='placeholder'>Could not decode this file — the UI accepts Linear PCM WAV only.</span>";
        fileInfo.textContent = `Decode failed: ${err && err.message ? err.message : "unsupported format"}`;
        startBtn.disabled  = false;
        hangupBtn.disabled = true;
      },
    );
  }

  function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    websocket = new WebSocket(`${protocol}//${location.host}/ws/call-stream`);
    websocket.binaryType = "arraybuffer";

    const targetLang = languageSelect ? languageSelect.value : "";

    websocket.onopen = () => {
      callStartTime = Date.now(); // clientT0
      websocket.send(JSON.stringify({
        type: "start_call",
        language: targetLang,
        audio: { sample_rate: 16000, channels: 1, encoding: "pcm_s16le" },
      }));
    };

    websocket.onmessage = event => {
      const data = JSON.parse(event.data);

      if (data.type === "call_ready") {
        if (streamModeLabel && data.stream_mode) {
          streamModeLabel.textContent = STREAM_MODE_LABEL[data.stream_mode] || data.stream_mode;
        }
        setState("buffering");
        transcriptBox.innerHTML = "<span class='placeholder'>Call active — streaming audio chunks…</span>";
        timerInterval = setInterval(updateCallTimer, 1000);
        startHealthPolling();
        if (decodedAudioBuffer) streamAudioBuffer(decodedAudioBuffer);

      } else if (data.type === "chunk_ack") {
        setMetric(mChunks, data.chunks_received);

      } else if (data.type === "transcript_delta") {
        if (currentState !== "transcribing") setState("transcribing");
        if (data.committed_text !== undefined) {
          renderTranscript(data.committed_text, data.tentative_text || "");
        } else {
          transcriptBox.textContent = data.full_text || "";
        }
        applyMetrics(data.metrics);

      } else if (data.type === "call_ended") {
        setState("completed");
        if (data.final_text) renderTranscript(data.final_text, "");
        else transcriptBox.textContent = "Call completed.";
        if (data.metrics) applyMetrics(data.metrics);
        _teardown();

      } else if (data.type === "error") {
        setState("error");
        transcriptBox.innerHTML = `<span class='placeholder'>${data.message || "Server rejected the call leg."}</span>`;
        _teardown();
      }
    };

    websocket.onerror = () => {
      setState("error");
      _teardown();
    };

    websocket.onclose = () => {
      if (currentState !== "completed" && currentState !== "idle") {
        setState("completed");
      }
      _teardown();
    };
  }

  /* ── Audio streaming ──────────────────────────────────── */
  /* Average every channel instead of keeping channel 0: on dual-channel call
     recordings each leg can sit on its own channel, so dropping one loses a
     speaker. decodeAudioData has already resampled to the 16 kHz baseline. */
  function downmixToMono(audioBuffer) {
    const channels = audioBuffer.numberOfChannels;
    if (channels === 1) return audioBuffer.getChannelData(0);

    const frames = audioBuffer.length;
    const mono   = new Float32Array(frames);
    for (let c = 0; c < channels; c++) {
      const data = audioBuffer.getChannelData(c);
      for (let i = 0; i < frames; i++) mono[i] += data[i];
    }
    for (let i = 0; i < frames; i++) mono[i] /= channels;
    return mono;
  }

  function streamAudioBuffer(audioBuffer) {
    if (audioContext) {
      if (audioContext.state === "suspended") audioContext.resume();
      audioSourceNode = audioContext.createBufferSource();
      audioSourceNode.buffer = audioBuffer;
      gainNode = audioContext.createGain();
      gainNode.gain.value = isMuted ? 0 : 1;
      audioSourceNode.connect(gainNode);
      gainNode.connect(audioContext.destination);
      audioSourceNode.start(0);
    }

    const channelData = downmixToMono(audioBuffer); // 16 kHz float32, single channel
    const chunkSize   = 8000;  // 0.5s at 16 kHz
    let offset        = 0;

    sendNextChunk();
    streamInterval = setInterval(sendNextChunk, 500);

    function sendNextChunk() {
      if (offset >= channelData.length || !websocket || websocket.readyState !== WebSocket.OPEN) {
        clearInterval(streamInterval);
        streamInterval = null;
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(JSON.stringify({ type: "end_call" }));
        }
        return;
      }
      const chunk   = channelData.subarray(offset, offset + chunkSize);
      const int16   = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) {
        int16[i] = Math.max(-1, Math.min(1, chunk[i])) * 0x7FFF;
      }
      websocket.send(int16.buffer);
      offset += chunkSize;
    }
  }

  /* ── Tear-down ────────────────────────────────────────── */
  function _teardown() {
    if (streamInterval) { clearInterval(streamInterval); streamInterval = null; }
    if (timerInterval)  { clearInterval(timerInterval);  timerInterval  = null; }
    stopHealthPolling();
    if (audioSourceNode) {
      try { audioSourceNode.stop(); } catch (_) {}
      audioSourceNode.disconnect();
      audioSourceNode = null;
    }
    if (gainNode) { try { gainNode.disconnect(); } catch (_) {} gainNode = null; }
    if (websocket) { try { websocket.close(); } catch (_) {} websocket = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    startBtn.disabled  = false;
    hangupBtn.disabled = true;
  }

  function endCallLeg() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({ type: "end_call" }));
    }
    setState("completed");
    _teardown();
  }

  function resetUI() {
    endCallLeg();
    setState("idle");
    resetMetrics();
    transcriptBox.innerHTML = "<span class='placeholder'>Recognition output will stream live here during the call leg…</span>";
    document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("selected"));
    selectedAudioArrayBuffer = null;
    decodedAudioBuffer       = null;
    if (fileInfo) fileInfo.textContent = "No file selected";
    startBtn.disabled = true;
  }
});
