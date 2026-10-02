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

  const modelStatusText = document.getElementById("model-status-text");

  let selectedAudioArrayBuffer = null;
  let websocket = null;
  let audioContext = null;
  let audioSourceNode = null;
  let timerInterval = null;
  let callStartTime = 0;
  let streamInterval = null;
  let decodedAudioBuffer = null;

  // Poll backend for model readiness status
  function checkModelHealth() {
    fetch("/api/health")
      .then(res => res.json())
      .then(data => {
        if (data.model_ready && modelStatusText) {
          modelStatusText.textContent = "MODEL: READY";
          modelStatusText.style.background = "#059669"; // Emerald green
        } else if (modelStatusText) {
          modelStatusText.textContent = "MODEL: LOADING...";
          setTimeout(checkModelHealth, 2000);
        }
      })
      .catch(() => {
        if (modelStatusText) setTimeout(checkModelHealth, 2000);
      });
  }
  checkModelHealth();

  const languageSelect = document.getElementById("language-select");

  // Fetch available sample audio files from backend
  fetch("/api/samples")
    .then(res => res.json())
    .then(data => {
      sampleButtonsContainer.innerHTML = "";
      if (data.samples && data.samples.length > 0) {
        // Group samples by language_name
        const groups = {};
        data.samples.forEach(sample => {
          const groupName = sample.language_name || "Other";
          if (!groups[groupName]) groups[groupName] = [];
          groups[groupName].push(sample);
        });

        Object.keys(groups).forEach(groupName => {
          const groupHeader = document.createElement("div");
          groupHeader.className = "lang-group-header";
          groupHeader.textContent = `🌐 ${groupName}`;
          sampleButtonsContainer.appendChild(groupHeader);

          groups[groupName].forEach(sample => {
            const btn = document.createElement("button");
            btn.className = "sample-btn";
            btn.innerHTML = `<span class="lang-badge">${sample.language_code.toUpperCase()}</span> 🎵 ${sample.name} (${(sample.size_bytes / 1024).toFixed(0)} KB)`;
            btn.onclick = (e) => {
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
      } else {
        sampleButtonsContainer.innerHTML = "<p class='loading-text'>No sample files found.</p>";
      }
    });

  function loadSampleAudio(samplePath, filename) {
    fileInfo.textContent = `Loading ${filename}...`;
    fetch(`/api/samples/${samplePath}`)
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
      document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("selected"));
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
    statusText.textContent = "CONNECTING...";
    transcriptBox.innerHTML = "<span class='placeholder'>Connecting call leg & warming pipeline...</span>";

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    
    // Pre-decode audio buffer before starting streaming
    audioContext.decodeAudioData(selectedAudioArrayBuffer.slice(0), (audioBuffer) => {
      decodedAudioBuffer = audioBuffer;
      connectWebSocketAndStart();
    });
  }

  function connectWebSocketAndStart() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    websocket = new WebSocket(`${protocol}//${location.host}/ws/call-stream`);
    websocket.binaryType = "arraybuffer";

    const targetLang = languageSelect ? languageSelect.value : "";

    websocket.onopen = () => {
      websocket.send(JSON.stringify({ type: "start_call", language: targetLang }));
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === "call_ready") {
        statusText.textContent = "CALL IN PROGRESS";
        transcriptBox.innerHTML = "<span class='placeholder'>Call active. Streaming audio...</span>";
        callStartTime = Date.now();
        timerInterval = setInterval(updateCallTimer, 1000);
        
        if (decodedAudioBuffer) {
          streamAudioBuffer(decodedAudioBuffer);
        }
      } else if (data.type === "chunk_ack") {
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
    // Start real-time audio playback through speakers synchronously with chunk streaming
    if (audioContext) {
      if (audioContext.state === "suspended") {
        audioContext.resume();
      }
      audioSourceNode = audioContext.createBufferSource();
      audioSourceNode.buffer = audioBuffer;
      audioSourceNode.connect(audioContext.destination);
      audioSourceNode.start(0);
    }

    const channelData = audioBuffer.getChannelData(0); // 16kHz float32
    const chunkSize = 8000; // 0.5s chunk at 16kHz (8000 samples)
    let offset = 0;

    // Send first chunk immediately
    sendNextChunk();

    streamInterval = setInterval(() => {
      sendNextChunk();
    }, 500); // 0.5s real-time pacing

    function sendNextChunk() {
      if (offset >= channelData.length || !websocket || websocket.readyState !== WebSocket.OPEN) {
        if (streamInterval) {
          clearInterval(streamInterval);
          streamInterval = null;
        }
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(JSON.stringify({ type: "end_call" }));
        }
        return;
      }

      const chunkFloat32 = channelData.subarray(offset, offset + chunkSize);
      const int16Buffer = new Int16Array(chunkFloat32.length);
      for (let i = 0; i < chunkFloat32.length; i++) {
        int16Buffer[i] = Math.max(-1, Math.min(1, chunkFloat32[i])) * 0x7FFF;
      }

      websocket.send(int16Buffer.buffer);
      offset += chunkSize;
    }
  }

  function updateCallTimer() {
    const elapsedSec = Math.floor((Date.now() - callStartTime) / 1000);
    const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
    const secs = String(elapsedSec % 60).padStart(2, '0');
    metricDuration.textContent = `${mins}:${secs}`;
  }

  function endCallLeg() {
    if (streamInterval) { clearInterval(streamInterval); streamInterval = null; }
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    if (audioSourceNode) {
      try { audioSourceNode.stop(); } catch (e) {}
      audioSourceNode.disconnect();
      audioSourceNode = null;
    }
    if (websocket) { websocket.close(); websocket = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    startBtn.disabled = false;
    hangupBtn.disabled = true;
    statusDot.classList.remove("active");
    statusText.textContent = "COMPLETED";
  }
});
