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

  let selectedAudioArrayBuffer = null;
  let websocket = null;
  let audioContext = null;
  let timerInterval = null;
  let callStartTime = 0;
  let streamInterval = null;

  // Fetch available sample audio files from backend
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
        sampleButtonsContainer.innerHTML = "<p class='loading-text'>No sample files found.</p>";
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

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    
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

    streamInterval = setInterval(() => {
      if (offset >= channelData.length || !websocket || websocket.readyState !== WebSocket.OPEN) {
        clearInterval(streamInterval);
        streamInterval = null;
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
    }, 500); // 0.5s real-time pacing
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
    if (websocket) { websocket.close(); websocket = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    startBtn.disabled = false;
    hangupBtn.disabled = true;
    statusDot.classList.remove("active");
    statusText.textContent = "COMPLETED";
  }
});
