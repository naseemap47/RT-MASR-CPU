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
