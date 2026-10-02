import numpy as np

# ── Constants ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128


# ── VAD-based Long Audio Chunking ────────────────────────────────────
SILENCE_THRESHOLD_DB = -40
SILENCE_HOP_SEC = 0.1

# ── Mel Spectrogram (Whisper-compatible, no PyTorch) ────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio file as mono 16kHz float32."""
    import librosa
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def compute_mel_spectrogram(wav: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """
    Compute log-mel spectrogram (Whisper-compatible).

    Args:
        wav: [T] float32 audio
        mel_filters: [n_mels, n_fft//2+1] mel filterbank

    Returns:
        mel: [n_mels, n_frames] float32
    """
    import librosa

    # STFT
    stft = librosa.stft(
        wav, n_fft=N_FFT, hop_length=HOP_LENGTH,
        window="hann", center=True, pad_mode="reflect",
    )
    magnitudes = np.abs(stft) ** 2

    # Apply mel filterbank
    mel_spec = mel_filters @ magnitudes

    # Log scale (Whisper-style)
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.astype(np.float32)


def get_mel_filters() -> np.ndarray:
    """Get Whisper-compatible mel filterbank using librosa."""
    import librosa
    mel_filters = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
        fmin=0, fmax=SAMPLE_RATE // 2, norm="slaney", htk=False,
    )
    return mel_filters.astype(np.float32)


def get_feat_extract_output_lengths(input_lengths: np.ndarray) -> np.ndarray:
    """Compute output lengths after 3x stride-2 convolution."""
    lengths = input_lengths
    for _ in range(3):
        lengths = (lengths - 1) // 2 + 1
    return lengths


def find_silence_split_points(wav: np.ndarray, target_sec: int = 30) -> list:
    """Find sample indices where audio can be split at silence boundaries.

    Uses RMS energy to detect silence — no external VAD model needed.
    Splits long audio into chunks at the nearest silent frame.

    Args:
        wav: Audio samples (float32, 16kHz mono).
        target_sec: Target chunk length in seconds (default 30).
                     Min = target/2, Max = target*1.5.
    """
    import librosa

    min_sec = target_sec // 2
    max_sec = int(target_sec * 1.5)

    total_samples = len(wav)
    if total_samples <= max_sec * SAMPLE_RATE:
        return []

    hop_samples = int(SILENCE_HOP_SEC * SAMPLE_RATE)
    rms = librosa.feature.rms(
        y=wav, frame_length=hop_samples * 2, hop_length=hop_samples,
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    is_silent = rms_db < SILENCE_THRESHOLD_DB

    split_points = []
    cursor = 0

    while cursor + max_sec * SAMPLE_RATE < total_samples:
        search_start_sec = max(0, cursor / SAMPLE_RATE + min_sec)
        search_end_sec = cursor / SAMPLE_RATE + max_sec
        target_abs_sec = cursor / SAMPLE_RATE + target_sec

        frame_start = int(search_start_sec / SILENCE_HOP_SEC)
        frame_end = min(int(search_end_sec / SILENCE_HOP_SEC), len(is_silent))
        frame_target = int(target_abs_sec / SILENCE_HOP_SEC)

        silent_frames = np.where(is_silent[frame_start:frame_end])[0] + frame_start

        if len(silent_frames) > 0:
            best_idx = int(np.argmin(np.abs(silent_frames - frame_target)))
            split_frame = silent_frames[best_idx]
            split_sample = int(split_frame * hop_samples)
        else:
            split_sample = int(target_abs_sec * SAMPLE_RATE)

        split_sample = min(split_sample, total_samples)
        split_points.append(split_sample)
        cursor = split_sample

    return split_points
