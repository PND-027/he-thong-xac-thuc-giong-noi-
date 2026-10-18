"""
Module xử lý âm thanh & VAD (Voice Activity Detection).

Pipeline:
    Tín hiệu thô (bất kỳ sr, mono/stereo, int16/float)
      -> Chuyển Mono
      -> Resample về 16 kHz (polyphase filter, chống aliasing)
      -> Loại bỏ thành phần DC
      -> Peak Normalization về [-1, 1]
      -> Silero VAD (fallback: VAD năng lượng) cắt bỏ khoảng lặng
      -> Kiểm tra thời lượng giọng nói >= MIN_SPEECH_SEC
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import config

log = logging.getLogger(__name__)


class SpeechTooShortError(ValueError):
    """Ném ra khi thời lượng giọng nói sau VAD nhỏ hơn ngưỡng tối thiểu."""

    def __init__(self, speech_sec: float, min_sec: float = config.MIN_SPEECH_SEC):
        self.speech_sec = speech_sec
        self.min_sec = min_sec
        super().__init__(
            f"Câu nói quá ngắn: chỉ phát hiện {speech_sec:.2f}s giọng nói "
            f"(yêu cầu ≥ {min_sec:.1f}s). Hãy nói lại dài hơn."
        )


@dataclass
class AudioResult:
    """Kết quả của toàn bộ pipeline tiền xử lý."""

    raw: np.ndarray                    # Tín hiệu sau chuẩn hóa (16k, mono, peak-norm)
    speech: np.ndarray                 # Tín hiệu chỉ còn giọng nói (sau VAD)
    segments: List[Tuple[int, int]]    # Các đoạn speech [start, end) theo mẫu
    sample_rate: int = config.SAMPLE_RATE
    vad_backend: str = "silero"
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.raw) / self.sample_rate

    @property
    def speech_duration(self) -> float:
        return len(self.speech) / self.sample_rate

    @property
    def speech_ratio(self) -> float:
        return self.speech_duration / self.duration if self.duration > 0 else 0.0


# =============================================================================
# 1. Đọc & chuẩn hóa tín hiệu
# =============================================================================
def to_float32(data: np.ndarray) -> np.ndarray:
    """Chuyển mọi kiểu dữ liệu PCM (int16, int32, uint8, float64) về float32 [-1, 1]."""
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if info.min == 0:  # uint8
            data = (data.astype(np.float32) - (info.max + 1) / 2) / ((info.max + 1) / 2)
        else:
            data = data.astype(np.float32) / max(abs(info.min), info.max)
    return data.astype(np.float32, copy=False)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Trộn các kênh về Mono bằng trung bình cộng."""
    if data.ndim == 1:
        return data
    # Gradio / soundfile trả về (n_samples, n_channels)
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data.mean(axis=1)


def resample(data: np.ndarray, orig_sr: int, target_sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Resample bằng bộ lọc đa pha (polyphase) – có sẵn lọc thông thấp chống aliasing."""
    if orig_sr == target_sr:
        return data
    frac = Fraction(target_sr, orig_sr).limit_denominator(1000)
    return resample_poly(data, frac.numerator, frac.denominator).astype(np.float32)


def peak_normalize(data: np.ndarray, target: float = config.PEAK_TARGET) -> np.ndarray:
    """Peak Normalization: x_norm = target * x / max|x|."""
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak < 1e-6:
        return data
    return (data * (target / peak)).astype(np.float32)


def standardize(data: np.ndarray, sr: int) -> np.ndarray:
    """Chuẩn hóa tín hiệu: Mono -> 16 kHz -> bỏ DC -> Peak Normalization."""
    x = to_mono(to_float32(data))
    x = resample(x, sr, config.SAMPLE_RATE)
    x = x - np.mean(x) if x.size else x          # Loại bỏ DC offset
    return peak_normalize(x)


def load_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """Đọc file âm thanh (wav/flac/ogg) trả về (data, sample_rate) gốc."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return data, sr


# =============================================================================
# 2. Voice Activity Detection
# =============================================================================
_silero_model = None
_silero_failed = False


def _get_silero():
    """Tải Silero VAD (lazy, chỉ 1 lần). Trả về None nếu không khả dụng."""
    global _silero_model, _silero_failed
    if _silero_model is not None or _silero_failed:
        return _silero_model
    try:
        import io
        from importlib import resources

        import torch

        # Không dùng silero_vad.load_silero_vad() trực tiếp: torch.jit.load(<path>)
        # lỗi trên Windows khi đường dẫn chứa ký tự Unicode (vd: "xử lý tiếng nói").
        # -> Đọc bytes bằng Python rồi nạp từ bộ nhớ.
        data = resources.files("silero_vad").joinpath("data", "silero_vad.jit").read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _silero_model = torch.jit.load(io.BytesIO(data), map_location="cpu")
        _silero_model.eval()
        log.info("Đã tải Silero VAD.")
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        log.warning("Không tải được Silero VAD (%s) -> dùng VAD năng lượng.", exc)
        _silero_failed = True
    return _silero_model


def silero_vad(x: np.ndarray, sr: int = config.SAMPLE_RATE) -> List[Tuple[int, int]]:
    """Phát hiện các đoạn giọng nói bằng mạng nơ-ron Silero VAD."""
    import torch
    from silero_vad import get_speech_timestamps

    model = _get_silero()
    ts = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        model,
        sampling_rate=sr,
        threshold=config.VAD_THRESHOLD,
        min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
        speech_pad_ms=config.VAD_SPEECH_PAD_MS,
    )
    return [(int(t["start"]), int(t["end"])) for t in ts]


def energy_vad(
    x: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> List[Tuple[int, int]]:
    """
    VAD dựa trên năng lượng ngắn hạn (Short-Time Energy) – phương án dự phòng.

    Ngưỡng thích nghi: max(noise_floor + 12 dB, peak - 45 dB), trong đó
    noise_floor là phân vị 10% năng lượng các khung.
    """
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < frame:
        return []
    n_frames = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hamming(frame)
    energy_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)

    noise_floor = np.percentile(energy_db, 10)
    thr = max(noise_floor + 12.0, energy_db.max() - 45.0)
    active = energy_db > thr

    # Làm mượt: hangover để không cắt vụn giữa các âm tiết
    hang = int(config.VAD_MIN_SILENCE_MS / hop_ms)
    smoothed = active.copy()
    last = -10**9
    for i, a in enumerate(active):
        if a:
            last = i
        elif i - last <= hang:
            smoothed[i] = True

    segments: List[Tuple[int, int]] = []
    pad = int(sr * config.VAD_SPEECH_PAD_MS / 1000)
    min_len = int(sr * config.VAD_MIN_SPEECH_MS / 1000)
    start = None
    for i, a in enumerate(np.append(smoothed, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            s = max(0, start * hop - pad)
            e = min(len(x), (i - 1) * hop + frame + pad)
            if e - s >= min_len:
                segments.append((s, e))
            start = None
    return _merge_segments(segments)


def _merge_segments(segs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for s, e in sorted(segs):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def apply_vad(x: np.ndarray, sr: int = config.SAMPLE_RATE) -> Tuple[np.ndarray, List[Tuple[int, int]], str]:
    """Cắt bỏ khoảng lặng, trả về (speech, segments, backend)."""
    backend = "silero"
    segments: Optional[List[Tuple[int, int]]] = None
    if _get_silero() is not None:
        try:
            segments = silero_vad(x, sr)
        except Exception as exc:  # pragma: no cover
            log.warning("Silero VAD lỗi (%s) -> dùng VAD năng lượng.", exc)
    if segments is None:
        backend = "energy"
        segments = energy_vad(x, sr)

    if not segments:
        return np.zeros(0, dtype=np.float32), [], backend
    speech = np.concatenate([x[s:e] for s, e in segments]).astype(np.float32)
    return speech, segments, backend


# =============================================================================
# 3. Pipeline hoàn chỉnh
# =============================================================================
def process(
    data: np.ndarray,
    sr: int,
    min_speech_sec: float = config.MIN_SPEECH_SEC,
    check_length: bool = True,
) -> AudioResult:
    """Tiền xử lý đầy đủ + VAD. Ném SpeechTooShortError nếu giọng nói quá ngắn."""
    x = standardize(data, sr)
    speech, segments, backend = apply_vad(x)
    result = AudioResult(raw=x, speech=speech, segments=segments, vad_backend=backend)
    if check_length and result.speech_duration < min_speech_sec:
        raise SpeechTooShortError(result.speech_duration, min_speech_sec)
    return result


def process_file(path: str | Path, **kwargs) -> AudioResult:
    data, sr = load_audio(path)
    return process(data, sr, **kwargs)
