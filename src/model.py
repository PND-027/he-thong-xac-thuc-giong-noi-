"""
Module trích xuất đặc trưng người nói (Speaker Embedding) bằng ECAPA-TDNN.

Mô hình: speechbrain/spkrec-ecapa-voxceleb (huấn luyện sẵn trên VoxCeleb 1+2,
hơn 7.000 người nói). Đầu ra: vector x-vector 192 chiều, chuẩn hóa L2.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from . import config

log = logging.getLogger(__name__)


class SpeakerEncoder:
    """Bọc mô hình ECAPA-TDNN pretrained. Dùng `SpeakerEncoder.get()` để lấy singleton."""

    _instance: Optional["SpeakerEncoder"] = None
    _lock = threading.Lock()

    def __init__(self, source: str = config.ECAPA_SOURCE, device: str = "cpu"):
        import torch

        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.device = device
        self.source = source
        t0 = time.perf_counter()
        self.model = self._load(source, device)
        self.model.eval()
        log.info("Đã tải ECAPA-TDNN trong %.1fs", time.perf_counter() - t0)

    @staticmethod
    def _load(source: str, device: str):
        """Tải mô hình từ HuggingFace (lần đầu) hoặc từ cache cục bộ."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # SpeechBrain < 1.0
            from speechbrain.pretrained import EncoderClassifier  # type: ignore

        savedir = config.MODEL_DIR / source.split("/")[-1]
        kwargs = dict(source=source, savedir=str(savedir), run_opts={"device": device})
        # Windows không cho tạo symlink nếu chưa bật Developer Mode -> copy file
        try:
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        try:
            return EncoderClassifier.from_hparams(**kwargs)
        except TypeError:
            kwargs.pop("local_strategy", None)
            return EncoderClassifier.from_hparams(**kwargs)

    @classmethod
    def get(cls) -> "SpeakerEncoder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """
        Trích xuất embedding từ tín hiệu 16 kHz mono (float32).

        Returns:
            np.ndarray shape (192,), đã chuẩn hóa ||E||_2 = 1.
        """
        import torch

        if wav.size == 0:
            raise ValueError("Tín hiệu rỗng, không thể trích xuất đặc trưng.")
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            emb = self.model.encode_batch(x.to(self.device))  # (1, 1, 192)
        emb = emb.squeeze().cpu().numpy().astype(np.float32)
        return l2_normalize(emb)


def l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Chuẩn hóa L2: E / ||E||_2."""
    return (v / (np.linalg.norm(v) + eps)).astype(np.float32)


def extract_embedding(wav: np.ndarray) -> np.ndarray:
    """Hàm tiện ích: trích xuất embedding bằng encoder singleton."""
    return SpeakerEncoder.get().embed(wav)
