"""
Kiểm tra nhanh môi trường (Bước 1.3 trong PROJECT_PLAN):
  - Tải Silero VAD và ECAPA-TDNN về máy, chạy thử trên CPU.
  - Đo thời gian trích xuất embedding.

Chạy:  python check_setup.py
"""
import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> int:
    import torch

    print(f"Python  : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    from src import audio
    from src.model import SpeakerEncoder

    # Tín hiệu giả lập 3s: tiếng "nguyên âm" tổng hợp có điều biến + khoảng lặng
    sr = 16_000
    t = np.arange(int(3 * sr)) / sr
    f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t)
    voiced = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / sr) / k for k in range(1, 15))
    envelope = (np.sin(2 * np.pi * 2 * t) > -0.3).astype(float)
    x = np.r_[np.zeros(sr // 2), 0.3 * voiced * envelope, np.zeros(sr // 2)].astype(np.float32)
    x += 0.002 * np.random.randn(len(x)).astype(np.float32)

    print(f"Silero VAD: {'OK' if audio._get_silero() is not None else 'KHÔNG KHẢ DỤNG (dùng VAD năng lượng)'}")
    r = audio.process(x, sr, check_length=False)
    print(f"VAD backend={r.vad_backend} | tổng {r.duration:.2f}s -> giọng nói {r.speech_duration:.2f}s")

    t0 = time.perf_counter()
    enc = SpeakerEncoder.get()
    print(f"Tải ECAPA-TDNN: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    emb = enc.embed(x)
    dt = time.perf_counter() - t0
    print(f"Embedding shape={emb.shape} | ||E||={np.linalg.norm(emb):.4f} | thời gian={dt*1000:.0f} ms")
    assert emb.shape == (192,)
    print("\n✅ Môi trường sẵn sàng!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
