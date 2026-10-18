"""Các hàm vẽ biểu đồ (dùng chung cho giao diện demo và báo cáo)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import stft  # noqa: E402

from . import config  # noqa: E402

# Bảng màu thống nhất với giao diện (Dark Slate & Tech Blue)
BG = "#090d16"
PANEL = "#111827"
FG = "#cbd5e1"
GRID = "#1e293b"
ACCENT = "#2563eb"
ACCENT2 = "#38bdf8"
OK = "#10b981"
BAD = "#ef4444"


def _style(ax, title: str = ""):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color="white", fontsize=10, loc="left", fontweight="bold")


def _fig(rows: int = 1, height: float = 2.4):
    fig, axes = plt.subplots(rows, 1, figsize=(9, height * rows), facecolor=BG)
    return fig, np.atleast_1d(axes)


def plot_vad(
    raw: np.ndarray,
    speech: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    sr: int = config.SAMPLE_RATE,
    backend: str = "silero",
):
    """Sóng âm trước VAD (tô các đoạn speech) và sau VAD."""
    fig, (a1, a2) = _fig(2, 2.1)
    t = np.arange(len(raw)) / sr
    a1.plot(t, raw, color="#475569", linewidth=0.5)
    for s, e in segments:
        a1.axvspan(s / sr, e / sr, color=ACCENT, alpha=0.22, linewidth=0)
        a1.plot(t[s:e], raw[s:e], color=ACCENT2, linewidth=0.6)
    _style(a1, f"Trước VAD  •  {len(raw)/sr:.2f}s  •  vùng tô = giọng nói ({backend})")
    a1.set_xlim(0, max(t[-1] if len(t) else 1, 1e-3))
    a1.set_ylim(-1.05, 1.05)

    t2 = np.arange(len(speech)) / sr
    a2.plot(t2, speech, color=OK, linewidth=0.5)
    _style(a2, f"Sau VAD  •  {len(speech)/sr:.2f}s giọng nói thực")
    a2.set_xlim(0, max(t2[-1] if len(t2) else 1, 1e-3))
    a2.set_ylim(-1.05, 1.05)
    a2.set_xlabel("Thời gian (s)")
    fig.tight_layout()
    return fig


def plot_spectrogram(x: np.ndarray, sr: int = config.SAMPLE_RATE, title: str = "Phổ (Spectrogram)"):
    """Spectrogram STFT: cửa sổ Hann 25 ms, bước nhảy 10 ms, thang dB."""
    fig, (ax,) = _fig(1, 2.8)
    nper = int(0.025 * sr)
    f, t, Z = stft(x, fs=sr, window="hann", nperseg=nper, noverlap=nper - int(0.010 * sr), nfft=512)
    S = 20 * np.log10(np.abs(Z) + 1e-8)
    S = np.clip(S, S.max() - 80, None)
    im = ax.pcolormesh(t, f / 1000, S, shading="auto", cmap="Blues_r")
    _style(ax, title)
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Tần số (kHz)")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.ax.tick_params(colors=FG, labelsize=7)
    cb.set_label("dB", color=FG)
    fig.tight_layout()
    return fig


def plot_ranking(ranking: List[Tuple[str, float]], threshold: float, top_k: int = 8):
    """Biểu đồ thanh điểm Cosine của các ứng viên (Nhận dạng 1:N)."""
    data = ranking[:top_k][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * max(len(data), 2) + 0.9), facecolor=BG)
    if not data:
        _style(ax, "Chưa có dữ liệu")
        return fig
    names = [n for n, _ in data]
    scores = [s for _, s in data]
    colors = [OK if s >= threshold else "#334155" for s in scores]
    if data[-1][1] >= threshold:
        colors[-1] = ACCENT2
    bars = ax.barh(names, scores, color=colors, height=0.6)
    ax.axvline(threshold, color=BAD, linestyle="--", linewidth=1.2, label=f"Ngưỡng τ = {threshold:.2f}")
    for b, s in zip(bars, scores):
        ax.text(max(s, 0) + 0.01, b.get_y() + b.get_height() / 2, f"{s:.3f}",
                va="center", color="white", fontsize=9)
    _style(ax, "Điểm Cosine Similarity với từng người trong CSDL")
    ax.set_xlim(min(0, min(scores) - 0.05), 1.08)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_score_distribution(genuine, imposter, threshold: Optional[float] = None, eer_thr: Optional[float] = None,
                            light: bool = False):
    """Histogram phân phối điểm Genuine vs Imposter."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white" if light else BG)
    bins = np.linspace(-0.2, 1.0, 49)
    ax.hist(imposter, bins=bins, alpha=0.65, color=BAD, label=f"Imposter (khác người) – {len(imposter)} cặp", density=True)
    ax.hist(genuine, bins=bins, alpha=0.65, color=OK, label=f"Genuine (cùng người) – {len(genuine)} cặp", density=True)
    if threshold is not None:
        ax.axvline(threshold, color=ACCENT, linestyle="--", label=f"τ mặc định = {threshold:.2f}")
    if eer_thr is not None:
        ax.axvline(eer_thr, color="#f59e0b", linestyle=":", label=f"τ tại EER = {eer_thr:.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Mật độ")
    if light:
        ax.set_title("Phân phối điểm Genuine / Imposter", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend()
    else:
        _style(ax, "Phân phối điểm Genuine / Imposter")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=8)
    fig.tight_layout()
    return fig
