"""
Module các mô hình cơ sở (Baseline) để đối sánh thực nghiệm với ECAPA-TDNN:

1. MFCC + Statistical Pooling (Supervector 120-D) + Cosine Similarity:
   - Tiền nhấn (Pre-emphasis 0.97), khung 25 ms / bước 10 ms, cửa sổ Hamming.
   - Ngân hàng 40 bộ lọc Mel (80 - 7600 Hz) + DCT-II -> 20 hệ số MFCC.
   - Đạo hàm bậc 1 (Delta) và bậc 2 (Delta-Delta) -> 60 đặc trưng/khung.
   - Chuẩn hóa trung bình & phương sai cepstral (CMVN) theo câu.
   - Gộp thống kê (Mean + Std pooling) -> Vector 120 chiều chuẩn hóa L2.

2. MFCC + GMM-UBM (Gaussian Mixture Model - Universal Background Model):
   - Huấn luyện mô hình nền UBM (hỗn hợp Gaussian hiệp phương sai đường chéo)
     trên toàn bộ các khung âm thanh.
   - Thích nghi hậu nghiệm cực đại (MAP adaptation - Reynolds et al., 2000)
     vector kỳ vọng của UBM sang từng người nói đăng ký.
   - Chấm điểm bằng tỉ số hợp lý logarit (Log-Likelihood Ratio - LLR).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.fftpack import dct
from scipy.signal import stft

from . import config


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int = config.SAMPLE_RATE,
    n_fft: int = 512,
    n_mels: int = 40,
    fmin: float = 80.0,
    fmax: float = 7600.0,
) -> np.ndarray:
    """Tạo ma trận ngân hàng bộ lọc tam giác trên thang tần số Mel."""
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(min(fmax, sr / 2)), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


_FB_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Tính đạo hàm theo thời gian (Delta) của ma trận đặc trưng (T, D)."""
    if feat.shape[0] < 2:
        return np.zeros_like(feat)
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, width + 1))
    out = np.zeros_like(feat)
    for n in range(1, width + 1):
        out += n * (padded[width + n : width + n + len(feat)] - padded[width - n : width - n + len(feat)])
    return out / denom


def extract_mfcc_frames(
    wav: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mfcc: int = 20,
    n_mels: int = 40,
    include_deltas: bool = True,
    cmvn: bool = True,
) -> np.ndarray:
    """
    Trích xuất đặc trưng MFCC cấp khung (T, D).
    Mặc định: 20 MFCC + 20 Delta + 20 Delta-Delta = 60 chiều/khung.
    """
    if wav.size < int(0.025 * sr):
        dim = n_mfcc * (3 if include_deltas else 1)
        return np.zeros((1, dim), dtype=np.float32)

    # 1. Pre-emphasis
    x = np.append(wav[0], wav[1:] - 0.97 * wav[:-1]).astype(np.float32)

    # 2. STFT (khung 25 ms, bước 10 ms, cửa sổ Hamming)
    nperseg = int(0.025 * sr)
    noverlap = nperseg - int(0.010 * sr)
    n_fft = 512
    _, _, Z = stft(x, fs=sr, window="hamming", nperseg=nperseg, noverlap=noverlap, nfft=n_fft, boundary=None)
    power = (np.abs(Z) ** 2) / n_fft  # (F, T)

    # 3. Mel-filterbank + Log + DCT-II
    key = (sr, n_fft, n_mels)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(sr, n_fft, n_mels)
    mel_energy = np.dot(_FB_CACHE[key], power).T  # (T, n_mels)
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc].astype(np.float32)

    if include_deltas:
        d1 = _delta(mfcc, width=2)
        d2 = _delta(d1, width=2)
        feat = np.concatenate([mfcc, d1, d2], axis=1)
    else:
        feat = mfcc

    if cmvn and feat.shape[0] > 1:
        feat = (feat - feat.mean(axis=0, keepdims=True)) / (feat.std(axis=0, keepdims=True) + 1e-8)

    return feat.astype(np.float32)


def extract_mfcc_supervector(wav: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Baseline 1: Gộp thống kê (Mean + Std pooling) các khung MFCC 60-D
    thành vector 120 chiều chuẩn hóa L2.
    """
    frames = extract_mfcc_frames(wav, sr=sr, n_mfcc=20, include_deltas=True, cmvn=False)
    mean = frames.mean(axis=0)
    std = frames.std(axis=0)
    vec = np.concatenate([mean, std]).astype(np.float32)
    return (vec / (np.linalg.norm(vec) + 1e-10)).astype(np.float32)


class GMMUBMVerifier:
    """
    Baseline 2: MFCC + GMM-UBM với thích nghi MAP (Reynolds et al., 2000).
    """

    def __init__(self, n_components: int = 16, relevance_factor: float = 10.0, random_state: int = 42):
        from sklearn.mixture import GaussianMixture

        self.n_components = n_components
        self.relevance_factor = relevance_factor
        self.ubm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            max_iter=100,
            reg_covar=1e-3,
            random_state=random_state,
        )

    def fit_ubm(self, frame_list: Sequence[np.ndarray]) -> "GMMUBMVerifier":
        """Huấn luyện mô hình nền UBM trên tập hợp các khung MFCC."""
        pooled = np.vstack(frame_list)
        if len(pooled) > 25_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pooled), size=25_000, replace=False)
            pooled = pooled[idx]
        self.ubm.fit(pooled)
        return self

    def adapt_speaker_means(self, frames: np.ndarray) -> np.ndarray:
        """
        Thích nghi MAP kỳ vọng các thành phần Gaussian từ UBM sang người nói cụ thể:
            mu_k_spk = alpha_k * E_k(X) + (1 - alpha_k) * mu_k_ubm
            với alpha_k = n_k / (n_k + r)
        """
        post = self.ubm.predict_proba(frames)  # (T, K)
        n_k = post.sum(axis=0, keepdims=True).T + 1e-10  # (K, 1)
        e_k = (post.T @ frames) / n_k  # (K, D)
        alpha = n_k / (n_k + self.relevance_factor)  # (K, 1)
        return (alpha * e_k + (1.0 - alpha) * self.ubm.means_).astype(np.float32)

    def _log_likelihood_with_means(self, frames: np.ndarray, means: np.ndarray) -> np.ndarray:
        """Tính log p(x_t | lambda) cho từng khung với kỳ vọng thích nghi `means`."""
        from scipy.special import logsumexp

        precisions = self.ubm.precisions_  # (K, D) = 1 / sigma^2
        log_det = np.sum(np.log(precisions), axis=1)  # (K,)
        d = frames.shape[1]
        # (T, K, D)
        diff = frames[:, None, :] - means[None, :, :]
        maha = np.sum((diff ** 2) * precisions[None, :, :], axis=2)  # (T, K)
        log_prob = -0.5 * (d * np.log(2 * np.pi) - log_det[None, :] + maha)
        log_weights = np.log(np.maximum(self.ubm.weights_, 1e-12))[None, :]
        return logsumexp(log_prob + log_weights, axis=1)

    def score_llr(self, spk_means: np.ndarray, test_frames: np.ndarray) -> float:
        """Tính điểm Log-Likelihood Ratio (LLR) trung bình trên các khung kiểm thử."""
        ll_spk = self._log_likelihood_with_means(test_frames, spk_means)
        ll_ubm = self.ubm.score_samples(test_frames)
        return float(np.mean(ll_spk - ll_ubm))
