"""
Module so khớp & ra quyết định.

    Score = E_enroll · E_test = cos(E_enroll, E_test)     (vì ||E|| = 1)
    Score >= tau  -> ACCEPT (Genuine)
    Score <  tau  -> REJECT (Imposter)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Độ tương đồng Cosine giữa 2 vector (tự chuẩn hóa nếu cần)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class VerificationResult:
    name: str
    score: float
    threshold: float

    @property
    def accepted(self) -> bool:
        return self.score >= self.threshold

    @property
    def label(self) -> str:
        return "ACCEPT" if self.accepted else "REJECT"


def verify(
    test_emb: np.ndarray,
    enroll_emb: np.ndarray,
    name: str = "",
    threshold: float = config.DEFAULT_THRESHOLD,
) -> VerificationResult:
    """Xác thực 1:1 – người nói có đúng là `name` hay không."""
    return VerificationResult(name, cosine_similarity(test_emb, enroll_emb), threshold)


@dataclass
class IdentificationResult:
    ranking: List[Tuple[str, float]]   # [(name, score)] giảm dần
    threshold: float

    @property
    def best(self) -> Optional[Tuple[str, float]]:
        return self.ranking[0] if self.ranking else None

    @property
    def is_known(self) -> bool:
        """Open-set: nếu điểm cao nhất < tau thì là người lạ (không có trong DB)."""
        return bool(self.ranking) and self.ranking[0][1] >= self.threshold


def identify(
    test_emb: np.ndarray,
    gallery: Dict[str, np.ndarray],
    threshold: float = config.DEFAULT_THRESHOLD,
) -> IdentificationResult:
    """Nhận dạng 1:N – so khớp với toàn bộ CSDL, xếp hạng theo điểm Cosine."""
    if not gallery:
        return IdentificationResult([], threshold)
    names = list(gallery.keys())
    mat = np.stack([gallery[n] for n in names])                 # (N, 192)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
    t = test_emb / (np.linalg.norm(test_emb) + 1e-10)
    scores = mat @ t                                            # (N,)
    order = np.argsort(-scores)
    return IdentificationResult([(names[i], float(scores[i])) for i in order], threshold)


# =============================================================================
# Các độ đo đánh giá (phục vụ báo cáo)
# =============================================================================
def compute_eer(genuine: Sequence[float], imposter: Sequence[float]) -> Tuple[float, float]:
    """
    Tính Equal Error Rate (EER) – điểm mà FAR = FRR.

    Returns:
        (eer, threshold_at_eer)
    """
    from sklearn.metrics import roc_curve

    y = np.r_[np.ones(len(genuine)), np.zeros(len(imposter))]
    s = np.r_[genuine, imposter]
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2)
    return eer, float(thr[i])


def far_frr(genuine: Sequence[float], imposter: Sequence[float], threshold: float) -> Tuple[float, float]:
    """FAR = tỉ lệ kẻ giả mạo được chấp nhận; FRR = tỉ lệ người thật bị từ chối."""
    g, i = np.asarray(genuine), np.asarray(imposter)
    far = float(np.mean(i >= threshold)) if i.size else 0.0
    frr = float(np.mean(g < threshold)) if g.size else 0.0
    return far, frr
