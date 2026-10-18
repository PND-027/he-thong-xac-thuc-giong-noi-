"""
Đánh giá hệ thống & vẽ biểu đồ cho báo cáo (Bước 5 – PROJECT_PLAN).

Chuẩn bị dữ liệu:
    data/eval/
      ├── An/      an_1.wav, an_2.wav, an_3.wav ...
      ├── Binh/    binh_1.wav ...
      └── ...      (mỗi thư mục = 1 người nói, ≥ 2 file/người)

Chạy:
    python evaluate_demo.py                       # dùng data/eval
    python evaluate_demo.py --data <thư_mục> --threshold 0.65

Kết quả (thư mục reports/):
    score_distribution.png   – Histogram điểm Genuine vs Imposter
    roc_curve.png            – Đường cong ROC + điểm EER
    far_frr.png              – FAR / FRR theo ngưỡng τ
    embeddings_tsne.png      – Không gian embedding 2D (t-SNE)
    confusion_matrix.png     – Ma trận nhầm lẫn nhận dạng 1:N (leave-one-out)
    metrics.json             – Các chỉ số định lượng
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import audio, config  # noqa: E402
from src.model import SpeakerEncoder  # noqa: E402
from src.verification import compute_eer, far_frr  # noqa: E402
from src.visualize import plot_score_distribution  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})


# =============================================================================
def load_dataset(root: Path):
    """Đọc dữ liệu, chạy pipeline (chuẩn hóa + VAD + ECAPA) cho từng file."""
    speakers = sorted(d for d in root.iterdir() if d.is_dir())
    if len(speakers) < 2:
        sys.exit(f"❌ Cần ít nhất 2 người nói trong '{root}' (mỗi người 1 thư mục con).")

    enc = SpeakerEncoder.get()
    labels, embs, files, skipped, times = [], [], [], [], []
    for spk in speakers:
        for f in sorted(p for p in spk.iterdir() if p.suffix.lower() in AUDIO_EXT):
            try:
                r = audio.process_file(f)
            except audio.SpeechTooShortError as e:
                skipped.append((str(f), str(e)))
                continue
            except Exception as e:  # noqa: BLE001
                skipped.append((str(f), repr(e)))
                continue
            t0 = time.perf_counter()
            embs.append(enc.embed(r.speech))
            times.append(time.perf_counter() - t0)
            labels.append(spk.name)
            files.append(f)
            print(f"  ✓ {spk.name:<15} {f.name:<28} giọng nói {r.speech_duration:5.2f}s / {r.duration:5.2f}s")
    for f, why in skipped:
        print(f"  ✘ Bỏ qua {f}: {why}")
    return np.stack(embs), np.array(labels), files, float(np.mean(times))


def pair_scores(embs: np.ndarray, labels: np.ndarray):
    S = embs @ embs.T
    genuine, imposter = [], []
    for i, j in itertools.combinations(range(len(labels)), 2):
        (genuine if labels[i] == labels[j] else imposter).append(float(S[i, j]))
    return np.array(genuine), np.array(imposter)


def leave_one_out_identification(embs: np.ndarray, labels: np.ndarray):
    """Mỗi file làm test, centroid các file còn lại của từng người làm enrollment."""
    spks = sorted(set(labels))
    y_true, y_pred = [], []
    for i in range(len(labels)):
        cents = []
        for s in spks:
            m = (labels == s)
            m[i] = False
            if not m.any():
                cents.append(np.zeros(embs.shape[1]))
                continue
            c = embs[m].mean(0)
            cents.append(c / (np.linalg.norm(c) + 1e-10))
        scores = np.stack(cents) @ embs[i]
        y_true.append(labels[i])
        y_pred.append(spks[int(np.argmax(scores))])
    return spks, np.array(y_true), np.array(y_pred)


# =============================================================================
def plot_roc(genuine, imposter, eer, out: Path):
    from sklearn.metrics import auc, roc_curve

    y = np.r_[np.ones(len(genuine)), np.zeros(len(imposter))]
    fpr, tpr, _ = roc_curve(y, np.r_[genuine, imposter])
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#7c5cff", lw=2.2, label=f"ECAPA-TDNN (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Ngẫu nhiên")
    ax.plot([0, 1], [1, 0], ":", color="#f59e0b", lw=1)
    ax.scatter([eer], [1 - eer], color="#ef4444", zorder=5, s=60, label=f"EER = {eer*100:.2f}%")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (1 − FRR)")
    ax.set_title("Đường cong ROC", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    return a


def plot_far_frr(genuine, imposter, tau, eer, eer_thr, out: Path):
    ts = np.linspace(-0.2, 1.0, 500)
    far = [np.mean(imposter >= t) for t in ts]
    frr = [np.mean(genuine < t) for t in ts]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ts, np.array(far) * 100, color="#ef4444", lw=2, label="FAR – chấp nhận nhầm kẻ giả mạo")
    ax.plot(ts, np.array(frr) * 100, color="#10b981", lw=2, label="FRR – từ chối nhầm người thật")
    ax.axvline(tau, color="#2563eb", ls="--", label=f"τ mặc định = {tau:.2f}")
    ax.scatter([eer_thr], [eer * 100], color="black", zorder=5, label=f"EER = {eer*100:.2f}% @ τ = {eer_thr:.3f}")
    ax.set_xlabel("Ngưỡng τ (Cosine)")
    ax.set_ylabel("Tỉ lệ lỗi (%)")
    ax.set_title("FAR / FRR theo ngưỡng quyết định", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def plot_tsne(embs, labels, out: Path):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = len(labels)
    if n >= 6:
        pts = TSNE(n_components=2, perplexity=max(2, min(30, (n - 1) // 3)), init="pca",
                   random_state=0).fit_transform(embs)
        method = "t-SNE"
    else:
        pts = PCA(n_components=2).fit_transform(embs)
        method = "PCA"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = plt.get_cmap("tab10" if len(set(labels)) <= 10 else "tab20")
    for k, s in enumerate(sorted(set(labels))):
        m = labels == s
        ax.scatter(pts[m, 0], pts[m, 1], s=70, color=cmap(k % cmap.N), label=s, edgecolor="white", lw=0.8)
    ax.set_title(f"Không gian Speaker Embedding 192-D ({method} → 2D)", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(spks, y_true, y_pred, out: Path):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=spks)
    fig, ax = plt.subplots(figsize=(1 + 0.7 * len(spks), 0.8 + 0.6 * len(spks)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(spks)), spks, rotation=45, ha="right")
    ax.set_yticks(range(len(spks)), spks)
    for i, j in itertools.product(range(len(spks)), repeat=2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title("Ma trận nhầm lẫn – Nhận dạng 1:N", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out)
    plt.close(fig)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Đánh giá hệ thống xác thực người nói")
    ap.add_argument("--data", type=Path, default=config.EVAL_DIR)
    ap.add_argument("--out", type=Path, default=config.REPORT_DIR)
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.data.exists() or not any(args.data.iterdir()):
        sys.exit(f"❌ Chưa có dữ liệu trong '{args.data}'. Xem hướng dẫn ở đầu file evaluate_demo.py.")

    print(f"📂 Đọc dữ liệu từ {args.data}")
    embs, labels, files, t_emb = load_dataset(args.data)
    genuine, imposter = pair_scores(embs, labels)
    if len(genuine) == 0:
        sys.exit("❌ Cần ít nhất 2 file cho một người nói để tạo cặp Genuine.")

    eer, eer_thr = compute_eer(genuine, imposter)
    far, frr = far_frr(genuine, imposter, args.threshold)
    spks, y_true, y_pred = leave_one_out_identification(embs, labels)
    acc = float(np.mean(y_true == y_pred))

    plot_score_distribution(genuine, imposter, args.threshold, eer_thr, light=True).savefig(args.out / "score_distribution.png")
    plt.close("all")
    auc = plot_roc(genuine, imposter, eer, args.out / "roc_curve.png")
    plot_far_frr(genuine, imposter, args.threshold, eer, eer_thr, args.out / "far_frr.png")
    plot_tsne(embs, labels, args.out / "embeddings_tsne.png")
    plot_confusion(spks, y_true, y_pred, args.out / "confusion_matrix.png")

    metrics = {
        "n_speakers": len(spks),
        "n_utterances": int(len(labels)),
        "n_genuine_pairs": int(len(genuine)),
        "n_imposter_pairs": int(len(imposter)),
        "genuine_mean": float(genuine.mean()), "genuine_std": float(genuine.std()),
        "imposter_mean": float(imposter.mean()), "imposter_std": float(imposter.std()),
        "EER": eer, "threshold_at_EER": eer_thr, "AUC": float(auc),
        "threshold": args.threshold, "FAR_at_threshold": far, "FRR_at_threshold": frr,
        "accuracy_at_threshold": float((np.sum(genuine >= args.threshold) + np.sum(imposter < args.threshold))
                                       / (len(genuine) + len(imposter))),
        "identification_top1_accuracy_LOO": acc,
        "avg_embedding_time_sec": t_emb,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f" KẾT QUẢ ĐÁNH GIÁ  ({len(spks)} người nói, {len(labels)} file)")
    print("=" * 62)
    print(f" Cặp Genuine / Imposter   : {len(genuine)} / {len(imposter)}")
    print(f" Điểm Genuine (TB ± std)  : {genuine.mean():.3f} ± {genuine.std():.3f}")
    print(f" Điểm Imposter (TB ± std) : {imposter.mean():.3f} ± {imposter.std():.3f}")
    print(f" EER                      : {eer*100:.2f}%  (τ = {eer_thr:.3f})")
    print(f" AUC                      : {auc:.4f}")
    print(f" Tại τ = {args.threshold:.2f}           : FAR = {far*100:.2f}%  |  FRR = {frr*100:.2f}%")
    print(f" Nhận dạng 1:N (LOO)      : Top-1 accuracy = {acc*100:.2f}%")
    print(f" Thời gian trích xuất TB  : {t_emb*1000:.0f} ms / file (CPU)")
    print("=" * 62)
    print(f"📊 Đã lưu biểu đồ & metrics.json vào: {args.out}")


if __name__ == "__main__":
    main()
