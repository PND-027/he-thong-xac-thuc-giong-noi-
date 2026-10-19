"""
Kịch bản thực nghiệm toàn diện phục vụ Báo cáo Đồ án Xử lý Tiếng nói:
- Tiêu chí 4 (1.4): Phân tích ảnh hưởng cải tiến qua thực nghiệm (Ablation Study:
  Không VAD vs. VAD Năng lượng vs. Silero VAD; Đăng ký đơn mẫu vs. đa mẫu Centroid).
- Tiêu chí 9 (2.4): So sánh đối đầu trực tiếp (side-by-side) giữa ECAPA-TDNN với
  2 mô hình tham chiếu (MFCC + Statistical Pooling + Cosine và MFCC + GMM-UBM MAP).
- Tiêu chí 10 (2.5): Đánh giá mở rộng trên tiếng Việt (VIVOS - data/eval), đa mức nhiễu
  (SNR = Clean, 20, 15, 10, 5, 0 dB với White, Pink, Babble noise) và phân tích giới tính.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.metrics import auc, roc_curve

import evaluate_demo
from src import audio, baseline, config, visualize
from src.database import SpeakerDB
from src.model import SpeakerEncoder, l2_normalize
from src.verification import compute_eer, far_frr

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


def load_raw_and_speech(root: Path):
    """Đọc toàn bộ file âm thanh, chuẩn hóa và trích xuất Silero VAD + ECAPA-TDNN 1 lần."""
    enc = SpeakerEncoder.get()
    speakers = sorted(d for d in root.iterdir() if d.is_dir())
    raw_wavs, speech_wavs, embs, labels, files, times = [], [], [], [], [], []
    for spk in speakers:
        for f in sorted(p for p in spk.iterdir() if p.suffix.lower() in evaluate_demo.AUDIO_EXT):
            data, sr = audio.load_audio(f)
            x = audio.standardize(data, sr)
            t0 = time.perf_counter()
            sp, _, _ = audio.apply_vad(x)
            if len(sp) < int(0.5 * config.SAMPLE_RATE):
                sp = x
            emb = enc.embed(sp)
            dt = time.perf_counter() - t0
            raw_wavs.append(x)
            speech_wavs.append(sp)
            embs.append(emb)
            labels.append(spk.name)
            files.append(f)
            times.append(dt)
    return raw_wavs, speech_wavs, np.stack(embs), np.array(labels), files, float(np.mean(times))


def generate_noise(kind: str, length: int, rng: np.random.Generator, pool_wavs: List[np.ndarray] | None = None) -> np.ndarray:
    """Sinh tín hiệu nhiễu chuẩn hóa năng lượng đơn vị: white, pink, hoặc babble."""
    if kind == "white":
        n = rng.standard_normal(length).astype(np.float32)
    elif kind == "pink":
        white = rng.standard_normal(length)
        X = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        scale = np.ones_like(freqs)
        scale[1:] = 1.0 / np.sqrt(freqs[1:])
        n = np.fft.irfft(X * scale, n=length).astype(np.float32)
    elif kind == "babble":
        assert pool_wavs and len(pool_wavs) >= 4
        n = np.zeros(length, dtype=np.float32)
        picks = rng.choice(len(pool_wavs), size=min(6, len(pool_wavs)), replace=False)
        for idx in picks:
            w = pool_wavs[int(idx)]
            if len(w) < length:
                reps = int(np.ceil(length / len(w))) + 1
                w = np.tile(w, reps)
            offset = int(rng.integers(0, max(1, len(w) - length)))
            n += w[offset : offset + length]
    else:
        raise ValueError(f"Unknown noise kind: {kind}")

    rms = float(np.sqrt(np.mean(n ** 2)) + 1e-10)
    return (n / rms).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise_unit: np.ndarray, snr_db: float) -> np.ndarray:
    """Trộn tín hiệu sạch với nhiễu theo tỉ số SNR (dB) xác định."""
    p_sig = float(np.mean(clean ** 2)) + 1e-10
    p_noise_target = p_sig / (10.0 ** (snr_db / 10.0))
    noisy = clean + noise_unit[: len(clean)] * np.sqrt(p_noise_target)
    return audio.peak_normalize(noisy.astype(np.float32))


def evaluate_embeddings(embs: np.ndarray, labels: np.ndarray, threshold: float = 0.45) -> dict:
    """Tính toàn bộ chỉ số EER, AUC, FAR, FRR, Top-1 LOO từ ma trận embedding."""
    genuine, imposter = evaluate_demo.pair_scores(embs, labels)
    eer, eer_thr = compute_eer(genuine, imposter)
    far, frr = far_frr(genuine, imposter, threshold)
    y = np.r_[np.ones(len(genuine)), np.zeros(len(imposter))]
    fpr, tpr, _ = roc_curve(y, np.r_[genuine, imposter])
    roc_auc = float(auc(fpr, tpr))
    _, y_true, y_pred = evaluate_demo.leave_one_out_identification(embs, labels)
    top1 = float(np.mean(y_true == y_pred))
    return {
        "EER": float(eer),
        "threshold_at_EER": float(eer_thr),
        "AUC": roc_auc,
        "genuine_mean": float(genuine.mean()),
        "genuine_std": float(genuine.std()),
        "imposter_mean": float(imposter.mean()),
        "imposter_std": float(imposter.std()),
        "separation_gap": float(genuine.mean() - imposter.mean()),
        "FAR_at_thr": float(far),
        "FRR_at_thr": float(frr),
        "top1_accuracy_LOO": top1,
    }


def run_vietnamese_standard_eval(embs: np.ndarray, labels: np.ndarray, t_emb: float):
    print("\n" + "=" * 70, flush=True)
    print(" [BƯỚC 1] ĐÁNH GIÁ TIÊU CHUẨN TRÊN DỮ LIỆU TIẾNG VIỆT (data/eval)", flush=True)
    print("=" * 70, flush=True)
    out_vn = config.REPORT_DIR / "vietnamese"
    out_vn.mkdir(parents=True, exist_ok=True)

    genuine, imposter = evaluate_demo.pair_scores(embs, labels)
    eer, eer_thr = compute_eer(genuine, imposter)
    far, frr = far_frr(genuine, imposter, config.DEFAULT_THRESHOLD)
    spks, y_true, y_pred = evaluate_demo.leave_one_out_identification(embs, labels)
    acc = float(np.mean(y_true == y_pred))

    for target_dir in (out_vn, config.REPORT_DIR):
        visualize.plot_score_distribution(
            genuine, imposter, config.DEFAULT_THRESHOLD, eer_thr, light=True
        ).savefig(target_dir / "score_distribution.png")
        plt.close("all")
        roc_auc = evaluate_demo.plot_roc(genuine, imposter, eer, target_dir / "roc_curve.png")
        evaluate_demo.plot_far_frr(genuine, imposter, config.DEFAULT_THRESHOLD, eer, eer_thr, target_dir / "far_frr.png")
        evaluate_demo.plot_tsne(embs, labels, target_dir / "embeddings_tsne.png")
        evaluate_demo.plot_confusion(spks, y_true, y_pred, target_dir / "confusion_matrix.png")

        metrics = {
            "dataset": "VIVOS Vietnamese Speech Corpus (10 speakers x 5 utterances)",
            "n_speakers": len(spks),
            "n_utterances": int(len(labels)),
            "n_genuine_pairs": int(len(genuine)),
            "n_imposter_pairs": int(len(imposter)),
            "genuine_mean": float(genuine.mean()),
            "genuine_std": float(genuine.std()),
            "imposter_mean": float(imposter.mean()),
            "imposter_std": float(imposter.std()),
            "EER": float(eer),
            "threshold_at_EER": float(eer_thr),
            "AUC": float(roc_auc),
            "threshold": config.DEFAULT_THRESHOLD,
            "FAR_at_threshold": float(far),
            "FRR_at_threshold": float(frr),
            "accuracy_at_threshold": float(
                (np.sum(genuine >= config.DEFAULT_THRESHOLD) + np.sum(imposter < config.DEFAULT_THRESHOLD))
                / (len(genuine) + len(imposter))
            ),
            "identification_top1_accuracy_LOO": acc,
            "avg_embedding_time_sec": t_emb,
        }
        (target_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    db = SpeakerDB()
    vn_names = {
        "VIVOSDEV02": "Nguyễn Minh Tuấn (VIVOS-02)",
        "VIVOSDEV03": "Trần Thị Mai (VIVOS-03)",
        "VIVOSDEV05": "Lê Hoàng Nam (VIVOS-05)",
        "VIVOSDEV09": "Phạm Thu Hà (VIVOS-09)",
        "VIVOSDEV11": "Vũ Đức Anh (VIVOS-11)",
    }
    for spk_code, display_name in vn_names.items():
        mask = labels == spk_code
        if mask.any():
            db.enroll(display_name, list(embs[mask][:3]), [3.8, 4.1, 3.9], append=False)

    print(f"  -> Tiếng Việt (VIVOS): EER = {eer*100:.2f}% @ tau={eer_thr:.3f}, AUC = {roc_auc:.4f}, Top-1 = {acc*100:.2f}%", flush=True)
    return metrics


def run_ablation_study(
    wavs_vn: List[np.ndarray],
    speech_vn: List[np.ndarray],
    embs_vn_clean: np.ndarray,
    labels_vn: np.ndarray,
    t_vn_clean: float,
    embs_ls_clean: np.ndarray,
    labels_ls: np.ndarray,
):
    print("\n" + "=" * 70, flush=True)
    print(" [BƯỚC 2 - TIÊU CHÍ 4] PHÂN TÍCH ẢNH HƯỞNG CẢI TIẾN (ABLATION STUDY)", flush=True)
    print("=" * 70, flush=True)
    out_dir = config.REPORT_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = SpeakerEncoder.get()
    rng = np.random.default_rng(42)

    # Kịch bản thu âm qua Micro có 1.5s khoảng chờ đầu + 1.5s cuối câu kèm tạp âm phòng & tiếng ồn nền
    padded_wavs = []
    pad_len = int(1.5 * config.SAMPLE_RATE)
    for w in wavs_vn:
        room_lead = (
            generate_noise("pink", pad_len, rng) * 0.14
            + generate_noise("babble", pad_len, rng, pool_wavs=wavs_vn) * 0.11
        )
        room_trail = (
            generate_noise("pink", pad_len, rng) * 0.14
            + generate_noise("babble", pad_len, rng, pool_wavs=wavs_vn) * 0.11
        )
        w_pad = np.concatenate([room_lead, w, room_trail]).astype(np.float32)
        padded_wavs.append(audio.peak_normalize(w_pad))

    def eval_vad_mode(
        input_wavs: List[np.ndarray],
        mode: str,
        precomputed_embs: np.ndarray | None = None,
        cross_with_clean: bool = False,
    ) -> dict:
        raw_durs = [len(w) / config.SAMPLE_RATE for w in input_wavs]
        if precomputed_embs is not None:
            speech_durs = [len(s) / config.SAMPLE_RATE for s in speech_vn]
            embs_arr = precomputed_embs
            avg_ms = float(t_vn_clean * 1000.0)
        else:
            embs, speech_durs, total_times = [], [], []
            for w in input_wavs:
                t0 = time.perf_counter()
                if mode == "no_vad":
                    sp = w
                elif mode == "energy_vad":
                    segs = audio.energy_vad(w, config.SAMPLE_RATE)
                    sp = np.concatenate([w[s:e] for s, e in segs]).astype(np.float32) if segs else w
                elif mode == "silero_vad":
                    segs = audio.silero_vad(w, config.SAMPLE_RATE)
                    sp = np.concatenate([w[s:e] for s, e in segs]).astype(np.float32) if segs else w
                else:
                    raise ValueError(mode)
                if len(sp) < int(0.5 * config.SAMPLE_RATE):
                    sp = w
                speech_durs.append(len(sp) / config.SAMPLE_RATE)
                embs.append(enc.embed(sp))
                total_times.append(time.perf_counter() - t0)
            embs_arr = np.stack(embs)
            avg_ms = float(np.mean(total_times) * 1000.0)

        if cross_with_clean:
            S = embs_vn_clean @ embs_arr.T
            gen, imp = [], []
            for i in range(len(labels_vn)):
                for j in range(len(labels_vn)):
                    if i == j:
                        continue
                    (gen if labels_vn[i] == labels_vn[j] else imp).append(float(S[i, j]))
            gen_arr, imp_arr = np.array(gen), np.array(imp)
            eer, eer_thr = compute_eer(gen_arr, imp_arr)
            far, frr = far_frr(gen_arr, imp_arr, 0.45)
            y = np.r_[np.ones(len(gen_arr)), np.zeros(len(imp_arr))]
            fpr, tpr, _ = roc_curve(y, np.r_[gen_arr, imp_arr])
            spks = sorted(set(labels_vn))
            correct = 0
            for i in range(len(labels_vn)):
                cents = [l2_normalize(embs_vn_clean[(labels_vn == s) & (np.arange(len(labels_vn)) != i)].mean(0)) for s in spks]
                if spks[int(np.argmax(np.stack(cents) @ embs_arr[i]))] == labels_vn[i]:
                    correct += 1
            res = {
                "EER": float(eer),
                "threshold_at_EER": float(eer_thr),
                "AUC": float(auc(fpr, tpr)),
                "genuine_mean": float(gen_arr.mean()),
                "genuine_std": float(gen_arr.std()),
                "imposter_mean": float(imp_arr.mean()),
                "imposter_std": float(imp_arr.std()),
                "separation_gap": float(gen_arr.mean() - imp_arr.mean()),
                "FAR_at_thr": float(far),
                "FRR_at_thr": float(frr),
                "top1_accuracy_LOO": float(correct / len(labels_vn)),
            }
        else:
            res = evaluate_embeddings(embs_arr, labels_vn, threshold=0.45)

        res["avg_raw_duration_sec"] = float(np.mean(raw_durs))
        res["avg_speech_duration_sec"] = float(np.mean(speech_durs))
        res["duration_reduction_pct"] = float((1.0 - np.mean(speech_durs) / np.mean(raw_durs)) * 100.0)
        res["avg_cpu_time_ms"] = avg_ms
        return res

    vad_results = {
        "clean_dataset": {
            "no_vad": eval_vad_mode(wavs_vn, "no_vad"),
            "energy_vad": eval_vad_mode(wavs_vn, "energy_vad"),
            "silero_vad": eval_vad_mode(wavs_vn, "silero_vad", precomputed_embs=embs_vn_clean),
        },
        "realistic_mic_padded": {
            "no_vad": eval_vad_mode(padded_wavs, "no_vad", cross_with_clean=True),
            "energy_vad": eval_vad_mode(padded_wavs, "energy_vad", cross_with_clean=True),
            "silero_vad": eval_vad_mode(padded_wavs, "silero_vad", cross_with_clean=True),
        },
    }

    for cond, modes in vad_results.items():
        print(f"  * Kịch bản [{cond}]:", flush=True)
        for m_name, r in modes.items():
            print(
                f"    - {m_name:<12}: EER={r['EER']*100:5.2f}%, AUC={r['AUC']:.4f}, "
                f"Gen={r['genuine_mean']:.3f}, Time={r['avg_cpu_time_ms']:.0f}ms, "
                f"Speech={r['avg_speech_duration_sec']:.2f}s (-{r['duration_reduction_pct']:.1f}%)",
                flush=True,
            )

    def eval_centroid_k(embs: np.ndarray, labels: np.ndarray) -> Dict[int, dict]:
        spks = sorted(set(labels))
        out_k = {}
        for k in (1, 2, 3, 4):
            gen_scores, imp_scores = [], []
            correct_top1, total_tests = 0, 0
            for s in spks:
                idx_s = np.where(labels == s)[0]
                for pos in range(len(idx_s)):
                    test_i = idx_s[pos]
                    rem = np.delete(idx_s, pos)
                    enroll_idx = rem[:k]
                    c = l2_normalize(embs[enroll_idx].mean(axis=0))
                    gen_scores.append(float(c @ embs[test_i]))

                    gallery_scores = []
                    for other_s in spks:
                        if other_s == s:
                            gallery_scores.append((s, float(c @ embs[test_i])))
                        else:
                            other_idx = np.where(labels == other_s)[0][:k]
                            c_other = l2_normalize(embs[other_idx].mean(axis=0))
                            sc = float(c_other @ embs[test_i])
                            imp_scores.append(sc)
                            gallery_scores.append((other_s, sc))
                    pred_s = max(gallery_scores, key=lambda x: x[1])[0]
                    if pred_s == s:
                        correct_top1 += 1
                    total_tests += 1

            eer, eer_thr = compute_eer(gen_scores, imp_scores)
            y = np.r_[np.ones(len(gen_scores)), np.zeros(len(imp_scores))]
            fpr, tpr, _ = roc_curve(y, np.r_[gen_scores, imp_scores])
            out_k[k] = {
                "k_samples": k,
                "EER": float(eer),
                "threshold_at_EER": float(eer_thr),
                "AUC": float(auc(fpr, tpr)),
                "genuine_mean": float(np.mean(gen_scores)),
                "genuine_std": float(np.std(gen_scores)),
                "imposter_mean": float(np.mean(imp_scores)),
                "imposter_std": float(np.std(imp_scores)),
                "top1_accuracy": float(correct_top1 / total_tests),
            }
        return out_k

    centroid_results = {
        "librispeech": eval_centroid_k(embs_ls_clean, labels_ls),
        "vietnamese": eval_centroid_k(embs_vn_clean, labels_vn),
    }

    ablation_summary = {
        "vad_ablation": vad_results,
        "centroid_ablation": centroid_results,
    }
    (out_dir / "ablation_metrics.json").write_text(
        json.dumps(ablation_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Biểu đồ 1: Không VAD vs Energy VAD vs Silero VAD
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    modes_keys = ["no_vad", "energy_vad", "silero_vad"]
    modes_labels = ["Không VAD\n(No VAD)", "VAD Năng lượng\n(Energy VAD)", "Silero VAD\n(Đề xuất)"]
    x = np.arange(len(modes_keys))
    w_bar = 0.34

    eer_clean = [vad_results["clean_dataset"][k]["EER"] * 100 for k in modes_keys]
    eer_pad = [vad_results["realistic_mic_padded"][k]["EER"] * 100 for k in modes_keys]
    b1 = axes[0].bar(x - w_bar / 2, eer_clean, w_bar, label="Bản thu chuẩn", color="#3b82f6")
    b2 = axes[0].bar(x + w_bar / 2, eer_pad, w_bar, label="Bản thu có khoảng chờ 1.5s + nhiễu nền", color="#ef4444")
    axes[0].set_xticks(x, modes_labels)
    axes[0].set_ylabel("Tỉ lệ lỗi ngang bằng EER (%) ↓")
    axes[0].set_title("a) Ảnh hưởng của VAD tới EER (%)", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, max(eer_clean + eer_pad) * 1.3 + 0.8)

    gap_clean = [vad_results["clean_dataset"][k]["genuine_mean"] for k in modes_keys]
    gap_pad = [vad_results["realistic_mic_padded"][k]["genuine_mean"] for k in modes_keys]
    b3 = axes[1].bar(x - w_bar / 2, gap_clean, w_bar, label="Bản thu chuẩn", color="#10b981")
    b4 = axes[1].bar(x + w_bar / 2, gap_pad, w_bar, label="Bản thu có khoảng chờ 1.5s + nhiễu nền", color="#f59e0b")
    axes[1].set_xticks(x, modes_labels)
    axes[1].set_ylabel("Điểm Cosine Genuine TB ↑")
    axes[1].set_title("b) Độ tương đồng cùng người nói (Genuine Score)", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, 0.92)

    t_pad = [vad_results["realistic_mic_padded"][k]["avg_cpu_time_ms"] for k in modes_keys]
    dur_pad = [vad_results["realistic_mic_padded"][k]["avg_speech_duration_sec"] for k in modes_keys]
    b5 = axes[2].bar(x, t_pad, 0.45, color=["#64748b", "#0ea5e9", "#2563eb"])
    axes[2].set_xticks(x, modes_labels)
    axes[2].set_ylabel("Thời gian xử lý TB trên CPU (ms/file) ↓")
    axes[2].set_title("c) Thời gian suy luận CPU (cắt bỏ khoảng lặng)", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)
    for bar, d_s in zip(b5, dur_pad):
        h = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            h + 8,
            f"{h:.0f} ms\n({d_s:.2f}s speech)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[2].set_ylim(0, max(t_pad) * 1.28)

    fig.tight_layout()
    fig.savefig(out_dir / "vad_ablation.png")
    plt.close(fig)

    # Biểu đồ 2: Đa mẫu Centroid K = 1..4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = [1, 2, 3, 4]
    eer_ls_k = [centroid_results["librispeech"][k]["EER"] * 100 for k in ks]
    eer_vn_k = [centroid_results["vietnamese"][k]["EER"] * 100 for k in ks]
    gen_ls_k = [centroid_results["librispeech"][k]["genuine_mean"] for k in ks]
    gen_vn_k = [centroid_results["vietnamese"][k]["genuine_mean"] for k in ks]

    ax1.plot(ks, eer_ls_k, "o-", lw=2.2, color="#2563eb", label="LibriSpeech (Tiếng Anh)")
    ax1.plot(ks, eer_vn_k, "s-", lw=2.2, color="#10b981", label="VIVOS (Tiếng Việt)")
    for k_v, y1, y2 in zip(ks, eer_ls_k, eer_vn_k):
        ax1.annotate(f"{y1:.2f}%", (k_v, y1), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax1.annotate(f"{y2:.2f}%", (k_v, y2), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8)
    ax1.set_xticks(ks, [f"K = {k} mẫu" for k in ks])
    ax1.set_ylabel("EER (%) ↓")
    ax1.set_title("a) Giảm tỉ lệ lỗi EER khi tăng số mẫu đăng ký K", fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(ks, gen_ls_k, "o-", lw=2.2, color="#2563eb", label="LibriSpeech (Tiếng Anh)")
    ax2.plot(ks, gen_vn_k, "s-", lw=2.2, color="#10b981", label="VIVOS (Tiếng Việt)")
    for k_v, y1, y2 in zip(ks, gen_ls_k, gen_vn_k):
        ax2.annotate(f"{y1:.3f}", (k_v, y1), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax2.annotate(f"{y2:.3f}", (k_v, y2), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8)
    ax2.set_xticks(ks, [f"K = {k} mẫu" for k in ks])
    ax2.set_ylabel("Điểm Cosine Genuine trung bình ↑")
    ax2.set_title("b) Tăng độ ổn định Centroid khi gộp đa mẫu", fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "centroid_ablation.png")
    plt.close(fig)

    # Biểu đồ 3: Sóng âm & Spectrogram trước/sau VAD
    sample_wav = padded_wavs[0]
    sp_seg, segs, backend = audio.apply_vad(sample_wav)
    fig_vad = visualize.plot_vad(sample_wav, sp_seg, segs, backend=backend)
    fig_vad.savefig(out_dir / "signal_vad_waveform.png")
    plt.close(fig_vad)

    fig_spec = visualize.plot_spectrogram(sp_seg, title="Phổ tần số (STFT Spectrogram) của tín hiệu tiếng nói sau Silero VAD")
    fig_spec.savefig(out_dir / "signal_vad_spectrogram.png")
    plt.close(fig_spec)

    return ablation_summary


def run_baseline_comparison(
    speech_ls: List[np.ndarray],
    embs_ls: np.ndarray,
    labels_ls: np.ndarray,
    t_ls: float,
    speech_vn: List[np.ndarray],
    embs_vn: np.ndarray,
    labels_vn: np.ndarray,
    t_vn: float,
):
    print("\n" + "=" * 70, flush=True)
    print(" [BƯỚC 3 - TIÊU CHÍ 9] SO SÁNH ĐỐI ĐẦU VỚI MÔ HÌNH BASELINE", flush=True)
    print("=" * 70, flush=True)
    out_dir = config.REPORT_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "librispeech": (speech_ls, embs_ls, labels_ls, t_ls),
        "vietnamese": (speech_vn, embs_vn, labels_vn, t_vn),
    }
    comp_results = {}
    roc_curves_data = {}

    for ds_name, (speech_list, ecapa_embs, labels_arr, t_ecapa) in datasets.items():
        spks = sorted(set(labels_arr))

        # --- Model 1: MFCC (60-D) + Statistical Pooling (120-D) + Cosine ---
        t0 = time.perf_counter()
        mfcc_embs = np.stack([baseline.extract_mfcc_supervector(w) for w in speech_list])
        t_mfcc_cos = (time.perf_counter() - t0) / len(speech_list)
        gen_m1, imp_m1 = evaluate_demo.pair_scores(mfcc_embs, labels_arr)
        eer_m1, thr_m1 = compute_eer(gen_m1, imp_m1)
        _, yt_m1, yp_m1 = evaluate_demo.leave_one_out_identification(mfcc_embs, labels_arr)
        top1_m1 = float(np.mean(yt_m1 == yp_m1))
        y_m1 = np.r_[np.ones(len(gen_m1)), np.zeros(len(imp_m1))]
        fpr_m1, tpr_m1, _ = roc_curve(y_m1, np.r_[gen_m1, imp_m1])
        auc_m1 = float(auc(fpr_m1, tpr_m1))

        # --- Model 2: MFCC (60-D CMVN) + GMM-UBM (MAP Adaptation, Reynolds 2000) ---
        t0 = time.perf_counter()
        frames_list = [baseline.extract_mfcc_frames(w, cmvn=True) for w in speech_list]
        gmm_ubm = baseline.GMMUBMVerifier(n_components=16, relevance_factor=10.0)
        gmm_ubm.fit_ubm(frames_list)
        adapted_means = [gmm_ubm.adapt_speaker_means(fr) for fr in frames_list]
        t_gmm = (time.perf_counter() - t0) / len(speech_list)

        gen_m2, imp_m2 = [], []
        for i, j in itertools.combinations(range(len(labels_arr)), 2):
            s_ij = 0.5 * (gmm_ubm.score_llr(adapted_means[i], frames_list[j]) + gmm_ubm.score_llr(adapted_means[j], frames_list[i]))
            (gen_m2 if labels_arr[i] == labels_arr[j] else imp_m2).append(s_ij)
        gen_m2_arr, imp_m2_arr = np.array(gen_m2), np.array(imp_m2)
        eer_m2, thr_m2 = compute_eer(gen_m2_arr, imp_m2_arr)
        y_m2 = np.r_[np.ones(len(gen_m2_arr)), np.zeros(len(imp_m2_arr))]
        fpr_m2, tpr_m2, _ = roc_curve(y_m2, np.r_[gen_m2_arr, imp_m2_arr])
        auc_m2 = float(auc(fpr_m2, tpr_m2))

        correct_gmm = 0
        for i in range(len(labels_arr)):
            scores_spk = []
            for s in spks:
                m = labels_arr == s
                m[i] = False
                enroll_frames = np.vstack([frames_list[k] for k in np.where(m)[0]])
                spk_means = gmm_ubm.adapt_speaker_means(enroll_frames)
                scores_spk.append(gmm_ubm.score_llr(spk_means, frames_list[i]))
            if spks[int(np.argmax(scores_spk))] == labels_arr[i]:
                correct_gmm += 1
        top1_m2 = float(correct_gmm / len(labels_arr))

        # --- Model 3: ECAPA-TDNN Pretrained (192-D) + Cosine ---
        gen_m3, imp_m3 = evaluate_demo.pair_scores(ecapa_embs, labels_arr)
        eer_m3, thr_m3 = compute_eer(gen_m3, imp_m3)
        _, yt_m3, yp_m3 = evaluate_demo.leave_one_out_identification(ecapa_embs, labels_arr)
        top1_m3 = float(np.mean(yt_m3 == yp_m3))
        y_m3 = np.r_[np.ones(len(gen_m3)), np.zeros(len(imp_m3))]
        fpr_m3, tpr_m3, _ = roc_curve(y_m3, np.r_[gen_m3, imp_m3])
        auc_m3 = float(auc(fpr_m3, tpr_m3))

        comp_results[ds_name] = {
            "mfcc_cosine": {
                "name": "MFCC (60-D) + Stat Pooling + Cosine",
                "dim": 120,
                "EER": float(eer_m1),
                "threshold_at_EER": float(thr_m1),
                "AUC": auc_m1,
                "top1_accuracy_LOO": top1_m1,
                "cpu_time_ms": float(t_mfcc_cos * 1000.0),
            },
            "mfcc_gmm_ubm": {
                "name": "MFCC (60-D CMVN) + GMM-UBM (MAP, M=16)",
                "dim": "16x60 GMM",
                "EER": float(eer_m2),
                "threshold_at_EER": float(thr_m2),
                "AUC": auc_m2,
                "top1_accuracy_LOO": top1_m2,
                "cpu_time_ms": float(t_gmm * 1000.0),
            },
            "ecapa_tdnn": {
                "name": "ECAPA-TDNN Pretrained (192-D) + Cosine",
                "dim": 192,
                "EER": float(eer_m3),
                "threshold_at_EER": float(thr_m3),
                "AUC": auc_m3,
                "top1_accuracy_LOO": top1_m3,
                "cpu_time_ms": float(t_ecapa * 1000.0),
            },
        }
        roc_curves_data[ds_name] = {
            "mfcc_cosine": (fpr_m1, tpr_m1, eer_m1, auc_m1),
            "mfcc_gmm_ubm": (fpr_m2, tpr_m2, eer_m2, auc_m2),
            "ecapa_tdnn": (fpr_m3, tpr_m3, eer_m3, auc_m3),
        }

        print(f"  * Tập dữ liệu [{ds_name}]:", flush=True)
        for k, v in comp_results[ds_name].items():
            print(
                f"    - {v['name']:<40}: EER={v['EER']*100:5.2f}%, "
                f"AUC={v['AUC']:.4f}, Top-1={v['top1_accuracy_LOO']*100:5.1f}%, "
                f"Time={v['cpu_time_ms']:.1f}ms",
                flush=True,
            )

    (out_dir / "baseline_comparison.json").write_text(
        json.dumps(comp_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = {
        "librispeech": "a) Đường cong ROC trên LibriSpeech (Tiếng Anh)",
        "vietnamese": "b) Đường cong ROC trên VIVOS (Tiếng Việt)",
    }
    for ax, ds_name in zip(axes, ("librispeech", "vietnamese")):
        d = roc_curves_data[ds_name]
        for key, label, color, ls in [
            ("mfcc_cosine", "Baseline 1: MFCC + StatPool + Cosine", "#f59e0b", "--"),
            ("mfcc_gmm_ubm", "Baseline 2: MFCC + GMM-UBM (MAP)", "#0ea5e9", "-."),
            ("ecapa_tdnn", "Đề xuất: ECAPA-TDNN (192-D)", "#2563eb", "-"),
        ]:
            fpr, tpr, eer_v, auc_v = d[key]
            ax.plot(fpr, tpr, color=color, ls=ls, lw=2.2, label=f"{label}\n(EER={eer_v*100:.2f}%, AUC={auc_v:.3f})")
            ax.scatter([eer_v], [1 - eer_v], color=color, s=45, zorder=5)
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
        ax.set_xlabel("False Acceptance Rate (FAR)")
        ax.set_ylabel("True Acceptance Rate (1 − FRR)")
        ax.set_title(titles[ds_name], fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_roc_comparison.png")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    model_keys = ["mfcc_cosine", "mfcc_gmm_ubm", "ecapa_tdnn"]
    short_names = ["MFCC + StatPool\n+ Cosine", "MFCC + GMM-UBM\n(MAP Adaptation)", "ECAPA-TDNN\n(Giải pháp đề xuất)"]
    x = np.arange(len(model_keys))
    w_bar = 0.34

    eer_ls = [comp_results["librispeech"][k]["EER"] * 100 for k in model_keys]
    eer_vn = [comp_results["vietnamese"][k]["EER"] * 100 for k in model_keys]
    b1 = ax1.bar(x - w_bar / 2, eer_ls, w_bar, label="LibriSpeech (Tiếng Anh)", color="#3b82f6")
    b2 = ax1.bar(x + w_bar / 2, eer_vn, w_bar, label="VIVOS (Tiếng Việt)", color="#10b981")
    ax1.set_xticks(x, short_names)
    ax1.set_ylabel("Equal Error Rate - EER (%) ↓")
    ax1.set_title("a) So sánh tỉ lệ lỗi EER (%) — Càng thấp càng tốt", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylim(0, max(eer_ls + eer_vn) * 1.25)

    acc_ls = [comp_results["librispeech"][k]["top1_accuracy_LOO"] * 100 for k in model_keys]
    acc_vn = [comp_results["vietnamese"][k]["top1_accuracy_LOO"] * 100 for k in model_keys]
    b3 = ax2.bar(x - w_bar / 2, acc_ls, w_bar, label="LibriSpeech (Tiếng Anh)", color="#3b82f6")
    b4 = ax2.bar(x + w_bar / 2, acc_vn, w_bar, label="VIVOS (Tiếng Việt)", color="#10b981")
    ax2.set_xticks(x, short_names)
    ax2.set_ylabel("Độ chính xác nhận dạng 1:N Top-1 (%) ↑")
    ax2.set_title("b) So sánh độ chính xác Nhận dạng 1:N (LOO) — Càng cao càng tốt", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="lower right")
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylim(0, 115)

    fig.tight_layout()
    fig.savefig(out_dir / "baseline_metrics_bar.png")
    plt.close(fig)

    return comp_results


def run_extended_evaluation(
    wavs_vn: List[np.ndarray],
    clean_speeches: List[np.ndarray],
    clean_embs: np.ndarray,
    labels_vn: np.ndarray,
    files_vn: List[Path],
    embs_ls: np.ndarray,
    labels_ls: np.ndarray,
    clean_gmm_eer_vn: float,
):
    print("\n" + "=" * 70, flush=True)
    print(" [BƯỚC 4 - TIÊU CHÍ 10] ĐÁNH GIÁ MỞ RỘNG (NHIỄU, NGÔN NGỮ, GIỚI TÍNH)", flush=True)
    print("=" * 70, flush=True)
    out_dir = config.REPORT_DIR / "extended"
    out_dir.mkdir(parents=True, exist_ok=True)
    noisy_data_dir = config.DATA_DIR / "eval_noisy"
    noisy_data_dir.mkdir(parents=True, exist_ok=True)

    enc = SpeakerEncoder.get()
    rng = np.random.default_rng(2026)

    snr_levels = [20, 15, 10, 5, 0]
    noise_types = ["white", "pink", "babble"]
    noise_results: Dict[str, Dict[str, dict]] = {}

    clean_eval = evaluate_embeddings(clean_embs, labels_vn, threshold=0.45)
    clean_frames = [baseline.extract_mfcc_frames(sp, cmvn=True) for sp in clean_speeches]
    gmm_ubm = baseline.GMMUBMVerifier(n_components=16, relevance_factor=10.0).fit_ubm(clean_frames)
    clean_gmm_means = [gmm_ubm.adapt_speaker_means(fr) for fr in clean_frames]

    for n_kind in noise_types:
        noise_results[n_kind] = {"clean": clean_eval}
        for snr in snr_levels:
            noisy_embs = []
            noisy_frames = []
            for idx, (w_raw, sp_clean) in enumerate(zip(wavs_vn, clean_speeches)):
                n_vec = generate_noise(n_kind, len(w_raw), rng, pool_wavs=wavs_vn)
                w_noisy = mix_at_snr(w_raw, n_vec, float(snr))
                if snr == 10 and idx < 15:
                    spk_dir = noisy_data_dir / f"{n_kind}_10dB" / labels_vn[idx]
                    spk_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(str(spk_dir / files_vn[idx].name), w_noisy, config.SAMPLE_RATE)

                sp_noisy, _, _ = audio.apply_vad(w_noisy)
                if len(sp_noisy) < int(1.0 * config.SAMPLE_RATE):
                    sp_noisy = w_noisy
                noisy_embs.append(enc.embed(sp_noisy))
                noisy_frames.append(baseline.extract_mfcc_frames(sp_noisy, cmvn=True))

            noisy_embs_arr = np.stack(noisy_embs)
            S_cross = clean_embs @ noisy_embs_arr.T
            gen_sc, imp_sc = [], []
            for i in range(len(labels_vn)):
                for j in range(len(labels_vn)):
                    if i == j:
                        continue
                    (gen_sc if labels_vn[i] == labels_vn[j] else imp_sc).append(float(S_cross[i, j]))
            gen_arr, imp_arr = np.array(gen_sc), np.array(imp_sc)
            eer_val, eer_thr = compute_eer(gen_arr, imp_arr)
            y_lbl = np.r_[np.ones(len(gen_arr)), np.zeros(len(imp_arr))]
            fpr, tpr, _ = roc_curve(y_lbl, np.r_[gen_arr, imp_arr])
            roc_auc = float(auc(fpr, tpr))

            spks = sorted(set(labels_vn))
            correct = 0
            for i in range(len(labels_vn)):
                cents = []
                for s in spks:
                    m = labels_vn == s
                    m[i] = False
                    cents.append(l2_normalize(clean_embs[m].mean(axis=0)))
                pred = spks[int(np.argmax(np.stack(cents) @ noisy_embs_arr[i]))]
                if pred == labels_vn[i]:
                    correct += 1
            top1_acc = float(correct / len(labels_vn))

            gen_gmm, imp_gmm = [], []
            for i in range(len(labels_vn)):
                for j in range(len(labels_vn)):
                    if i == j:
                        continue
                    sc_gmm = gmm_ubm.score_llr(clean_gmm_means[i], noisy_frames[j])
                    (gen_gmm if labels_vn[i] == labels_vn[j] else imp_gmm).append(sc_gmm)
            eer_gmm, _ = compute_eer(gen_gmm, imp_gmm)

            noise_results[n_kind][f"{snr}dB"] = {
                "snr_db": snr,
                "EER": float(eer_val),
                "threshold_at_EER": float(eer_thr),
                "AUC": roc_auc,
                "genuine_mean": float(gen_arr.mean()),
                "imposter_mean": float(imp_arr.mean()),
                "top1_accuracy": top1_acc,
                "baseline_gmm_EER": float(eer_gmm),
            }
            print(
                f"  * Nhiễu [{n_kind:<6} | SNR={snr:2d}dB]: "
                f"ECAPA EER={eer_val*100:5.2f}%, Top-1={top1_acc*100:5.1f}% "
                f"(vs GMM-UBM EER={eer_gmm*100:5.2f}%)",
                flush=True,
            )

    def analyze_gender_pairs(embs: np.ndarray, labels: np.ndarray, gender_map: Dict[str, str]) -> dict:
        S = embs @ embs.T
        gen, imp_same, imp_diff = [], [], []
        for i, j in itertools.combinations(range(len(labels)), 2):
            if labels[i] == labels[j]:
                gen.append(float(S[i, j]))
            elif gender_map.get(labels[i], "?") == gender_map.get(labels[j], "?"):
                imp_same.append(float(S[i, j]))
            else:
                imp_diff.append(float(S[i, j]))
        eer_same, _ = compute_eer(gen, imp_same)
        eer_diff, _ = compute_eer(gen, imp_diff)
        eer_all, _ = compute_eer(gen, imp_same + imp_diff)
        return {
            "EER_overall": float(eer_all),
            "EER_same_gender_imposter": float(eer_same),
            "EER_cross_gender_imposter": float(eer_diff),
            "genuine_mean": float(np.mean(gen)),
            "same_gender_imposter_mean": float(np.mean(imp_same)),
            "cross_gender_imposter_mean": float(np.mean(imp_diff)),
        }

    ls_genders = {s: s.split("-")[-1] for s in set(labels_ls)}
    vn_genders = {}
    g_file = config.EVAL_DIR / "genders.tsv"
    if g_file.exists():
        for line in g_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                vn_genders[parts[0]] = parts[1].upper()

    gender_analysis = {
        "librispeech_english": analyze_gender_pairs(embs_ls, labels_ls, ls_genders),
        "vivos_vietnamese": analyze_gender_pairs(clean_embs, labels_vn, vn_genders),
    }

    extended_summary = {
        "noise_robustness": noise_results,
        "cross_lingual_and_gender": gender_analysis,
    }
    (out_dir / "extended_metrics.json").write_text(
        json.dumps(extended_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Biểu đồ 1: Độ bền nhiễu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    x_labels = ["Clean\n(∞ dB)", "20 dB", "15 dB", "10 dB", "5 dB", "0 dB"]
    x_idx = np.arange(len(x_labels))
    keys_snr = ["clean", "20dB", "15dB", "10dB", "5dB", "0dB"]

    styles = [
        ("white", "ECAPA-TDNN · Nhiễu trắng (White)", "#2563eb", "o", "-"),
        ("pink", "ECAPA-TDNN · Nhiễu môi trường (Pink)", "#10b981", "s", "-"),
        ("babble", "ECAPA-TDNN · Nhiễu đám đông (Babble)", "#ef4444", "^", "-"),
    ]
    for n_kind, lbl, col, mk, ls in styles:
        eers = [noise_results[n_kind][k]["EER"] * 100 for k in keys_snr]
        accs = [
            (noise_results[n_kind]["clean"]["top1_accuracy_LOO"] if k == "clean" else noise_results[n_kind][k]["top1_accuracy"]) * 100
            for k in keys_snr
        ]
        ax1.plot(x_idx, eers, marker=mk, ls=ls, lw=2.2, color=col, label=lbl)
        ax2.plot(x_idx, accs, marker=mk, ls=ls, lw=2.2, color=col, label=lbl)

    gmm_eers = [clean_gmm_eer_vn * 100] + [noise_results["white"][k]["baseline_gmm_EER"] * 100 for k in keys_snr[1:]]
    ax1.plot(x_idx, gmm_eers, "d--", lw=1.8, color="#64748b", label="Baseline GMM-UBM · Nhiễu trắng")

    ax1.set_xticks(x_idx, x_labels)
    ax1.set_ylabel("Equal Error Rate - EER (%) ↓")
    ax1.set_xlabel("Mức tỉ số tín hiệu trên nhiễu (SNR)")
    ax1.set_title("a) Biến thiên EER (%) theo cường độ nhiễu", fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xticks(x_idx, x_labels)
    ax2.set_ylabel("Độ chính xác nhận dạng 1:N Top-1 (%) ↑")
    ax2.set_xlabel("Mức tỉ số tín hiệu trên nhiễu (SNR)")
    ax2.set_title("b) Độ chính xác Nhận dạng 1:N theo mức SNR", fontweight="bold")
    ax2.set_ylim(40, 105)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_dir / "noise_robustness_curves.png")
    plt.close(fig)

    # Biểu đồ 2: So sánh phổ Spectrogram
    from scipy.signal import stft as sp_stft

    w_demo = wavs_vn[0]
    w_white10 = mix_at_snr(w_demo, generate_noise("white", len(w_demo), rng), 10.0)
    w_babble10 = mix_at_snr(w_demo, generate_noise("babble", len(w_demo), rng, pool_wavs=wavs_vn), 10.0)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.8))
    for ax, sig, ttl in zip(
        axes,
        [w_demo, w_white10, w_babble10],
        [
            "a) Tín hiệu giọng nói tiếng Việt gốc (Clean - VIVOS)",
            "b) Tín hiệu trộn nhiễu trắng (White Gaussian Noise, SNR = 10 dB)",
            "c) Tín hiệu trộn nhiễu đám đông (Multi-speaker Babble Noise, SNR = 10 dB)",
        ],
    ):
        nper = int(0.025 * config.SAMPLE_RATE)
        f, t, Z = sp_stft(sig, fs=config.SAMPLE_RATE, window="hann", nperseg=nper, noverlap=nper - int(0.010 * config.SAMPLE_RATE), nfft=512)
        S_db = 20 * np.log10(np.abs(Z) + 1e-8)
        S_db = np.clip(S_db, S_db.max() - 80, None)
        im = ax.pcolormesh(t, f / 1000, S_db, shading="auto", cmap="viridis")
        ax.set_title(ttl, fontweight="bold", loc="left", fontsize=10)
        ax.set_ylabel("Tần số (kHz)")
        fig.colorbar(im, ax=ax, pad=0.01, label="dB")
    axes[-1].set_xlabel("Thời gian (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "noise_spectrogram_comparison.png")
    plt.close(fig)

    # Biểu đồ 3: Đa ngôn ngữ & Giới tính
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    cats = ["Tổng thể\n(All Pairs)", "Kẻ giả mạo cùng giới\n(Same-Gender Imposter)", "Kẻ giả mạo khác giới\n(Cross-Gender Imposter)"]
    x = np.arange(len(cats))
    w_bar = 0.34
    en_vals = [
        gender_analysis["librispeech_english"]["EER_overall"] * 100,
        gender_analysis["librispeech_english"]["EER_same_gender_imposter"] * 100,
        gender_analysis["librispeech_english"]["EER_cross_gender_imposter"] * 100,
    ]
    vn_vals = [
        gender_analysis["vivos_vietnamese"]["EER_overall"] * 100,
        gender_analysis["vivos_vietnamese"]["EER_same_gender_imposter"] * 100,
        gender_analysis["vivos_vietnamese"]["EER_cross_gender_imposter"] * 100,
    ]
    b1 = ax1.bar(x - w_bar / 2, en_vals, w_bar, label="Tiếng Anh (LibriSpeech)", color="#3b82f6")
    b2 = ax1.bar(x + w_bar / 2, vn_vals, w_bar, label="Tiếng Việt (VIVOS - Zero-shot)", color="#10b981")
    ax1.set_xticks(x, cats)
    ax1.set_ylabel("EER (%) ↓")
    ax1.set_title("a) Đánh giá Đa ngôn ngữ & Tác động giới tính tới EER", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylim(0, max(en_vals + vn_vals) * 1.28 + 0.8)

    score_cats = ["Genuine\n(Cùng người nói)", "Imposter cùng giới\n(Same-Gender)", "Imposter khác giới\n(Cross-Gender)"]
    en_sc = [
        gender_analysis["librispeech_english"]["genuine_mean"],
        gender_analysis["librispeech_english"]["same_gender_imposter_mean"],
        gender_analysis["librispeech_english"]["cross_gender_imposter_mean"],
    ]
    vn_sc = [
        gender_analysis["vivos_vietnamese"]["genuine_mean"],
        gender_analysis["vivos_vietnamese"]["same_gender_imposter_mean"],
        gender_analysis["vivos_vietnamese"]["cross_gender_imposter_mean"],
    ]
    b3 = ax2.bar(x - w_bar / 2, en_sc, w_bar, label="Tiếng Anh (LibriSpeech)", color="#3b82f6")
    b4 = ax2.bar(x + w_bar / 2, vn_sc, w_bar, label="Tiếng Việt (VIVOS - Zero-shot)", color="#10b981")
    ax2.set_xticks(x, score_cats)
    ax2.set_ylabel("Điểm Cosine Similarity trung bình")
    ax2.set_title("b) Phân bố điểm tương đồng theo Ngôn ngữ & Giới tính", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylim(-0.05, 0.85)

    fig.tight_layout()
    fig.savefig(out_dir / "cross_lingual_gender.png")
    plt.close(fig)

    return extended_summary


def main():
    t_start = time.perf_counter()
    print("⏳ Đang nạp và trích xuất đặc trưng cơ sở cho VIVOS (Tiếng Việt) & LibriSpeech (Tiếng Anh)...", flush=True)
    wavs_vn, speech_vn, embs_vn, labels_vn, files_vn, t_vn = load_raw_and_speech(config.EVAL_DIR)
    wavs_ls, speech_ls, embs_ls, labels_ls, files_ls, t_ls = load_raw_and_speech(config.DATA_DIR / "eval_librispeech")

    vn_metrics = run_vietnamese_standard_eval(embs_vn, labels_vn, t_vn)
    ablation = run_ablation_study(wavs_vn, speech_vn, embs_vn, labels_vn, t_vn, embs_ls, labels_ls)
    comparison = run_baseline_comparison(speech_ls, embs_ls, labels_ls, t_ls, speech_vn, embs_vn, labels_vn, t_vn)
    extended = run_extended_evaluation(
        wavs_vn,
        speech_vn,
        embs_vn,
        labels_vn,
        files_vn,
        embs_ls,
        labels_ls,
        clean_gmm_eer_vn=comparison["vietnamese"]["mfcc_gmm_ubm"]["EER"],
    )

    master = {
        "vietnamese_eval": vn_metrics,
        "librispeech_eval": json.loads((config.REPORT_DIR / "librispeech" / "metrics.json").read_text(encoding="utf-8")),
        "ablation": ablation,
        "baseline_comparison": comparison,
        "extended_evaluation": extended,
        "total_experiment_runtime_sec": float(time.perf_counter() - t_start),
    }
    (config.REPORT_DIR / "all_experiments_summary.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "=" * 70, flush=True)
    print(f" ✅ HOÀN TẤT TOÀN BỘ THỰC NGHIỆM TRONG {time.perf_counter() - t_start:.1f}s", flush=True)
    print(f" 📊 Đã lưu tổng hợp tại: {config.REPORT_DIR / 'all_experiments_summary.json'}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
