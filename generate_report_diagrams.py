"""
Vẽ các sơ đồ khối kỹ thuật trực quan chất lượng cao (220 DPI) phục vụ Báo cáo Đồ án:
1. diagram_pipeline.png: Sơ đồ luồng xử lý tổng thể.
2. diagram_ecapa.png: Sơ đồ kiến trúc mạng ECAPA-TDNN.
3. diagram_dsp_vad.png: Quy trình tiền xử lý tín hiệu số & cơ chế lọc khoảng lặng VAD.
4. diagram_ui_overview.png: Cấu trúc giao diện ứng dụng Web và cơ sở dữ liệu SQLite.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src import config


def _box(ax, x, y, w, h, title, subtitle="", facecolor="#eff6ff", edgecolor="#2563eb", title_color="#1e3a8a", sub_color="#334155", fontsize=9.5):
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=fontsize - 1.3, color=sub_color)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color)


def _arrow(ax, x1, y1, x2, y2, label="", color="#475569"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", lw=1.6, color=color),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label, ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")


def draw_pipeline_diagram(out: Path):
    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=220)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    ax.text(
        6.25,
        5.28,
        "SƠ ĐỒ KIẾN TRÚC VÀ LUỒNG XỬ LÝ HỆ THỐNG XÁC THỰC NGƯỜI NÓI VOICEID",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )

    _box(ax, 0.3, 3.35, 2.0, 1.25, "1. Tín hiệu Đầu vào", "Micro trực tiếp / File WAV\nĐa tần số, Mono hoặc Stereo", "#f8fafc", "#64748b", "#0f172a")
    _arrow(ax, 2.3, 3.97, 2.85, 3.97)

    _box(ax, 2.85, 3.35, 2.35, 1.25, "2. Chuẩn hóa Tín hiệu", "Trộn Mono · Resample 16 kHz\nKhử DC · Peak Norm 0.95", "#eff6ff", "#2563eb", "#1e3a8a")
    _arrow(ax, 5.20, 3.97, 5.75, 3.97)

    _box(ax, 5.75, 3.35, 2.35, 1.25, "3. Lọc khoảng lặng VAD", "Silero VAD kết hợp Dự phòng\nKiểm tra thời lượng ≥ 2.0s", "#ecfdf5", "#059669", "#065f46")
    _arrow(ax, 8.10, 3.97, 8.65, 3.97)

    _box(ax, 8.65, 3.35, 3.55, 1.25, "4. Trích xuất ECAPA-TDNN", "Mô hình tiền huấn luyện VoxCeleb\nVector đặc trưng E ∈ R^192, ||E||₂ = 1", "#f5f3ff", "#7c3aed", "#4c1d95")

    _arrow(ax, 10.42, 3.35, 2.1, 2.15, "")
    _arrow(ax, 10.42, 3.35, 6.25, 2.15, "")
    _arrow(ax, 10.42, 3.35, 10.35, 2.15, "")

    _box(ax, 0.35, 0.45, 3.5, 1.65, "Luồng A: Đăng ký", "Thu K mẫu giọng → Tính Centroid:\nĒ = Σ E_k / ||Σ E_k||₂\nLưu trữ vào SQLite", "#f0fdf4", "#16a34a", "#14532d")
    _box(ax, 4.45, 0.45, 3.6, 1.65, "Luồng B: Xác thực 1:1", "So khớp với Centroid người khai báo:\nScore = cos(E_test, Ē_claim)\nScore ≥ τ → Hợp lệ | < τ → Từ chối", "#eff6ff", "#2563eb", "#1e3a8a")
    _box(ax, 8.60, 0.45, 3.6, 1.65, "Luồng C: Nhận dạng 1:N Tập mở", "Nhân ma trận với toàn bộ N người:\nS = Gallery_N×192 · E_test\nMax(S) ≥ τ → Định danh | < τ → Người lạ", "#fffbeb", "#d97706", "#78350f")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_ecapa_diagram(out: Path):
    fig, ax = plt.subplots(figsize=(12.5, 4.0), dpi=220)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    ax.text(
        6.25,
        3.62,
        "KIẾN TRÚC MẠNG TRÍCH XUẤT ĐẶC TRƯNG GIỌNG NÓI ECAPA-TDNN",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )

    steps = [
        (0.25, 1.25, 1.65, 1.6, "Tín hiệu sau VAD", "16 kHz Mono\nKhung 25ms / 10ms", "#f8fafc", "#64748b", "#0f172a"),
        (2.20, 1.25, 1.75, 1.6, "80-Mel Filterbank", "Phổ Log-Mel 80 chiều\nChuẩn hóa CMVN", "#eff6ff", "#2563eb", "#1e3a8a"),
        (4.25, 1.25, 2.05, 1.6, "3 × SE-Res2Block", "Tích chập giãn nở đa tỉ lệ\nSqueeze-and-Excitation", "#eef2ff", "#4f46e5", "#312e81"),
        (6.60, 1.25, 1.85, 1.6, "Gộp Đa tầng MFA", "Nối đặc trưng 3 khối\nConv1D 1×1, 1536 kênh", "#f5f3ff", "#7c3aed", "#4c1d95"),
        (8.75, 1.25, 1.85, 1.6, "Attentive Pooling", "Trọng số chú ý theo khung\nGộp Mean μ và Std σ", "#fdf2f8", "#db2777", "#831843"),
        (10.85, 1.25, 1.45, 1.6, "Embedding L2", "FC + BatchNorm\nVector E ∈ R^192", "#ecfdf5", "#059669", "#065f46"),
    ]
    for i, (x, y, w, h, t, s, fc, ec, tc) in enumerate(steps):
        _box(ax, x, y, w, h, t, s, fc, ec, tc, fontsize=9.0)
        if i < len(steps) - 1:
            nx = steps[i + 1][0]
            _arrow(ax, x + w, y + h / 2, nx, y + h / 2)

    ax.annotate(
        "Tổng hợp đặc trưng đa mức từ các tầng trung gian",
        xy=(7.52, 1.25),
        xytext=(5.27, 0.45),
        ha="center",
        fontsize=8.5,
        color="#6d28d9",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.25", color="#6d28d9", lw=1.4, ls="--"),
    )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_dsp_vad_diagram(out: Path):
    fig, ax = plt.subplots(figsize=(12.5, 4.2), dpi=220)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    ax.text(
        6.25,
        3.82,
        "QUY TRÌNH XỬ LÝ TÍN HIỆU SỐ VÀ PHÁT HIỆN HOẠT ĐỘNG TIẾNG NÓI",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )

    steps = [
        (0.25, 1.4, 1.7, 1.55, "Chuẩn hóa PCM", "Đưa về float32 [-1, 1]\nTrộn kênh trung bình Mono", "#f8fafc", "#64748b", "#0f172a"),
        (2.25, 1.4, 1.85, 1.55, "Lấy mẫu lại 16 kHz", "Bộ lọc đa pha Polyphase\nChống hiện tượng chồng phổ", "#eff6ff", "#2563eb", "#1e3a8a"),
        (4.40, 1.4, 1.75, 1.55, "Khử DC & Peak Norm", "x ← x − x̄\nx_norm = 0.95 · x / max|x|", "#eff6ff", "#2563eb", "#1e3a8a"),
        (6.45, 1.4, 2.05, 1.55, "Phân đoạn VAD", "Nhánh chính: Silero VAD\nDự phòng: STE Thích nghi", "#ecfdf5", "#059669", "#065f46"),
        (8.80, 1.4, 1.75, 1.55, "Ghép nối & Đệm", "Loại bỏ khoảng lặng ≥ 150ms\nĐệm biên 60ms bảo toàn âm", "#f0fdf4", "#16a34a", "#14532d"),
        (10.85, 1.4, 1.4, 1.55, "Kiểm định ≥ 2.0s", "Đạt: Trích xuất đặc trưng\nNgắn: Yêu cầu thu lại", "#fffbeb", "#d97706", "#78350f"),
    ]
    for i, (x, y, w, h, t, s, fc, ec, tc) in enumerate(steps):
        _box(ax, x, y, w, h, t, s, fc, ec, tc, fontsize=8.8)
        if i < len(steps) - 1:
            nx = steps[i + 1][0]
            _arrow(ax, x + w, y + h / 2, nx, y + h / 2)

    ax.text(
        6.25,
        0.55,
        "Thông số Silero VAD: ngưỡng xác suất p = 0.5 · đoạn tiếng nói tối thiểu 250 ms · khoảng lặng tách đoạn 150 ms · vùng đệm hai đầu 60 ms",
        ha="center",
        va="center",
        fontsize=9,
        color="#334155",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_ui_overview_diagram(out: Path):
    fig, ax = plt.subplots(figsize=(12.5, 4.6), dpi=220)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    ax.text(
        6.25,
        4.22,
        "CẤU TRÚC GIAO DIỆN ỨNG DỤNG WEB GRADIO VÀ CƠ SỞ DỮ LIỆU SQLITE",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )

    tabs = [
        (0.3, 2.2, 2.2, 1.45, "Tab 1: Đăng ký", "Thu 1–3 mẫu giọng\nKiểm tra độ nhất quán\nTính vector Centroid", "#ecfdf5", "#059669", "#065f46"),
        (2.75, 2.2, 2.2, 1.45, "Tab 2: Xác thực 1:1", "Chọn danh tính khai báo\nThanh đo điểm Cosine\nQuyết định theo ngưỡng τ", "#eff6ff", "#2563eb", "#1e3a8a"),
        (5.2, 2.2, 2.2, 1.45, "Tab 3: Nhận dạng 1:N", "Truy vấn tập mở Open-set\nXếp hạng ứng viên\nPhát hiện người lạ", "#f5f3ff", "#7c3aed", "#4c1d95"),
        (7.65, 2.2, 2.2, 1.45, "Tab 4: Phân tích Tín hiệu", "Sóng âm trước/sau VAD\nPhổ tần số Spectrogram\nƯớc lượng tỉ số SNR", "#fffbeb", "#d97706", "#78350f"),
        (10.1, 2.2, 2.1, 1.45, "Tab 5: Quản lý CSDL", "Danh sách người dùng\nLịch sử thời gian\nThêm hoặc xoá bản ghi", "#f8fafc", "#64748b", "#0f172a"),
    ]
    for x, y, w, h, t, s, fc, ec, tc in tabs:
        _box(ax, x, y, w, h, t, s, fc, ec, tc, fontsize=9.0)

    _box(
        ax,
        1.8,
        0.35,
        8.9,
        1.2,
        "Tầng Lưu trữ Cơ sở Dữ liệu Quan hệ SQLite — data/database.db",
        "Bảng users: id, name, embedding Centroid 192×float32 chuẩn hóa L2, n_samples, created_at, updated_at\n"
        "Bảng samples: id, user_id, embedding từng mẫu riêng lẻ, duration — hỗ trợ cập nhật Centroid tức thì không cần thu lại từ đầu",
        "#eff6ff",
        "#1d4ed8",
        "#1e3a8a",
        fontsize=9.2,
    )

    for x, y, w, _, _, _, _, _, _ in tabs[:3] + [tabs[4]]:
        _arrow(ax, x + w / 2, 2.2, 6.25, 1.55)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    out_dir = config.REPORT_DIR / "diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_pipeline_diagram(out_dir / "diagram_pipeline.png")
    draw_ecapa_diagram(out_dir / "diagram_ecapa.png")
    draw_dsp_vad_diagram(out_dir / "diagram_dsp_vad.png")
    draw_ui_overview_diagram(out_dir / "diagram_ui_overview.png")
    print(f"✅ Đã xuất 4 sơ đồ kỹ thuật tại: {out_dir}")


if __name__ == "__main__":
    main()
