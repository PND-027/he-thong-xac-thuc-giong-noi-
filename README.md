# 🎙️ VoiceID — Hệ thống Xác thực Người nói Độc lập Văn bản

**Đồ án môn học: Xử lý tiếng nói (Speech Processing)**

Hệ thống xác thực $1:1$ và nhận dạng $1:N$ tập mở vận hành hoàn toàn ngoại tuyến trên CPU, ứng dụng mạng học sâu **ECAPA-TDNN** tiền huấn luyện trên VoxCeleb kết hợp bộ lọc khoảng lặng hai tầng **Silero VAD** và **Short-Time Energy VAD**.
Xem chi tiết thiết kế kiến trúc và kết quả thực nghiệm tại [ARCHITECTURE.md](ARCHITECTURE.md) và [PROJECT_PLAN.md](PROJECT_PLAN.md).

```
Mic / File ─► Mono · 16 kHz · Peak Norm ─► Silero VAD (≥ 2s) ─► ECAPA-TDNN (192-D, L2) ─► Cosine ─► τ ─► Accept / Reject
```

## 1. Cài đặt môi trường

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python check_setup.py        # Kiểm tra môi trường & mô hình tiền huấn luyện
```

## 2. Khởi chạy ứng dụng Web Demo

```powershell
python app.py                # Mở giao diện tại http://127.0.0.1:7860
```

| Phân trang | Chức năng kỹ thuật |
|---|---|
| ① Đăng ký | Nhập danh tính + thu 1–3 mẫu giọng → cập nhật vector trọng tâm Centroid chuẩn hóa $L_2$ vào SQLite |
| ② Xác thực 1:1 | Chọn danh tính khai báo → thu âm câu bất kỳ → tính điểm Cosine và quyết định **HỢP LỆ / TỪ CHỐI** |
| ③ Nhận dạng 1:N | Thu âm câu ẩn danh → xếp hạng toàn bộ CSDL và phát hiện **người lạ** theo cơ chế tập mở |
| ④ Phân tích tín hiệu | Trực quan hóa sóng âm trước/sau VAD, phổ tần số STFT và ước lượng tỉ số SNR |
| ⑤ Quản lý CSDL | Xem danh sách người dùng, số mẫu đã đăng ký và xoá bản ghi |

## 3. Thực nghiệm đánh giá & Xuất báo cáo học thuật

### 3.1. Chuẩn bị dữ liệu kiểm thử đa ngôn ngữ
```powershell
python prepare_eval_data.py --dataset librispeech   # 10 người nói tiếng Anh × 5 câu → data/eval_librispeech/
python prepare_eval_data.py --dataset vivos         # 10 người nói tiếng Việt × 5 câu → data/eval/
```

### 3.2. Chạy toàn bộ thực nghiệm đối sánh (Tiêu chí 4, 9, 10)
```powershell
python generate_report_diagrams.py   # Vẽ 4 sơ đồ khối kiến trúc & quy trình xử lý tín hiệu
python run_experiments.py            # Chạy Ablation VAD/Centroid, Đối sánh Baseline & Đa mức nhiễu SNR
python export_report.py              # Xuất file báo cáo chính thức Bao_cao_Do_an_Xu_ly_tieng_noi.docx
```

## 4. Cấu trúc mã nguồn

```
├── src/
│   ├── config.py                  # Tham số hệ thống: 16 kHz, ngưỡng τ, tham số VAD, đường dẫn
│   ├── audio.py                   # Chuẩn hóa tín hiệu số + Silero VAD + VAD năng lượng dự phòng
│   ├── model.py                   # ECAPA-TDNN → vector đặc trưng 192 chiều chuẩn hóa L2
│   ├── baseline.py                # Mô hình tham chiếu: MFCC + StatPool + Cosine & MFCC + GMM-UBM
│   ├── verification.py            # Độ đo Cosine, xác thực 1:1, nhận dạng 1:N, tính EER/FAR/FRR
│   ├── database.py                # Quản lý SQLite: bảng users (centroid) và bảng samples
│   └── visualize.py               # Biểu đồ sóng âm, phổ STFT, xếp hạng và phân phối điểm
├── assets/style.css               # Giao diện tối hiện đại
├── app.py                         # Ứng dụng Web tương tác Gradio
├── evaluate_demo.py               # Đánh giá tiêu chuẩn trên một bộ dữ liệu bất kỳ
├── run_experiments.py             # Thực nghiệm toàn diện: Ablation, Baseline, Đa mức nhiễu
├── generate_report_diagrams.py    # Sinh sơ đồ khối kỹ thuật phục vụ báo cáo
├── export_report.py               # Xuất báo cáo học thuật hoàn chỉnh (.docx)
├── prepare_eval_data.py           # Tải luồng dữ liệu LibriSpeech (Anh) & VIVOS (Việt)
└── data/                          # database.db, data/eval/, data/eval_librispeech/, data/eval_noisy/
```
