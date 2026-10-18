# 🎙️ VoiceID — Hệ thống Xác thực Người nói Độc lập Văn bản

**Đồ án môn học: Xử lý tiếng nói (Speech Processing)**

Hệ thống xác thực / nhận dạng người nói chạy **offline trên CPU**, dùng mô hình
**ECAPA-TDNN** huấn luyện sẵn (SpeechBrain, VoxCeleb) — **không cần train lại**.
Xem chi tiết thiết kế hệ thống và luồng xử lý tại [ARCHITECTURE.md](ARCHITECTURE.md).

```
Mic / File ─► Mono · 16 kHz · Peak Norm ─► Silero VAD (≥ 2s) ─► ECAPA-TDNN (192-D, L2) ─► Cosine ─► τ ─► Accept / Reject
```

## 1. Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # bản CPU, nhẹ hơn
pip install -r requirements.txt
python check_setup.py        # tải mô hình (~80 MB, chỉ lần đầu) & kiểm tra môi trường
```

## 2. Chạy demo

```powershell
python app.py                # tự mở http://127.0.0.1:7860
```

| Tab | Chức năng |
|---|---|
| ① Đăng ký | Nhập tên + thu 1–3 mẫu giọng → lưu embedding trung bình vào SQLite |
| ② Xác thực 1:1 | Chọn tên → nói 1 câu bất kỳ → điểm Cosine + **HỢP LỆ / TỪ CHỐI** |
| ③ Nhận dạng 1:N | Nói 1 câu → hệ thống tự tìm người khớp nhất (hoặc báo *người lạ*) |
| ④ Phân tích tín hiệu | Sóng âm trước/sau VAD, spectrogram, SNR ước lượng |
| ⑤ Quản lý CSDL | Xem / xoá người dùng |

> Trình duyệt chỉ cho dùng micro trên `localhost`/`127.0.0.1` hoặc HTTPS.

## 3. Đánh giá & vẽ biểu đồ cho báo cáo

### 3a. Đánh giá nhanh trên LibriSpeech (không cần thu âm)

```powershell
python prepare_eval_data.py          # tải ~80 MB, lấy 10 người (5 nam/5 nữ) × 5 câu → data/eval_librispeech/
python evaluate_demo.py --data data\eval_librispeech --out reports\librispeech
```

### 3b. Đánh giá trên dữ liệu tự thu của nhóm

1. Thu âm 5–10 người, mỗi người 3–5 câu (3–6 giây/câu), lưu dạng:
   ```
   data/eval/An/1.wav, 2.wav, ...
   data/eval/Binh/1.wav, ...
   ```
2. Chạy:
   ```powershell
   python evaluate_demo.py              # tuỳ chọn: --threshold 0.65 --data <thư mục>
   ```

Kết quả trong `reports/` (hoặc thư mục `--out`): `score_distribution.png`, `roc_curve.png`, `far_frr.png`,
`embeddings_tsne.png`, `confusion_matrix.png`, `metrics.json` (EER, AUC, FAR, FRR, Top-1 accuracy).

## 4. Cấu trúc mã nguồn

```
├── src/
│   ├── config.py         # Tham số: 16 kHz, ngưỡng τ, tham số VAD, đường dẫn
│   ├── audio.py          # Chuẩn hóa tín hiệu + Silero VAD (+ VAD năng lượng dự phòng)
│   ├── model.py          # ECAPA-TDNN → embedding 192-D chuẩn hóa L2
│   ├── verification.py   # Cosine, xác thực 1:1, nhận dạng 1:N, EER/FAR/FRR
│   ├── database.py       # SQLite: bảng users (centroid) + samples
│   └── visualize.py      # Waveform, spectrogram, biểu đồ xếp hạng
├── assets/style.css      # Giao diện dark / glassmorphism
├── app.py                # Web demo Gradio
├── evaluate_demo.py      # Đánh giá + biểu đồ báo cáo
├── prepare_eval_data.py  # Tải tập con LibriSpeech dev-clean để đánh giá
├── check_setup.py        # Kiểm tra môi trường
└── data/                 # database.db, data/eval/, data/eval_librispeech/
```

## 5. Ghi chú kỹ thuật

- **Đường dẫn có dấu tiếng Việt**: `torch.jit.load(path)` lỗi trên Windows với đường dẫn Unicode,
  nên Silero VAD được nạp từ bộ nhớ (`BytesIO`) — xem `src/audio.py::_get_silero`.
- **Symlink trên Windows**: SpeechBrain được cấu hình `LocalStrategy.COPY` để không cần quyền Admin/Developer Mode.
- **Chọn ngưỡng τ**: chạy `evaluate_demo.py` trên dữ liệu thật, lấy `threshold_at_EER` trong `metrics.json`
  rồi cập nhật `DEFAULT_THRESHOLD` trong `src/config.py`.
