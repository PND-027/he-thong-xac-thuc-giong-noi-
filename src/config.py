"""Cấu hình chung cho toàn bộ hệ thống xác thực giọng nói."""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Đường dẫn
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = Path(os.getenv("VOICEID_DB", DATA_DIR / "database.db"))
EVAL_DIR = DATA_DIR / "eval"              # data/eval/<ten_nguoi_noi>/*.wav
REPORT_DIR = ROOT_DIR / "reports"         # Biểu đồ phục vụ báo cáo
MODEL_DIR = ROOT_DIR / "pretrained_models"

# ----------------------------------------------------------------------------
# Tiền xử lý tín hiệu
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16_000        # 16 kHz – chuẩn của ECAPA-TDNN (VoxCeleb)
MIN_SPEECH_SEC = 2.0        # Thời lượng giọng nói tối thiểu sau VAD
PEAK_TARGET = 0.95          # Biên độ đỉnh sau Peak Normalization

# ----------------------------------------------------------------------------
# VAD
# ----------------------------------------------------------------------------
VAD_THRESHOLD = 0.5         # Ngưỡng xác suất speech của Silero VAD
VAD_MIN_SPEECH_MS = 250     # Bỏ các đoạn speech ngắn hơn giá trị này
VAD_MIN_SILENCE_MS = 150    # Khoảng lặng tối thiểu để tách 2 đoạn
VAD_SPEECH_PAD_MS = 60      # Đệm thêm 2 đầu mỗi đoạn speech

# ----------------------------------------------------------------------------
# Mô hình & Ra quyết định
# ----------------------------------------------------------------------------
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIM = 192
DEFAULT_THRESHOLD = 0.65    # Ngưỡng tau mặc định cho Cosine Similarity

for _d in (DATA_DIR, EVAL_DIR, REPORT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
