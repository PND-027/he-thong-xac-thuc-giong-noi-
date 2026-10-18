"""
Giao diện Web Demo – Hệ thống Xác thực Người nói Độc lập Văn bản.

Chạy:  python app.py      ->  mở http://127.0.0.1:7860
"""
from __future__ import annotations

import html
import inspect
import itertools
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import audio, config, visualize  # noqa: E402
from src.database import SpeakerDB  # noqa: E402
from src.model import SpeakerEncoder  # noqa: E402
from src.verification import cosine_similarity, identify, verify  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for noisy in ("httpx", "speechbrain", "huggingface_hub", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("app")

db = SpeakerDB()
AudioInput = Any


# =============================================================================
# Tiện ích
# =============================================================================
def _warmup():
    """Tải trước mô hình ở luồng nền để lần bấm đầu tiên không phải chờ."""
    try:
        audio._get_silero()
        SpeakerEncoder.get()
        log.info("Mô hình đã sẵn sàng.")
    except Exception:
        log.exception("Không thể tải mô hình")


class NoAudioError(ValueError):
    pass


def _prepare(a: AudioInput, check_length: bool = True) -> audio.AudioResult:
    if a is None:
        raise NoAudioError(
            "Chưa có âm thanh. Hãy bấm 🎙️ để ghi âm (nhớ bấm nút ⏹️ Dừng sau khi nói xong) hoặc tải file âm thanh lên."
        )

    # Xử lý đa dạng kiểu dữ liệu trả về từ Gradio (tuple numpy, filepath string/Path, dict)
    if isinstance(a, (str, Path)):
        data, sr = audio.load_audio(a)
    elif isinstance(a, dict):
        if "data" in a and "sample_rate" in a:
            sr, data = a["sample_rate"], a["data"]
        elif "path" in a:
            data, sr = audio.load_audio(a["path"])
        else:
            raise NoAudioError("Định dạng âm thanh không hợp lệ.")
    elif isinstance(a, (tuple, list)) and len(a) == 2:
        sr, data = a
    else:
        raise NoAudioError("Định dạng âm thanh không hợp lệ.")

    if data is None or len(data) == 0:
        raise NoAudioError("File âm thanh rỗng.")
    return audio.process(data, sr, check_length=check_length)


def _embed(r: audio.AudioResult) -> np.ndarray:
    return SpeakerEncoder.get().embed(r.speech)


def _close(*figs):
    for f in figs:
        if f is not None:
            plt.close(f)


def _esc(s: str) -> str:
    return html.escape(str(s))


def _idle(msg: str) -> str:
    return f'<div class="vx-card idle">{msg}</div>'


def _error(title: str, msg: str, kind: str = "bad") -> str:
    icon = "⚠️" if kind == "warn" else "⛔"
    return (f'<div class="vx-card {kind}"><div class="vx-body">'
            f'<div class="vx-title">{icon} {_esc(title)}</div>'
            f'<div class="vx-sub">{_esc(msg)}</div></div></div>')


def _gauge(score: float, color: str, caption: str = "cosine") -> str:
    pct = max(0.0, min(1.0, score)) * 100
    return (f'<div class="vx-gauge" style="--p:{pct:.1f};--c:{color}">'
            f'<div><b>{score:.3f}</b><small>{caption}</small></div></div>')


def _meter(score: float, thr: float, color: str) -> str:
    w = max(0.0, min(1.0, score)) * 100
    return (f'<div class="vx-meter"><div class="fill" style="width:{w:.1f}%;'
            f'background:linear-gradient(90deg,{color}88,{color})"></div>'
            f'<div class="tick" style="left:{thr*100:.1f}%" data-l="τ = {thr:.2f}"></div></div>')


def _chips(*items: str) -> str:
    return '<div class="vx-chips">' + "".join(f'<span class="vx-chip">{i}</span>' for i in items) + "</div>"


def prompt_html(idx: int, name: str = "") -> str:
    name_clean = (name or "").strip()
    display_name = html.escape(name_clean) if name_clean else "[Họ và tên của bạn]"
    sentences = [
        "“Bản thu âm này đang được ghi trong không gian yên tĩnh, tín hiệu rõ ràng và không bị lẫn tạp âm môi trường.”",
        f"“Tôi là <b class='vx-prompt-name'>{display_name}</b>, đang thực hiện thu âm mẫu giọng nói để đăng ký vào hệ thống xác thực sinh trắc học.”",
        "“Dữ liệu âm thanh sẽ được trích xuất vector đặc trưng ECAPA-TDNN nhằm đảm bảo bảo mật và chống giả mạo.”",
    ]
    meta = [
        ("Mẫu 1 (Bắt buộc) · Khai báo danh tính", "~4–5 giây"),
        ("Mẫu 2 (Khuyến khích) · Điều kiện môi trường", "~4–5 giây"),
        ("Mẫu 3 (Tuỳ chọn) · Đặc trưng sinh trắc", "~4–5 giây"),
    ]
    tag, dur = meta[idx - 1]
    return (
        f'<div class="vx-prompt-box">'
        f'<div class="vx-prompt-meta"><span class="vx-prompt-tag">📖 {tag}</span>'
        f'<span class="vx-prompt-dur">⏱ {dur}</span></div>'
        f'<div class="vx-prompt-text">{sentences[idx - 1]}</div>'
        f'</div>'
    )


def update_prompts(name: str):
    return prompt_html(1, name), prompt_html(2, name), prompt_html(3, name)


def hero_html() -> str:
    n = len(db)
    return f"""
<div class="vx-hero">
  <div class="vx-badge"><span class="vx-dot"></span> Đồ án môn học · Xử lý tiếng nói</div>
  <h1>VoiceID — Hệ thống Xác thực Người nói</h1>
  <p>Text-Independent Speaker Verification · ECAPA-TDNN (SpeechBrain, pretrained VoxCeleb)
     · Silero VAD · Cosine Similarity · Chạy hoàn toàn offline trên CPU</p>
  <div class="vx-stats">
    <div class="vx-stat"><b>{n}</b><span>Người đã đăng ký</span></div>
    <div class="vx-stat"><b>192-D</b><span>Speaker embedding (L2-norm)</span></div>
    <div class="vx-stat"><b>16 kHz</b><span>Mono · Peak-normalized</span></div>
    <div class="vx-stat"><b>τ = {config.DEFAULT_THRESHOLD:.2f}</b><span>Ngưỡng quyết định mặc định</span></div>
  </div>
</div>"""


def _users_table():
    rows = db.list_users()
    return [[i + 1, u["name"], u["n_samples"], u["created_at"].replace("T", " "),
             u["updated_at"].replace("T", " ")] for i, u in enumerate(rows)]


def _refresh_all():
    names = db.names()
    return (
        hero_html(),
        gr.update(choices=names, value=names[0] if names else None),
        gr.update(choices=names, value=None),
        _users_table(),
    )


# =============================================================================
# Tab 1 – Đăng ký
# =============================================================================
def enroll_fn(name: str, a1: AudioInput, a2: AudioInput, a3: AudioInput, overwrite: bool):
    name = (name or "").strip()
    if not name:
        return _error("Thiếu tên", "Vui lòng nhập tên người dùng trước khi đăng ký.", "warn"), None, *_refresh_all()

    t0 = time.perf_counter()
    embs, durs, lines, first = [], [], [], None
    for idx, a in enumerate((a1, a2, a3), start=1):
        if a is None:
            continue
        try:
            r = _prepare(a)
            embs.append(_embed(r))
            durs.append(r.speech_duration)
            first = first or r
            lines.append(f'<li><span class="vx-ok">✔</span> Mẫu {idx}: {r.duration:.2f}s ghi âm → '
                         f'<b>{r.speech_duration:.2f}s</b> giọng nói ({len(r.segments)} đoạn, VAD {r.vad_backend})</li>')
        except (audio.SpeechTooShortError, NoAudioError) as e:
            lines.append(f'<li><span class="vx-bad">✘</span> Mẫu {idx}: {_esc(e)}</li>')
        except Exception as e:  # noqa: BLE001
            log.exception("Lỗi xử lý mẫu %s", idx)
            lines.append(f'<li><span class="vx-bad">✘</span> Mẫu {idx}: {_esc(e)}</li>')

    if not embs:
        body = "".join(lines) or "<li>Chưa có mẫu giọng nói nào. Hãy thu âm ít nhất 1 mẫu (≥ 2 giây).</li>"
        return (f'<div class="vx-card bad"><div class="vx-body"><div class="vx-title">⛔ Đăng ký thất bại</div>'
                f'<ul class="vx-list">{body}</ul></div></div>'), None, *_refresh_all()

    # Độ nhất quán giữa các mẫu (cùng 1 người thì cosine giữa các mẫu phải cao)
    consistency = ""
    if len(embs) > 1:
        sims = [cosine_similarity(x, y) for x, y in itertools.combinations(embs, 2)]
        lo = min(sims)
        cls = "vx-ok" if lo >= config.DEFAULT_THRESHOLD else "vx-warn"
        consistency = (f'<li><span class="{cls}">◆</span> Độ nhất quán giữa các mẫu: trung bình '
                       f'<b>{np.mean(sims):.3f}</b>, thấp nhất <b>{lo:.3f}</b>'
                       + (" — có thể do nhiễu hoặc khác người nói, nên thu âm lại." if lo < config.DEFAULT_THRESHOLD else "")
                       + "</li>")

    existed = name in db.names()
    total = db.enroll(name, embs, durs, append=not overwrite)
    dt = time.perf_counter() - t0
    action = "Ghi đè" if (existed and overwrite) else ("Bổ sung mẫu cho" if existed else "Đăng ký mới")
    card = (
        f'<div class="vx-card ok">{_gauge(1.0, "#10b981", "đã lưu")}<div class="vx-body">'
        f'<div class="vx-title">✅ {action}: {_esc(name)}</div>'
        f'<div class="vx-sub">Embedding đại diện = trung bình {total} vector mẫu, chuẩn hóa L2, lưu vào SQLite.</div>'
        f'<ul class="vx-list">{"".join(lines)}{consistency}</ul>'
        f'{_chips(f"{len(embs)} mẫu mới", f"Tổng {total} mẫu", f"Xử lý {dt:.2f}s")}'
        f'</div></div>'
    )
    fig = visualize.plot_vad(first.raw, first.speech, first.segments, backend=first.vad_backend)
    out = (card, fig, *_refresh_all())
    _close(fig)
    return out


# =============================================================================
# Tab 2 – Xác thực 1:1
# =============================================================================
def verify_fn(name: str, a: AudioInput, thr: float):
    if not name:
        return _error("Chưa chọn người dùng", "Hãy đăng ký ít nhất 1 người ở tab Đăng ký.", "warn"), None, None
    enroll_emb = db.get(name)
    if enroll_emb is None:
        return _error("Không tìm thấy", f"Người dùng '{name}' không tồn tại.", "warn"), None, None
    try:
        t0 = time.perf_counter()
        r = _prepare(a)
        res = verify(_embed(r), enroll_emb, name, thr)
        dt = time.perf_counter() - t0
    except audio.SpeechTooShortError as e:
        return _error("Câu nói quá ngắn", str(e), "warn"), None, None
    except NoAudioError as e:
        return _error("Thiếu âm thanh", str(e), "warn"), None, None
    except Exception as e:  # noqa: BLE001
        log.exception("verify")
        return _error("Lỗi xử lý", str(e)), None, None

    color = "#10b981" if res.accepted else "#ef4444"
    title = "✅ HỢP LỆ — Đúng người nói" if res.accepted else "⛔ TỪ CHỐI — Nghi ngờ giả mạo"
    verdict = "Genuine" if res.accepted else "Imposter"
    card = (
        f'<div class="vx-card {"ok" if res.accepted else "bad"}">{_gauge(res.score, color)}'
        f'<div class="vx-body"><div class="vx-title">{title}</div>'
        f'<div class="vx-sub">Danh tính khai báo: <b>{_esc(name)}</b> · Quyết định: '
        f'<b>{verdict}</b> (Score {"≥" if res.accepted else "<"} τ)</div>'
        f'{_meter(res.score, thr, color)}'
        f'{_chips(f"Cosine = {res.score:.4f}", f"Độ tương đồng ≈ {max(res.score, 0)*100:.1f}%", f"τ = {thr:.2f}", f"Giọng nói {r.speech_duration:.2f}s", f"Xử lý {dt:.2f}s")}'
        f'</div></div>'
    )
    f1 = visualize.plot_vad(r.raw, r.speech, r.segments, backend=r.vad_backend)
    f2 = visualize.plot_spectrogram(r.speech, title="Phổ của đoạn giọng nói (sau VAD)")
    out = (card, f1, f2)
    _close(f1, f2)
    return out


# =============================================================================
# Tab 3 – Nhận dạng 1:N
# =============================================================================
def identify_fn(a: AudioInput, thr: float):
    gallery = db.all_embeddings()
    if not gallery:
        return _error("CSDL trống", "Hãy đăng ký ít nhất 1 người ở tab Đăng ký.", "warn"), None
    try:
        t0 = time.perf_counter()
        r = _prepare(a)
        res = identify(_embed(r), gallery, thr)
        dt = time.perf_counter() - t0
    except audio.SpeechTooShortError as e:
        return _error("Câu nói quá ngắn", str(e), "warn"), None
    except NoAudioError as e:
        return _error("Thiếu âm thanh", str(e), "warn"), None
    except Exception as e:  # noqa: BLE001
        log.exception("identify")
        return _error("Lỗi xử lý", str(e)), None

    best_name, best = res.best
    margin = best - res.ranking[1][1] if len(res.ranking) > 1 else None
    if res.is_known:
        cls, color, title = "ok", "#10b981", f"🎯 Người đang nói là: {_esc(best_name)}"
        sub = f"Điểm cao nhất vượt ngưỡng τ = {thr:.2f} → chấp nhận nhận dạng."
    else:
        cls, color, title = "bad", "#ef4444", "❓ Người lạ — không có trong CSDL"
        sub = (f"Ứng viên gần nhất là <b>{_esc(best_name)}</b> nhưng điểm {best:.3f} &lt; τ = {thr:.2f} "
               f"(Open-set: từ chối thay vì đoán bừa).")
    chips = [f"Top-1 = {best:.4f}", f"So khớp {len(gallery)} người", f"Xử lý {dt:.2f}s"]
    if margin is not None:
        chips.insert(1, f"Cách biệt Top-2 = {margin:.3f}")
    card = (f'<div class="vx-card {cls}">{_gauge(best, color, "top-1")}<div class="vx-body">'
            f'<div class="vx-title">{title}</div><div class="vx-sub">{sub}</div>'
            f'{_meter(best, thr, color)}{_chips(*chips)}</div></div>')
    fig = visualize.plot_ranking(res.ranking, thr)
    out = (card, fig)
    _close(fig)
    return out


# =============================================================================
# Tab 4 – Phân tích tín hiệu
# =============================================================================
def analyze_fn(a: AudioInput):
    try:
        r = _prepare(a, check_length=False)
    except Exception as e:  # noqa: BLE001
        return _error("Lỗi", str(e), "warn"), None, None

    # Ước lượng SNR: năng lượng vùng speech so với vùng im lặng
    mask = np.zeros(len(r.raw), dtype=bool)
    for s, e in r.segments:
        mask[s:e] = True
    p_speech = float(np.mean(r.raw[mask] ** 2)) if mask.any() else 0.0
    p_noise = float(np.mean(r.raw[~mask] ** 2)) if (~mask).any() else 0.0
    snr = 10 * np.log10(p_speech / p_noise) if p_noise > 1e-12 and p_speech > 0 else float("nan")
    ok = r.speech_duration >= config.MIN_SPEECH_SEC
    cls = "ok" if ok else "warn"
    title = "✅ Đủ điều kiện xác thực" if ok else f"⚠️ Giọng nói < {config.MIN_SPEECH_SEC:.1f}s — chưa đủ điều kiện"
    card = (
        f'<div class="vx-card {cls}">{_gauge(r.speech_ratio, "#2563eb", "tỉ lệ speech")}<div class="vx-body">'
        f'<div class="vx-title">{title}</div>'
        f'<div class="vx-sub">Tiền xử lý: Mono → 16 kHz (polyphase) → bỏ DC → Peak Norm → VAD ({r.vad_backend}).</div>'
        + _chips(f"Tổng {r.duration:.2f}s", f"Giọng nói {r.speech_duration:.2f}s",
                 f"Im lặng bị cắt {r.duration - r.speech_duration:.2f}s", f"{len(r.segments)} đoạn speech",
                 f"SNR ước lượng ≈ {snr:.1f} dB" if np.isfinite(snr) else "SNR: không đủ vùng lặng")
        + "</div></div>"
    )
    f1 = visualize.plot_vad(r.raw, r.speech, r.segments, backend=r.vad_backend)
    f2 = visualize.plot_spectrogram(r.raw, title="Phổ tín hiệu gốc (STFT · Hann 25 ms · hop 10 ms)")
    out = (card, f1, f2)
    _close(f1, f2)
    return out


# =============================================================================
# Tab 5 – Quản lý
# =============================================================================
def delete_fn(name: str):
    if not name:
        msg = _error("Chưa chọn", "Chọn người dùng cần xoá.", "warn")
    elif db.delete(name):
        msg = f'<div class="vx-card ok"><div class="vx-body"><div class="vx-title">🗑️ Đã xoá {_esc(name)}</div></div></div>'
    else:
        msg = _error("Không tìm thấy", name, "warn")
    return (msg, *_refresh_all())


# =============================================================================
# Bố cục giao diện
# =============================================================================
THEME = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
).set(
    body_background_fill_dark="#090d16",
    background_fill_primary_dark="#090d16",
    background_fill_secondary_dark="#111827",
    block_background_fill_dark="#111827",
    block_border_color_dark="#1e293b",
    block_border_width="1px",
    block_radius="6px",
    input_background_fill_dark="#0b0f19",
    input_border_color_dark="#1e293b",
    input_radius="4px",
    button_large_radius="4px",
    button_medium_radius="4px",
    button_small_radius="4px",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_dark="#2563eb",
    button_primary_border_color="#1d4ed8",
    button_primary_border_color_dark="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    table_radius="4px",
    checkbox_border_radius="3px",
)
CSS = (config.ROOT_DIR / "assets" / "style.css").read_text(encoding="utf-8")
FORCE_DARK = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

AUDIO_KW = dict(sources=["microphone", "upload"], type="numpy")
PIPELINE_HTML = """
<div class="vx-pipeline">
  <div class="vx-step">🎙️ Micro / File<small>bất kỳ sr, stereo</small></div><span class="vx-arrow">→</span>
  <div class="vx-step">Chuẩn hóa<small>Mono · 16 kHz · Peak Norm</small></div><span class="vx-arrow">→</span>
  <div class="vx-step">Silero VAD<small>cắt khoảng lặng · ≥ 2s</small></div><span class="vx-arrow">→</span>
  <div class="vx-step">ECAPA-TDNN<small>embedding 192-D · L2</small></div><span class="vx-arrow">→</span>
  <div class="vx-step">Cosine Score<small>so với ngưỡng τ</small></div><span class="vx-arrow">→</span>
  <div class="vx-step">✅ / ⛔<small>Accept · Reject</small></div>
</div>"""

GUIDE_MD = f"""
### Kiến trúc hệ thống
{PIPELINE_HTML}

| Thành phần | Kỹ thuật | Ghi chú |
|---|---|---|
| Tiền xử lý | Mono → Resample 16 kHz (polyphase, chống aliasing) → bỏ DC → Peak Normalization | `src/audio.py` |
| VAD | **Silero VAD** (mạng nơ-ron, ~2 MB); dự phòng: VAD năng lượng ngắn hạn | Cắt im lặng, hơi thở; yêu cầu ≥ {config.MIN_SPEECH_SEC:.0f}s giọng nói |
| Trích đặc trưng | **ECAPA-TDNN** (SE-Res2Net + Multi-layer Feature Aggregation + Attentive Statistics Pooling) | Pretrained VoxCeleb 1+2 (7.000+ người), **không cần huấn luyện lại** |
| So khớp | $\\text{{Score}} = E_{{enroll}} \\cdot E_{{test}} = \\cos(E_{{enroll}}, E_{{test}})$ | Vector đã chuẩn hóa L2 |
| Quyết định | Score ≥ τ → **Accept**, ngược lại **Reject** | τ mặc định = {config.DEFAULT_THRESHOLD} (chỉnh bằng thanh trượt) |
| Lưu trữ | SQLite: bảng `users` (centroid) + `samples` (từng mẫu) | `data/database.db` |

### Mẹo demo
1. **Đăng ký**: mỗi người thu 2–3 câu, mỗi câu 3–6 giây, nói tự nhiên, nơi yên tĩnh.
2. **Xác thực**: nói một câu **bất kỳ** (khác nội dung lúc đăng ký) → hệ thống vẫn nhận ra vì *độc lập văn bản*.
3. **Thử giả mạo**: chọn tên bạn A nhưng để bạn B nói → điểm thấp, bị **Từ chối**.
4. **Đánh giá**: chạy `python evaluate_demo.py` để vẽ histogram Genuine/Imposter, ROC, EER cho báo cáo.
"""


def build_ui() -> gr.Blocks:
    blocks_kw, launch_kw = {}, {}
    style = dict(theme=THEME, css=CSS, js=FORCE_DARK)
    if "css" in inspect.signature(gr.Blocks.launch).parameters:   # Gradio >= 6
        launch_kw.update(style)
    else:                                                          # Gradio 4/5
        blocks_kw.update(style)

    with gr.Blocks(title="VoiceID – Xác thực Người nói", **blocks_kw) as demo:
        hero = gr.HTML(hero_html())

        with gr.Tabs():
            # ------------------------------------------------------------ Tab 1
            with gr.Tab("① Đăng ký", id="tab-enroll"):
                gr.HTML('<p class="vx-hint">Nhập tên và thu <b>1–3 mẫu</b> giọng nói (mỗi mẫu ≥ 2 giây giọng nói thực). '
                        'Hãy đọc theo <b>3 câu mẫu kỹ thuật</b> bên dưới (mỗi câu ~4–5 giây, ghép lại thành bản khai báo liền mạch) '
                        'để đạt chất lượng nhận dạng chuẩn xác nhất.<br>'
                        '💡 <b>Lưu ý khi thu âm:</b> Bấm biểu tượng 🎙️ (Micro) để bắt đầu (cấp quyền micro nếu trình duyệt hỏi) → '
                        'Đọc to câu mẫu → <b>Bấm nút ⏹️ (Dừng) để kết thúc ghi âm</b> (chờ sóng âm xuất hiện) → Bấm <b>Lưu giọng nói</b>.</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="vx-panel"):
                        en_name = gr.Textbox(label="Tên người dùng", placeholder="VD: Nguyễn Văn An", elem_id="enroll-name")
                        en_p1 = gr.HTML(prompt_html(1, ""))
                        en_a1 = gr.Audio(label="Thu âm Mẫu 1 (bắt buộc)", elem_id="enroll-audio-1", **AUDIO_KW)
                        en_p2 = gr.HTML(prompt_html(2, ""))
                        en_a2 = gr.Audio(label="Thu âm Mẫu 2 (khuyến khích)", elem_id="enroll-audio-2", **AUDIO_KW)
                        en_p3 = gr.HTML(prompt_html(3, ""))
                        en_a3 = gr.Audio(label="Thu âm Mẫu 3 (tuỳ chọn)", elem_id="enroll-audio-3", **AUDIO_KW)
                        en_over = gr.Checkbox(label="Ghi đè nếu tên đã tồn tại (bỏ chọn = bổ sung thêm mẫu)", value=False,
                                              elem_id="enroll-overwrite")
                        en_btn = gr.Button("💾 Lưu giọng nói", variant="primary", size="lg", elem_id="enroll-btn")
                    with gr.Column(scale=6):
                        en_out = gr.HTML(_idle("Kết quả đăng ký sẽ hiển thị ở đây"))
                        en_plot = gr.Plot(label="Sóng âm trước / sau VAD (mẫu đầu tiên)")

            # ------------------------------------------------------------ Tab 2
            with gr.Tab("② Xác thực 1:1", id="tab-verify"):
                gr.HTML('<p class="vx-hint">Chọn danh tính cần kiểm tra → nói <b>một câu bất kỳ</b> → hệ thống so khớp '
                        'Cosine với giọng đã đăng ký và quyết định <b>Chấp nhận / Từ chối</b>.<br>'
                        '💡 <b>Thu âm:</b> Bấm 🎙️ để nói → <b>Bấm ⏹️ (Dừng)</b> để hoàn tất → Bấm <b>Xác thực</b>.</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="vx-panel"):
                        names = db.names()
                        ve_name = gr.Dropdown(choices=names, value=names[0] if names else None,
                                              label="Tôi là…", elem_id="verify-name")
                        ve_audio = gr.Audio(label="Giọng nói cần xác thực", elem_id="verify-audio", **AUDIO_KW)
                        ve_thr = gr.Slider(0.0, 1.0, value=config.DEFAULT_THRESHOLD, step=0.01,
                                           label="Ngưỡng quyết định τ", elem_id="verify-threshold",
                                           info="Tăng τ → an toàn hơn (ít FAR) nhưng dễ từ chối nhầm (FRR tăng)")
                        ve_btn = gr.Button("🔐 Xác thực", variant="primary", size="lg", elem_id="verify-btn")
                    with gr.Column(scale=6):
                        ve_out = gr.HTML(_idle("Kết quả xác thực sẽ hiển thị ở đây"))
                        ve_plot = gr.Plot(label="Sóng âm trước / sau VAD")
                        ve_spec = gr.Plot(label="Spectrogram")

            # ------------------------------------------------------------ Tab 3
            with gr.Tab("③ Nhận dạng 1:N", id="tab-identify"):
                gr.HTML('<p class="vx-hint">Không cần khai báo tên — hệ thống tự tìm <b>ai đang nói</b> trong CSDL. '
                        'Nếu điểm cao nhất &lt; τ → kết luận <b>người lạ</b> (open-set).<br>'
                        '💡 <b>Thu âm:</b> Bấm 🎙️ để nói → <b>Bấm ⏹️ (Dừng)</b> để hoàn tất → Bấm <b>Ai đang nói?</b>.</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="vx-panel"):
                        id_audio = gr.Audio(label="Giọng nói", elem_id="identify-audio", **AUDIO_KW)
                        id_thr = gr.Slider(0.0, 1.0, value=config.DEFAULT_THRESHOLD, step=0.01,
                                           label="Ngưỡng τ (open-set)", elem_id="identify-threshold")
                        id_btn = gr.Button("🔎 Ai đang nói?", variant="primary", size="lg", elem_id="identify-btn")
                    with gr.Column(scale=6):
                        id_out = gr.HTML(_idle("Kết quả nhận dạng sẽ hiển thị ở đây"))
                        id_plot = gr.Plot(label="Xếp hạng ứng viên")

            # ------------------------------------------------------------ Tab 4
            with gr.Tab("④ Phân tích tín hiệu", id="tab-analyze"):
                gr.HTML('<p class="vx-hint">Trực quan hoá các bước <b>xử lý tín hiệu</b>: sóng âm, vùng giọng nói do VAD '
                        'phát hiện, phổ tần số — dùng cho phần thuyết trình.<br>'
                        '💡 <b>Thu âm:</b> Bấm 🎙️ để nói → <b>Bấm ⏹️ (Dừng)</b> để hoàn tất → Bấm <b>Phân tích</b>.</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, elem_classes="vx-panel"):
                        an_audio = gr.Audio(label="Âm thanh", elem_id="analyze-audio", **AUDIO_KW)
                        an_btn = gr.Button("📊 Phân tích", variant="primary", size="lg", elem_id="analyze-btn")
                    with gr.Column(scale=7):
                        an_out = gr.HTML(_idle("Thông số tín hiệu sẽ hiển thị ở đây"))
                        an_plot = gr.Plot(label="Voice Activity Detection")
                        an_spec = gr.Plot(label="Spectrogram")

            # ------------------------------------------------------------ Tab 5
            with gr.Tab("⑤ Quản lý CSDL", id="tab-manage"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, elem_classes="vx-panel"):
                        users = gr.Dataframe(value=_users_table(), interactive=False, elem_id="users-table",
                                             headers=["#", "Tên", "Số mẫu", "Đăng ký lúc", "Cập nhật lúc"])
                        refresh_btn = gr.Button("🔄 Làm mới", elem_id="refresh-btn")
                    with gr.Column(scale=4, elem_classes="vx-panel"):
                        del_name = gr.Dropdown(choices=db.names(), label="Chọn người dùng cần xoá", elem_id="delete-name")
                        del_btn = gr.Button("🗑️ Xoá người dùng", variant="stop", elem_id="delete-btn")
                        del_out = gr.HTML()

            # ------------------------------------------------------------ Tab 6
            with gr.Tab("ℹ️ Giới thiệu", id="tab-about"):
                gr.Markdown(GUIDE_MD, latex_delimiters=[{"left": "$", "right": "$", "display": False}])

        refresh_outputs = [hero, ve_name, del_name, users]
        en_name.input(update_prompts, inputs=en_name, outputs=[en_p1, en_p2, en_p3])
        en_name.change(update_prompts, inputs=en_name, outputs=[en_p1, en_p2, en_p3])
        en_btn.click(enroll_fn, [en_name, en_a1, en_a2, en_a3, en_over], [en_out, en_plot, *refresh_outputs])
        ve_btn.click(verify_fn, [ve_name, ve_audio, ve_thr], [ve_out, ve_plot, ve_spec])
        id_btn.click(identify_fn, [id_audio, id_thr], [id_out, id_plot])
        an_btn.click(analyze_fn, [an_audio], [an_out, an_plot, an_spec])
        del_btn.click(delete_fn, [del_name], [del_out, *refresh_outputs])
        refresh_btn.click(_refresh_all, None, refresh_outputs)
        demo.load(_refresh_all, None, refresh_outputs)

    demo._vx_launch_kw = launch_kw  # type: ignore[attr-defined]
    return demo


if __name__ == "__main__":
    threading.Thread(target=_warmup, daemon=True).start()
    app = build_ui()
    app.queue().launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", 7860)),
                       inbrowser=os.getenv("NO_BROWSER") is None, **app._vx_launch_kw)
