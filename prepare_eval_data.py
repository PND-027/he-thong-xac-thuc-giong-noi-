"""
Tải một tập con nhỏ của LibriSpeech dev-clean để đánh giá hệ thống (Bước 5.1).

LibriSpeech (OpenSLR #12, giấy phép CC BY 4.0) – tiếng Anh đọc sách, 16 kHz, FLAC.
Script đọc file .tar.gz theo kiểu *streaming*: chỉ tải phần đầu của kho (~vài chục MB)
cho đến khi gom đủ số người nói cần thiết rồi dừng, không cần tải cả 337 MB.

Chạy:
    python prepare_eval_data.py                         # 10 người (5 nam / 5 nữ) × 5 câu
    python prepare_eval_data.py --speakers 12 --utts 6 --out data/eval_librispeech

Kết quả:
    data/eval_librispeech/
      ├── LS-1272-M/   1272-128104-0000.flac ...
      ├── LS-1462-F/   ...
      └── speakers.json  (giới tính, chương sách, thời lượng từng câu)

Sau đó:
    python evaluate_demo.py --data data/eval_librispeech --out reports/librispeech
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import requests
import soundfile as sf

from src import config

MIRRORS = [
    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "https://us.openslr.org/resources/12/dev-clean.tar.gz",
    "https://openslr.trmal.net/resources/12/dev-clean.tar.gz",
]
MIN_DUR, MAX_DUR = 3.0, 12.0      # Chọn câu dài 3–12 s (sau VAD vẫn ≥ 2 s)
MAX_CANDIDATES_PER_CHAPTER = 8


class _CountingReader(io.RawIOBase):
    """Bọc luồng HTTP để hiển thị số MB đã tải."""

    def __init__(self, raw):
        self.raw, self.n = raw, 0

    def readable(self):
        return True

    def readinto(self, b):
        chunk = self.raw.read(len(b))
        if not chunk:
            return 0
        b[: len(chunk)] = chunk
        self.n += len(chunk)
        if self.n // (5 << 20) != (self.n - len(chunk)) // (5 << 20):
            print(f"\r  ⇣ đã tải {self.n / 2**20:6.1f} MB", end="", flush=True)
        return len(chunk)


def parse_speakers_txt(text: str) -> dict:
    """SPEAKERS.TXT: ID | SEX | SUBSET | MINUTES | NAME"""
    out = {}
    for line in text.splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[2] == "dev-clean":
            out[parts[0]] = parts[1]
    return out


def select(cands: dict, k: int) -> list:
    """Chọn k câu, xoay vòng qua các chương sách để đa dạng nội dung/phiên ghi âm."""
    chapters = [sorted(v) for _, v in sorted(cands.items())]
    picked, i = [], 0
    while len(picked) < k and any(chapters):
        ch = chapters[i % len(chapters)]
        if ch:
            picked.append(ch.pop(0))
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser(description="Tải tập con LibriSpeech dev-clean để đánh giá")
    ap.add_argument("--speakers", type=int, default=10, help="Số người nói (chia đều nam/nữ)")
    ap.add_argument("--utts", type=int, default=5, help="Số câu mỗi người")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "eval_librispeech")
    args = ap.parse_args()

    need = {"F": args.speakers // 2, "M": args.speakers - args.speakers // 2}
    genders: dict = {}
    # spk -> chapter -> [(name, bytes, dur)]
    pool: dict = defaultdict(lambda: defaultdict(list))
    done: dict = {}                 # spk -> list đã chọn
    order: list = []                # thứ tự người nói xuất hiện trong luồng

    def finalize(spk):
        if spk in done or spk not in pool:
            return
        g = genders.get(spk, "?")
        if not genders:             # Không có SPEAKERS.TXT -> bỏ qua cân bằng giới tính
            g = "F" if need["F"] > 0 else "M"
        total = sum(len(v) for v in pool[spk].values())
        if total >= args.utts and need.get(g, 0) > 0:
            done[spk] = select(pool[spk], args.utts)
            need[g] -= 1
            print(f"\r  ✓ Người nói {spk:>5} ({g}) – {len(pool[spk])} chương, chọn {args.utts} câu")
        pool.pop(spk, None)

    for url in MIRRORS:
        try:
            print(f"🌐 Kết nối {url}")
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"  ✘ {e}")
    else:
        sys.exit("❌ Không tải được LibriSpeech từ các mirror.")

    reader = _CountingReader(resp.raw)
    with tarfile.open(fileobj=io.BufferedReader(reader, 1 << 20), mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            if m.name.endswith("SPEAKERS.TXT"):
                genders = parse_speakers_txt(tar.extractfile(m).read().decode("utf-8", "replace"))
                continue
            if not m.name.endswith(".flac") or len(parts) < 4:
                continue
            spk, chap = parts[-3], parts[-2]
            if spk in done:
                continue
            if spk not in order:
                for prev in order:      # Người nói trước đã kết thúc trong luồng
                    finalize(prev)
                order.append(spk)
                if all(v <= 0 for v in need.values()):
                    break               # Đủ người nói -> dừng tải
            if genders and need.get(genders.get(spk, "?"), 0) <= 0:
                continue
            if len(pool[spk][chap]) >= MAX_CANDIDATES_PER_CHAPTER:
                continue
            data = tar.extractfile(m).read()
            dur = sf.info(io.BytesIO(data)).duration
            if MIN_DUR <= dur <= MAX_DUR:
                pool[spk][chap].append((Path(m.name).name, data, dur))
        for prev in order:
            finalize(prev)
    resp.close()
    print(f"\n  Tổng dung lượng đã tải: {reader.n / 2**20:.1f} MB")

    if len(done) < 2:
        sys.exit("❌ Không gom đủ người nói.")

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for spk, items in done.items():
        g = genders.get(spk, "U")
        d = args.out / f"LS-{spk}-{g}"
        d.mkdir(exist_ok=True)
        for name, data, _ in items:
            (d / name).write_bytes(data)
        meta[d.name] = {
            "librispeech_id": spk, "gender": g,
            "chapters": sorted({n.split("-")[1] for n, _, _ in items}),
            "files": {n: round(du, 2) for n, _, du in items},
        }
    (args.out / "speakers.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    n_utt = sum(len(v) for v in done.values())
    print(f"✅ Đã lưu {len(done)} người nói / {n_utt} câu vào {args.out}")
    print(f"👉 Tiếp theo: python evaluate_demo.py --data {args.out.relative_to(config.ROOT_DIR)} "
          f"--out reports/librispeech")


if __name__ == "__main__":
    main()
