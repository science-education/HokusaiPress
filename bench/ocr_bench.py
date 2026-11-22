"""OCR accuracy/speed benchmark for HokusaiPress OCR engines.

The page images and ground truth are NOT in this repository (they are
copyrighted book pages). Point --data at a folder laid out as:

    <data>/<book>-pages/<book>_p0001.png ...     page images
    <data>/<book>-gt-ordered/<book>_p0001.txt    ground truth in reading order
    <data>/<book>-gt/<book>_p0001.txt            (used when -gt-ordered is absent)

On the main Windows machine the data lives in C:\\Users\\user\\dev\\ocr-bench-data
(set HOKUSAI_BENCH_DATA to use another place).

Each engine is run through hokusai_press.ocr.get_ocr_engine, i.e. exactly as
`hokusai-press run` uses it, and scored per book:

    CER    edit distance / ground-truth characters (whitespace removed)
    miss   characters in the ground truth the output lacks (order-free)
    extra  characters in the output the ground truth lacks (order-free)

CER mixes reading-order and recognition errors; miss/extra do not depend on
order, so a high CER with low miss/extra means a reading-order problem.

usage:
    python bench/ocr_bench.py --engines ndlocr hybrid --device cpu
    python bench/ocr_bench.py --engines hybrid --device npu --openvino-cache-dir C:\\tmp\\ovcache
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

DEFAULT_DATA = os.environ.get(
    "HOKUSAI_BENCH_DATA", r"C:\Users\user\dev\ocr-bench-data"
)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def normalize(text: str) -> str:
    return re.sub(r"\s", "", text)


def score(gt: str, hyp: str) -> dict:
    gt, hyp = normalize(gt), normalize(hyp)
    cg, ch = Counter(gt), Counter(hyp)
    return {
        "chars": len(gt),
        "edits": levenshtein(gt, hyp),
        "miss": sum((cg - ch).values()),
        "extra": sum((ch - cg).values()),
    }


def find_books(data: Path) -> dict:
    books = {}
    for pages in sorted(data.glob("*-pages")):
        book = pages.name[: -len("-pages")]
        gt_dir = data / f"{book}-gt-ordered"
        if not gt_dir.is_dir():
            gt_dir = data / f"{book}-gt"
        if not gt_dir.is_dir():
            continue
        items = []
        for gt in sorted(gt_dir.glob("*.txt")):
            imgs = [p for p in pages.glob(gt.stem + ".*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
            if imgs:
                items.append((imgs[0], gt))
        if items:
            books[book] = items
    return books


def run_engine(name: str, args, books: dict) -> dict:
    import cv2

    from hokusai_press.ocr import get_ocr_engine

    t0 = time.time()
    engine = get_ocr_engine(
        name,
        args.model_dir,
        args.device,
        args.openvino_cache_dir,
        args.runtime,
    )
    load_s = time.time() - t0
    out = {"engine": name, "device": args.device, "load_s": round(load_s, 1), "books": {}}
    for book, items in books.items():
        total = {"chars": 0, "edits": 0, "miss": 0, "extra": 0, "pages": 0, "ocr_s": 0.0}
        for img_path, gt_path in items:
            img = cv2.imdecode(
                __import__("numpy").fromfile(str(img_path), dtype="uint8"), cv2.IMREAD_COLOR
            )
            t = time.time()
            result = engine(img)
            total["ocr_s"] += time.time() - t
            hyp = "\n".join((ln.get("text") or "") for ln in result.get("lines", []))
            s = score(gt_path.read_text(encoding="utf-8"), hyp)
            for k in ("chars", "edits", "miss", "extra"):
                total[k] += s[k]
            total["pages"] += 1
            if args.save_text:
                d = Path(args.save_text) / f"{book}-{name}-{args.device}"
                d.mkdir(parents=True, exist_ok=True)
                (d / (gt_path.stem + ".txt")).write_text(hyp, encoding="utf-8")
        n = max(total["chars"], 1)
        out["books"][book] = {
            "pages": total["pages"],
            "CER%": round(100 * total["edits"] / n, 2),
            "miss%": round(100 * total["miss"] / n, 2),
            "extra%": round(100 * total["extra"] / n, 2),
            "s_per_page": round(total["ocr_s"] / max(total["pages"], 1), 2),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--engines", nargs="+", default=["ndlocr", "hybrid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--openvino-cache-dir", default=None)
    ap.add_argument("--save-text", default=None, help="folder to keep each engine's page text")
    ap.add_argument("--json", default=None, help="write results to this JSON file")
    args = ap.parse_args(argv)

    data = Path(args.data)
    books = find_books(data)
    if not books:
        print(f"no benchmark data under {data}", file=sys.stderr)
        return 2
    results = []
    for name in args.engines:
        r = run_engine(name, args, books)
        results.append(r)
        for book, s in r["books"].items():
            print(
                f"{name:10s} {args.device:5s} {book:8s} pages={s['pages']:2d} "
                f"CER={s['CER%']:5.2f}% miss={s['miss%']:5.2f}% extra={s['extra%']:5.2f}% "
                f"{s['s_per_page']:.2f}s/page (load {r['load_s']}s)",
                flush=True,
            )
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
