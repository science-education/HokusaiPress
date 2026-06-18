from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import cv2

from hokusai_press.source import load_page
from hybrid_ocr.cropper import crop_line
from hybrid_ocr.pipeline import HybridOCR


def summarize(name: str, times: list[float]) -> None:
    warm = times[1:] if len(times) > 1 else times
    print(
        f"{name}_warm_sec "
        f"avg={statistics.mean(warm):.3f} "
        f"median={statistics.median(warm):.3f} "
        f"min={min(warm):.3f} max={max(warm):.3f}",
        flush=True,
    )


def time_loop(
    name: str,
    iters: int,
    fn: Callable[[], tuple[int, int]],
) -> list[float]:
    times: list[float] = []
    for i in range(iters):
        start = time.perf_counter()
        lines, texts = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(
            f"{name}_iter={i + 1} sec={elapsed:.3f} lines={lines} texts={texts}",
            flush=True,
        )
    summarize(name, times)
    return times


def make_crops(engine: HybridOCR, img):
    quads, scores = engine.detector.detect(img)
    crops = {}
    for i, quad in enumerate(quads):
        crop = crop_line(img, quad, engine.margin_ratio)
        if crop is not None:
            crops[i] = crop
    return quads, scores, crops


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--openvino-cache-dir")
    parser.add_argument("--full-iters", type=int, default=3)
    parser.add_argument("--det-iters", type=int, default=4)
    parser.add_argument("--rec-iters", type=int, default=4)
    args = parser.parse_args()

    source, _original, ocr_img, _dpi = next(load_page(args.pdf, {args.page}))
    if args.scale != 1.0:
        h, w = ocr_img.shape[:2]
        ocr_img = cv2.resize(
            ocr_img,
            (max(1, int(w * args.scale)), max(1, int(h * args.scale))),
            interpolation=cv2.INTER_AREA,
        )
    print(f"image shape={ocr_img.shape} source_scale={source.ocr_scale}", flush=True)

    start = time.perf_counter()
    engine = HybridOCR(
        model_dir=args.model_dir,
        device=args.device,
        openvino_cache_dir=args.openvino_cache_dir,
    )
    print(f"construct_sec={time.perf_counter() - start:.3f}", flush=True)

    quads, _scores, crops = make_crops(engine, ocr_img)
    print(f"baseline lines={len(quads)} crops={len(crops)}", flush=True)

    def full_once() -> tuple[int, int]:
        result = engine(ocr_img)
        lines = result.get("lines", [])
        return len(lines), sum(1 for line in lines if line.get("text"))

    def det_once() -> tuple[int, int]:
        next_quads, _next_scores = engine.detector.detect(ocr_img)
        return len(next_quads), 0

    def rec_once() -> tuple[int, int]:
        texts = engine.recognizer.read_lines(crops)
        return len(crops), sum(1 for text in texts.values() if text)

    time_loop("full", args.full_iters, full_once)
    time_loop("yomitoku_det", args.det_iters, det_once)
    time_loop("ndlocr_rec", args.rec_iters, rec_once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
