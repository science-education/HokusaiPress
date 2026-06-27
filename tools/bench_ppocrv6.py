from __future__ import annotations

import argparse
import statistics
import time

import cv2

from hokusai_press.ocr.paddle import PPOCRv6Engine
from hokusai_press.source import load_page


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--runtime", default="onnxruntime")
    parser.add_argument("--device", default="npu")
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

    t0 = time.perf_counter()
    engine = PPOCRv6Engine(None, args.device, args.runtime)
    print(f"construct_sec={time.perf_counter() - t0:.3f}", flush=True)

    times = []
    for i in range(args.iters):
        t = time.perf_counter()
        result = engine(ocr_img)
        dt = time.perf_counter() - t
        lines = result.get("lines", [])
        text_count = sum(1 for line in lines if line.get("text"))
        times.append(dt)
        print(
            f"iter={i + 1} sec={dt:.3f} lines={len(lines)} texts={text_count}",
            flush=True,
        )

    warm = times[1:] if len(times) > 1 else times
    print(
        "warm_sec "
        f"avg={statistics.mean(warm):.3f} "
        f"median={statistics.median(warm):.3f} "
        f"min={min(warm):.3f} max={max(warm):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
