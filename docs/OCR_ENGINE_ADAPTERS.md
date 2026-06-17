# OCR Engine Adapters

Status: implemented in isolated worktree `HokusaiPress-ocr-paddle`.

## Goal

HokusaiPress can now select OCR engines without changing the downstream
content pipeline:

- `hybrid` / `yomitoku` / `ndlocr`: existing Yomitoku + NDL-OCR-Lite
  `hybrid_ocr` engine. This remains the default.
- `ppocr-v6` / `paddleocr-v6`: PaddleOCR PP-OCRv6 adapter.
- `paddle-vl` / `paddleocr-vl`: PaddleOCR-VL-1.6 document parser adapter,
  including layout boxes for figure/photo/table-like regions.

All adapters normalize output to:

```python
{
    "lines": [
        {
            "polygon": [[x, y], ...],
            "box": (x0, y0, x1, y1),
            "text": "...",
            "det_score": 0.98,
            "source": "ppocr-v6",
        }
    ],
    "layout_boxes": [
        {
            "box": (x0, y0, x1, y1),
            "label": "table",
            "score": 0.90,
            "source": "paddle-vl",
        }
    ],
}
```

`content.analyze()` maps both text and layout boxes back through
`SourceRef.ocr_scale`, so all stored `Region` coordinates remain original-image
pixels.

## CLI

```bash
hokusai-press run scan.pdf --out book.pdf --ocr-engine hybrid --device npu
hokusai-press run scan.pdf --out book.pdf --ocr-engine ppocr-v6 --device npu
hokusai-press run scan.pdf --out book.pdf --ocr-engine paddle-vl --device npu
```

PaddleOCR backend can be selected with:

```bash
hokusai-press run scan.pdf --out book.pdf --ocr-engine paddle-vl --paddle-engine paddle
hokusai-press run scan.pdf --out book.pdf --ocr-engine paddle-vl --paddle-engine transformers
```

## NPU Behavior

`src/hokusai_press/ocr/factory.py` keeps one process-wide OCR engine instance
per configuration. Paddle adapters additionally serialize their `predict()`
calls with a per-engine lock because Paddle pipelines should not be assumed
thread-safe.

This preserves the existing HokusaiPress rule: threaded multi-file execution is
allowed, but only one OCR/NPU context should be active inside the process.

## Paddle Runtime Notes

PaddleOCR's documented `device="npu:0"` support is hardware-runtime dependent.
The adapter maps HokusaiPress `--device npu` to Paddle's `npu:0`, but the
installed Paddle runtime must actually support that device. If it does not,
construction should fail loudly instead of silently falling back for a whole
batch.

PaddleOCR-VL is heavier than PP-OCRv6. Use it when layout parsing is needed;
use PP-OCRv6 when text OCR throughput matters more than document structure.
