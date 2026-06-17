# OCR Engine Adapters

Status: implemented in isolated worktree `HokusaiPress-ocr-paddle`.

## Goal

HokusaiPress can now select layout and text OCR engines independently without
changing the downstream content pipeline:

- layout `yomitoku`: existing Yomitoku RT-DETR layout provider. This is the
  default layout selector and is optional at import time.
- text `ndlocr`: existing NDL-OCR/hybrid text OCR path. This is the default
  text selector.
- layout `pp-structurev3`: PaddleOCR PP-StructureV3 document structure parser.
- layout/text `paddle-vl`: PaddleOCR-VL-1.6 document parser.
- text `ppocr-v6`: PaddleOCR PP-OCRv6 adapter.

The old `--ocr-engine` option remains as a compatibility shortcut:

| `--ocr-engine` | Internal expansion |
| --- | --- |
| `hybrid`, `auto`, `yomitoku`, `ndlocr` | `layout=yomitoku`, `text=ndlocr` |
| `ppocr-v6` | `layout=none`, `text=ppocr-v6` |
| `pp-structurev3` | `layout=pp-structurev3`, `text=none` |
| `paddle-vl` | `layout=paddle-vl`, `text=paddle-vl` |

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
hokusai-press run scan.pdf --out book.pdf --layout-engine yomitoku --text-engine ndlocr --device npu
hokusai-press run scan.pdf --out book.pdf --layout-engine pp-structurev3 --text-engine ppocr-v6 --runtime onnxruntime --device npu
hokusai-press run scan.pdf --out book.pdf --layout-engine paddle-vl --text-engine ppocr-v6 --runtime onnxruntime --device npu
```

Runtime/backend can be selected with:

```bash
hokusai-press run scan.pdf --out book.pdf --layout-engine pp-structurev3 --text-engine ppocr-v6 --runtime paddle
hokusai-press run scan.pdf --out book.pdf --layout-engine pp-structurev3 --text-engine ppocr-v6 --runtime onnxruntime
hokusai-press run scan.pdf --out book.pdf --layout-engine paddle-vl --text-engine ppocr-v6 --runtime transformers
```

## NPU Behavior

`src/hokusai_press/ocr/factory.py` keeps one process-wide composite OCR engine
instance per layout+text+runtime configuration. Paddle adapters additionally
serialize their `predict()` calls with a per-engine lock because Paddle
pipelines should not be assumed thread-safe.

This preserves the existing HokusaiPress rule: threaded multi-file execution is
allowed, but only one OCR/NPU context should be active inside the process.

## Paddle Runtime Notes

PaddleOCR's documented `device="npu:0"` support is hardware-runtime dependent.
The adapter maps HokusaiPress `--device npu` to Paddle's `npu:0`, but the
installed Paddle runtime must actually support that device. If it does not,
construction should fail loudly instead of silently falling back for a whole
batch.

For Intel NPU, the preferred Paddle path is expected to be ONNX Runtime +
OpenVINO Execution Provider after converting Paddle models to ONNX. PP-OCRv6
is the first engine to validate; PP-StructureV3 layout is the next practical
target. PaddleOCR-VL is heavier and should be validated after the lighter
layout/text split works.
