import sys
import types

import numpy as np

from hokusai_press.content import analyze
from hokusai_press.model import Flag, RegionKind, SourceRef
from hokusai_press.ocr.paddle import (
    PaddleOCRVLEngine,
    _runtime_kwargs,
    normalize_paddle_ocr_result,
    normalize_paddle_vl_result,
)


def test_paddle_vl_mlx_configures_recognition_server(monkeypatch):
    captured = {}

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    paddleocr = types.ModuleType("paddleocr")
    paddleocr.PaddleOCRVL = FakePaddleOCRVL
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)
    monkeypatch.setenv("HOKUSAI_MLX_MODEL", "mlx-community/PaddleOCR-VL-4bit")
    monkeypatch.setenv("HOKUSAI_MLX_SERVER_URL", "http://127.0.0.1:9123/")

    PaddleOCRVLEngine(device="mps", engine="mlx")

    assert captured["engine"] == "paddle"
    assert captured["device"] == "cpu"
    assert captured["vl_rec_backend"] == "mlx-vlm-server"
    assert captured["vl_rec_server_url"] == "http://127.0.0.1:9123/"
    assert captured["vl_rec_api_model_name"] == "mlx-community/PaddleOCR-VL-4bit"


def test_normalize_ppocr_v6_result_arrays():
    output = [{
        "res": {
            "rec_polys": np.array([[[1, 2], [11, 2], [11, 8], [1, 8]]]),
            "rec_boxes": np.array([[1, 2, 11, 8]]),
            "rec_texts": ["北斎"],
            "rec_scores": np.array([0.98]),
        }
    }]

    result = normalize_paddle_ocr_result(output)

    assert result["lines"][0]["box"] == (1.0, 2.0, 11.0, 8.0)
    assert result["lines"][0]["text"] == "北斎"
    assert result["lines"][0]["source"] == "ppocr-v6"


def test_runtime_kwargs_maps_onnxruntime_npu_to_openvino_ep():
    kwargs = _runtime_kwargs("onnxruntime", "npu")

    assert kwargs["device"] == "cpu"
    assert kwargs["engine_config"]["providers"] == [
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert kwargs["engine_config"]["provider_options"][0]["device_type"] == "NPU"


def test_normalize_ppocr_v6_flat_single_box():
    output = [{
        "res": {
            "rec_boxes": [1, 2, 11, 8],
            "rec_texts": ["一行"],
            "rec_scores": [0.91],
        }
    }]

    result = normalize_paddle_ocr_result(output)

    assert result["lines"][0]["box"] == (1.0, 2.0, 11.0, 8.0)
    assert result["lines"][0]["text"] == "一行"


def test_normalize_paddle_vl_layout_boxes():
    output = [{
        "res": {
            "layout_det_res": {
                "boxes": [
                    {"label": "table", "coordinate": [20, 30, 120, 80], "score": 0.9},
                    {"label": "text", "coordinate": [0, 0, 10, 10], "score": 0.8},
                ]
            }
        }
    }]

    result = normalize_paddle_vl_result(output)

    assert result["layout_boxes"] == [{
        "box": (20.0, 30.0, 120.0, 80.0),
        "label": "table",
        "score": 0.9,
        "source": "paddle-vl",
    }]


def test_normalize_paddle_vl_dedupes_nested_ocr_lines():
    line_data = {
        "rec_boxes": [[1, 2, 11, 8]],
        "rec_texts": ["重複"],
        "rec_scores": [0.8],
    }
    output = [{"res": {**line_data, "overall_ocr_res": line_data}}]

    result = normalize_paddle_vl_result(output)

    assert len(result["lines"]) == 1
    assert result["lines"][0]["text"] == "重複"


def test_normalize_paddle_vl_dedupes_layout_and_parsing_image():
    output = [{"res": {
        "layout_det_res": {"boxes": [{
            "label": "image", "coordinate": [20, 30, 120, 80], "score": 0.9,
        }]},
        "parsing_res_list": [{
            "block_bbox": [20, 30, 120, 80],
            "block_label": "image",
            "block_content": "",
        }],
    }}]

    result = normalize_paddle_vl_result(output)

    assert len(result["layout_boxes"]) == 1
    assert result["layout_boxes"][0]["label"] == "figure"


def test_normalize_paddle_vl_16_parsing_blocks():
    output = [{
        "res": {
            "parsing_res_list": [
                {
                    "block_bbox": np.array([10, 20, 210, 80]),
                    "block_label": "text",
                    "block_content": "縦書き本文",
                    "block_score": 0.97,
                },
                {
                    "block_bbox": [[20, 100], [220, 100], [220, 260], [20, 260]],
                    "block_label": "image",
                    "block_content": "",
                    "block_score": 0.91,
                },
                {
                    "block_bbox": [230, 300, 260, 330],
                    "block_label": "number",
                    "block_content": "31",
                },
            ]
        }
    }]

    result = normalize_paddle_vl_result(output)

    assert [line["text"] for line in result["lines"]] == ["縦書き本文", "31"]
    assert result["lines"][0]["box"] == (10.0, 20.0, 210.0, 80.0)
    assert result["layout_boxes"][0]["label"] == "text"
    assert result["layout_boxes"][1]["label"] == "figure"
    assert result["layout_boxes"][2]["label"] == "page_number"


def test_normalize_paddle_vl_16_parsing_block_objects():
    class FakePaddleOCRVLBlock:
        label = "vertical_text"
        bbox = [10, 20, 210, 80]
        content = "オブジェクト形式の本文"
        polygon_points = None

    output = [{"res": {"parsing_res_list": [FakePaddleOCRVLBlock()]}}]

    result = normalize_paddle_vl_result(output)

    assert result["lines"][0]["text"] == "オブジェクト形式の本文"
    assert result["lines"][0]["box"] == (10.0, 20.0, 210.0, 80.0)
    assert result["layout_boxes"][0]["label"] == "text"


def test_content_accepts_layout_boxes_from_ocr_engine(monkeypatch):
    class FakeEngine:
        def __call__(self, image_bgr):
            return {
                "lines": [],
                "layout_boxes": [{
                    "box": (50, 200, 250, 350),
                    "label": "image",
                    "source": "paddle-vl",
                }],
            }

    def fake_get_ocr_engine(*args, **kwargs):
        return FakeEngine()

    import hokusai_press.ocr

    monkeypatch.setattr(hokusai_press.ocr, "get_ocr_engine", fake_get_ocr_engine)

    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    grad = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (150, 1))
    img[200:350, 50:250] = np.dstack([grad, grad, grad])

    regions, flags = analyze(
        img, img, SourceRef(path="x", ocr_scale=1.0),
        use_ocr=True, ocr_engine="paddle-vl",
    )

    assert regions[0].source == "paddle-vl"
    assert regions[0].kind == RegionKind.PHOTO
    assert regions[0].box.x0 == 50
    assert Flag.OCR_LOW_COVERAGE not in flags
