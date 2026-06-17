import numpy as np

from hokusai_press.content import analyze
from hokusai_press.model import RegionKind, SourceRef
from hokusai_press.ocr.paddle import (
    normalize_paddle_ocr_result,
    normalize_paddle_vl_result,
)


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

    regions, _ = analyze(
        img, img, SourceRef(path="x", ocr_scale=1.0),
        use_ocr=True, ocr_engine="paddle-vl",
    )

    assert regions[0].source == "paddle-vl"
    assert regions[0].kind == RegionKind.PHOTO
    assert regions[0].box.x0 == 50
