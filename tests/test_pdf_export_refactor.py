import sys
from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
import pikepdf
import pytest

from hokusai_press.pdf_export import (
    SearchablePdfBuilder,
    encode_page_pdf,
    build_text_overlay,
    decide_page_mode,
    UnsupportedCodecError,
    BackendUnavailableError,
    MissingFontError,
)
from hokusai_press.pdf_export.backend import (
    _encode_g4,
    _encode_jpeg,
    discover_font,
)


def test_backend_selection_auto():
    # hybrid_ocr が利用可能かどうかに関わらず、auto はエラーにならないはず
    builder = SearchablePdfBuilder(backend="auto")
    assert builder._builder is not None


def test_backend_selection_unavailable():
    # hybrid_ocr をインポート不可にする
    with patch.dict(sys.modules, {"hybrid_ocr.pdf_export": None, "hybrid_ocr": None}):
        with pytest.raises(BackendUnavailableError):
            SearchablePdfBuilder(backend="hybrid_ocr")

        # auto の場合は internal にフォールバックするはず
        builder = SearchablePdfBuilder(backend="auto")
        from hokusai_press.pdf_export import InternalSearchablePdfBuilder
        assert isinstance(builder._builder, InternalSearchablePdfBuilder)


def test_invalid_inputs():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 無効なモード
    with pytest.raises(UnsupportedCodecError):
        encode_page_pdf(img, mode="invalid", compress="g4", backend="internal")

    # 無効な圧縮形式
    with pytest.raises(UnsupportedCodecError):
        encode_page_pdf(img, mode="bw", compress="invalid", backend="internal")

    # pyjbig2 がない環境での jbig2 指定
    with patch.dict(sys.modules, {"pyjbig2": None, "pyjbig2.api": None}):
        with pytest.raises(UnsupportedCodecError):
            encode_page_pdf(img, mode="bw", compress="jbig2", backend="internal")
            
        with pytest.raises(UnsupportedCodecError):
            SearchablePdfBuilder(compress="jbig2", backend="internal")


def test_exact_mediabox_and_filters_g4():
    # 200x300 二値画像
    img = np.zeros((300, 200), dtype=np.uint8)
    img[50:250, 50:150] = 255  # 一部白

    pdf_bytes = encode_page_pdf(img, mode="bw", compress="g4", backend="internal")
    
    # pikepdf で検証
    with pikepdf.open(BytesIO(pdf_bytes)) as pdf:
        assert len(pdf.pages) == 1
        page = pdf.pages[0]
        
        # 物理 MediaBox (width, height)
        # pikepdf の Page.MediaBox / Page.trimbox / Page.cropbox などは [0, 0, 200, 300]
        # (1px = 1pt 物理スケール)
        box = [float(v) for v in page.MediaBox]
        assert box == [0.0, 0.0, 200.0, 300.0]

        # XObject 画像オブジェクトの検証
        images = list(page.Resources.XObject.values())
        assert len(images) == 1
        img_obj = images[0]
        
        assert img_obj.Filter == pikepdf.Name("/CCITTFaxDecode")
        assert img_obj.Width == 200
        assert img_obj.Height == 300
        assert img_obj.ColorSpace == pikepdf.Name("/DeviceGray")
        assert img_obj.BitsPerComponent == 1
        
        # DecodeParms
        parms = img_obj.DecodeParms
        assert parms.K == -1
        assert parms.Columns == 200
        assert parms.Rows == 300
        assert parms.BlackIs1 == True


def test_exact_mediabox_and_filters_jpeg():
    # 150x250 カラー画像
    img = np.zeros((250, 150, 3), dtype=np.uint8)
    img[50:200, 30:120, 0] = 255  # 青い四角

    # Color JPEG
    pdf_bytes_color = encode_page_pdf(img, mode="color", compress="g4", backend="internal")
    with pikepdf.open(BytesIO(pdf_bytes_color)) as pdf:
        page = pdf.pages[0]
        box = [float(v) for v in page.MediaBox]
        assert box == [0.0, 0.0, 150.0, 250.0]
        
        img_obj = list(page.Resources.XObject.values())[0]
        assert img_obj.Filter == pikepdf.Name("/DCTDecode")
        assert img_obj.Width == 150
        assert img_obj.Height == 250
        assert img_obj.ColorSpace == pikepdf.Name("/DeviceRGB")
        assert img_obj.BitsPerComponent == 8

    # Grayscale JPEG
    pdf_bytes_gray = encode_page_pdf(img, mode="gray", compress="g4", backend="internal")
    with pikepdf.open(BytesIO(pdf_bytes_gray)) as pdf:
        page = pdf.pages[0]
        box = [float(v) for v in page.MediaBox]
        assert box == [0.0, 0.0, 150.0, 250.0]
        
        img_obj = list(page.Resources.XObject.values())[0]
        assert img_obj.Filter == pikepdf.Name("/DCTDecode")
        assert img_obj.Width == 150
        assert img_obj.Height == 250
        assert img_obj.ColorSpace == pikepdf.Name("/DeviceGray")
        assert img_obj.BitsPerComponent == 8


def test_empty_text_page():
    # テキスト行が空
    pages = [
        (200, 300, []),
    ]
    overlay_bytes = build_text_overlay(pages, backend="internal")
    with pikepdf.open(BytesIO(overlay_bytes)) as pdf:
        assert len(pdf.pages) == 1
        page = pdf.pages[0]
        box = [float(v) for v in page.MediaBox]
        assert box == [0.0, 0.0, 200.0, 300.0]
        # XObject (テキスト描画) はないはず
        assert not hasattr(page.Resources, "XObject")


def test_text_overlay_horizontal_and_vertical():
    lines = [
        # 横書きテキスト
        {"text": "Hello World", "box": [10, 20, 100, 40], "direction": "h"},
        # 縦書きテキスト
        {"text": "日本語縦書き", "box": [120, 20, 140, 180], "direction": "v"},
    ]
    pages = [
        (200, 300, lines),
    ]
    
    overlay_bytes = build_text_overlay(pages, backend="internal")
    with pikepdf.open(BytesIO(overlay_bytes)) as pdf:
        assert len(pdf.pages) == 1
        page = pdf.pages[0]
        
        # コンテンツストリームの内容をデコードして検証
        contents_obj = page.obj.Contents
        if isinstance(contents_obj, pikepdf.Array):
            content_bytes = b"".join(stream.read_bytes() for stream in contents_obj)
        else:
            content_bytes = contents_obj.read_bytes()
        content_str = content_bytes.decode("utf-8", errors="ignore")
        
        # MPLUS1p-Medium フォントが使用されていること
        font_resources = page.Resources.Font
        has_font = False
        for font_key, font_val in font_resources.items():
            if "MPLUS1p-Medium" in str(font_val.BaseFont):
                has_font = True
                break
        assert has_font
        
        # Text Render Mode が 3 (invisible) に設定されていること
        assert "3 Tr" in content_str
        
        # 縦書きの各文字が回転配置されているため、rotate(-90) すなわち三角関数マトリックス
        # reportlabは通常、回転操作として "0 -1 1 0" または "0 -1.0 1.0 0" を含む変換マトリックスを出力する
        # （縦書き一文字ずつ回転するため、複数回現れる）
        assert "0 -1" in content_str or "0 -1.0" in content_str


def test_font_discovery_missing():
    # 存在しないフォントを指定した場合に MissingFontError になるか
    with pytest.raises(MissingFontError):
        discover_font("C:\\path\\to\\nonexistent\\font.ttf")


def test_searchable_pdf_builder_internal():
    img1 = np.zeros((300, 200, 3), dtype=np.uint8)
    img2 = np.zeros((400, 300, 3), dtype=np.uint8)
    
    # 模擬OCRデータ
    lines1 = [{"text": "Page 1 Line 1", "box": [10, 10, 100, 30], "direction": "h"}]
    lines2 = [{"text": "Page 2 Line 1", "box": [20, 20, 150, 45], "direction": "h"}]

    builder = SearchablePdfBuilder(mode="bw", compress="g4", backend="internal")
    builder.add_page(img1, lines1)
    builder.add_page(img2, lines2)
    
    # 一時ファイルに保存
    import tempfile
    import os
    
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        builder.save(temp_path)
        
        # pikepdf で再オープンして検証
        with pikepdf.open(temp_path) as pdf:
            assert len(pdf.pages) == 2
            
            # 1ページ目
            page1 = pdf.pages[0]
            box1 = [float(v) for v in page1.MediaBox]
            assert box1 == [0.0, 0.0, 200.0, 300.0]
            
            # 2ページ目
            page2 = pdf.pages[1]
            box2 = [float(v) for v in page2.MediaBox]
            assert box2 == [0.0, 0.0, 300.0, 400.0]
            
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_mrc_page_builder_internal(tmp_path):
    # Simulating hybrid_ocr is absent
    with patch.dict(sys.modules, {"hybrid_ocr": None, "hybrid_ocr.pdf_export": None}):
        from hokusai_press.mrc import MrcPageBuilder
        # Simple test image
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        img[20:80, 20:80] = 0  # Black square

        builder = MrcPageBuilder(compress="g4", target_dpi=600)
        lines = [{"text": "テスト", "box": [20, 20, 80, 40], "direction": "h"}]
        builder.add_page(img, lines, [], mode="bw")

        out_pdf = tmp_path / "mrc_internal.pdf"
        builder.save(str(out_pdf))

        with pikepdf.open(out_pdf) as pdf:
            assert len(pdf.pages) == 1
            page = pdf.pages[0]
            box = [float(v) for v in page.MediaBox]
            assert box == [0.0, 0.0, 100.0, 100.0]


def test_backend_auto_routing():
    # 1. When compress='g4', backend='auto' should always select internal
    with patch.dict(sys.modules, {"hybrid_ocr.pdf_export": None, "hybrid_ocr": None}):
        builder = SearchablePdfBuilder(compress="g4", backend="auto")
        from hokusai_press.pdf_export import InternalSearchablePdfBuilder
        assert isinstance(builder._builder, InternalSearchablePdfBuilder)

    # 2. When compress='jbig2', if pyjbig2 is present, backend='auto' selects internal
    with patch.dict(sys.modules, {"pyjbig2": sys.modules.get("pyjbig2") or object()}):
        builder = SearchablePdfBuilder(compress="jbig2", backend="auto")
        assert isinstance(builder._builder, InternalSearchablePdfBuilder)

    # 3. When compress='jbig2' and pyjbig2 is absent:
    # If hybrid_ocr is present, backend='auto' routes to hybrid_ocr
    mock_hybrid = object()
    with patch.dict(sys.modules, {"pyjbig2": None, "pyjbig2.api": None}):
        # Mocking SearchablePdfBuilder on hybrid_ocr.pdf_export
        class MockHybridBuilder:
            def __init__(self, **kwargs):
                pass

        class MockModule:
            SearchablePdfBuilder = MockHybridBuilder

        with patch.dict(sys.modules, {"hybrid_ocr.pdf_export": MockModule, "hybrid_ocr": MockModule}):
            builder = SearchablePdfBuilder(compress="jbig2", backend="auto")
            assert builder._builder.__class__.__name__ == "MockHybridBuilder"


def test_input_validation():
    # Invalid image type
    with pytest.raises(TypeError):
        decide_page_mode("not an array", [])
    with pytest.raises(TypeError):
        encode_page_pdf("not an array", "bw", "g4")

    # Empty array
    with pytest.raises(ValueError):
        decide_page_mode(np.array([]), [])
    with pytest.raises(ValueError):
        encode_page_pdf(np.array([]), "bw", "g4")

    # Invalid dimension
    with pytest.raises(ValueError):
        decide_page_mode(np.zeros((10,)), [])
    with pytest.raises(ValueError):
        encode_page_pdf(np.zeros((10,)), "bw", "g4")
