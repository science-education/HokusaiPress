import numpy as np

from hokusai_press.content import analyze
from hokusai_press.model import Box, RegionKind, SourceRef


class FakeLayout:
    """Returns a fixed figure box in image pixels."""

    def __init__(self, box):
        self._box = box

    def figures(self, img_bgr):
        return [self._box]


def _page_with_gradient():
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    grad = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (150, 1))
    img[200:350, 50:250] = np.dstack([grad, grad, grad])  # continuous tone
    return img


def test_layout_provider_adds_photo_region():
    img = _page_with_gradient()
    provider = FakeLayout(Box(50, 200, 250, 350))
    regions, flags = analyze(
        img, img, SourceRef(path="x", ocr_scale=1.0),
        use_ocr=False, layout_provider=provider,
    )
    rt = [r for r in regions if r.source == "rtdetr"]
    assert len(rt) == 1
    assert rt[0].kind == RegionKind.PHOTO   # gradient -> continuous tone
    # padded outward a few px (soft/anti-aliased figure edges can fall just
    # outside the raw detector box) -- so x0 shrinks and y1 grows from input
    assert rt[0].box.x0 < 50 and rt[0].box.y1 > 350


def test_layout_provider_line_art_is_figure():
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    img[210:212, 60:240] = 0      # thin lines = line art (2 levels)
    img[260:262, 60:240] = 0
    provider = FakeLayout(Box(50, 200, 250, 300))
    regions, _ = analyze(
        img, img, SourceRef(path="x", ocr_scale=1.0),
        use_ocr=False, layout_provider=provider,
    )
    rt = [r for r in regions if r.source == "rtdetr"]
    assert rt and rt[0].kind == RegionKind.FIGURE


def test_ocr_scale_maps_layout_box_to_original():
    img = _page_with_gradient()
    provider = FakeLayout(Box(50, 200, 250, 350))
    regions, _ = analyze(
        img, img, SourceRef(path="x", ocr_scale=0.5),  # ocr is half-size
        use_ocr=False, layout_provider=provider,
    )
    rt = [r for r in regions if r.source == "rtdetr"][0]
    # divided by ocr_scale, then a few px of padding (applied pre-scale) shows
    # up scaled by 1/ocr_scale too -- so x0 is a bit under 100, x1 a bit over 500
    assert 90 < rt.box.x0 < 100
    assert 500 < rt.box.x1 < 510
