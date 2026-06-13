import numpy as np
import pytest

pytest.importorskip("pypdfium2")
pytest.importorskip("PIL")


def _make_pdf(path, w=300, h=400):
    from PIL import Image

    arr = np.full((h, w, 3), 240, dtype=np.uint8)
    arr[40:80, 30:270] = 0  # a black bar so the page isn't blank
    Image.fromarray(arr).save(str(path), "PDF", resolution=150.0)
    return str(path)


def test_load_single_extracts_embedded_image(tmp_path):
    from hokusai_press.source import load_single

    pdf = _make_pdf(tmp_path / "scan.pdf")
    original, dpi = load_single(pdf, 0)
    assert original.ndim == 3 and original.shape[2] == 3
    assert original.shape[0] > 0 and original.shape[1] > 0


def test_load_single_is_thread_safe(tmp_path):
    # pdfium is not thread-safe; without the module lock, concurrent load_single
    # calls raise "Data format error" / "Failed to load page" or crash the
    # process. Hammer it and assert every call returns a well-formed image.
    import threading

    from hokusai_press.source import load_single

    pdf = _make_pdf(tmp_path / "scan.pdf")
    errors: list[Exception] = []
    shapes: list[tuple] = []

    def worker():
        try:
            for _ in range(15):
                original, _ = load_single(pdf, 0)
                assert original.ndim == 3
                shapes.append(original.shape)
        except Exception as e:  # pragma: no cover - only on regression
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    assert len(set(shapes)) == 1  # every extraction identical
