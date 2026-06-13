import os

from hokusai_press.cli import _resolve_out


def test_out_file_path_is_used_verbatim(tmp_path):
    out = str(tmp_path / "book.pdf")
    assert _resolve_out(out, "C:/scans/chufu.pdf") == out


def test_existing_dir_uses_source_stem(tmp_path):
    out = _resolve_out(str(tmp_path), "C:/scans/chufu.pdf")
    assert out == os.path.join(str(tmp_path), "chufu.pdf")


def test_trailing_separator_is_a_folder(tmp_path):
    folder = str(tmp_path / "out") + os.sep
    out = _resolve_out(folder, "/data/shinto.pdf")
    assert out == os.path.join(folder, "shinto.pdf")
    assert os.path.isdir(folder)            # created on demand


def test_extensionless_is_treated_as_folder(tmp_path):
    folder = str(tmp_path / "results")
    out = _resolve_out(folder, "img20260420_0057.pdf")
    assert out == os.path.join(folder, "img20260420_0057.pdf")
    assert os.path.isdir(folder)


def test_image_source_becomes_pdf(tmp_path):
    out = _resolve_out(str(tmp_path), "/data/page01.png")
    assert out == os.path.join(str(tmp_path), "page01.pdf")
