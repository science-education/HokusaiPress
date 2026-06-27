from hokusai_press.vecfill import fill_rect_ops, px_to_pt


def test_fill_rect_ops_contains_pdf_fill_rectangle_and_gray_ops():
    ops = fill_rect_ops(10, 20, 40, 50, 0.25, page_h_pt=100.0, scale=1.0)

    assert b" g " in ops
    assert b" re " in ops
    assert b" f" in ops


def test_fill_rect_ops_flips_top_left_pixel_y_to_high_pdf_y():
    ops = fill_rect_ops(10, 10, 50, 20, 0.5, page_h_pt=800.0, scale=1.0)
    parts = ops.decode("ascii").split()

    y_pt = float(parts[3])

    assert y_pt == 780.0
    assert y_pt > 0.95 * 800.0


def test_fill_rect_ops_output_is_small():
    ops = fill_rect_ops(0, 0, 20, 10, 1.0, page_h_pt=100.0, scale=0.5)

    assert len(ops) < 64


def test_px_to_pt_uses_pdf_dpi_scale():
    assert px_to_pt(600, 600) == 72.0
