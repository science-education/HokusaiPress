"""Contract for WP-21: page similarity for correction propagation.

docs/VIEWER_LIBRARY_PLAN.md WP-21. Pure logic (numpy only, no image I/O in this
cut — images are a later enhancement). Given a corrected page, rank other pages
by structural similarity so WP-22 can propose the same fix to the near ones.
Skip until the module lands.

Design fixed here (do not re-litigate):
- `page_vector(params)` builds a feature vector reusing learn.page_features.
- `rank_similar(target, candidates, k=None)` z-scores each dimension over the
  pool (target + candidates) so heterogeneous features are comparable, then
  ranks candidates by ascending Euclidean distance. Zero-variance dimensions
  must not produce NaN.
"""

import numpy as np
import pytest

from hokusai_press.model import Box, Deskew, PageParams, Region, RegionKind, SourceRef

try:
    from hokusai_press.similarity import page_vector, rank_similar
except ImportError:
    pytest.skip(
        "WP-21 unimplemented: hokusai_press.similarity missing "
        "(see docs/tasks/WP-21-codex.md)",
        allow_module_level=True,
    )


def _page(angle=0.0, n_text=1, n_photo=0):
    regions = [Region(kind=RegionKind.TEXT, box=Box(0, 0, 50, 10))
               for _ in range(n_text)]
    regions += [Region(kind=RegionKind.PHOTO, box=Box(0, 20, 80, 90))
                for _ in range(n_photo)]
    return PageParams(
        source=SourceRef(path="s.pdf", page_index=0),
        deskew=Deskew(angle_deg=angle, confidence=1.0),
        regions=regions,
    )


def test_page_vector_is_numeric_and_fixed_length():
    v0 = page_vector(_page(n_text=1))
    v1 = page_vector(_page(n_text=5))
    assert isinstance(v0, np.ndarray)
    assert v0.shape == v1.shape
    assert not np.isnan(v0).any()


def test_identical_page_ranks_first_with_zero_distance():
    target = _page(angle=0.5, n_text=3)
    candidates = [
        ("same", _page(angle=0.5, n_text=3)),
        ("different", _page(angle=8.0, n_text=1, n_photo=4)),
    ]
    ranked = rank_similar(target, candidates)
    assert ranked[0][0] == "same"
    assert ranked[0][1] == pytest.approx(0.0, abs=1e-9)
    # ascending distance
    assert ranked[0][1] <= ranked[1][1]
    assert ranked[-1][0] == "different"


def test_k_limits_results():
    target = _page(n_text=2)
    candidates = [(f"p{i}", _page(n_text=i)) for i in range(6)]
    ranked = rank_similar(target, candidates, k=3)
    assert len(ranked) == 3


def test_constant_dimension_does_not_nan():
    # All pages share identical features -> zero variance in every dimension.
    target = _page(n_text=2)
    candidates = [("a", _page(n_text=2)), ("b", _page(n_text=2))]
    ranked = rank_similar(target, candidates)
    assert all(not np.isnan(d) for _, d in ranked)
    assert {k for k, _ in ranked} == {"a", "b"}


def test_empty_candidates_returns_empty():
    assert rank_similar(_page(), []) == []
