"""Contract for WP-41: OPF metadata export for Calibre interop.

docs/VIEWER_LIBRARY_PLAN.md WP-41. Emit a Dublin-Core OPF 2.0 package document
from a book metadata dict so Calibre's "add from folder" ingests bibliography
alongside the exported PDF. Pure function, well-formed XML. Skip until landed.
"""

import xml.etree.ElementTree as ET

import pytest

try:
    from hokusai_press.export_opf import build_opf
except ImportError:
    pytest.skip(
        "WP-41 unimplemented: hokusai_press.export_opf.build_opf missing "
        "(see docs/tasks/WP-41-codex.md)",
        allow_module_level=True,
    )

_DC = "http://purl.org/dc/elements/1.1/"


def _dc_texts(root, tag):
    return [e.text for e in root.iter(f"{{{_DC}}}{tag}")]


def test_opf_is_well_formed_and_carries_metadata():
    xml = build_opf({
        "title": "吾輩は猫である",
        "author": "夏目漱石",
        "publisher": "岩波書店",
        "year": 1905,
        "isbn": "9784001010015",
    })
    root = ET.fromstring(xml)  # raises if not well-formed

    assert _dc_texts(root, "title") == ["吾輩は猫である"]
    assert _dc_texts(root, "creator") == ["夏目漱石"]
    assert _dc_texts(root, "publisher") == ["岩波書店"]
    assert any("1905" in (d or "") for d in _dc_texts(root, "date"))
    assert any("9784001010015" in (i or "") for i in _dc_texts(root, "identifier"))


def test_missing_fields_are_omitted_not_crashed():
    xml = build_opf({"title": "無名の本"})
    root = ET.fromstring(xml)
    assert _dc_texts(root, "title") == ["無名の本"]
    # No author/publisher provided -> those DC elements simply absent.
    assert _dc_texts(root, "creator") == []
    assert _dc_texts(root, "publisher") == []


def test_special_characters_are_escaped():
    # Ampersands / angle brackets in metadata must not break the XML.
    xml = build_opf({"title": "A & B <test>", "author": "X"})
    root = ET.fromstring(xml)
    assert _dc_texts(root, "title") == ["A & B <test>"]
