"""OPF bibliography export for Calibre interoperability."""

from __future__ import annotations

import xml.etree.ElementTree as ET


_OPF_NAMESPACE = "http://www.idpf.org/2007/opf"
_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"

ET.register_namespace("", _OPF_NAMESPACE)
ET.register_namespace("dc", _DC_NAMESPACE)


def build_opf(book: dict) -> str:
    """Return an OPF 2.0 package document for the supplied bibliography."""
    package = ET.Element(f"{{{_OPF_NAMESPACE}}}package", {"version": "2.0"})
    metadata = ET.SubElement(package, f"{{{_OPF_NAMESPACE}}}metadata")

    fields = {
        "title": "title",
        "author": "creator",
        "publisher": "publisher",
        "year": "date",
        "isbn": "identifier",
    }
    for field, dc_element in fields.items():
        value = book.get(field)
        if value is None:
            continue
        element = ET.SubElement(metadata, f"{{{_DC_NAMESPACE}}}{dc_element}")
        element.text = str(value)

    return ET.tostring(package, encoding="unicode")
