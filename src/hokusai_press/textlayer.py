"""Invisible text layer: drawing, removing, and adding it to an existing PDF.

The one place HokusaiPress writes OCR text into a PDF (consolidated from the
hybrid-ocr and NDL-OCR-Lite-NPU writers, 2026-11-22):

* draw_text_pages(): an invisible-text-only PDF (text render mode 3), one
  page per (width, height, lines). Used by the MRC builder for `run`.
* strip_text(): removes all text-showing operators from a page and the form
  XObjects it uses; images and vector graphics stay untouched.
* overlay_text(): adds a text layer to every page of an existing PDF without
  re-encoding its images (used by the `ocr` subcommand).
* image_only_copy(): a copy of a PDF with its text layer removed (the
  "本文二値化" stage output).

Fonts: the bundled Noto Sans JP first, then system fonts, chosen per
character, because a character no registered font maps cannot be encoded
and silently disappears from the layer (measured: 214 of 158,168 characters
of one book with a single font). CJK compatibility ideographs (U+F900...,
which NDL's recognizer emits, e.g. U+FA5B for 者) are folded to the unified
form so that searching for the ordinary character hits.
"""

from __future__ import annotations

import logging
import os
import unicodedata
from collections import Counter
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

# System fonts tried after the bundled one. Meiryo / Yu cover JIS X 0213;
# MS Gothic misses a few old-form kanji (彥產强說稅戶歲).
FALLBACK_FONT_PATHS = [
    "C:\\Windows\\Fonts\\meiryo.ttc",
    "C:\\Windows\\Fonts\\YuGothM.ttc",
    "C:\\Windows\\Fonts\\yumin.ttf",
    "C:\\Windows\\Fonts\\msgothic.ttc",
    "C:\\Windows\\Fonts\\msmincho.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/ipaexfont/ipaexg.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]

_REGISTERED: dict = {}  # font path -> reportlab font name, or None if unusable


def fold_char(ch: str) -> str:
    """CJK compatibility ideograph -> unified character (else unchanged)."""
    cp = ord(ch)
    if 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F:
        folded = unicodedata.normalize("NFKC", ch)
        if len(folded) == 1:
            return folded
    return ch


def _register(path: str):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if path not in _REGISTERED:
        name = None
        p = Path(path)
        if p.is_file() and p.stat().st_size > 0:
            candidate = f"hp-text-{len(_REGISTERED)}"
            try:
                pdfmetrics.registerFont(TTFont(candidate, str(p), subfontIndex=0))
                name = candidate
            except Exception as e:  # CFF outlines, broken file, ...
                logger.debug("font %s unusable: %s", path, e)
        _REGISTERED[path] = name
    return _REGISTERED[path]


class Fonts:
    """Picks, per character, a registered font that maps it."""

    def __init__(self, primary: str | None = None, fallbacks=None):
        if primary is None:
            from .pdf_export.backend import discover_font

            primary = discover_font(None)
        self._pending = [primary] + list(
            FALLBACK_FONT_PATHS if fallbacks is None else fallbacks)
        self._names: list = []
        self.missing = Counter()

    @staticmethod
    def _maps(name: str, ch: str) -> bool:
        from reportlab.pdfbase import pdfmetrics

        return ord(ch) in pdfmetrics.getFont(name).face.charToGlyph

    def _load_next(self) -> bool:
        while self._pending:
            name = _register(self._pending.pop(0))
            if name and name not in self._names:
                self._names.append(name)
                return True
        return False

    def primary(self) -> str:
        if not self._names and not self._load_next():
            from .pdf_export.exceptions import MissingFontError

            raise MissingFontError("no usable Japanese font for the text layer")
        return self._names[0]

    def font_for(self, ch: str):
        self.primary()
        for name in self._names:
            if self._maps(name, ch):
                return name
        while self._load_next():
            if self._maps(self._names[-1], ch):
                return self._names[-1]
        self.missing[ch] += 1
        return None

    def warn_missing(self) -> None:
        if self.missing:
            kinds = "".join(sorted(self.missing))
            logger.warning(
                "text layer: %d character(s) of %d kind(s) have no glyph in any "
                "font and were left out: %s",
                sum(self.missing.values()), len(self.missing), kinds[:80],
            )


def _is_vertical(line: dict) -> bool:
    d = line.get("direction")
    if d in ("v", "h"):
        return d == "v"
    if "vertical" in line:
        return bool(line["vertical"])
    x0, y0, x1, y1 = line["box"][:4]
    return (y1 - y0) > (x1 - x0)


def draw_lines(c, fonts: Fonts, lines, page_h: float, sx: float = 1.0,
               sy: float = 1.0, ox: float = 0.0, oy: float = 0.0) -> None:
    """Draw invisible text for lines whose boxes are in top-left pixel
    coordinates; (sx, sy) scale pixels to points, (ox, oy) offset the area."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    for line in lines:
        text = "".join(fold_char(ch) for ch in (line.get("text") or "") if not ch.isspace())
        if not text:
            continue
        box = line.get("box")
        if not box or len(box) < 4:
            continue
        x0, y0, x1, y1 = box[:4]
        left, right = ox + x0 * sx, ox + x1 * sx
        top, bottom = oy + page_h - y0 * sy, oy + page_h - y1 * sy
        bw, bh = right - left, top - bottom
        if bw <= 0 or bh <= 0:
            continue
        n = len(text)
        if _is_vertical(line):
            # one character per slot down the column, glyph turned so that
            # text selection follows the column
            slot = bh / n
            size = max(1.0, min(bw, slot))
            for j, ch in enumerate(text):
                name = fonts.font_for(ch)
                if name is None:
                    continue  # keep the slot so later characters stay in place
                cy = top - slot * (j + 0.5)
                c.saveState()
                c.translate(left + (bw - size) / 2.0, cy + size / 2.0)
                c.rotate(-90)
                t = c.beginText(0, 0)
                t.setTextRenderMode(3)
                t.setFont(name, size)
                t.textOut(ch)
                c.drawText(t)
                c.restoreState()
        else:
            size = max(1.0, bh * 0.85)
            runs: list = []
            for ch in text:
                name = fonts.font_for(ch)
                if runs and runs[-1][0] == name:
                    runs[-1][1] += ch
                else:
                    runs.append([name, ch])
            natural = sum(
                stringWidth(s, name, size) if name else size * len(s) for name, s in runs
            )
            t = c.beginText(left, bottom + (bh - size) / 2.0)
            t.setTextRenderMode(3)
            if natural > 0:
                t.setHorizScale(100.0 * bw / natural)
            for name, s in runs:
                if name is None:
                    t.moveCursor(size * len(s) * bw / max(natural, 1e-6), 0)
                    continue
                t.setFont(name, size)
                t.textOut(s)
            c.drawText(t)


def draw_text_pages(pages, font_path: str | None = None) -> bytes:
    """Invisible-text-only PDF; pages = [(width, height, lines)] with line
    boxes in the page's own units (1 px = 1 pt)."""
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pageCompression=1)
    fonts = None
    for w, h, lines in pages:
        c.setPageSize((w, h))
        if any((ln.get("text") or "").strip() for ln in lines):
            if fonts is None:
                fonts = Fonts(font_path)
            draw_lines(c, fonts, lines, h)
        c.showPage()
    c.save()
    if fonts is not None:
        fonts.warn_missing()
    return buf.getvalue()


# --- removing a text layer ---------------------------------------------------

def _strip_stream(pdf, owner) -> bool:
    import pikepdf

    ops = pikepdf.parse_content_stream(owner)
    kept, depth, changed = [], 0, False
    for operands, op in ops:
        name = str(op)
        if name == "BT":
            depth += 1
            changed = True
            continue
        if name == "ET":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            kept.append((operands, op))
    if changed:
        data = pikepdf.unparse_content_stream(kept)
        if isinstance(owner, pikepdf.Page):
            owner.obj.Contents = pdf.make_stream(data)
        else:
            owner.write(data)
    return changed


def strip_text(pdf, page) -> int:
    """Remove all text-showing operators from a page and the form XObjects it
    draws (recursively). Returns the number of streams changed.

    This removes visible text too: use it only on scanned PDFs, where all
    text is an OCR layer."""
    import pikepdf

    changed = 1 if _strip_stream(pdf, page) else 0
    seen = set()

    def walk(resources):
        nonlocal changed
        if resources is None or "/XObject" not in resources:
            return
        for _, xo in resources.XObject.items():
            if xo.get("/Subtype") != pikepdf.Name.Form:
                continue
            if xo.objgen in seen:
                continue
            seen.add(xo.objgen)
            if _strip_stream(pdf, xo):
                changed += 1
            walk(xo.get("/Resources"))

    walk(page.obj.get("/Resources"))
    return changed


def image_only_copy(src: str, dst: str) -> None:
    """Write src without its text layer to dst (images untouched)."""
    import pikepdf
    from .model import ensure_parent_dir, refuse_overwrite

    refuse_overwrite(src, dst)
    ensure_parent_dir(dst)
    with pikepdf.open(src) as pdf:
        for page in pdf.pages:
            strip_text(pdf, page)
        pdf.save(dst, deterministic_id=True)


# --- adding a text layer to an existing PDF ----------------------------------

def _page_rotation(page) -> int:
    """Return effective /Rotate (0, 90, 180, 270), resolving inherited /Parent."""
    node = page.obj
    while node is not None:
        if "/Rotate" in node:
            return int(node["/Rotate"]) % 360
        node = node.get("/Parent")
    return 0


def check_no_rotated_pages(src: str) -> None:
    """Raise NotImplementedError before OCR if any page in *src* has /Rotate."""
    import pikepdf

    with pikepdf.open(src) as pdf:
        for page in pdf.pages:
            rot = _page_rotation(page)
            if rot:
                raise NotImplementedError(f"{src}: /Rotate {rot} pages are not supported")


def overlay_text(src: str, pages: list, dst: str, strip_existing: bool = True,
                 font_path: str | None = None) -> None:
    """Write src with an invisible text layer to dst.

    pages[i] = (img_width, img_height, lines) for source page i: the size in
    pixels of the rendering the OCR ran on (pypdfium2 renders the crop box)
    and line boxes in those pixels; None leaves the page as it is."""
    import pikepdf
    from reportlab.pdfgen import canvas
    from .model import ensure_parent_dir, refuse_overwrite

    refuse_overwrite(src, dst)
    ensure_parent_dir(dst)
    with pikepdf.open(src) as pdf:
        if len(pdf.pages) != len(pages):
            raise ValueError(f"{src}: {len(pdf.pages)} pages but {len(pages)} OCR results")
        for page in pdf.pages:
            rot = _page_rotation(page)
            if rot:
                raise NotImplementedError(f"{src}: /Rotate {rot} pages are not supported")
        if strip_existing:
            for page in pdf.pages:
                strip_text(pdf, page)
        buf = BytesIO()
        c = canvas.Canvas(buf, pageCompression=1)
        fonts = None
        for page, ocr in zip(pdf.pages, pages):
            mb = [float(v) for v in page.mediabox]
            cb = [float(v) for v in page.cropbox]
            c.setPageSize((mb[2] - mb[0], mb[3] - mb[1]))
            if ocr is not None:
                iw, ih, lines = ocr
                cw, ch = cb[2] - cb[0], cb[3] - cb[1]
                if lines:
                    if fonts is None:
                        fonts = Fonts(font_path)
                    draw_lines(c, fonts, lines, ch, cw / float(iw), ch / float(ih),
                               cb[0] - mb[0], cb[1] - mb[1])
            c.showPage()
        c.save()
        if fonts is not None:
            fonts.warn_missing()
        with pikepdf.open(BytesIO(buf.getvalue())) as text_pdf:
            for page, tpage, ocr in zip(pdf.pages, text_pdf.pages, pages):
                if ocr is None or not ocr[2]:
                    continue
                mb = [float(v) for v in page.mediabox]
                page.add_overlay(tpage, pikepdf.Rectangle(*mb))
            pdf.save(dst, deterministic_id=True)
