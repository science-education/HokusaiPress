import logging
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import zlib

from .exceptions import UnsupportedCodecError, MissingFontError

logger = logging.getLogger(__name__)

JPEG_QUALITY = 85
AUTO_FIGURE_AREA_RATIO = 0.01
AUTO_FIGURE_FILL_RATIO = 0.2
AUTO_TEXT_DILATE_PX = 5
AUTO_COLOR_CHROMA_STD = 16.0


def is_bilevel(img_bgr: np.ndarray) -> bool:
    """True if the image only contains pure black and pure white pixels."""
    vals = np.unique(img_bgr)
    return len(vals) <= 2 and bool(np.isin(vals, (0, 255)).all())


def binarize(img_bgr: np.ndarray) -> np.ndarray:
    """Return a {0,255} uint8 single-channel image (Otsu; bilevel passthrough)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if is_bilevel(gray):
        return gray.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _has_photo(img_bgr: np.ndarray, lines: list) -> bool:
    """True when non-text ink contains a dense blob (photo/halftone)."""
    binary = binarize(img_bgr)
    ink = (binary == 0).astype(np.uint8)

    mask = np.zeros_like(ink)
    for ln in lines:
        poly = np.array(ln["polygon"], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 1)
    if AUTO_TEXT_DILATE_PX > 0:
        kernel = np.ones((AUTO_TEXT_DILATE_PX * 2 + 1,) * 2, np.uint8)
        mask = cv2.dilate(mask, kernel)
    ink[mask > 0] = 0

    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    page_h, page_w = ink.shape
    page_area = page_h * page_w
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w == page_w or y + h == page_h:
            continue  # scan surround / black margin
        if area < AUTO_FIGURE_AREA_RATIO * page_area:
            continue
        if area / float(w * h) >= AUTO_FIGURE_FILL_RATIO:
            return True
    return False


def _is_color(img_bgr: np.ndarray) -> bool:
    """True when the page has real color content, not just paper tint."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    chroma_std = (
        lab[..., 1].astype(np.float32).std() + lab[..., 2].astype(np.float32).std()
    )
    return chroma_std > AUTO_COLOR_CHROMA_STD


def decide_page_mode(img_bgr: np.ndarray, lines: list) -> str:
    """auto mode: 'bw' unless non-text ink suggests photos/figures."""
    if is_bilevel(img_bgr):
        return "bw"
    if _has_photo(img_bgr, lines):
        return "color" if _is_color(img_bgr) else "gray"
    return "bw"


def _encode_g4(binary: np.ndarray) -> bytes:
    """Encode binary image to CCITT G4 PDF Stream and return PDF bytes."""
    import pikepdf

    if not isinstance(binary, np.ndarray):
        raise TypeError("binary must be a numpy.ndarray")
    if binary.ndim != 2 or binary.dtype != np.uint8:
        raise ValueError("binary must be an HxW uint8 array")

    h, w = binary.shape
    pil = Image.fromarray(binary).convert("1")
    buf = BytesIO()
    pil.save(
        buf, format="TIFF", compression="group4",
        tiffinfo={278: h},
    )
    tiff_bytes = buf.getvalue()

    # Extract raw CCITT G4 strip data from TIFF
    buf.seek(0)
    with Image.open(buf) as t:
        offset = t.tag_v2[273][0]
        byte_count = t.tag_v2[279][0]
        g4_data = tiff_bytes[offset:offset+byte_count]

    pdf = pikepdf.new()
    decode_parms = pikepdf.Dictionary(
        K=-1,
        Columns=w,
        Rows=h,
        BlackIs1=True
    )
    image = pikepdf.Stream(
        pdf,
        g4_data,
        Filter=pikepdf.Name("/CCITTFaxDecode"),
        DecodeParms=decode_parms,
        Type=pikepdf.Name("/XObject"),
        Subtype=pikepdf.Name("/Image"),
        Width=w,
        Height=h,
        ColorSpace=pikepdf.Name("/DeviceGray"),
        BitsPerComponent=1
    )
    content = pikepdf.Stream(pdf, f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode())
    page = pdf.add_blank_page(page_size=(w, h))
    page.obj.Resources = pikepdf.Dictionary(
        XObject=pikepdf.Dictionary(Im0=pdf.make_indirect(image))
    )
    page.obj.Contents = pdf.make_indirect(content)

    out_buf = BytesIO()
    pdf.save(out_buf)
    return out_buf.getvalue()


def _encode_jpeg(img_bgr: np.ndarray, color_mode: str, jpeg_quality: int = 85) -> bytes:
    """Encode grayscale/color image to JPEG PDF Stream and return PDF bytes."""
    import pikepdf

    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a numpy.ndarray")
    if img_bgr.ndim not in (2, 3) or img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr must be an HxW (uint8) or HxWxC (uint8) array")

    h, w = img_bgr.shape[:2]
    if color_mode == "gray":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        ok, jpg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        colorspace = "/DeviceGray"
    elif color_mode == "color":
        if img_bgr.ndim != 3:
            raise UnsupportedCodecError("Color mode requested but image is not 3-channel BGR")
        ok, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        colorspace = "/DeviceRGB"
    else:
        raise UnsupportedCodecError(f"Unsupported color mode for JPEG: {color_mode}")

    if not ok:
        raise RuntimeError("JPEG encoding failed")

    jpg_bytes = jpg.tobytes()

    pdf = pikepdf.new()
    image = pikepdf.Stream(
        pdf,
        jpg_bytes,
        Filter=pikepdf.Name("/DCTDecode"),
        Type=pikepdf.Name("/XObject"),
        Subtype=pikepdf.Name("/Image"),
        Width=w,
        Height=h,
        ColorSpace=pikepdf.Name(colorspace),
        BitsPerComponent=8
    )
    content = pikepdf.Stream(pdf, f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode())
    page = pdf.add_blank_page(page_size=(w, h))
    page.obj.Resources = pikepdf.Dictionary(
        XObject=pikepdf.Dictionary(Im0=pdf.make_indirect(image))
    )
    page.obj.Contents = pdf.make_indirect(content)

    out_buf = BytesIO()
    pdf.save(out_buf)
    return out_buf.getvalue()


def encode_page_pdf(img_bgr: np.ndarray, mode: str, compress: str, jpeg_quality: int = 85) -> bytes:
    """Encode one page image as a single-page PDF (no text layer) using internal backend."""
    if mode not in ("bw", "gray", "color"):
        raise UnsupportedCodecError(f"Unsupported PDF mode: {mode}")

    if mode == "bw":
        binary = binarize(img_bgr)
        if compress == "jbig2":
            try:
                from pyjbig2.api import page_pdf_jbig2
            except ImportError as e:
                raise UnsupportedCodecError(
                    "jbig2 compression requested but 'pyjbig2' package is not available."
                ) from e
            return page_pdf_jbig2(binary)
        elif compress == "g4":
            return _encode_g4(binary)
        else:
            raise UnsupportedCodecError(f"Unsupported compression codec for bw mode: {compress}")

    # mode in ("gray", "color")
    return _encode_jpeg(img_bgr, mode, jpeg_quality)


def _encode_gray_flate_page_pdf(gray: np.ndarray) -> bytes:
    """Single-page PDF containing an 8-bit DeviceGray FlateDecode image."""
    import pikepdf

    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("gray must be an HxW uint8 array")
    h, w = gray.shape
    pdf = pikepdf.new()
    image = pikepdf.Stream(
        pdf,
        zlib.compress(np.ascontiguousarray(gray).tobytes()),
        Filter=pikepdf.Name("/FlateDecode"),
    )
    image.Type = pikepdf.Name("/XObject")
    image.Subtype = pikepdf.Name("/Image")
    image.Width = w
    image.Height = h
    image.ColorSpace = pikepdf.Name("/DeviceGray")
    image.BitsPerComponent = 8
    content = pikepdf.Stream(pdf, f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode())
    page = pdf.add_blank_page(page_size=(w, h))
    page.obj.Resources = pikepdf.Dictionary(
        XObject=pikepdf.Dictionary(Im0=pdf.make_indirect(image))
    )
    page.obj.Contents = pdf.make_indirect(content)
    buf = BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def discover_font(font_path: str | None = None) -> str:
    """Discover a valid Japanese TrueType/OpenType font file."""
    if font_path:
        if os.path.exists(font_path):
            return font_path
        raise MissingFontError(f"Specified font path does not exist: {font_path}")

    # 1. Check hybrid_ocr package resource
    try:
        import hybrid_ocr
        hybrid_ocr_dir = os.path.dirname(hybrid_ocr.__file__)
        resource_font = os.path.join(hybrid_ocr_dir, "resource", "MPLUS1p-Medium.ttf")
        if os.path.exists(resource_font):
            return resource_font
    except ImportError:
        pass

    # 2. Check System Fonts (Windows, Linux, macOS)
    candidates = []

    # Windows Fonts
    windir = os.environ.get("WINDIR") or "C:\\Windows"
    candidates.extend([
        os.path.join(windir, "Fonts", "msgothic.ttc"),
        os.path.join(windir, "Fonts", "msmincho.ttc"),
        os.path.join(windir, "Fonts", "yugothm.ttc"),
        os.path.join(windir, "Fonts", "meiryo.ttc"),
    ])

    # Linux (Ubuntu / Debian / CentOS etc.) Fonts
    linux_paths = [
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
        "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    candidates.extend(linux_paths)

    # macOS Fonts
    macos_paths = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    candidates.extend(macos_paths)

    for fp in candidates:
        if os.path.exists(fp):
            return fp

    # 3. Last fallback: check a few other common locations or generic names in current workdir
    generic_paths = [
        "MPLUS1p-Medium.ttf",
        "msgothic.ttc",
    ]
    for fp in generic_paths:
        if os.path.exists(fp):
            return fp

    raise MissingFontError(
        "Could not discover any suitable Japanese TrueType/OpenType font. "
        "Please ensure system fonts are accessible or specify a valid font_path."
    )


def _calc_font_size(stringWidth, content, bbox_height, bbox_width, font_name):
    """Best font size so the string fills the box."""
    best, min_diff = None, np.inf
    for rate in np.arange(0.5, 1.0, 0.01):
        font_size = bbox_height * rate
        diff = abs(stringWidth(content, font_name, font_size) - bbox_width)
        if diff < min_diff:
            min_diff, best = diff, font_size
    return best


def build_text_overlay(pages: list, font_path: str | None = None) -> bytes:
    """Invisible-text-only PDF: one page per (width, height, lines) tuple."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    # Check if there is any non-empty text across all pages
    has_any_text = False
    for w, h, lines in pages:
        for ln in lines:
            if ln.get("text", "").strip():
                has_any_text = True
                break
        if has_any_text:
            break

    packet = BytesIO()
    c = canvas.Canvas(packet)

    if not has_any_text:
        for w, h, lines in pages:
            c.setPageSize((w, h))
            c.showPage()
        c.save()
        return packet.getvalue()

    font_file = discover_font(font_path)
    font_name = "MPLUS1p-Medium"

    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, font_file))

    for w, h, lines in pages:
        c.setPageSize((w, h))
        for ln in lines:
            text = ln.get("text", "")
            if not text:
                continue
            box = ln.get("box")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            bw, bh = x2 - x1, y2 - y1
            vertical = ln.get("direction") == "v"
            font_size = (
                _calc_font_size(stringWidth, text, bw, bh, font_name)
                if vertical
                else _calc_font_size(stringWidth, text, bh, bw, font_name)
            )
            if not font_size:
                continue
            if vertical:
                char_h = bh / len(text)
                for j, ch in enumerate(text):
                    char_x = x1 + (bw - font_size) / 2
                    char_y = (h - y1) - (j * char_h) - char_h / 2
                    c.saveState()
                    c.translate(char_x, char_y + font_size / 2)
                    c.rotate(-90)
                    t = c.beginText(0, 0)
                    t.setTextRenderMode(3)  # invisible
                    t.setFont(font_name, font_size)
                    t.textOut(ch)
                    c.drawText(t)
                    c.restoreState()
            else:
                t = c.beginText(x1, h - y2 + (bh - font_size) * 0.5)
                t.setTextRenderMode(3)
                t.setFont(font_name, font_size)
                t.textOut(text)
                c.drawText(t)
        c.showPage()
    c.save()
    return packet.getvalue()
