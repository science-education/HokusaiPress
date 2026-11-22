"""Identify a scanned book and propose a file name from its bibliographic record.

Steps (`hokusai-press identify`):

1. Read the PDF's text layer. ISBN candidates come from the last pages (the
   colophon, 奥付) and the first pages; OCR noise is tolerated (spaces,
   dashes, O/0, l/1) and every candidate must pass the ISBN check digit.
2. Look every candidate up in NDL Search (OpenSearch, ``isbn=``). A back page
   often advertises other books with their ISBNs, so the record whose title
   best matches the text of the opening pages (cover, title page) wins, with
   a bonus for an ISBN printed next to colophon words (発行, 印刷, ...).
3. Format the name like the existing library:
   ``『本題ー副題ー』（著者 著、訳者 訳）.pdf``.

Nothing is renamed here; callers show the proposals for a person to confirm.
Only ISBNs are sent to NDL Search.
"""

from __future__ import annotations

import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

ENDPOINT = "https://ndlsearch.ndl.go.jp/api/opensearch"
USER_AGENT = "HokusaiPress/0.1 (book scan tool)"
NS = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcndl": "http://ndl.go.jp/dcndl/terms/",
    "dcterms": "http://purl.org/dc/terms/",
}
XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"
COLOPHON_WORDS = ("発行", "印刷", "製本", "著者", "発行所", "初版", "第1刷", "第一刷")
ROLE_RE = re.compile(r"(著|編|訳|監修|監訳|編著|共著|解説|画|絵|写真|作|文|原作|責任編集)\]?\s*$")
ILLEGAL = re.compile(r'[\\/:*?"<>|]')


@dataclass
class Record:
    isbn: str
    title: str
    authors: str            # display form, e.g. "アレクシス・ウィギンズ 著、吉田新一郎 訳"
    publisher: str = ""
    issued: str = ""
    volume: str = ""
    edition: str = ""
    raw_author: str = ""


@dataclass
class Proposal:
    path: str
    name: str | None
    record: Record | None
    isbn_pages: dict = field(default_factory=dict)   # isbn -> page numbers (1-based)
    score: float = 0.0
    note: str = ""


# --- ISBN extraction ------------------------------------------------------------

_NOISE = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1",
                        "S": "5", "B": "8", "ー": "-", "—": "-", "‐": "-", "－": "-",
                        "−": "-", "–": "-"})


def isbn_ok(digits: str) -> bool:
    if len(digits) == 13 and digits.isdigit() and digits[:3] in ("978", "979"):
        s = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:12]))
        return (10 - s % 10) % 10 == int(digits[12])
    if len(digits) == 10 and digits[:9].isdigit() and (digits[9].isdigit() or digits[9] in "Xx"):
        s = sum(int(d) * (10 - i) for i, d in enumerate(digits[:9]))
        check = (11 - s % 11) % 11
        return (str(check) if check < 10 else "X") == digits[9].upper()
    return False


def find_isbns(text: str) -> list[str]:
    """Valid ISBNs in OCR text, in order of appearance (no duplicates)."""
    text = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    for m in re.finditer(r"[I1l]\s{0,3}[S5]\s{0,3}[B8]\s{0,3}N\s*[:：]?\s*([0-9OolIDSB|\-\s‐－−–—ー]{9,80}[0-9Xx])", text):
        raw = m.group(1).translate(_NOISE)
        digits = re.sub(r"[^0-9Xx]", "", raw)
        for cand in (digits[:13], digits[:10]):
            if isbn_ok(cand) and cand not in out:
                out.append(cand)
                break
    return out


# --- NDL lookup -----------------------------------------------------------------

def _display_authors(author_field: str, creators: list[str]) -> str:
    """NDL's author field is the creators' headings, then the display form:
    'Wiggins, Alexis,吉田, 新一郎,アレクシス・ウィギンズ [著],吉田新一郎 訳'
    -> 'アレクシス・ウィギンズ 著、吉田新一郎 訳';
    '...,飯窪真也, 齊藤萌木, 白水始 著' -> '飯窪真也・齊藤萌木・白水始 著'."""
    field_ = (author_field or "").strip()
    heads = ",".join(creators)
    display = field_[len(heads):].lstrip(", ") if heads and field_.startswith(heads) else field_
    groups, names = [], []
    for p in (x.strip() for x in display.split(",")):
        if not p:
            continue
        m = ROLE_RE.search(p)
        if m:
            name = re.sub(r"\s*\[?" + re.escape(m.group(1)) + r"\]?\s*$", "", p).strip()
            if name:
                names.append(name)
            if names:
                groups.append("・".join(names) + " " + m.group(1))
            names = []
        else:
            names.append(p)
    if groups:
        return "、".join(re.sub(r"\s+", " ", g) for g in groups)
    # no display form: "鈴木, 秀幸, 1948-" -> "鈴木秀幸"
    out = []
    for c in creators:
        bits = [b.strip() for b in c.split(",") if b.strip() and not re.match(r"^\d{4}-", b.strip())]
        out.append("".join(bits[:2]))
    return "、".join(n for n in out if n)


def parse_record(item, isbn: str) -> Record:
    title = (item.findtext("dc:title", namespaces=NS) or item.findtext("title") or "").strip()
    creators = [(e.text or "").strip() for e in item.findall("dc:creator", NS)]
    author = (item.findtext("author") or "").strip()
    return Record(
        isbn=isbn,
        title=title,
        authors=_display_authors(author, creators),
        publisher=", ".join((e.text or "").strip() for e in item.findall("dc:publisher", NS)),
        issued=(item.findtext("dcterms:issued", namespaces=NS) or "").strip(),
        volume=(item.findtext("dcndl:volume", namespaces=NS) or "").strip(),
        edition=(item.findtext("dcndl:edition", namespaces=NS) or "").strip(),
        raw_author=author,
    )


def lookup_isbn(isbn: str, opener=None, retries: int = 2) -> Record | None:
    url = ENDPOINT + "?" + urllib.parse.urlencode({"isbn": isbn, "cnt": 5})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with (opener or urllib.request.urlopen)(req, timeout=20) as r:
                root = ET.fromstring(r.read())
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(5 * (attempt + 1))
    books = [it for it in root.iter("item")
             if any((c.text or "") == "図書" for c in it.findall("category"))]
    items = books or list(root.iter("item"))
    return parse_record(items[0], isbn) if items else None


# --- matching and naming --------------------------------------------------------

def _bigrams(s: str) -> set:
    s = re.sub(r"\s", "", unicodedata.normalize("NFKC", s))
    return {s[i:i + 2] for i in range(len(s) - 1)}


def title_score(title: str, text: str) -> float:
    """Share of the title's character bigrams found in the text (0..1)."""
    main = title.split(" : ")[0]
    tb = _bigrams(main)
    if not tb:
        return 0.0
    return len(tb & _bigrams(text)) / len(tb)


def format_name(rec: Record, suffix: str = "") -> str:
    # " = Parallel title" (an English title printed alongside) is dropped
    parts = [p.split(" = ")[0].strip() for p in rec.title.split(" : ")]
    parts = [p for p in parts if p]
    title = parts[0]
    if len(parts) > 1:
        title += "ー" + "ー".join(parts[1:]) + "ー"
    if rec.volume and rec.volume not in title:
        title += " " + rec.volume
    if rec.edition and rec.edition not in title:
        title += " " + rec.edition
    name = f"『{title}』"
    if rec.authors:
        name += f"（{rec.authors}）"
    name = ILLEGAL.sub("＿", name)
    return name + suffix + ".pdf"


def page_texts(path: str, first: int = 8, last: int = 15) -> tuple[dict, str]:
    """({page_number: text} for the last pages, text of the first pages)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(path)
    try:
        n = len(doc)
        tail = {}
        for i in range(max(0, n - last), n):
            tail[i + 1] = doc[i].get_textpage().get_text_bounded()
        head = "\n".join(doc[i].get_textpage().get_text_bounded() for i in range(min(first, n)))
        for i in range(min(first, n)):
            tail.setdefault(i + 1, doc[i].get_textpage().get_text_bounded())
        return tail, head
    finally:
        doc.close()


def identify(path: str, lookup=lookup_isbn, suffix: str = "") -> Proposal:
    pages, head = page_texts(path)
    isbn_pages: dict = {}
    colophon: set = set()
    for no in sorted(pages, reverse=True):
        t = pages[no]
        found = find_isbns(t)
        for isbn in found:
            isbn_pages.setdefault(isbn, []).append(no)
            if any(w in t for w in COLOPHON_WORDS):
                colophon.add(isbn)
    if not isbn_pages:
        return Proposal(path, None, None, {}, 0.0, "ISBNが見つからない")
    best: tuple = (-1.0, None)
    for isbn in isbn_pages:
        rec = lookup(isbn)
        if rec is None:
            continue
        score = title_score(rec.title, head) + (0.3 if isbn in colophon else 0.0)
        if score > best[0]:
            best = (score, rec)
    score, rec = best
    if rec is None:
        return Proposal(path, None, None, isbn_pages, 0.0, "NDLに該当なし")
    note = "" if score >= 0.5 else "書名が表紙・扉の文字と合わない（要確認）"
    return Proposal(path, format_name(rec, suffix), rec, isbn_pages, round(score, 2), note)
