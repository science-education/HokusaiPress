"""Book identification from the text layer (no network: lookups are faked)."""

from hokusai_press import naming
from hokusai_press.naming import Record, find_isbns, format_name, isbn_ok


def test_isbn_check_digits():
    assert isbn_ok("9784810057881")
    assert not isbn_ok("9784810057882")
    assert isbn_ok("4794805918")


def test_find_isbns_tolerates_ocr_noise():
    text = ("発行所 図書文化社\nISBN978-4-8100-5788-1C3037\n"
            "ＩＳＢＮ 978－4－7948－1093－9\nISBN978-4-7948-1080-91")
    found = find_isbns(text)
    assert found[:2] == ["9784810057881", "9784794810939"]
    assert "9784794810809" in found   # trailing OCR junk digit ignored


def test_display_authors_and_name():
    a = naming._display_authors(
        "Wiggins, Alexis,吉田, 新一郎,アレクシス・ウィギンズ [著],吉田新一郎 訳",
        ["Wiggins, Alexis", "吉田, 新一郎"])
    assert a == "アレクシス・ウィギンズ 著、吉田新一郎 訳"
    assert naming._display_authors("", ["鈴木, 秀幸, 1948-"]) == "鈴木秀幸"
    rec = Record("x", "教育評価を学ぶ : いま問われる「評価」の本質", "梶田叡一 著")
    assert format_name(rec, "_HP") == "『教育評価を学ぶーいま問われる「評価」の本質ー』（梶田叡一 著）_HP.pdf"
    assert format_name(Record("x", 'A/B: "C"', "")) == "『A＿B＿ ＿C＿』.pdf"


def test_identify_prefers_the_book_matching_the_title_page(tmp_path, monkeypatch):
    pages = {270: "ISBN978-4-7948-1093-9\nISBN978-4-7948-1016-8"}
    head = "成績をハックする 評価を学びにいかす10の方法"
    monkeypatch.setattr(naming, "page_texts", lambda path, **k: (pages, head))
    db = {"9784794810939": Record("9784794810939", "最高の授業 : スパイダー討論", "A 著"),
          "9784794810168": Record("9784794810168", "成績をハックする : 評価を学びにいかす10の方法", "B 著")}
    p = naming.identify("x.pdf", lookup=db.get, suffix="_HP")
    assert p.record.isbn == "9784794810168"
    assert p.name == "『成績をハックするー評価を学びにいかす10の方法ー』（B 著）_HP.pdf"
    assert p.note == ""


def test_identify_without_isbn(monkeypatch):
    monkeypatch.setattr(naming, "page_texts", lambda path, **k: ({1: "no number"}, "title"))
    p = naming.identify("x.pdf", lookup=lambda i: None)
    assert p.name is None and "ISBN" in p.note
