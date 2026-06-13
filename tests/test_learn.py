import json

from hokusai_press import learn
from hokusai_press.model import (
    Box,
    Deskew,
    Margin,
    PageParams,
    Region,
    RegionKind,
    SourceRef,
)


def _decision(features, label):
    return {"features": json.dumps(features), "new_value": json.dumps(label)}


def test_train_insufficient_data_returns_none():
    decisions = [_decision({"n_photo": 0}, "bw")] * 2
    assert learn.train(decisions) is None  # only one class, too few


def test_train_and_predict_separates_bw_from_color():
    bw = [_decision({"n_text": 30, "n_photo": 0, "photo_area_frac": 0.0}, "bw")
          for _ in range(4)]
    color = [_decision({"n_text": 2, "n_photo": 3, "photo_area_frac": 0.6}, "color")
             for _ in range(4)]
    model = learn.train(bw + color)
    assert model is not None
    assert set(model["counts"]) == {"bw", "color"}

    label_bw, conf_bw = learn.predict(
        model, {"n_text": 28, "n_photo": 0, "photo_area_frac": 0.02})
    label_col, conf_col = learn.predict(
        model, {"n_text": 1, "n_photo": 4, "photo_area_frac": 0.7})
    assert label_bw == "bw" and learn.is_confident(conf_bw)
    assert label_col == "color" and learn.is_confident(conf_col)


def test_page_features_extracts_photo_area():
    params = PageParams(
        source=SourceRef(path="x"),
        deskew=Deskew(angle_deg=0.5, confidence=3.0),
        margin=Margin(content=Box(0, 0, 100, 100), confidence=0.8),
        regions=[
            Region(kind=RegionKind.TEXT, box=Box(0, 0, 50, 10)),
            Region(kind=RegionKind.PHOTO, box=Box(0, 50, 100, 100)),  # 50% area
        ],
    )
    f = learn.page_features(params)
    assert f["n_text"] == 1 and f["n_photo"] == 1
    assert abs(f["photo_area_frac"] - 0.5) < 1e-6
    assert f["deskew_conf"] == 3.0


def test_save_load_round_trip(tmp_path):
    model = learn.train(
        [_decision({"n_photo": 0}, "bw") for _ in range(3)]
        + [_decision({"n_photo": 5}, "color") for _ in range(3)]
    )
    p = tmp_path / "m.json"
    learn.save(model, str(p))
    assert learn.load(str(p))["counts"] == model["counts"]
    assert learn.load(str(tmp_path / "missing.json")) is None
