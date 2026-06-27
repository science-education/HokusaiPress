"""JSON Lines worker for running NDL DEIM/PARSeq out of process."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


NDL_SRC = Path(
    os.environ.get("HOKUSAI_NDL_SRC")
    or str(Path(__file__).resolve().parents[2] / "engines" / "ndl-ocr" / "src")
)

# DEIM 17-class (ndl.yaml) -> canonical layout vocabulary (see base.FIGURE_LIKE_LABELS).
# The consumer maps from canonical labels; raw_label carries the native name.
DEIM_CANONICAL = {
    "text_block": "text", "line_main": "text", "line_caption": "caption",
    "line_ad": "advertisement", "line_note": "note", "line_note_tochu": "note",
    "block_fig": "figure", "block_ad": "advertisement", "block_pillar": "running_head",
    "block_folio": "page_number", "block_rubi": "rubi", "block_chart": "chart",
    "block_eqn": "equation", "block_cfm": "colophon", "block_eng": "text",
    "block_table": "table", "line_title": "title",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _load_ndl_modules():
    sys.path.insert(0, str(NDL_SRC))
    with contextlib.redirect_stdout(sys.stderr):
        from ndl_parser import convert_to_xml_string3
        from ocr import RecogLine, get_detector, get_recognizer, process_cascade
        from reading_order.xy_cut.eval import eval_xml

    return RecogLine, get_detector, get_recognizer, process_cascade, convert_to_xml_string3, eval_xml


def _make_args(device: str, openvino_cache_dir: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        det_weights=str(NDL_SRC / "model" / "deim-s-1024x1024.onnx"),
        det_classes=str(NDL_SRC / "config" / "ndl.yaml"),
        det_score_threshold=0.2,
        det_conf_threshold=0.25,
        det_iou_threshold=0.2,
        rec_weights30=str(
            NDL_SRC / "model" / "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx"
        ),
        rec_weights50=str(
            NDL_SRC / "model" / "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx"
        ),
        rec_weights=str(
            NDL_SRC / "model" / "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"
        ),
        rec_classes=str(NDL_SRC / "config" / "NDLmoji.yaml"),
        device=device,
        rec_workers=None,
        rec_batch_size=None,
        rec_batch_require_fixed_model=False,
        openvino_cache_dir=openvino_cache_dir,
        enable_tcy=False,
        simple_mode=False,
        json_only=True,
    )


class DeimWorker:
    def __init__(self, device: str = "cpu", openvino_cache_dir: str | None = None):
        (
            self.RecogLine,
            get_detector,
            get_recognizer,
            self.process_cascade,
            self.convert_to_xml_string3,
            self.eval_xml,
        ) = _load_ndl_modules()
        self.args = _make_args(device, openvino_cache_dir)
        start = time.time()
        print(f"[deim-worker] loading models device={device}", file=sys.stderr, flush=True)
        with contextlib.redirect_stdout(sys.stderr):
            self.detector = get_detector(self.args)
            self.recognizer100 = get_recognizer(self.args)
            self.recognizer30 = get_recognizer(self.args, weights_path=self.args.rec_weights30)
            self.recognizer50 = get_recognizer(self.args, weights_path=self.args.rec_weights50)
        print(f"[deim-worker] models loaded in {time.time() - start:.3f}s", file=sys.stderr, flush=True)

    def infer_path(self, image_npy: str) -> dict[str, Any]:
        npimg_bgr = np.load(image_npy)
        if npimg_bgr.dtype != np.uint8 or npimg_bgr.ndim != 3 or npimg_bgr.shape[2] != 3:
            raise ValueError(f"expected BGR uint8 HxWx3 image, got {npimg_bgr.dtype} {npimg_bgr.shape}")
        # NDL's reference path reads images through PIL RGB. The HokusaiPress contract is BGR.
        img = np.ascontiguousarray(npimg_bgr[:, :, ::-1])

        start = time.time()
        with contextlib.redirect_stdout(sys.stderr):
            detections = self.detector.detect(img)
        classeslist = list(self.detector.classes.values())
        layout_boxes = self._layout_boxes(detections)
        # Folio/running-head text is NOT part of NDL's reading-order LINEs, so it
        # never appears in `lines`. Recognize those regions directly and attach the
        # text to the layout box so the consumer (nombre) can read the page number.
        self._enrich_furniture_text(img, layout_boxes)
        lines = self._recognize_lines(img, detections, classeslist)
        print(
            f"[deim-worker] processed {Path(image_npy).name}: "
            f"detections={len(detections)} lines={len(lines)} layout={len(layout_boxes)} "
            f"elapsed={time.time() - start:.3f}s",
            file=sys.stderr,
            flush=True,
        )
        return {"lines": lines, "layout_boxes": layout_boxes}

    def _layout_boxes(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Full fidelity: emit every detected region with its canonical label and the
        # native DEIM class (raw_label/class_id). The consumer (HokusaiPress) decides
        # how to use each class — figure/table/chart for MRC, page_number/running_head
        # for nombre/margin, etc.
        boxes = []
        for det in detections:
            class_id = int(det.get("class_index", -1))
            raw = str(det.get("class_name", ""))
            x0, y0, x1, y1 = [int(v) for v in det["box"]]
            boxes.append({
                "box": [x0, y0, x1, y1],
                "label": DEIM_CANONICAL.get(raw, "text"),
                "raw_label": raw,
                "class_id": class_id,
                "score": float(det.get("confidence", 0.0)),
                "source": "deim",
            })
        return boxes

    def _enrich_furniture_text(self, img: np.ndarray, layout_boxes: list[dict[str, Any]]) -> None:
        """Recognize page_number / running_head regions (furniture excluded from the
        reading-order LINEs) and attach the text to their layout box."""
        for lb in layout_boxes:
            if lb.get("label") not in ("page_number", "running_head"):
                continue
            x0, y0, x1, y1 = [int(v) for v in lb["box"]]
            crop = img[max(0, y0):y1, max(0, x0):x1, :]
            if crop.size == 0:
                continue
            # PARSeq duplicates a single isolated glyph on a tight crop (a lone "3"
            # reads as "33"). Pad with white so the digit sits in a line-like field.
            ch, cw = crop.shape[:2]
            padw = max(cw, int(ch * 1.5))   # generous side margin (thin "1" needs more)
            padv = int(ch * 0.3)            # a little top/bottom too
            padded = np.full((ch + 2 * padv, cw + 2 * padw, 3), 255, dtype=np.uint8)
            padded[padv:padv + ch, padw:padw + cw] = crop
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    lb["text"] = self.recognizer30.read(padded)
            except Exception as exc:  # never fail the whole page on furniture
                print(f"[deim-worker] furniture recog failed: {exc}", file=sys.stderr, flush=True)

    def _recognize_lines(
        self,
        img: np.ndarray,
        detections: list[dict[str, Any]],
        classeslist: list[str],
    ) -> list[dict[str, Any]]:
        img_h, img_w = img.shape[:2]
        resultobj = [dict(), dict()]
        resultobj[0][0] = []
        for i in range(17):
            resultobj[1][i] = []
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            conf = float(det["confidence"])
            char_count = float(det.get("pred_char_count", 100.0))
            class_index = int(det["class_index"])
            if class_index == 0:
                resultobj[0][0].append([xmin, ymin, xmax, ymax])
            resultobj[1][class_index].append([xmin, ymin, xmax, ymax, conf, char_count])

        xmlstr = self.convert_to_xml_string3(img_w, img_h, "request.npy", classeslist, resultobj)
        root = ET.fromstring("<OCRDATASET>" + xmlstr + "</OCRDATASET>")
        with contextlib.redirect_stdout(sys.stderr):
            self.eval_xml(root, logger=None)

        line_elems = list(root.findall(".//LINE"))
        alllineobj = []
        for idx, lineobj in enumerate(line_elems):
            xmin = int(lineobj.get("X"))
            ymin = int(lineobj.get("Y"))
            line_w = int(lineobj.get("WIDTH"))
            line_h = int(lineobj.get("HEIGHT"))
            pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT") or 100.0)
            lineimg = img[ymin : ymin + line_h, xmin : xmin + line_w, :]
            if lineimg.size == 0:
                continue
            alllineobj.append(self.RecogLine(lineimg, idx, pred_char_cnt))

        if not alllineobj and detections:
            page = root.find("PAGE")
            if page is not None:
                for idx, det in enumerate(detections):
                    xmin, ymin, xmax, ymax = [int(v) for v in det["box"]]
                    line_w = xmax - xmin
                    line_h = ymax - ymin
                    if line_w <= 0 or line_h <= 0:
                        continue
                    line_elem = ET.SubElement(page, "LINE")
                    c_idx = int(det["class_index"])
                    type_name = classeslist[c_idx] if c_idx < len(classeslist) else "本文"
                    line_elem.set("TYPE", type_name)
                    line_elem.set("X", str(xmin))
                    line_elem.set("Y", str(ymin))
                    line_elem.set("WIDTH", str(line_w))
                    line_elem.set("HEIGHT", str(line_h))
                    line_elem.set("CONF", f"{float(det.get('confidence', 0.0)):0.3f}")
                    pred_char_cnt = float(det.get("pred_char_count", 100.0))
                    line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")
                    lineimg = img[ymin:ymax, xmin:xmax, :]
                    if lineimg.size:
                        alllineobj.append(self.RecogLine(lineimg, idx, pred_char_cnt))
                line_elems = list(root.findall(".//LINE"))

        if not alllineobj:
            return []

        rec_batch_size = self.args.rec_batch_size if self.args.device in ("npu", "openvino-auto") else None
        rec_workers = 1 if self.args.device in ("npu", "openvino-auto") else self.args.rec_workers
        with contextlib.redirect_stdout(sys.stderr):
            texts = self.process_cascade(
                alllineobj,
                self.recognizer30,
                self.recognizer50,
                self.recognizer100,
                is_cascade=True,
                rec_workers=rec_workers,
                rec_batch_size=rec_batch_size,
            )

        out = []
        for idx, (lineobj, text) in enumerate(zip(line_elems, texts)):
            xmin = int(lineobj.get("X"))
            ymin = int(lineobj.get("Y"))
            line_w = int(lineobj.get("WIDTH"))
            line_h = int(lineobj.get("HEIGHT"))
            conf = float(lineobj.get("CONF") or 0.0)
            out.append({
                "box": [xmin, ymin, xmin + line_w, ymin + line_h],
                "text": text,
                "score": conf,
                "source": "deim",
            })
        return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "npu", "openvino-auto"])
    parser.add_argument("--openvino-cache-dir", default=None)
    args = parser.parse_args()

    try:
        worker = DeimWorker(device=args.device, openvino_cache_dir=args.openvino_cache_dir)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 2

    for raw in sys.stdin:
        try:
            req = json.loads(raw)
            result = worker.infer_path(req["image_npy"])
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {"lines": [], "layout_boxes": [], "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, default=_json_default), flush=True)
    print("[deim-worker] stdin closed; exiting", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
