import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from hokusai_press.ocr.native import (
    NdlocrLiteEngine,
    YomitokuOCREngine,
)


class _Word:
    points = [[1, 2], [11, 2], [11, 12], [1, 12]]
    content = "科学"
    det_score = 0.9
    rec_score = 0.8


class _YomitokuResult:
    words = [_Word()]


class _FakeYomitokuOCR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.recognizer = types.SimpleNamespace(
            _cfg=types.SimpleNamespace(data=types.SimpleNamespace(num_workers=4))
        )
        self.detector = object()

    def __call__(self, image):
        return _YomitokuResult(), None


class _FakeDetector:
    def detect(self, image):
        return [{
            "box": [20, 30, 80, 90],
            "class_index": 6,
            "class_name": "block_fig",
            "confidence": 0.95,
        }]


class _FakeNdlModule(types.ModuleType):
    __file__ = "/tmp/ndlocr-lite/ocr.py"

    @staticmethod
    def get_detector(args):
        return _FakeDetector()

    @staticmethod
    def get_recognizer(args, weights_path=None):
        return object()

    @staticmethod
    def _run_ocr_on_image_array(**kwargs):
        kwargs["detector"].detect(kwargs["img"])
        return {
            "json_lines": [{
                "boundingBox": [[2, 3], [2, 13], [12, 3], [12, 13]],
                "text": "研究",
                "confidence": 0.85,
            }],
        }


def test_yomitoku_native_engine_normalizes_words(monkeypatch):
    module = types.ModuleType("yomitoku.ocr")
    module.OCR = _FakeYomitokuOCR
    monkeypatch.setitem(sys.modules, "yomitoku.ocr", module)

    engine = YomitokuOCREngine(device="cpu")
    result = engine(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result["lines"] == [{
        "polygon": [[1.0, 2.0], [11.0, 2.0], [11.0, 12.0], [1.0, 12.0]],
        "box": (1.0, 2.0, 11.0, 12.0),
        "text": "科学",
        "det_score": 0.9,
        "source": "yomitoku",
    }]
    assert engine._engine.recognizer._cfg.data.num_workers == 0


def test_ndlocr_native_engine_normalizes_lines_and_layout(monkeypatch):
    monkeypatch.setitem(sys.modules, "ocr", _FakeNdlModule("ocr"))

    engine = NdlocrLiteEngine(device="cpu")
    result = engine(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result["lines"][0]["text"] == "研究"
    assert result["lines"][0]["box"] == (2.0, 3.0, 12.0, 13.0)
    assert result["layout_boxes"] == [{
        "box": (20.0, 30.0, 80.0, 90.0),
        "label": "figure",
        "raw_label": "block_fig",
        "class_id": 6,
        "score": 0.95,
        "source": "ndlocr-lite",
    }]


def test_ndlocr_native_engine_does_not_mix_layout_boxes_across_threads(monkeypatch):
    page1_left_lock = threading.Event()
    page2_finished = threading.Event()

    class _SyncLines(list):
        def __init__(self, items, page_id):
            super().__init__(items)
            self._page_id = page_id

        def __iter__(self):
            if self._page_id == 1:
                page1_left_lock.set()
                assert page2_finished.wait(timeout=5.0)
            return super().__iter__()

    class _PerPageDetector:
        def detect(self, image):
            page_id = int(image[0, 0, 0])
            return [{
                "box": [page_id * 100, 10, page_id * 100 + 50, 60],
                "class_index": 6,
                "class_name": "block_fig",
                "confidence": 0.95,
            }]

    class _PerPageNdlModule(types.ModuleType):
        __file__ = "/tmp/ndlocr-lite/ocr.py"

        @staticmethod
        def get_detector(args):
            return _PerPageDetector()

        @staticmethod
        def get_recognizer(args, weights_path=None):
            return object()

        @staticmethod
        def _run_ocr_on_image_array(**kwargs):
            img = kwargs["img"]
            page_id = int(img[0, 0, 0])
            kwargs["detector"].detect(img)
            return {
                "json_lines": _SyncLines([{
                    "boundingBox": [[page_id, 0], [page_id + 10, 0], [page_id + 10, 10], [page_id, 10]],
                    "text": f"page-{page_id}",
                    "confidence": 0.9,
                }], page_id),
            }

    monkeypatch.setitem(sys.modules, "ocr", _PerPageNdlModule("ocr"))
    engine = NdlocrLiteEngine(device="cpu")

    def run_page1():
        img = np.full((10, 10, 3), 1, dtype=np.uint8)
        return engine(img)

    def run_page2():
        assert page1_left_lock.wait(timeout=5.0)
        img = np.full((10, 10, 3), 2, dtype=np.uint8)
        res = engine(img)
        page2_finished.set()
        return res

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(run_page1)
        f2 = pool.submit(run_page2)
        res1 = f1.result(timeout=5.0)
        res2 = f2.result(timeout=5.0)

    assert res1["lines"][0]["text"] == "page-1"
    assert res1["layout_boxes"][0]["box"] == (100.0, 10.0, 150.0, 60.0)
    assert res2["lines"][0]["text"] == "page-2"
    assert res2["layout_boxes"][0]["box"] == (200.0, 10.0, 250.0, 60.0)


def test_ndlocr_native_engine_parallel_threads_preserve_per_page_layout(monkeypatch):
    class _PerPageDetector:
        def detect(self, image):
            page_id = int(image[0, 0, 0])
            return [{
                "box": [page_id * 10, 0, page_id * 10 + 5, 5],
                "class_index": 6,
                "class_name": "block_fig",
                "confidence": 0.9,
            }]

    class _PerPageNdlModule(types.ModuleType):
        __file__ = "/tmp/ndlocr-lite/ocr.py"

        @staticmethod
        def get_detector(args):
            return _PerPageDetector()

        @staticmethod
        def get_recognizer(args, weights_path=None):
            return object()

        @staticmethod
        def _run_ocr_on_image_array(**kwargs):
            img = kwargs["img"]
            page_id = int(img[0, 0, 0])
            kwargs["detector"].detect(img)
            return {
                "json_lines": [{
                    "boundingBox": [[0, 0], [5, 0], [5, 5], [0, 5]],
                    "text": f"page-{page_id}",
                    "confidence": 0.9,
                }],
            }

    monkeypatch.setitem(sys.modules, "ocr", _PerPageNdlModule("ocr"))
    engine = NdlocrLiteEngine(device="cpu")

    def worker(page_id):
        img = np.full((8, 8, 3), page_id, dtype=np.uint8)
        res = engine(img)
        assert res["lines"][0]["text"] == f"page-{page_id}"
        assert res["layout_boxes"][0]["box"] == (page_id * 10.0, 0.0, page_id * 10.0 + 5.0, 5.0)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, i) for i in range(1, 25)]
        for f in futures:
            f.result(timeout=5.0)


def test_hybrid_engine_serializes_concurrent_calls(monkeypatch):
    from hokusai_press.ocr.hybrid import HybridOCREngine

    state_lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0
    first_entered = threading.Event()

    class _FakeHybridOCR:
        def __init__(self, **kwargs):
            pass

        def __call__(self, image_bgr):
            nonlocal in_flight, max_in_flight
            with state_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            try:
                if int(image_bgr[0, 0, 0]) == 1:
                    first_entered.set()
                    threading.Event().wait(0.05)
                return {"lines": [{"text": f"p{int(image_bgr[0, 0, 0])}"}], "layout_boxes": []}
            finally:
                with state_lock:
                    in_flight -= 1

    fake_pipeline_mod = types.ModuleType("hybrid_ocr.pipeline")
    fake_pipeline_mod.HybridOCR = _FakeHybridOCR
    monkeypatch.setitem(sys.modules, "hybrid_ocr", types.ModuleType("hybrid_ocr"))
    monkeypatch.setitem(sys.modules, "hybrid_ocr.pipeline", fake_pipeline_mod)

    engine = HybridOCREngine(model_dir=".", device="cpu")

    def call1():
        return engine(np.full((4, 4, 3), 1, dtype=np.uint8))

    def call2():
        assert first_entered.wait(timeout=5.0)
        return engine(np.full((4, 4, 3), 2, dtype=np.uint8))

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(call1)
        f2 = pool.submit(call2)
        r1 = f1.result(timeout=5.0)
        r2 = f2.result(timeout=5.0)

    assert max_in_flight == 1
    assert r1["lines"][0]["text"] == "p1"
    assert r2["lines"][0]["text"] == "p2"


def test_oop_engine_serializes_concurrent_calls(tmp_path):
    import json
    import queue as queue_mod
    from hokusai_press.ocr.oop import OutOfProcessEngine

    engine = OutOfProcessEngine()
    req1_sent = threading.Event()
    in_flight_lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    class _FakeStdin:
        def __init__(self, out_q):
            self._out_q = out_q

        def write(self, data):
            nonlocal in_flight, max_in_flight
            req = json.loads(data)
            img = np.load(req["image_npy"])
            page_id = int(img[0, 0, 0])
            with in_flight_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)

            def respond():
                nonlocal in_flight
                if page_id == 1:
                    req1_sent.set()
                    threading.Event().wait(0.05)
                resp = {
                    "lines": [{"box": [0, 0, 10, 10], "text": f"page-{page_id}", "score": 0.9}],
                    "layout_boxes": [{"box": [page_id * 10, 0, page_id * 10 + 5, 5], "label": "figure"}],
                }
                with in_flight_lock:
                    in_flight -= 1
                self._out_q.put(json.dumps(resp) + "\n")

            threading.Thread(target=respond, daemon=True).start()

        def flush(self):
            pass

        def close(self):
            pass

    class _FakeProc:
        def __init__(self, out_q):
            self.stdin = _FakeStdin(out_q)
            self.stdout = object()

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    engine._stdout_queue = queue_mod.Queue()
    engine._proc = _FakeProc(engine._stdout_queue)
    engine._start = lambda: None

    def call1():
        return engine(np.full((4, 4, 3), 1, dtype=np.uint8))

    def call2():
        assert req1_sent.wait(timeout=5.0)
        return engine(np.full((4, 4, 3), 2, dtype=np.uint8))

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(call1)
        f2 = pool.submit(call2)
        r1 = f1.result(timeout=5.0)
        r2 = f2.result(timeout=5.0)

    assert max_in_flight == 1
    assert r1["lines"][0]["text"] == "page-1"
    assert r1["layout_boxes"][0]["box"] == (10.0, 0.0, 15.0, 5.0)
    assert r2["lines"][0]["text"] == "page-2"
    assert r2["layout_boxes"][0]["box"] == (20.0, 0.0, 25.0, 5.0)
