"""Out-of-process OCR adapter for dependency-isolated engines."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .base import LayoutBox, OCRLine, OCRResult


DEFAULT_WORKER_PYTHON = r"C:\Users\user\dev\NDL-OCR-Lite-NPU\.venv-openvino\Scripts\python.exe"
DEFAULT_WORKER_SCRIPT = (
    Path(__file__).resolve().parents[3] / "engines" / "deim" / "deim_worker.py"
)


class OutOfProcessEngine:
    def __init__(
        self,
        worker_python: str | None = None,
        worker_script: str | Path | None = None,
        device: str = "cpu",
        response_timeout: float = 300.0,
    ):
        self.worker_python = str(worker_python or DEFAULT_WORKER_PYTHON)
        self.worker_script = str(worker_script or DEFAULT_WORKER_SCRIPT)
        self.device = device
        self.response_timeout = response_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "deim-oop",
            "role": "both",
            "speed_score": 0.5,
            "accuracy_score": 0.7,
            "supports": {
                "npu": False,
                "vertical": True,
                "layout_classes": True,
            },
            "license": "CC-BY-4.0",
        }

    def __call__(self, image_bgr: np.ndarray) -> OCRResult:
        result = self._infer(image_bgr)
        return {
            "lines": self._normalize_lines(result.get("lines", [])),
            "layout_boxes": self._normalize_layout_boxes(result.get("layout_boxes", [])),
            "raw": result,
        }

    def recognize_text(self, image_bgr: np.ndarray) -> list[OCRLine]:
        return self(image_bgr).get("lines", [])

    def analyze_layout(self, image_bgr: np.ndarray) -> list[LayoutBox]:
        return self(image_bgr).get("layout_boxes", [])

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [
            self.worker_python,
            self.worker_script,
            "--device",
            self.device,
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
        )
        self._stdout_queue = queue.Queue()
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_thread.start()

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._stdout_queue.put(line)
        self._stdout_queue.put(None)

    def _infer(self, image_bgr: np.ndarray) -> dict[str, Any]:
        self._start()
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("worker process is not available")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="deim_oop_", suffix=".npy", delete=False) as tmp:
                tmp_path = tmp.name
                np.save(tmp, np.ascontiguousarray(image_bgr, dtype=np.uint8))
            request = {"image_npy": str(Path(tmp_path).resolve())}
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            try:
                line = self._stdout_queue.get(timeout=self.response_timeout)
            except queue.Empty:
                self._proc.kill()
                raise TimeoutError(f"worker response timed out after {self.response_timeout}s")
            if not line:
                rc = self._proc.poll()
                raise RuntimeError(f"worker exited without response; returncode={rc}")
            result = json.loads(line)
            if "error" in result:
                raise RuntimeError(f"worker error: {result['error']}")
            return result
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _normalize_lines(items: Any) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for item in items or []:
            box = tuple(float(v) for v in item.get("box", ()))
            if len(box) != 4:
                continue
            lines.append({
                "box": box,  # type: ignore[typeddict-item]
                "text": item.get("text"),
                "det_score": None if item.get("score") is None else float(item.get("score")),
                "source": item.get("source", "deim"),
            })
        return lines

    @staticmethod
    def _normalize_layout_boxes(items: Any) -> list[LayoutBox]:
        boxes: list[LayoutBox] = []
        for item in items or []:
            box = tuple(float(v) for v in item.get("box", ()))
            if len(box) != 4:
                continue
            entry: LayoutBox = {
                "box": box,  # type: ignore[typeddict-item]
                "label": str(item.get("label", "figure")),
                "score": None if item.get("score") is None else float(item.get("score")),
                "source": item.get("source", "deim"),
            }
            if item.get("raw_label") is not None:
                entry["raw_label"] = str(item["raw_label"])
            if item.get("class_id") is not None:
                entry["class_id"] = int(item["class_id"])
            if item.get("text") is not None:
                entry["text"] = str(item["text"])  # furniture text (folio / running head)
            boxes.append(entry)
        return boxes
