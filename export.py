import torch
from yomitoku.text_detector import TextDetector
import os

os.makedirs("models", exist_ok=True)
d = TextDetector(device="cpu", visualize=False)
d.model.eval()

x = torch.randn(1, 3, 640, 640)
onnx_path = "models/yomitoku-text-detector-dbnet-v2_1.onnx"

torch.onnx.export(
    d.model,
    x,
    onnx_path,
    opset_version=14,
    input_names=["input"],
    output_names=["preds"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "preds": {0: "batch_size", 2: "height", 3: "width"}
    }
)
print("Export complete:", onnx_path)
