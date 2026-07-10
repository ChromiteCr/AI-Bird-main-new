"""
识别台后端 —— 从 bird_classification_v2.ipynb 提取而来。

预处理、模型与推理逻辑与 notebook 保持一致（ResNet50, ImageNet 预训练,
Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize），只是把交互方式
从"读本地目录批量打印"换成了一个给 index.html 调用的小型 HTTP 接口。

运行：
    python3 recognize_server.py
默认监听 http://localhost:5050，index.html 的识别台会自动尝试连接这个地址。
"""

import io
import urllib.request
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image

TOP_K = 3
PORT = 5050
CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
CLASSES_PATH = Path(__file__).parent / "imagenet_classes.txt"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"使用设备: {DEVICE}")

print("正在加载 ResNet50（ImageNet 预训练）……")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = model.to(DEVICE)
model.eval()
print("模型加载完成")

if not CLASSES_PATH.exists():
    print("正在下载 ImageNet 类别标签……")
    try:
        urllib.request.urlretrieve(CLASSES_URL, CLASSES_PATH)
    except Exception:
        mirror_url = "https://ghproxy.com/" + CLASSES_URL
        urllib.request.urlretrieve(mirror_url, CLASSES_PATH)
    print("下载完成")

with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"已加载 {len(class_names)} 个类别，其中鸟类约占 60 种")

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(img: Image.Image) -> torch.Tensor:
    return PREPROCESS(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def predict_bird(img_tensor, top_k=TOP_K):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_indices = torch.topk(probs, top_k)
    results = []
    for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
        results.append({"label": class_names[idx], "confidence": prob.item()})
    return results


app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "num_classes": len(class_names)})


@app.route("/recognize", methods=["POST", "OPTIONS"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400
    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"无法解析图片: {e}"}), 400

    top_k = int(request.args.get("top_k", TOP_K))
    try:
        tensor = preprocess_image(img)
        predictions = predict_bird(tensor, top_k=top_k)
    except Exception as e:
        return jsonify({"error": f"识别失败: {e}"}), 500

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    print(f"识别台后端已启动 http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
