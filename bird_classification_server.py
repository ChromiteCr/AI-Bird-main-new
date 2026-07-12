"""
识别台后端 —— 基于同学重构的 bird_classification/bird_classification.py。

把那边的批处理管线（YOLOv8 检测 -> DeepLabV3 分割 -> CLIP 品种识别 ->
DeepSeek 中文百科 -> 连续距离衰减美学评分 -> DeepSeek AI 点评 -> 懒人模式
一键裁剪+微调）原样复用，只是从"读文件夹、写报告"改成"收一张图、
吐一个 JSON"，给 index.html 的识别台调用。

运行：
    python3 bird_classification_server.py
默认监听 http://localhost:5051。
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).parent / "bird_classification"))
import bird_classification as bc  # noqa: E402

PORT = 5051

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"使用设备: {DEVICE}")

print("正在加载 YOLOv8 检测模型……")
bc.YOLODetector.load_model()
print("YOLOv8 加载完成")

print("正在加载 DeepLabV3 分割模型（首次启动需要联网下载权重，约 160MB）……")
bc.DeepLabSegmentor.load_model()
print("DeepLabV3 加载完成")

print("正在加载 CLIP 模型（首次加载可能需要几分钟）……")
bc.BirdSpeciesClassifier.load_model()
print("CLIP 加载完成，候选物种数：", len(bc.DEFAULT_CANDIDATE_SPECIES))


def _to_native(obj):
    """递归把 numpy 标量/数组转成原生 Python 类型，否则 Flask 的 jsonify 会在
    contour 坐标、美学评分这些 numpy 计算结果上直接报 TypeError。"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _ndarray_to_data_url(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def analyze_single_image(orig_bgr, lazy_mode=False):
    cleaned_img = bc.DataPreprocessing.image_cleaning(orig_bgr)
    cropped_blocks, bboxes = bc.YOLODetector.detect_and_crop(cleaned_img)

    result_img = orig_bgr.copy()
    birds = []

    for crop_img, (x1, y1, x2, y2) in zip(cropped_blocks, bboxes):
        padded_img, x_off, y_off, new_w, new_h = bc.DataPreprocessing.pad_to_square_resize(
            crop_img, bc.Config.DL_INPUT_SIZE
        )
        main_contour, _ = bc.DeepLabSegmentor.refine_contour(crop_img, x_off, y_off, new_w, new_h)

        contour_points = None
        if main_contour is not None:
            global_contour = main_contour + np.array([x1, y1])
            cv2.drawContours(result_img, [global_contour], -1, (0, 255, 0), 2)
            contour_points = [[int(p[0]), int(p[1])] for p in global_contour.reshape(-1, 2)]
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        species, conf = bc.BirdSpeciesClassifier.classify_species_from_array(crop_img)

        info = None
        if species not in ("Unknown", "Fail", "Error") and conf > 0.3:
            info = bc.BirdInfoGenerator.get_species_info(species)
            cv2.putText(result_img, species, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        birds.append({
            "bbox": [x1, y1, x2, y2],
            "contour": contour_points,
            "species_en": species,
            "species_confidence": conf,
            "crop_data_url": _ndarray_to_data_url(crop_img),
            "info": info,
        })

    aesthetic_data = bc.AestheticEvaluator.analyze_image(cleaned_img)
    ai_critique = bc.AIArtisticCritic.get_feedback(aesthetic_data)

    lazy_result = {"applied": False, "optimized_image_data_url": None}
    if lazy_mode and bboxes:
        optimized_crop_img = bc.AutoCropper.generate_optimized_crop(orig_bgr, bboxes)
        if optimized_crop_img is not None and optimized_crop_img.size > 0:
            final_opt_img = bc.AutoEnhancer.conservative_enhance(optimized_crop_img, aesthetic_data)
            lazy_result = {
                "applied": True,
                "optimized_image_data_url": _ndarray_to_data_url(final_opt_img),
            }

    return {
        "birds": birds,
        "result_image_data_url": _ndarray_to_data_url(result_img),
        "aesthetic": {
            "composition": aesthetic_data["composition"],
            "richness": aesthetic_data["richness"],
            "purity": aesthetic_data["purity"],
            "total": aesthetic_data["total"],
        },
        "ai_critique": ai_critique,
        "lazy_mode": lazy_result,
        "error": None,
    }


app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "yolo_loaded": bc._yolo_model is not None,
        "deeplab_loaded": bc._deeplab_model is not None,
        "clip_loaded": bc._clip_model is not None,
    })


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400
    file = request.files["image"]
    lazy_mode = request.form.get("lazy_mode", "false").lower() == "true"

    try:
        data_bytes = file.read()
        arr = np.frombuffer(data_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解析图片")
    except Exception as e:
        return jsonify({"error": f"无法解析图片: {e}"}), 400

    try:
        result = analyze_single_image(img, lazy_mode=lazy_mode)
    except Exception as e:
        return jsonify({"error": f"分析失败: {e}"}), 500

    return jsonify(_to_native(result))


if __name__ == "__main__":
    print(f"鸟类分析后端已启动 http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=False)
