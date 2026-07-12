import os
import json
import time
import requests
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ==========================================
# 0. 全局变量与依赖声明
# ==========================================
try:
    import clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[警告] CLIP未安装，跳过品种识别。")

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


# ==========================================
# 1. 配置参数区 (Config)
# ==========================================
class Config:
    INPUT_DIR = "test_images"
    OUTPUT_DIR = "test_results"

    # 锚定到本模块所在目录的绝对路径，避免服务器进程的 CWD 不是
    # bird_classification/ 时重新下载模型 / 把缓存写到别处。
    YOLO_MODEL_NAME = str(Path(__file__).parent / "yolov8n.pt")
    YOLO_CONF_THRESHOLD = 0.5
    DL_INPUT_SIZE = 520
    PAD_COLOR = (0, 0, 0)

    # DeepSeek API 配置：从环境变量 / .env 读取，不再硬编码在源码里
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    CACHE_FILE = str(Path(__file__).parent / "bird_info_cache.json")

    MIN_BLOCK_AREA = 100
    HUE_THRESHOLD = 15
    MORPH_KERNEL = 5

    # 懒人模式控制
    LAZY_MODE = False


# ==========================================
# 2. 模型全局缓存
# ==========================================
_yolo_model = None
_deeplab_model = None
_clip_model = None
_clip_preprocess = None


# ==========================================
# 3. 模块一：数据清洗与预处理
# ==========================================
class DataPreprocessing:
    @staticmethod
    def image_cleaning(image):
        if image is None: return None
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        denoised_img = cv2.bilateralFilter(enhanced_img, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised_img

    @staticmethod
    def pad_to_square_resize(img, target_size):
        h, w = img.shape[:2]
        ratio = min(target_size / h, target_size / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized_img = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        canvas[:] = Config.PAD_COLOR
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
        return canvas, x_offset, y_offset, new_w, new_h


# ==========================================
# 4. 模块二：YOLO 定位与裁切
# ==========================================
class YOLODetector:
    @staticmethod
    def load_model():
        global _yolo_model
        if _yolo_model is None:
            _yolo_model = YOLO(Config.YOLO_MODEL_NAME)
        return _yolo_model

    @staticmethod
    def detect_and_crop(image):
        model = YOLODetector.load_model()
        results = model(image, conf=Config.YOLO_CONF_THRESHOLD)
        cropped_blocks, bboxes = [], []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 14:  # 鸟类
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    h, w, _ = image.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        cropped_blocks.append(image[y1:y2, x1:x2])
                        bboxes.append((x1, y1, x2, y2))
        return cropped_blocks, bboxes


# ==========================================
# 5. 模块三：DeepLabV3 精细分割
# ==========================================
class DeepLabSegmentor:
    @staticmethod
    def load_model():
        global _deeplab_model
        if _deeplab_model is None:
            _deeplab_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
            _deeplab_model.eval()
            if torch.cuda.is_available(): _deeplab_model = _deeplab_model.cuda()
        return _deeplab_model

    @staticmethod
    def get_bird_mask(image_roi):
        model = DeepLabSegmentor.load_model()
        rgb_img = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
        transform = T.Compose(
            [T.ToPILImage(), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_tensor = transform(rgb_img).unsqueeze(0)
        if torch.cuda.is_available(): input_tensor = input_tensor.cuda()
        with torch.no_grad(): output = model(input_tensor)['out'][0]
        pred = output.argmax(0).byte().cpu().numpy()
        return (pred == 3).astype(np.uint8) * 255

    @staticmethod
    def refine_contour(crop_img, x_offset, y_offset, new_w, new_h):
        padded_img, _, _, _, _ = DataPreprocessing.pad_to_square_resize(crop_img, Config.DL_INPUT_SIZE)
        mask_padded = DeepLabSegmentor.get_bird_mask(padded_img)
        mask_resized = mask_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
        orig_h, orig_w = crop_img.shape[:2]
        final_mask = cv2.resize(mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) > 50:
                return main_contour, final_mask
        return None, None


# ==========================================
# 6. 模块四：CLIP 品种识别
# ==========================================
# 候选物种列表：在同学原有 12 个通用类别基础上，补齐国内水边摄影常见的
# 涉禽/水鸟（覆盖样本照片实际出现的夜鹭等近似种），后续可以随时继续加。
DEFAULT_CANDIDATE_SPECIES = [
    # 原有 12 个通用类别
    "Sparrow", "Robin", "Eagle", "Owl", "Crane", "Kingfisher", "Pigeon", "Crow",
    "Woodpecker", "Duck", "Seagull", "Heron",
    # 补充的常见涉禽 / 水鸟
    "Black-crowned Night Heron", "Little Egret", "Great Egret", "Grey Heron",
    "Chinese Pond Heron", "Black Stork", "White Stork", "Eurasian Spoonbill",
    "Black Kite", "Osprey", "Mallard", "Common Coot", "Little Grebe",
    "Great Cormorant", "Common Tern", "Black-headed Gull",
]


class BirdSpeciesClassifier:
    @staticmethod
    def load_model():
        global _clip_model, _clip_preprocess
        if not CLIP_AVAILABLE: return None, None
        if _clip_model is None:
            print("[CLIP] 加载 CLIP 模型 (首次加载约需3-5分钟)...")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
                print("[CLIP] 模型加载成功！")
            except Exception as e:
                print(f"[CLIP] 加载失败: {e}");
                return None, None
        return _clip_model, _clip_preprocess

    @staticmethod
    def _classify_pil_image(image, candidate_species=None):
        if not CLIP_AVAILABLE: return "Unknown", 0.0
        if candidate_species is None:
            candidate_species = DEFAULT_CANDIDATE_SPECIES
        model, preprocess = BirdSpeciesClassifier.load_model()
        if model is None: return "Fail", 0.0
        try:
            image_input = preprocess(image).unsqueeze(0).to(next(model.parameters()).device)
            text_inputs = clip.tokenize([f"a photo of a {sp}" for sp in candidate_species]).to(
                next(model.parameters()).device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            return candidate_species[indices[0]], values[0].item()
        except Exception as e:
            print(f"CLIP Error: {e}")
            return "Error", 0.0

    @staticmethod
    def classify_species(crop_image_path, candidate_species=None):
        image = Image.open(crop_image_path).convert("RGB")
        return BirdSpeciesClassifier._classify_pil_image(image, candidate_species)

    @staticmethod
    def classify_species_from_array(crop_bgr_ndarray, candidate_species=None):
        """和 classify_species 相同，但直接吃内存里的 BGR ndarray（cv2 读的图），
        不用先落盘再读回来。给单图 HTTP 接口用。"""
        image = Image.fromarray(cv2.cvtColor(crop_bgr_ndarray, cv2.COLOR_BGR2RGB))
        return BirdSpeciesClassifier._classify_pil_image(image, candidate_species)


# ==========================================
# 7. 模块五：DeepSeek API 百科生成（自带缓存）
# ==========================================
class BirdInfoGenerator:
    @staticmethod
    def _load_cache():
        if os.path.exists(Config.CACHE_FILE):
            with open(Config.CACHE_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        return {}

    @staticmethod
    def _save_cache(cache_data):
        with open(Config.CACHE_FILE, 'w', encoding='utf-8') as f: json.dump(cache_data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def get_species_info(species_name):
        cache = BirdInfoGenerator._load_cache()
        if species_name in cache: return cache[species_name]

        headers = {"Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        prompt = f"请用中文介绍鸟类 '{species_name}'。严格按JSON返回: {{\"chinese_name\":\"中文名\",\"scientific_name\":\"学名\",\"introduction\":\"100-150字概述\",\"habitat\":\"栖息地\",\"diet\":\"食性\",\"conservation_status\":\"保护级别\"}}"
        data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
        try:
            r = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=20)
            if r.status_code == 200:
                try:
                    info = json.loads(r.json()['choices'][0]['message']['content'])
                    cache[species_name] = info;
                    BirdInfoGenerator._save_cache(cache);
                    return info
                except:
                    return {"chinese_name": "未知", "introduction": "API 返回格式错误，请重试。"}
            return {"chinese_name": "未知", "introduction": f"API失败: {r.status_code}"}
        except Exception as e:
            return {"chinese_name": "未知", "introduction": f"请求异常: {e}"}
        finally:
            time.sleep(0.5)


# ==========================================
# 8. 模块六：基于连续距离衰减的构图美学评价
# ==========================================
class AestheticEvaluator:
    @staticmethod
    def _classify_hue(h_value):
        hd = h_value * 2
        if hd < 15 or hd >= 345:
            return "Red"
        elif 15 <= hd < 45:
            return "Orange/Yellow"
        elif 45 <= hd < 165:
            return "Green"
        elif 165 <= hd < 255:
            return "Blue"
        else:
            return "Purple"

    @staticmethod
    def _extract_color_blocks(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.MORPH_KERNEL, Config.MORPH_KERNEL))
        all_blocks = []
        color_ranges = [
            ("Red", [(0, 50, 50), (10, 255, 255)], [(170, 50, 50), (179, 255, 255)]),
            ("Orange/Yellow", [(11, 50, 50), (37, 255, 255)]),
            ("Green", [(38, 50, 50), (82, 255, 255)]),
            ("Blue", [(83, 50, 50), (127, 255, 255)]),
            ("Purple", [(128, 50, 50), (169, 255, 255)])
        ]
        for color_name, *ranges in color_ranges:
            mask = np.zeros((h, w), dtype=np.uint8)
            for rng in ranges:
                if len(rng) == 2:
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(rng[0]), np.array(rng[1])))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < Config.MIN_BLOCK_AREA: continue
                cm = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(cm, [cnt], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=cm)[:3]
                h_val, s_val, v_val = mean_hsv
                col = AestheticEvaluator._classify_hue(h_val)
                x, y, wb, hb = cv2.boundingRect(cnt)
                all_blocks.append({
                    "color": col, "hue": h_val, "area": area, "position": (x, y, wb, hb), "mean_hsv": mean_hsv
                })
        return all_blocks

    @staticmethod
    def _circular_std(h_values):
        if len(h_values) < 2: return 0
        angles = np.deg2rad(np.array(h_values) * 2)
        cs = np.sum(np.exp(1j * angles))
        avg = np.angle(cs)
        diffs = np.abs(np.angle(np.exp(1j * (angles - avg))))
        return np.rad2deg(np.std(diffs))

    @staticmethod
    def score_composition(blocks, image_shape):
        h, w = image_shape[:2]
        if not blocks: return 0
        anchor_blocks = sorted(blocks, key=lambda b: (-b['mean_hsv'][1], b['mean_hsv'][2], -b['area']))[:3]
        third_points = [(w / 3, h / 3), (2 * w / 3, h / 3), (w / 3, 2 * h / 3), (2 * w / 3, 2 * h / 3)]
        max_dist = np.sqrt(w ** 2 + h ** 2) * 0.35
        scores = []
        for blk in anchor_blocks:
            x, y, wb, hb = blk['position']
            center = (x + wb / 2, y + hb / 2)
            min_dist = min(np.sqrt((center[0] - tp[0]) ** 2 + (center[1] - tp[1]) ** 2) for tp in third_points)
            anchor_score = 100 * (1 - min(min_dist, max_dist) / max_dist)
            scores.append(anchor_score)
        return round(np.mean(scores), 2)

    @staticmethod
    def score_richness(blocks):
        if not blocks: return 0
        color_area = defaultdict(float)
        for b in blocks: color_area[b['color']] += b['area']
        if len(color_area) < 3: return 0
        top = sorted(color_area.items(), key=lambda x: -x[1])[:3]
        total = sum(a for _, a in top)
        ratios = [a / total for _, a in top]
        ideal = [0.6, 0.3, 0.1]
        err = sum(abs(r - i) for r, i in zip(ratios, ideal))
        return max(0, 100 - err * 100)

    @staticmethod
    def score_purity(blocks):
        if not blocks: return 0
        groups = defaultdict(list)
        for b in blocks: groups[b['color']].append(b['hue'])
        scores = [100 * (1 - AestheticEvaluator._circular_std(hues) / 180) for hues in groups.values()]
        return np.mean(scores) if scores else 0

    @staticmethod
    def analyze_image(image):
        blocks = AestheticEvaluator._extract_color_blocks(image)
        comp = AestheticEvaluator.score_composition(blocks, image.shape)
        rich = AestheticEvaluator.score_richness(blocks)
        pure = AestheticEvaluator.score_purity(blocks)
        total = 0.3 * comp + 0.4 * rich + 0.3 * pure
        return {"blocks": blocks, "composition": round(comp, 2), "richness": round(rich, 2), "purity": round(pure, 2),
                "total": round(total, 2)}


# ==========================================
# 9. 模块七：基于量化的 DeepSeek AI 美学赏析与建议
# ==========================================
class AIArtisticCritic:
    @staticmethod
    def get_feedback(score_data):
        headers = {"Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        prompt = f"""
        你是一位专业的摄影与美学评论家。用户提供了一张鸟类摄影照片。
        系统已经通过算法给出了客观的量化美学得分，具体如下：
        1. 构图得分 (基于三分法锚点的距离衰减计算): {score_data['composition']}/100。
        2. 色彩丰富度得分 (基于主色/副色/点缀色的60/30/10理想比例): {score_data['richness']}/100。
        3. 色彩纯净度得分 (基于同类颜色的色调一致性): {score_data['purity']}/100。
        4. 综合美学得分: {score_data['total']}/100。

        **重要提示:**
        请严格基于以上提供的量化数据，不要凭空捏造。
        (1) 根据上述具体得分，对这张照片进行一段 100-150 字的整体美学赏析（好在哪里，不足在哪里）。
        (2) 给出 1-3 条具体、可操作、有针对性的修改建议（例如：裁剪改变主体位置；后期调整色彩比例等）。修改建议必须与上述量化得分严丝合缝、逻辑自洽，绝不能违背分数的指向。
        (3) 如果构图得分低于60，请建议“使用裁剪工具将主体向画面井字格交叉点靠拢”。
        输出格式为 Markdown 代码块。最后不要有额外的输出。
        """
        data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
        try:
            r = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=25)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            return f"AI美学解析失败，HTTP状态码：{r.status_code}"
        except Exception as e:
            return f"AI美学请求异常：{e}"
        finally:
            time.sleep(0.5)


# ==========================================
# 10. 模块八：【懒人模式核心】构图裁剪 + 色彩智能微调
# ==========================================
class AutoCropper:
    @staticmethod
    def generate_optimized_crop(original_image, bird_boxes):
        """根据YOLO框计算所有鸟的包围盒，向最近的三分点平移裁剪（保守偏移 <= 8%）"""
        if not bird_boxes: return None
        h, w = original_image.shape[:2]
        min_x = min(b[0] for b in bird_boxes)
        min_y = min(b[1] for b in bird_boxes)
        max_x = max(b[2] for b in bird_boxes)
        max_y = max(b[3] for b in bird_boxes)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        third_pts = [(w / 3, h / 3), (2 * w / 3, h / 3), (w / 3, 2 * h / 3), (2 * w / 3, 2 * h / 3)]
        nearest_pt = min(third_pts, key=lambda p: np.sqrt((center_x - p[0]) ** 2 + (center_y - p[1]) ** 2))
        target_x, target_y = nearest_pt
        offset_x = target_x - center_x
        offset_y = target_y - center_y
        # 保守限制：最大偏移量不超过画幅的8%
        max_offset_x = w * 0.08
        max_offset_y = h * 0.08
        offset_x = max(-max_offset_x, min(max_offset_x, offset_x))
        offset_y = max(-max_offset_y, min(max_offset_y, offset_y))
        new_x1 = max(0, int(offset_x))
        new_y1 = max(0, int(offset_y))
        new_x2 = min(w, int(w + offset_x))
        new_y2 = min(h, int(h + offset_y))
        cropped_img = original_image[new_y1:new_y2, new_x1:new_x2]
        return cropped_img


class AutoEnhancer:
    @staticmethod
    def conservative_enhance(image, aesthetic_data):
        """
        根据量化得分进行保守的色彩增强。
        如果色彩丰富度或综合得分偏低，自动适度增加饱和度和亮度。
        """
        if aesthetic_data['richness'] >= 60 or aesthetic_data['total'] >= 60:
            return image  # 分数够高，不做调整

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # 饱和度调整：分数低于40，增加15%；分数40-60，增加8%
        if aesthetic_data['richness'] < 40:
            s = s * 1.15
        elif aesthetic_data['richness'] < 60:
            s = s * 1.08
        s = np.clip(s, 0, 255)

        # 明度调整：综合得分低于50，稍微提亮7%
        if aesthetic_data['total'] < 50:
            v = v * 1.07
            v = np.clip(v, 0, 255)

        hsv_enhanced = cv2.merge([h, s, v]).astype(np.uint8)
        enhanced_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        return enhanced_img


# ==========================================
# 11. 主流程控制函数
# ==========================================
def process_folder():
    # 询问懒人模式
    user_choice = input("是否开启懒人模式（自动保存构图优化裁剪图+色彩微调）? (y/n): ").strip().lower()
    if user_choice in ['y', 'yes']:
        Config.LAZY_MODE = True
        print("[懒人模式] 已开启，将额外输出构图微调 + 色彩轻度增强图。")
    else:
        Config.LAZY_MODE = False
        print("[懒人模式] 未开启，仅输出报告。")

    input_path = Path(Config.INPUT_DIR)
    output_path = Path(Config.OUTPUT_DIR)
    if not input_path.exists():
        print(f"错误：未找到输入文件夹 '{Config.INPUT_DIR}'")
        return
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg"))
    print(f"\n=== 开始处理，共 {len(image_files)} 张图片 ===\n")

    report_path = output_path / "final_report.txt"

    for img_file in image_files:
        filename = img_file.stem
        print(f"-> 处理: {filename}")
        orig_img = cv2.imread(str(img_file))
        if orig_img is None: continue

        cleaned_img = DataPreprocessing.image_cleaning(orig_img)
        cropped_blocks, bboxes = YOLODetector.detect_and_crop(cleaned_img)
        found_birds = len(cropped_blocks)

        result_img = orig_img.copy()
        report_lines = [f"\n【{filename}】", f"检测到鸟类数量: {found_birds}"]

        for idx, (crop_img, (x1, y1, x2, y2)) in enumerate(zip(cropped_blocks, bboxes)):
            padded_img, x_off, y_off, new_w, new_h = DataPreprocessing.pad_to_square_resize(crop_img,
                                                                                            Config.DL_INPUT_SIZE)
            main_contour, _ = DeepLabSegmentor.refine_contour(crop_img, x_off, y_off, new_w, new_h)
            if main_contour is not None:
                global_contour = main_contour + np.array([x1, y1])
                cv2.drawContours(result_img, [global_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            crop_save = output_path / f"{filename}_crop_{idx + 1}.jpg"
            cv2.imwrite(str(crop_save), crop_img)

            species, conf = BirdSpeciesClassifier.classify_species(str(crop_save))
            if species != "Unknown" and conf > 0.3:
                info = BirdInfoGenerator.get_species_info(species)
                cn = info.get('chinese_name', species)
                intro = info.get('introduction', '暂无介绍')
                print(f"    [CLIP] {species} ({conf:.2f}) -> {cn}")
                report_lines.append(f"  {idx + 1}. {cn} ({species}) [置信度:{conf:.2f}]")
                report_lines.append(f"      简介: {intro}")
                cv2.putText(result_img, f"{species}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 量化美学分析（平滑连续算法）
        aesthetic_data = AestheticEvaluator.analyze_image(cleaned_img)
        print(
            f"    [量化] 构图:{aesthetic_data['composition']}, 丰富度:{aesthetic_data['richness']}, 纯净度:{aesthetic_data['purity']}")

        report_lines.append(f"\n  ----- 量化美学分析 -----")
        report_lines.append(f"  构图评分: {aesthetic_data['composition']}/100")
        report_lines.append(f"  色彩丰富度: {aesthetic_data['richness']}/100")
        report_lines.append(f"  色彩纯净度: {aesthetic_data['purity']}/100")
        report_lines.append(f"  综合得分: {aesthetic_data['total']}/100")

        # AI 赏析与建议
        print(f"    [AI批评家] 正在基于量化数据生成美学赏析与建议...")
        ai_feedback = AIArtisticCritic.get_feedback(aesthetic_data)

        report_lines.append(f"\n  ----- AI 美学赏析与修改建议 -----")
        report_lines.append(ai_feedback)
        report_lines.append("\n" + "=" * 50)

        cv2.imwrite(str(output_path / f"{filename}_result.jpg"), result_img)

        # 懒人模式核心：构图裁剪 + 色彩微调
        if Config.LAZY_MODE and bboxes:
            # 1. 先做构图裁剪（基于鸟类包围盒的平移）
            optimized_crop_img = AutoCropper.generate_optimized_crop(orig_img, bboxes)

            if optimized_crop_img is not None and optimized_crop_img.size > 0:
                # 2. 基于量化得分，做保守的色彩增强
                final_opt_img = AutoEnhancer.conservative_enhance(optimized_crop_img, aesthetic_data)

                opt_path = output_path / f"{filename}_optimized.jpg"
                cv2.imwrite(str(opt_path), final_opt_img)
                report_lines.append(f"  [懒人模式] 已生成构图微调 + 色彩轻度增强图：{opt_path.name}")
                print(f"    [懒人模式] 已生成构图微调 + 色彩轻度增强图。")

        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"    [完成] 结果已保存")

    print(f"\n=== 全部处理完毕！报告位于: {Config.OUTPUT_DIR}/final_report.txt ===")


if __name__ == "__main__":
    process_folder()