"""
BirdComposeAI - 智能鸟类构图布局系统
纯构图分割线版本，不包含文字标记和边界框
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import json
from pathlib import Path
import random
from datetime import datetime

class BirdComposeAI:
    def __init__(self):
        # 初始化参数
        self.output_dir = "output"
        self.image_dir = "images"
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 构图权重参数
        self.composition_weights = {
            'rule_of_thirds': 0.25,
            'golden_ratio': 0.25,
            'visual_balance': 0.2,
            'color_harmony': 0.15,
            'leading_lines': 0.15
        }
        
        # 颜色定义（用于构图线）
        self.line_colors = {
            'rule_of_thirds': (255, 100, 100, 180),      # 红色，半透明
            'golden_ratio': (255, 200, 50, 180),         # 金色，半透明
            'symmetry': (100, 255, 100, 180),            # 绿色，半透明
            'diagonal': (100, 150, 255, 180)             # 蓝色，半透明
        }
    
    def get_image_files(self):
        """获取images文件夹中的所有图片文件"""
        image_files = []
        for file_name in os.listdir(self.image_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                image_files.append(file_name)
        return sorted(image_files)
    
    def load_image(self, file_name):
        """加载单个图片"""
        file_path = os.path.join(self.image_dir, file_name)
        try:
            img = Image.open(file_path)
            print(f"已加载图片: {file_name} (尺寸: {img.size})")
            return img
        except Exception as e:
            print(f"加载图片 {file_name} 失败: {e}")
            return None
    
    def detect_bird(self, image):
        """检测鸟类主体"""
        # 转换为OpenCV格式
        if image.mode == 'RGBA':
            # 如果有透明通道，转换为RGB并保留透明度信息
            rgb_img = Image.new('RGB', image.size, (255, 255, 255))
            rgb_img.paste(image, mask=image.split()[3])
            img_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
            
            # 使用透明度通道作为掩码
            alpha = np.array(image.split()[-1])
            bird_mask = alpha > 10
        else:
            img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            # 使用自适应阈值
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            bird_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)
        
        # 找到轮廓
        contours, _ = cv2.findContours(bird_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 确保边界框在图片范围内
            h_img, w_img = img_cv.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(10, min(w, w_img - x))
            h = max(10, min(h, h_img - y))
            
            # 计算中心点
            bird_center = (x + w/2, y + h/2)
            
            return {
                'bbox': (x, y, w, h),
                'center': bird_center,
                'width': w,
                'height': h,
                'area': w * h
            }
        else:
            # 如果没有找到轮廓，假设整个图片是鸟类
            h_img, w_img = img_cv.shape[:2]
            return {
                'bbox': (0, 0, w_img, h_img),
                'center': (w_img/2, h_img/2),
                'width': w_img,
                'height': h_img,
                'area': w_img * h_img
            }
    
    def analyze_background(self, background_img):
        """分析背景图片的构图特征"""
        # 确保背景图片是RGB模式
        if background_img.mode != 'RGB':
            bg_img_rgb = background_img.convert("RGB")
        else:
            bg_img_rgb = background_img
        
        return {
            'size': background_img.size,
            'width': background_img.width,
            'height': background_img.height
        }
    
    def calculate_rule_of_thirds(self, bg_size, bird_info):
        """计算三分法构图位置"""
        width, height = bg_size
        
        # 九宫格交叉点
        points = [
            (width/3, height/3),      # 左上点
            (2*width/3, height/3),    # 右上点
            (width/3, 2*height/3),    # 左下点
            (2*width/3, 2*height/3)   # 右下点
        ]
        
        # 计算每个点的分数
        scores = []
        for point in points:
            # 距离画面中心的距离（不要太近）
            dist_to_center = np.sqrt((point[0] - width/2)**2 + (point[1] - height/2)**2)
            score = dist_to_center / (np.sqrt((width/2)**2 + (height/2)**2))
            scores.append(score)
        
        # 选择最佳点（分数最高的）
        best_idx = np.argmax(scores)
        best_point = points[best_idx]
        
        return {
            'position': best_point,
            'method': '三分法构图',
            'description': f'九宫格交叉点构图',
            'score': scores[best_idx],
            'grid_points': points  # 存储所有网格点，用于绘制
        }
    
    def calculate_golden_ratio(self, bg_size, bird_info):
        """计算黄金分割构图位置"""
        width, height = bg_size
        phi = 1.618
        
        # 黄金分割点
        points = [
            (width/phi, height/phi),           # 主黄金点
            (width - width/phi, height/phi),   # 对称黄金点
            (width/phi, height - height/phi),
            (width - width/phi, height - height/phi)
        ]
        
        # 计算每个点的分数
        scores = []
        for point in points:
            # 避免太靠近边缘
            margin_x = abs(point[0] - width/2) / (width/2)
            margin_y = abs(point[1] - height/2) / (height/2)
            score = (margin_x + margin_y) / 2
            scores.append(score)
        
        best_idx = np.argmax(scores)
        
        return {
            'position': points[best_idx],
            'method': '黄金分割构图',
            'description': '黄金比例构图',
            'score': scores[best_idx],
            'golden_points': points  # 存储所有黄金分割点
        }
    
    def calculate_symmetrical(self, bg_size, bird_info):
        """计算对称构图位置"""
        width, height = bg_size
        
        # 对称构图选项
        options = [
            {
                'position': (width/2, height/3),
                'type': 'vertical',
                'description': '垂直对称构图'
            },
            {
                'position': (width/3, height/2),
                'type': 'horizontal',
                'description': '水平对称构图'
            },
            {
                'position': (width/2, height/2),
                'type': 'central',
                'description': '中心对称构图'
            }
        ]
        
        # 为每个选项评分
        for i, option in enumerate(options):
            pos = option['position']
            
            # 计算平衡分数
            dist_to_center = np.sqrt((pos[0] - width/2)**2 + (pos[1] - height/2)**2)
            center_score = 1 - dist_to_center / (np.sqrt((width/2)**2 + (height/2)**2))
            
            # 如果是对称中心，分数更高
            if option['type'] == 'central':
                center_score *= 1.2
            
            option['score'] = center_score
        
        # 选择最佳选项
        best_option = max(options, key=lambda x: x['score'])
        
        return {
            'position': best_option['position'],
            'method': f'{best_option["type"].title()}对称构图',
            'description': best_option['description'],
            'score': best_option['score'],
            'symmetry_type': best_option['type']
        }
    
    def calculate_diagonal(self, bg_size, bird_info):
        """计算对角线构图位置"""
        width, height = bg_size
        
        # 对角线上的点（避免平均分割）
        diagonal_points = []
        
        # 主对角线（左上到右下）
        for ratio in [0.3, 0.4, 0.6, 0.7]:
            x = width * ratio
            y = height * ratio
            diagonal_points.append((x, y, '主对角线'))
        
        # 计算每个点的分数
        best_score = -1
        best_point = None
        
        for x, y, desc in diagonal_points:
            # 避免太靠近边缘
            edge_dist = min(x, width-x, y, height-y)
            edge_score = edge_dist / (min(width, height) / 2)
            
            # 动态感分数
            symmetry_score = abs((x/width) - (y/height))
            
            total_score = edge_score * 0.6 + symmetry_score * 0.4
            
            if total_score > best_score:
                best_score = total_score
                best_point = (x, y)
        
        return {
            'position': best_point,
            'method': '对角线构图',
            'description': '对角线构图',
            'score': best_score
        }
    
    def optimize_visual_tension(self, position, bird_info, bg_info):
        """根据视觉张力原则优化位置"""
        width, height = bg_info['size']
        x, y = position
        
        # 确保鸟类大小合适
        bg_area = width * height
        bird_area = bird_info['area']
        current_ratio = bird_area / bg_area
        target_ratio = 1/5
        
        scale_factor = 1.0
        if current_ratio < target_ratio * 0.8:
            scale_factor = min(2.0, target_ratio / current_ratio)
        elif current_ratio > target_ratio * 1.2:
            scale_factor = max(0.5, target_ratio / current_ratio)
        
        # 避免太靠近边缘
        margin = min(width, height) * 0.1
        x = max(margin, min(width - margin, x))
        y = max(margin, min(height - margin, y))
        
        return (x, y), scale_factor
    
    def generate_composition_options(self, bird_img, background_img):
        """生成多种构图方案"""
        # 分析鸟类和背景
        bird_info = self.detect_bird(bird_img)
        bg_info = self.analyze_background(background_img)
        
        options = {}
        
        # 1. 三分法构图
        options['rule_of_thirds'] = self.calculate_rule_of_thirds(bg_info['size'], bird_info)
        
        # 2. 黄金分割构图
        options['golden_ratio'] = self.calculate_golden_ratio(bg_info['size'], bird_info)
        
        # 3. 对称构图
        options['symmetrical'] = self.calculate_symmetrical(bg_info['size'], bird_info)
        
        # 4. 对角线构图
        options['diagonal'] = self.calculate_diagonal(bg_info['size'], bird_info)
        
        # 对每个选项进行优化和评分
        for key, option in options.items():
            # 优化位置
            optimized_pos, scale_factor = self.optimize_visual_tension(
                option['position'], bird_info, bg_info
            )
            option['optimized_position'] = optimized_pos
            option['scale_factor'] = scale_factor
            
            # 更新总分数
            option['total_score'] = option['score']
        
        return options, bird_info, bg_info
    
    def clean_bird_image(self, bird_img):
        """清理鸟类图片，确保背景透明"""
        if bird_img.mode == 'RGBA':
            # 已经是RGBA模式，检查透明度
            alpha = bird_img.split()[3]
            # 如果完全透明，可能需要处理
            return bird_img
        else:
            # 转换为RGBA，设置透明度
            bird_rgba = bird_img.convert("RGBA")
            
            # 创建一个白色背景的掩码
            data = np.array(bird_rgba)
            red, green, blue, alpha = data.T
            
            # 将白色背景（接近白色）设为透明
            white_areas = (red > 240) & (green > 240) & (blue > 240)
            data[..., :-1][white_areas.T] = (255, 255, 255)  # 保持白色
            data[..., 3][white_areas.T] = 0  # 设置为透明
            
            return Image.fromarray(data)
    
    def apply_composition(self, bird_img, background_img, composition_option):
        """应用构图方案，将鸟类合成到背景中"""
        # 每次都从原始背景图片开始，确保没有其他图片残留
        if background_img.mode != 'RGB':
            result_img = background_img.copy().convert("RGB")
        else:
            result_img = background_img.copy()
        
        # 转换为RGBA以便合成
        result_rgba = result_img.convert("RGBA")
        
        # 清理鸟类图片，确保背景透明
        bird_cleaned = self.clean_bird_image(bird_img)
        
        # 获取优化后的位置和缩放因子
        x, y = composition_option['optimized_position']
        scale_factor = composition_option['scale_factor']
        
        # 调整鸟类大小
        bird_width = int(bird_cleaned.width * scale_factor)
        bird_height = int(bird_cleaned.height * scale_factor)
        bird_resized = bird_cleaned.resize((bird_width, bird_height), Image.Resampling.LANCZOS)
        
        # 计算位置（使鸟类中心位于目标点）
        paste_x = int(x - bird_width / 2)
        paste_y = int(y - bird_height / 2)
        
        # 确保位置在画面内
        paste_x = max(0, min(result_rgba.width - bird_width, paste_x))
        paste_y = max(0, min(result_rgba.height - bird_height, paste_y))
        
        # 使用新创建的临时图像进行合成
        temp_img = Image.new('RGBA', result_rgba.size, (0, 0, 0, 0))
        temp_img.paste(bird_resized, (paste_x, paste_y), bird_resized)
        
        # 合成图像
        result_final = Image.alpha_composite(result_rgba, temp_img)
        
        return result_final, (paste_x, paste_y, bird_width, bird_height)
    
    def draw_pure_composition_lines(self, img, composition_option):
        """在图像上绘制纯构图分割线，无任何文字标记"""
        # 确保图像是RGBA模式以便绘制半透明线
        if img.mode != 'RGBA':
            draw_img = img.convert("RGBA")
        else:
            draw_img = img.copy()
        
        draw = ImageDraw.Draw(draw_img, 'RGBA')
        
        width, height = draw_img.size
        
        # 根据构图方法选择颜色
        method = composition_option['method']
        if '三分' in method:
            line_color = self.line_colors['rule_of_thirds']
        elif '黄金' in method:
            line_color = self.line_colors['golden_ratio']
        elif '对称' in method:
            line_color = self.line_colors['symmetry']
        else:
            line_color = self.line_colors['diagonal']
        
        # 绘制构图分割线
        if '三分法构图' in method:
            # 三分法网格线
            # 垂直线
            draw.line([(width/3, 0), (width/3, height)], fill=line_color, width=2)
            draw.line([(2*width/3, 0), (2*width/3, height)], fill=line_color, width=2)
            
            # 水平线
            draw.line([(0, height/3), (width, height/3)], fill=line_color, width=2)
            draw.line([(0, 2*height/3), (width, 2*height/3)], fill=line_color, width=2)
            
            # 交叉点标记
            for x in [width/3, 2*width/3]:
                for y in [height/3, 2*height/3]:
                    draw.ellipse([x-4, y-4, x+4, y+4], fill=line_color, outline=None)
        
        elif '黄金分割构图' in method:
            # 黄金分割线
            phi = 1.618
            golden_x = width / phi
            golden_y = height / phi
            
            # 垂直线
            draw.line([(golden_x, 0), (golden_x, height)], fill=line_color, width=2)
            draw.line([(width - golden_x, 0), (width - golden_x, height)], fill=line_color, width=2)
            
            # 水平线
            draw.line([(0, golden_y), (width, golden_y)], fill=line_color, width=2)
            draw.line([(0, height - golden_y), (width, height - golden_y)], fill=line_color, width=2)
            
            # 黄金分割点
            points = [
                (golden_x, golden_y),
                (width - golden_x, golden_y),
                (golden_x, height - golden_y),
                (width - golden_x, height - golden_y)
            ]
            
            for point in points:
                draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], 
                            fill=line_color, outline=None)
        
        elif '对称构图' in method:
            # 对称轴线
            symmetry_type = composition_option.get('symmetry_type', 'vertical')
            
            if symmetry_type == 'vertical':
                # 垂直对称线
                draw.line([(width/2, 0), (width/2, height)], fill=line_color, width=3)
            elif symmetry_type == 'horizontal':
                # 水平对称线
                draw.line([(0, height/2), (width, height/2)], fill=line_color, width=3)
            elif symmetry_type == 'central':
                # 中心对称（十字线）
                draw.line([(width/2, 0), (width/2, height)], fill=line_color, width=2)
                draw.line([(0, height/2), (width, height/2)], fill=line_color, width=2)
        
        elif '对角线构图' in method:
            # 对角线
            draw.line([(0, 0), (width, height)], fill=line_color, width=2)
            draw.line([(width, 0), (0, height)], fill=line_color, width=2)
            
            # 标记对角线上的点
            if 'position' in composition_option:
                x, y = composition_option['position']
                draw.ellipse([x-6, y-6, x+6, y+6], fill=line_color, outline=None)
        
        return draw_img
    
    def save_results(self, result_img, lines_img, composition_option, bg_filename, option_name):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bg_stem = Path(bg_filename).stem
        
        # 保存合成图像（无构图线）
        result_filename = f"{self.output_dir}/{bg_stem}_composed_{option_name}_{timestamp}.png"
        result_img.save(result_filename, format='PNG')
        
        # 保存带构图线的图像
        lines_filename = f"{self.output_dir}/{bg_stem}_lines_{option_name}_{timestamp}.png"
        lines_img.save(lines_filename, format='PNG')
        
        # 保存构图信息
        info_filename = f"{self.output_dir}/{bg_stem}_info_{option_name}_{timestamp}.json"
        with open(info_filename, 'w', encoding='utf-8') as f:
            info_dict = {
                'composition_method': composition_option['method'],
                'description': composition_option['description'],
                'optimized_position': composition_option['optimized_position'],
                'scale_factor': composition_option['scale_factor'],
                'score': composition_option['total_score'],
                'timestamp': timestamp,
                'background_file': bg_filename
            }
            json.dump(info_dict, f, ensure_ascii=False, indent=2)
        
        print(f"已保存结果: {result_filename}")
        print(f"已保存构图线图: {lines_filename}")
        
        return result_filename, lines_filename, info_filename
    
    def process_composition(self, bird_file, background_file):
        """处理单个构图组合"""
        print(f"\n处理构图: {bird_file} -> {background_file}")
        
        # 加载图片
        bird_img = self.load_image(bird_file)
        background_img = self.load_image(background_file)
        
        if bird_img is None or background_img is None:
            print(f"无法加载图片，跳过此组合")
            return False
        
        # 生成构图方案
        options, bird_detected_info, bg_analyzed_info = self.generate_composition_options(bird_img, background_img)
        
        # 显示可用的构图方案
        print("\n可用的构图方案:")
        for key, option in options.items():
            print(f"  {key}: {option['method']} - 分数: {option['total_score']:.2f}")
        
        # 选择最佳方案
        best_key = max(options.keys(), key=lambda k: options[k]['total_score'])
        best_option = options[best_key]
        
        print(f"\n最佳方案: {best_key} ({best_option['method']})")
        print(f"构图描述: {best_option['description']}")
        print(f"综合分数: {best_option['total_score']:.2f}")
        
        # 应用构图方案
        result_img, bird_bbox = self.apply_composition(bird_img, background_img, best_option)
        
        # 绘制纯构图分割线（无任何文字标记）
        lines_img = self.draw_pure_composition_lines(result_img, best_option)
        
        # 保存结果
        self.save_results(result_img, lines_img, best_option, background_file, best_key)
        
        return True
    
    def interactive_mode(self):
        """交互式模式：让用户选择鸟类和背景图片"""
        print("=" * 60)
        print("BirdComposeAI - 智能鸟类构图布局系统")
        print("纯构图分割线版本")
        print("=" * 60)
        
        # 获取所有图片文件
        image_files = self.get_image_files()
        
        if not image_files:
            print("在images文件夹中没有找到任何图片！")
            print("请确保将图片放在/images文件夹中")
            return
        
        print(f"\n找到 {len(image_files)} 张图片:")
        for i, file_name in enumerate(image_files):
            print(f"  {i+1}. {file_name}")
        
        # 让用户选择鸟类图片
        print("\n请选择鸟类图片:")
        bird_choice = input("输入鸟类图片编号（或输入逗号分隔的多个编号）: ").strip()
        
        if ',' in bird_choice:
            bird_indices = [int(idx.strip()) - 1 for idx in bird_choice.split(',') if idx.strip().isdigit()]
        else:
            bird_indices = [int(bird_choice) - 1] if bird_choice.isdigit() else []
        
        bird_files = []
        for idx in bird_indices:
            if 0 <= idx < len(image_files):
                bird_files.append(image_files[idx])
        
        if not bird_files:
            print("未选择有效的鸟类图片，使用第一张图片")
            bird_files = [image_files[0]] if image_files else []
        
        print(f"\n选择的鸟类图片: {bird_files}")
        
        # 让用户选择背景图片
        print("\n请选择背景图片:")
        print("0. 使用所有非鸟类图片作为背景")
        for i, file_name in enumerate(image_files):
            if file_name not in bird_files:
                print(f"  {i+1}. {file_name}")
        
        bg_choice = input("输入背景图片编号（或输入逗号分隔的多个编号，输入0使用所有）: ").strip()
        
        background_files = []
        if bg_choice == '0':
            # 使用所有非鸟类图片作为背景
            background_files = [f for f in image_files if f not in bird_files]
        elif ',' in bg_choice:
            bg_indices = [int(idx.strip()) - 1 for idx in bg_choice.split(',') if idx.strip().isdigit()]
            for idx in bg_indices:
                if 0 <= idx < len(image_files):
                    background_files.append(image_files[idx])
        elif bg_choice.isdigit():
            idx = int(bg_choice) - 1
            if 0 <= idx < len(image_files):
                background_files = [image_files[idx]]
        
        if not background_files:
            print("未选择有效的背景图片，使用所有非鸟类图片")
            background_files = [f for f in image_files if f not in bird_files]
        
        print(f"\n选择的背景图片: {background_files}")
        
        # 处理所有组合
        processed_count = 0
        
        for bird_file in bird_files:
            for background_file in background_files:
                if bird_file == background_file:
                    continue  # 跳过相同的文件
                
                print(f"\n{'='*40}")
                print(f"处理组合 {processed_count + 1}:")
                print(f"鸟类: {bird_file}")
                print(f"背景: {background_file}")
                print(f"{'='*40}")
                
                try:
                    success = self.process_composition(bird_file, background_file)
                    if success:
                        processed_count += 1
                except Exception as e:
                    print(f"处理时出错: {e}")
                    continue
        
        print(f"\n处理完成！共处理了 {processed_count} 个构图组合")
        print(f"所有结果已保存到 {self.output_dir} 文件夹")
    
    def auto_mode(self):
        """自动模式：自动识别和处理所有图片"""
        print("=" * 60)
        print("BirdComposeAI - 自动模式")
        print("=" * 60)
        
        # 获取所有图片文件
        image_files = self.get_image_files()
        
        if not image_files:
            print("在images文件夹中没有找到任何图片！")
            return
        
        print(f"\n找到 {len(image_files)} 张图片")
        
        # 自动分类：假设文件名包含"bird"的是鸟类图片
        bird_files = []
        background_files = []
        
        for file_name in image_files:
            file_lower = file_name.lower()
            if any(keyword in file_lower for keyword in ['bird', '鸟类', '鸟', 'bird_', '_bird']):
                bird_files.append(file_name)
            else:
                background_files.append(file_name)
        
        # 如果没有找到鸟类图片，使用第一张图片
        if not bird_files and len(image_files) >= 1:
            bird_files = [image_files[0]]
            background_files = image_files[1:]
        
        print(f"自动识别到 {len(bird_files)} 张鸟类图片")
        print(f"自动识别到 {len(background_files)} 张背景图片")
        
        # 处理所有组合
        processed_count = 0
        
        for bird_file in bird_files:
            for background_file in background_files:
                if bird_file == background_file:
                    continue
                
                print(f"\n处理组合: {bird_file} -> {background_file}")
                
                try:
                    success = self.process_composition(bird_file, background_file)
                    if success:
                        processed_count += 1
                except Exception as e:
                    print(f"处理时出错: {e}")
                    continue
        
        print(f"\n自动处理完成！共处理了 {processed_count} 个构图组合")
        print(f"所有结果已保存到 {self.output_dir} 文件夹")

def create_example_images():
    """创建示例图片"""
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    
    # 创建透明背景的鸟类图片
    bird_img = Image.new('RGBA', (200, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bird_img)
    
    # 绘制一只鸟
    draw.ellipse([50, 50, 150, 120], fill=(255, 100, 100, 255))  # 身体
    draw.ellipse([130, 30, 180, 70], fill=(255, 150, 150, 255))  # 头
    draw.polygon([(80, 80), (40, 60), (80, 40)], fill=(255, 200, 100, 255))  # 翅膀
    draw.ellipse([140, 45, 150, 55], fill=(0, 0, 0, 255))  # 眼睛
    
    bird_img.save(f"{image_dir}/bird.png")
    print("已创建示例鸟类图片: bird.png")
    
    # 创建背景图片1
    bg1 = Image.new('RGB', (800, 600), (100, 150, 200))
    draw = ImageDraw.Draw(bg1)
    draw.rectangle([100, 400, 700, 550], fill=(50, 120, 50))
    draw.rectangle([300, 200, 500, 400], fill=(150, 100, 50))
    draw.ellipse([250, 100, 550, 250], fill=(100, 200, 100))
    
    bg1.save(f"{image_dir}/background1.jpg")
    print("已创建背景图片: background1.jpg")
    
    # 创建背景图片2
    bg2 = Image.new('RGB', (800, 600), (200, 230, 255))
    draw = ImageDraw.Draw(bg2)
    draw.rectangle([0, 300, 800, 600], fill=(240, 240, 200))
    draw.ellipse([100, 100, 300, 300], fill=(255, 255, 150))
    draw.rectangle([400, 350, 600, 450], fill=(180, 150, 100))
    
    bg2.save(f"{image_dir}/background2.jpg")
    print("已创建背景图片: background2.jpg")
    
    # 创建背景图片3
    bg3 = Image.new('RGB', (800, 600), (150, 180, 220))
    draw = ImageDraw.Draw(bg3)
    
    # 绘制山景
    draw.polygon([(0, 400), (200, 200), (400, 400)], fill=(80, 100, 60))
    draw.polygon([(300, 450), (500, 250), (700, 450)], fill=(70, 90, 50))
    draw.rectangle([0, 450, 800, 600], fill=(60, 80, 40))
    
    bg3.save(f"{image_dir}/background3.jpg")
    print("已创建背景图片: background3.jpg")
    
    print("\n示例图片已创建完成！")

if __name__ == "__main__":
    # 检查images文件夹
    image_dir = "images"
    if not os.path.exists(image_dir) or len([f for f in os.listdir(image_dir) 
                                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) < 2:
        print("检测到images文件夹中没有足够的图片，正在创建示例图片...")
        create_example_images()
    
    # 创建构图器
    composer = BirdComposeAI()
    
    # 选择模式
    print("\n请选择运行模式:")
    print("1. 交互模式（手动选择鸟类和背景）")
    print("2. 自动模式（自动识别和处理）")
    
    mode_choice = input("请输入模式编号 (1-2): ").strip()
    
    if mode_choice == '1':
        composer.interactive_mode()
    elif mode_choice == '2':
        composer.auto_mode()
    else:
        print("无效的选择，使用自动模式")
        composer.auto_mode()
    
    print("\n程序运行结束！")