import os
import torch
import numpy as np
from bp_net.model import MLPmodel
from datetime import datetime


# 定义函数来裁剪图片的中心区域
def crop_center(image, n):
    y_center, x_center = image.shape[0] // 2, image.shape[1] // 2
    crop = image[y_center - n // 2: y_center + n // 2, x_center - n // 2: x_center + n // 2]
    return crop


# 计算灰度均值
def calculate_mean(grayscale_image):
    return np.mean(grayscale_image)


# 加载模型函数
def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)

    # 初始化模型
    model = MLPmodel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式

    # 加载归一化器
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']

    return model, scaler_X, scaler_y


# 计算指定区域的灰度均值
def calculate_region_mean(data, x, y, window_size):
    half = window_size // 2
    h, w = data.shape

    # 计算区域边界
    x_min = max(0, x - half)
    x_max = min(w, x + half + 1)
    y_min = max(0, y - half)
    y_max = min(h, y + half + 1)

    # 提取区域数据
    region = data[y_min:y_max, x_min:x_max]
    return np.mean(region)


# 从文件名解析时间信息
def parse_time_from_filename(filename):
    try:
        # 示例文件名：2025-03-26-18_52_41_1号.txt
        file_stem = os.path.splitext(filename)[0]
        time_str = file_stem[:19]  # 获取"2025-03-26-18"
        capture_time = datetime.strptime(time_str, "%Y-%m-%d-%H_%M_%S")
        return capture_time
    except Exception as e:
        print(f"无法解析文件名中的时间信息：{filename} - {e}")
        return None
