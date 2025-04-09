"""
用于探究红外偏振相机随工作时间发生的信号增益现象：
data文件夹中的txt文件以图像采集时间命名；
将data文件夹中的txt文件以图像中心进行裁剪，并计算灰度均值；
提取txt文件名，并以时间最早的txt文件名作为开始工作的时间，计算采集每一张图像时相机的工作时间；
以工作时间为x，平均灰度值为y，进行散点图绘制，观察信号增益;
利用BP-NET进行温度预测，观察预热多久后，温度稳定并接近于真实温度。
"""
import os
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from model import MLPmodel
import torch
import pandas as pd
from utils.utils import crop_center, calculate_mean, load_model


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


# 预测函数
def predict(gray_values, model, scaler_X, scaler_y):
    """
    输入：灰度值数组 (N,) 或 (N,1)
    输出：预测温度数组 (N,)
    """
    # 转换为二维数组
    if len(gray_values.shape) == 1:
        gray_values = gray_values.reshape(-1, 1)

    # 归一化
    scaled_X = scaler_X.transform(gray_values)

    # 转换为Tensor
    input_tensor = torch.tensor(scaled_X, dtype=torch.float32)

    # 预测
    with torch.no_grad():
        scaled_pred = model(input_tensor).numpy()

    # 反归一化
    final_pred = scaler_y.inverse_transform(scaled_pred.reshape(-1, 1))

    return final_pred.flatten()


# 参数设置
n = 20  # 中心裁剪区域尺寸，可根据需要修改
data_dir = r'E:\博一\【项目】\【测温功能开发】\【方案一】直接拟合\datasets\【2025.3.31】40℃工作时间相机增益\data_cali'  # 数据文件夹路径

# 准备存储数据
times = []
avg_gray_values = []

# 遍历处理所有txt文件
for filename in os.listdir(data_dir):
    if not filename.endswith('.txt'):
        continue

    try:
        # 解析拍摄时间
        file_stem = os.path.splitext(filename)[0]
        time_str = file_stem[:19]  # 提取前19个字符作为时间部分
        capture_time = datetime.strptime(time_str, "%Y-%m-%d-%H_%M_%S")

        # 读取图像数据
        file_path = os.path.join(data_dir, filename)
        image = np.loadtxt(file_path)

        # 裁剪中心区域
        cropped_image = crop_center(image, n)

        # 计算裁剪区域的灰度均值
        gray_mean = calculate_mean(cropped_image)

        # 计算平均灰度值（14bit数据范围：0-16383）
        gray_mean = calculate_mean(cropped_image)

        # 存储结果
        times.append(capture_time)
        avg_gray_values.append(gray_mean)

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")
        continue

# 计算工作时间（秒）
if not times:
    print("未找到有效数据，请检查文件格式和内容")
    exit()

min_time = min(times)
work_durations = [(t - min_time).total_seconds() for t in times]

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(work_durations, avg_gray_values, alpha=0.7)
plt.xlabel("相机工作时间（秒）")
plt.ylabel("中心区域平均灰度值")
plt.title("相机工作时间与中心灰度变化关系")
plt.grid(True)
plt.show()

excel_data = {
    "id": range(1, len(avg_gray_values)+1),
    "平均灰度": avg_gray_values,
    "工作时长(s)": work_durations
}

df = pd.DataFrame(excel_data)
# 定义保存路径（与数据文件夹同级）
save_path = os.path.join(os.path.dirname(data_dir), "camera_worktime_data_cali.xlsx")

try:
    df.to_excel(save_path, index=False)
    print(f"数据已成功保存至：{save_path}")
except Exception as e:
    print(f"保存Excel文件时出错：{str(e)}")

# 加载模型
model, scaler_X, scaler_y = load_model("saved_models/model_weights.pth")

# 示例输入（支持单样本和批量）
test_data = np.array(avg_gray_values)  # 一维数组

# 进行预测
predictions = predict(test_data, model, scaler_X, scaler_y)

# 打印结果
print("预测结果：")
for gray, temp in zip(test_data, predictions):
    print(f"灰度值 {gray:.1f} → 温度 {temp:.2f}℃")
