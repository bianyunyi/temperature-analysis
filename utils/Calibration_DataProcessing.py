"""
data文件夹中的txt文件以温度值命名；
将data文件夹中的txt文件以图像中心进行裁剪，并保存至output_folder位置；
计算裁剪位置灰度均值，并保存至excel文件中，格式为：id、temperature、gray_mean。
"""
import os
import numpy as np
from openpyxl import Workbook
from utils import crop_center, calculate_mean  # 从 utils.py 中导入函数


def process_images(input_folder, output_folder, n=20):
    os.makedirs(output_folder, exist_ok=True)
    excel_filename = os.path.join(output_folder, 'gray_means.xlsx')
    wb = Workbook()
    ws = wb.active
    ws.append(["ID", "Temperature", "Gray Mean"])

    image_id = 1
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            # 读取图片数据
            file_path = os.path.join(input_folder, filename)
            image = np.loadtxt(file_path)

            # 裁剪中心区域
            cropped_image = crop_center(image, n)

            # 计算裁剪区域的灰度均值
            gray_mean = calculate_mean(cropped_image)

            # 保存裁剪后的图片到新的文件夹
            output_path = os.path.join(output_folder, filename)
            np.savetxt(output_path, cropped_image, fmt='%d')

            # 去除文件名的后缀 .txt
            filename_without_ext = os.path.splitext(filename)[0]

            # 将结果写入Excel文件
            ws.append([image_id, filename_without_ext, gray_mean])

            image_id += 1

    wb.save(excel_filename)
    print(f"处理完成，灰度均值已保存至 {excel_filename}")
