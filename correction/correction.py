import numpy as np
import math
import os
from utils.utils import parse_time_from_filename


class TimeCorrection:
    def __init__(self, input_folder, output_folder, A=749.77015, B=857.9647, time_calibrate=1800):
        """
        初始化时间校正处理类

        :param input_folder: 输入数据文件夹路径
        :param output_folder: 输出校准后的图像文件夹路径
        :param A: 曲线参数A
        :param B: 曲线参数B
        :param time_calibrate: 固定的校准时间点
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.A = A
        self.B = B
        self.time_calibrate = time_calibrate

        # 确保输出文件夹存在
        os.makedirs(self.output_folder, exist_ok=True)

    def calibrate_image(self, image_array, t_current):
        """
        图像校准函数

        参数：
        image_array -- numpy数组格式的输入图像
        t_current -- 相机当前工作时间（秒）

        返回：
        calibrated_array -- 校准后的numpy数组图像
        """
        # 计算指数项
        exp_calibrate = math.exp(-self.time_calibrate / self.B)
        exp_current = math.exp(-t_current / self.B)

        # 计算灰度增量
        delta = self.A * (exp_calibrate - exp_current)

        # 应用校准（使用浮点运算避免溢出）
        calibrated = image_array.astype(np.float32) + delta

        # 处理超出范围的值（根据实际测量范围调整）
        calibrated = np.clip(calibrated, 0, 16383)  # 假设使用16位灰度

        return calibrated.astype(np.uint16)


    def process_data_folder(self):
        """处理整个数据文件夹"""
        # 收集所有TXT文件并解析时间
        files = []
        for fname in os.listdir(self.input_folder):
            if fname.endswith('.txt'):
                file_time = parse_time_from_filename(fname)
                if file_time:
                    files.append((file_time, fname))

        if not files:
            print("未找到有效的TXT文件")
            return

        # 按时间排序
        files.sort()
        start_time = files[0][0]

        # 处理每个文件
        for file_time, fname in files:
            # 计算工作时间（秒）
            t_current = (file_time - start_time).total_seconds()

            # 加载图像
            input_path = os.path.join(self.input_folder, fname)
            try:
                image_array = np.loadtxt(input_path, dtype=np.uint16)
            except Exception as e:
                print(f"加载失败：{fname} - {e}")
                continue

            # 执行校准
            calibrated_array = self.calibrate_image(image_array, t_current)

            # 保存结果
            output_path = os.path.join(self.output_folder, f"{fname}")
            np.savetxt(output_path, calibrated_array, fmt='%d', delimiter=' ')

            print(f"已处理：{fname} (工作时间：{t_current:.1f}秒)")


def emissivity_correction(t_n, lambda_wave, epsilon_env, epsilon_obj):
    """
    计算发射率修正后的温度。

    :param lambda_wave: 中心波长
    :param t_n: 测温仪的读数
    :param epsilon_env: 测温仪的发射率
    :param epsilon_obj: 被测物体的发射率
    :return: 修正后的温度
    """
    corrected_temp = t_n + (lambda_wave * ((273.15 + t_n) ** 2) * (epsilon_env - epsilon_obj)) / (1.4388e-2 * epsilon_obj)
    # C2: 第二辐射系数,单位：K·m
    return corrected_temp
