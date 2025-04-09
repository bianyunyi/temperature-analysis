"""
利用saved_models中保存的模型参数对图像温度进行预测：
分别输入14-bit和8-bit图像所在文件夹，保证二者命名一致即可进行交互式测温；
左键单机显示温度，空格键切换至文件夹中下一张图片。
"""
import os
import numpy as np
import torch
from PIL import Image
from utils.utils import load_model, calculate_region_mean
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from correction.correction import emissivity_correction, TimeCorrection


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


class CalibrationConfig:
    def __init__(self, is_emissivity_correction=False, lambda_wave=None, epsilon_env=None, epsilon_obj=None,
                       is_time_correction=False, output_folder=None):
        """
        初始化发射率校正配置

        :param is_emissivity_correction: 是否进行发射率校正
        :param lambda_wave: 中心波长
        :param epsilon_env: 测温仪的发射率
        :param epsilon_obj: 被测物体的发射率

        :param is_time_correction:是否进行工作时间校正
        :param output_folder: 校正后的数据保存文件夹
        """
        self.is_emissivity_correction = is_emissivity_correction
        self.lambda_wave = lambda_wave
        self.epsilon_env = epsilon_env
        self.epsilon_obj = epsilon_obj

        self.is_time_correction = is_time_correction
        self.output_folder = output_folder

    def is_emissivity_correction_needed(self):
        """判断是否需要发射率校正"""
        return self.is_emissivity_correction and self.epsilon_env is not None and self.lambda_wave is not None and self.epsilon_obj is not None

    def is_time_correction_needed(self):
        """判断是否需要工作时间校正"""
        return self.is_time_correction and self.output_folder is not None


class TemperaturePredictor:
    def __init__(self, model_path, data_dir, image_dir, window_size=5, calibration_config=None):
        """
        初始化预测系统
        :param model_path: 模型路径
        :param data_dir: 14位数据目录
        :param image_dir: 8位图像目录
        :param window_size: 采样窗口大小（奇数）
        """
        # 加载模型和归一化器
        self.model, self.scaler_X, self.scaler_y = load_model(model_path)

        # 配置路径参数
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.window_size = window_size
        self.calibration_config = calibration_config
        self.current_image = None
        self.current_data = None

        # 空格键切换标志
        self.space_pressed = False

        # 创建交互界面
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # 如果需要进行工作时间校正，进行处理
        if self.calibration_config.is_time_correction_needed():
            self.time_correction_process()

    def time_correction_process(self):
        """执行工作时间校正"""
        # 创建 TimeCorrection 类的实例
        time_correction = TimeCorrection(input_folder=self.data_dir, output_folder=self.calibration_config.output_folder)
        # 执行校正并保存校正后的数据
        time_correction.process_data_folder()

    def load_image_pair(self, image_name):
        """加载对应的图像和数据对"""
        # 加载8位图像
        image_path = os.path.join(self.image_dir, image_name)
        self.current_image = np.array(Image.open(image_path))

        # 根据是否需要时间校正，决定使用原始数据目录或校正后的数据目录
        if self.calibration_config.is_time_correction_needed():
            # 使用校正后的数据
            data_path = os.path.join(self.calibration_config.output_folder,  # 使用校正后的数据
                                     os.path.splitext(image_name)[0] + ".txt")
        else:
            # 使用原始数据
            data_path = os.path.join(self.data_dir,  # 使用原始数据
                                     os.path.splitext(image_name)[0] + ".txt")

        self.current_data = np.loadtxt(data_path)

        # 验证尺寸一致性
        if self.current_image.shape != self.current_data.shape:
            raise ValueError("图像和数据尺寸不匹配")

    def predict_temperature(self, gray_value):
        """执行温度预测"""
        scaled_X = self.scaler_X.transform([[gray_value]])
        input_tensor = torch.tensor(scaled_X, dtype=torch.float32)

        with torch.no_grad():
            scaled_pred = self.model(input_tensor).numpy()

        return self.scaler_y.inverse_transform(
            scaled_pred.reshape(-1, 1)).flatten()[0]

    def onclick(self, event):
        """鼠标点击事件处理"""
        if event.inaxes != self.ax:
            return

        # 获取点击坐标
        x = int(event.xdata)
        y = int(event.ydata)

        try:
            # 计算区域均值
            gray_mean = calculate_region_mean(self.current_data, x, y, self.window_size)

            # 执行预测
            temperature = self.predict_temperature(gray_mean)

            # 如果开启了发射率校正且配置了相关参数，则进行校正
            if self.calibration_config and self.calibration_config.is_emissivity_correction_needed():
                temperature = emissivity_correction(temperature,
                                                    self.calibration_config.lambda_wave,
                                                    self.calibration_config.epsilon_env,
                                                    self.calibration_config.epsilon_obj
                                                    )

            # 更新显示
            self.ax.clear()
            self.ax.imshow(self.current_image, cmap='gray')
            self.ax.scatter(x, y, c='r', s=50)
            self.ax.text(x + 10, y + 10,
                         f"{temperature:.2f}℃",
                         color='yellow', fontsize=12,
                         bbox=dict(facecolor='black', alpha=0.5))
            plt.draw()
        except Exception as e:
            print(f"预测失败: {str(e)}")

    def on_key_press(self, event):
        """键盘按键事件处理，检测空格键"""
        if event.key == ' ':
            self.space_pressed = True

    def run(self):
        """运行交互系统"""
        # 获取所有图像文件
        image_files = [f for f in os.listdir(self.image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 循环显示图像
        for img_file in image_files:
            try:
                self.load_image_pair(img_file)
                self.ax.imshow(self.current_image, cmap='gray')
                self.ax.set_title(f"当前图像: {img_file}")
                plt.pause(0.1)  # 等待用户点击

                # 重置空格键按下标志，并等待空格键触发
                self.space_pressed = False
                while not self.space_pressed:
                    plt.pause(0.1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"加载 {img_file} 失败: {str(e)}")

                # 最后一张图像也等待空格键后关闭
            self.space_pressed = False
            while not self.space_pressed:
                plt.pause(0.1)
