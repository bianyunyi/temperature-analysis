import os
import numpy as np
import torch
from bp_net.train import train_model
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


class TemperatureCalibration:
    def __init__(self, excel_path, save_dir, model_name):
        """
        初始化温度定标类
        :param excel_path: Excel 数据路径
        :param save_dir: 保存模型的目录
        :param model_name: 保存的模型文件名
        """
        self.excel_path = excel_path
        self.save_dir = save_dir
        self.model_name = model_name

        # 创建保存模型的目录
        os.makedirs(self.save_dir, exist_ok=True)

    def train_and_save_model(self):
        """训练模型并保存"""
        # 训练模型
        model, scaler_X, scaler_y, excel_data = train_model(self.excel_path)

        # 保存模型参数
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'input_dim': 1
        }, os.path.join(self.save_dir, self.model_name))

        print(f"模型已保存至 {self.save_dir}/{self.model_name}")
        return model, scaler_X, scaler_y, excel_data

    def visualize_results(self, model, scaler_X, scaler_y, excel_data):
        """可视化模型预测结果"""
        # 使用模型进行预测
        x_new = np.linspace(min(scaler_X.data_min_), max(scaler_X.data_max_), 100).reshape(-1, 1)
        x_new_scaled = scaler_X.transform(x_new)
        x_new_tensor = torch.tensor(x_new_scaled, dtype=torch.float32)

        y_new_scaled = model(x_new_tensor).detach().numpy().reshape(-1, 1)
        y_new = scaler_y.inverse_transform(y_new_scaled)

        plt.figure()
        plt.plot(x_new, y_new, 'b-', label='Predicted')
        plt.scatter(excel_data[['Gray Mean']].values, excel_data['Temperature'].values, color='r', label='True Data')
        plt.title("Gray Mean vs Temperature")
        plt.legend()
        plt.show()

    def run(self):
        """执行整个定标过程"""
        model, scaler_X, scaler_y, excel_data = self.train_and_save_model()
        self.visualize_results(model, scaler_X, scaler_y, excel_data)


if __name__ == "__main__":
    excel_path = r'E:\博一\【项目】\【测温功能开发】\【方案一】直接拟合\datasets\【2025.3.25】BP-NET拟合\data_centercrop\gray_means.xlsx'
    save_dir = "saved_models"
    model_name = "未预热.pth"

    # 创建定标对象并运行
    calibration = TemperatureCalibration(excel_path, save_dir, model_name)
    calibration.run_calibration()
