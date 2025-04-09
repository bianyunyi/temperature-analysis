import os
from Calibration import TemperatureCalibration
from Measurement import TemperaturePredictor,CalibrationConfig


mode = "TemperaturePredictor"  # "TemperatureCalibration"or "TemperaturePredictor"

if mode == "TemperatureCalibration":
    config = {
        "excel_path": r'E:\博一\【项目】\【测温功能开发】\【方案一】直接拟合\datasets\【2025.3.25】BP-NET拟合\data_centercrop\gray_means.xlsx',
        "save_dir": "saved_models",
        "model_name": "未预热.pth"
    }

    calibration = TemperatureCalibration(**config)
    calibration.run()

elif mode == "TemperaturePredictor":
    calibration_config = CalibrationConfig(
        is_emissivity_correction=False,  # 开启或关闭发射率校正
        lambda_wave=10e-6,  # 中心波长
        epsilon_env=0.95,  # 测温仪的发射率
        epsilon_obj=0.95,  # 被测物体的发射率

        is_time_correction=False,  # 开启或关闭工作时间校正
        output_folder=r"E:\博一\【项目】\【测温功能开发】\【方案一】直接拟合\datasets\【2025.3.26】65℃工作时间相机增益\1\data2"
    )

    config = {
        "model_path": "saved_models/未预热.pth",
        "data_dir": r"E:\博一\【项目】\【测温功能开发】\【方案一】直接拟合\datasets\【2025.3.26】65℃工作时间相机增益\1\data",
        "image_dir": r"E:\博一\【项目】\【测温功能开发】\【方案一】直接拟合\datasets\【2025.3.26】65℃工作时间相机增益\1\image",
        "window_size": 11,  # 采样窗口大小（奇数）
        "calibration_config": calibration_config
    }

    predictor = TemperaturePredictor(**config)
    predictor.run()
