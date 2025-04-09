
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import hiddenlayer as hl
from model import MLPmodel


def load_data(excel_path):
    # 加载Excel数据
    excel_data = pd.read_excel(excel_path)

    # 提取特征和标签
    X = excel_data[['Gray Mean']].values
    y = excel_data['Temperature'].values

    # 数据归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=123)

    # 转换为 PyTorch Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建数据加载器
    train_data = Data.TensorDataset(X_train_tensor, y_train_tensor)
    test_data = Data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=len(test_data))

    return train_loader, test_loader, scaler_X, scaler_y


def train_model(excel_path):
    # 加载数据
    excel_data = pd.read_excel(excel_path)  # 在这里加载 Excel 数据
    train_loader, test_loader, scaler_X, scaler_y = load_data(excel_path)

    # 实例化模型
    model = MLPmodel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()  # 最小平方根误差

    train_loss_all = []  # 输出每个批次训练的损失函数
    history = hl.History()  # 记录训练过程中每一轮（epoch）的损失和其他信息
    canvas = hl.Canvas()  # 使用Canvas进行可视化

    # 进行训练
    for epoch in range(300):
        for step, (b_x, b_y) in enumerate(train_loader):  # b_x是当前批次的输入数据，b_y是当前批次的目标值（标签）
            output = model(b_x).flatten()  # MLP在训练batch上的输出
            train_loss = loss_func(output, b_y)  # 平方根误差
            optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
            train_loss.backward()  # 损失的后向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化
            train_loss_all.append(train_loss.item())

        # 每10个epoch输出一次损失
        if (epoch + 1) % 10 == 0:
            output1 = model(b_x)
            test_loss = loss_func(output1, b_y)
            history.log(epoch, test_loss=test_loss)
            with canvas:
                canvas.draw_plot(history["test_loss"])

    # 绘制训练过程中的损失
    plt.figure()
    plt.plot(train_loss_all, "r-")
    plt.title("Train loss per iteration")
    plt.show()

    return model, scaler_X, scaler_y, excel_data
