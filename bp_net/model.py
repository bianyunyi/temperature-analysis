
import torch.nn as nn


class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(1, 200, bias=True)
        self.active1 = nn.ReLU()
        # 定义第一个隐藏层
        self.hidden2 = nn.Linear(200, 80)
        self.active2 = nn.ReLU()
        # 定义预测回归层
        self.hidden3 = nn.Linear(80, 10)
        self.active3 = nn.ReLU()
        self.regression = nn.Linear(10, 1)

    # 定义网络的向前传播路径
    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        x = self.hidden3(x)
        x = self.active3(x)
        output = self.regression(x).squeeze(-1)
        # 输出为output
        return output
