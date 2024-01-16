"""
@Author  :shw
@Date    :2024/1/12 14:44
"""
import torch
from torch import nn
from torch.nn import functional as F

"""
使用在3个通道的彩色图像上 数据集采用CIFAR-10 输入图像大小32*32
"""
class LeNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3,16,5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,32,5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
