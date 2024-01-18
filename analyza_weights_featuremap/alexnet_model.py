"""
@Author  :shw
@Date    :2024/1/12 16:41
"""
import torch
from torch import nn
from torch.nn import functional as F

# 对花图像数据进行分类
class AlexNet(nn.Module):
    # padding : (1,2,3,4)  左右上下
    def __init__(self,num_classes=1000,init_weights=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3,48,11,4,2)
        self.pool1 = nn.MaxPool2d(3,2)
        self.conv2 = nn.Conv2d(48,128,5,padding=2)
        self.pool2 = nn.MaxPool2d(3,2)
        self.conv3 = nn.Conv2d(128,192,3,padding=1)
        self.conv4 = nn.Conv2d(192,192,3,padding=1)
        self.conv5 = nn.Conv2d(192,128,3,padding=1)
        self.pool3 = nn.MaxPool2d(3,2)

        self.fc1 = nn.Linear(128*6*6,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3 = nn.Linear(2048,num_classes)
        # 是否进行初始化
        # 其实我们并不需要对其进行初始化，因为在pytorch中，对我们对卷积及全连接层，自动使用了凯明初始化方法进行了初始化
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        outputs = []  # 定义一个列表，返回我们要查看的哪一层的输出特征矩阵
        x = self.conv1(x)
        outputs.append(x)
        x = self.pool1(F.relu(x,inplace=True))
        x = self.conv2(x)
        outputs.append(x)
        x = self.pool2(F.relu(x,inplace=True))
        x = self.conv3(x)
        outputs.append(x)
        x = F.relu(x,inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        x = self.pool3(F.relu(self.conv5(x),inplace=True))
        x = x.view(-1,128*6*6)
        x = F.dropout(x,p=0.5)
        x = F.relu(self.fc1(x),inplace=True)
        x = F.dropout(x,p=0.5)
        x = F.relu(self.fc2(x),inplace=True)
        x = self.fc3(x)


        # for name,module in self.named_children():
        #     x = module(x)
        #     if name == ["conv1","conv2","conv3"]:
        #         outputs.append(x)
        return outputs

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                # 凯明初始化 - 何凯明
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0,0.01)  # 使用正态分布给权重赋值进行初始化
                nn.init.constant_(m.bias,0)


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = AlexNet()
    result = model(x)
    print(result.size())