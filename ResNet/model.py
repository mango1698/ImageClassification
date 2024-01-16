"""
@Author  :shw
@Date    :2024/1/16 11:24
"""
from torch import nn
from torch.nn import functional as F


cfgs = {
    'resnet18':[2,2,2,2],
    'resnet34':[3,4,6,3],
    'resnet50':[3,4,6,3],
    'resnet101':[3,4,23,3],
    'resnet152':[3,8,36,3]
}

class ShortCut(nn.Module):

    def __init__(self,in_channel,out_channel,num,kernel_size,fout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num = num
        if num==2:
            self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size)
            self.bn2 = nn.BatchNorm2d(out_channel)
        elif num==3:
            self.conv1 = nn.Conv2d(in_channel,out_channel,1)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size)
            self.bn2 = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel,fout,1)
            self.bn3 = nn.BatchNorm2d(fout)
        if num==2:
            self.conv1x1 = nn.Conv2d(in_channel,out_channel,kernel_size=1)
        elif num==3:
            self.conv1x1 = nn.Conv2d(in_channel,fout,kernel_size=1)

    def forward(self, x):
        if self.num==2:
            x1 = self.conv1(x)
            x1 = F.relu(self.bn1(x1))
            x1 = self.conv2(x1)
            x1 = self.bn2(x1)
            x2 = self.conv1x1(x)
            import torch
            x = torch.cat([x1,x2],dim=1)
            x = F.relu(x)
        elif self.num==3:
            x1 = self.conv1(x)
            x1 = F.relu(self.bn1(x1))
            x1 = self.conv2(x1)
            x1 = F.relu(self.bn2(x1))
            x1 = self.conv3(x1)
            x1 = self.bn3(x1)
            x2 = self.conv1x1(x)
            import torch
            x = torch.cat([x1, x2], dim=1)
            x = F.relu(x)
        return x

