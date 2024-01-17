"""
@Author  :shw
@Date    :2024/1/17 14:01
"""
import torch
from torch import nn

# 定义卷积、BN、ReLU6激活组合操作
class ConvBNReLU(nn.Sequential):

    # 通过设置groups来实现DW卷积
    # 当groups为1时，为常规卷积
    # 当将groups设置为输入特征图当深度时，也就是groups=in_channels，则为DW卷积
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU,self).__init__(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

# 定义倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self,in_channel,out_channel,stride,expand_radio, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_channel = in_channel*expand_radio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        layers = []
        if expand_radio !=1 :
            layers.append(ConvBNReLU(in_channel,hidden_channel,kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_channel,hidden_channel,stride=stride,groups=hidden_channel),
            nn.Conv2d(hidden_channel,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel)
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        residual = self.conv(x)
        if self.use_shortcut :
            residual = residual + x
        return residual

# 将 通道数ch调整为divisor的整数倍
def _make_divisible(ch,divisor=8,min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch,int(ch+divisor / 2)//divisor*divisor)
    if new_ch < ch*0.9:
        new_ch += divisor
    return new_ch

class MobileNetV2(nn.Module):

    def __init__(self,num_classes=1000,alpha=1.0,round_nearest=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        block = InvertedResidual
        input_channel = _make_divisible(32*alpha,round_nearest)
        last_channel = _make_divisible(1280*alpha,round_nearest)
        inverted_residual_setting = [
            # t  c  n  s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        features = []
        features.append(ConvBNReLU(3,input_channel,stride=2))
        for setting in inverted_residual_setting:
            for i in range(setting[2]):
                stride = setting[3] if i == 0 else 1
                out_channel = _make_divisible(setting[1]*alpha,round_nearest)
                features.append(block(input_channel,out_channel,stride=stride,expand_radio=setting[0]))
                input_channel = out_channel
        features.append(ConvBNReLU(input_channel,last_channel,kernel_size=1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel,num_classes)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x