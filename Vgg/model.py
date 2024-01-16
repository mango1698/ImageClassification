"""
@Author  :shw
@Date    :2024/1/15 21:18
"""
from torch import nn
import torch
from torch.nn import functional as F

cfgs = {
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}

def make_features(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2,2)]
        else:
            layers += [nn.Conv2d(in_channels,v,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  # 非关键字参数


class VGG(nn.Module):
    def __init__(self,features,class_num=1000,init_weights=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*7*7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,class_num)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x

def vgg(model_name="vgg16",**kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning")
        exit(-1)
    model = VGG(make_features(cfg),**kwargs)
    return model

if __name__ == '__main__':
    x = torch.rand(1,3,224,224)
    model = vgg("vgg16")
    res = model(x)
    print(res.size())