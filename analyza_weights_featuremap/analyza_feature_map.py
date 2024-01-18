"""
@Author  :shw
@Date    :2024/1/18 15:30
"""
import matplotlib.pyplot as plt
from torchvision import transforms
import alexnet_model
import torch
from PIL import Image
import numpy as np
from alexnet_model import AlexNet

# AlexNet 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 实例化模型
model = AlexNet(num_classes=5)
weights = torch.load("./alexnet_weight_20.pth", map_location="cpu")
model.load_state_dict(weights)

image = Image.open("./images/yjx.jpg")
image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)

for feature_map in output:
    # (N,C,W,H) -> (C,W,H)
    im = np.squeeze(feature_map.detach().numpy())
    # (C,W,H) -> (W,H,C)
    im = np.transpose(im,[1,2,0])
    plt.figure()
    # 展示当前层的前12个通道
    for i in range(12):
        ax = plt.subplot(3,4,i+1) # i+1: 每个图的索引
        plt.imshow(im[:,:,i])
    plt.show()






