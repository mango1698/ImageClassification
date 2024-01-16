"""
@Author  :shw
@Date    :2024/1/12 15:07
"""
import torch
from torchvision import datasets,transforms
from PIL import Image
from torch.nn import functional as F

test_data = datasets.CIFAR10("./data", train=False, download=True)
classes = test_data.classes

# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image = Image.open("./images/autimobile.jpeg")
transformer = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
image_tensor = transformer(image)  # C,H,W
image_tensor = torch.unsqueeze(image_tensor,dim=0) # N,C,H,W  增加一个批量的维度
model = torch.load("./model/net_20.pth", map_location=torch.device('cpu'))
with torch.no_grad():
    result = model(image_tensor)
    result = F.softmax(result,dim=1)  # 使用softmax激活函数
print(classes[torch.argmax(result,dim=1)])


