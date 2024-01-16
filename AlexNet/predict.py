"""
@Author  :shw
@Date    :2024/1/15 16:28
"""
import json

import torch
from torchvision import transforms
from PIL import Image
model = torch.load("model/net_19.pth",map_location="cpu")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
image = Image.open("./images/yjx.jpg")
image = transform(image)
image = torch.unsqueeze(image,dim=0)

model.eval()
with torch.no_grad():
    output = torch.squeeze(model(image))
    predict = torch.softmax(output,dim=0)
    predict_class = torch.argmax(predict).numpy()

# 读取类别文件
json_file = open("./class_indices.json","r")
class_indices = json.load(json_file)

print("predict class：",class_indices[str(predict_class)],"，confidence degree：%.3f"%max(predict).item())


