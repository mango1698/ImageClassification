"""
@Author  :shw
@Date    :2024/1/16 11:24
"""
import json
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import ResNet34_Weights

from model import resnet34

batch_size = 64
lr = 0.0001
epochs = 3

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
}

# os.getcwd()：获取当前文件所在目录
data_root = os.path.join(os.getcwd(), '..')
image_path = data_root + "/dataset/flower_data"
train_data = datasets.ImageFolder(root=image_path+"/train", transform=data_transform['train'])
train_size = len(train_data)

# ------------------处理类别索引信息------------------
# 获取 分类对名称对应对索引
flower_list = train_data.class_to_idx  # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
cla_dict = dict((val,key) for key,val in flower_list.items())  # {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
# 将类别及索引转为json格式别保持至文件中  方便我们在预测时，读取其对应对类别信息
json_str = json.dumps(cla_dict,indent=4)
with open('class_indices.json','w+') as json_file:
    json_file.write(json_str)
train_loader = DataLoader(train_data,batch_size,shuffle=True,num_workers=0)
# --------------------------------------------------

test_data = datasets.ImageFolder(root=image_path+"/val", transform=data_transform['val'])
test_size = len(test_data)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=0)

print("训练集大小为：{}".format(train_size))
print("测试集大小为：{}".format(test_size))

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
model = resnet34().to(device)
model_weight_path = './weights/resnet34-pre.pth'
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path,map_location="cpu"),strict=False)
in_channel = model.fc.in_features
model.fc = nn.Linear(in_channel, 5)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
writer = {
    "train": SummaryWriter(log_dir="./logs/train"),
    "test": SummaryWriter(log_dir="./logs/test")
}

train_step = 0
t1 = time.perf_counter()
for epoch in range(epochs):
    print("===========第 {} Epoch训练开始===========".format(epoch+1))
    model.train()
    train_loss = 0
    train_accuracy_num = 0
    for step,data in enumerate(train_loader,start=0):
        train_step += 1
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_accuracy_num += torch.sum(labels==torch.argmax(outputs,dim=1))

        # ----------训练进度----------
        rate = (step+1)/len(train_loader)
        a = "*"*int(rate*50)
        b = "."*int((1-rate)*50)
        # ^ 表示居中对其
        print("\rtran loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss),end="")
        # ---------------------------

    print("\nEpoch: {} Train Loss : {}".format(epoch+1, train_loss))
    print("Epoch: {} Train Accuracy : {}".format(epoch+1, train_accuracy_num/train_size))
    writer['train'].add_scalar("Loss", train_loss,epoch+1)
    writer['train'].add_scalar("Accuracy", train_accuracy_num/train_size,epoch+1)

    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_accuracy_num = 0
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            test_accuracy_num += torch.sum(labels == torch.argmax(outputs, dim=1))
        print("Epoch: {} Test Loss : {}".format(epoch + 1, test_loss))
        print("Epoch: {} Test Accuracy : {}".format(epoch + 1, test_accuracy_num / test_size))
        writer['test'].add_scalar("Loss", test_loss, epoch + 1)
        writer['test'].add_scalar("Accuracy", test_accuracy_num / test_size, epoch + 1)

        torch.save(model, "./model/net_{}.pth".format(epoch + 1))
print("Total Time：{}".format(time.perf_counter()-t1))
writer['train'].close()
writer['test'].close()
