"""
@Author  :shw
@Date    :2024/1/12 15:07
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from LeNet.model import LeNet

batch_size = 64
lr = 0.001
epochs = 20

transformer = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])

train_data = datasets.CIFAR10(root='./data', train=True,transform=transformer,download=True)
test_data = datasets.CIFAR10(root='./data', train=False,transform=transformer,download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
train_size = len(train_data)
test_size = len(test_data)
print("训练集大小为：{}".format(train_size))
print("测试集大小为：{}".format(test_size))


# myloader = DataLoader(test_data, batch_size=5, shuffle=True, num_workers=0)
# my_iter = iter(myloader)
# images, labels = next(my_iter)
# def imgshow(img):
#     img = img/2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()
#
# imgshow(torchvision.utils.make_grid(images))



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
writer = {
    "train": SummaryWriter(log_dir="./logs/train"),
    "test": SummaryWriter(log_dir="./logs/test")
}

train_step = 0

for epoch in range(epochs):
    print("===========第 {} Epoch训练开始===========".format(epoch+1))
    model.train()
    train_loss = 0
    train_accuracy_num = 0
    for data in train_loader:
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
        if train_step % 100 ==0:
            print("Train Step : {} Loss : {}".format(train_step, loss.item()))
            writer['train'].add_scalar("Train Step Loss", loss.item(), train_step)
    print("Epoch: {} Train Loss : {}".format(epoch+1, train_loss))
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

writer['train'].close()
writer['test'].close()



