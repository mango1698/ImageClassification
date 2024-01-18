"""
@Author  :shw
@Date    :2024/1/18 15:30
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from AlexNet.model import AlexNet

# 实例化模型
model = AlexNet(num_classes=5)
weights = torch.load("./alexnet_weight_20.pth", map_location="cpu")
model.load_state_dict(weights)

weights_keys = model.state_dict().keys()
for key in weights_keys:
    if "num_batches_tracked" in key:
        continue
    weight_t = model.state_dict()[key].numpy()
    weight_mean = weight_t.mean()
    weight_std = weight_t.std(ddof=1)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean, weight_std, weight_min, weight_max))

    weight_vec = np.reshape(weight_t,[-1])
    plt.hist(weight_vec,bins=50)
    plt.title(key)
    plt.show()