import time
import torch
from torch.nn import functional as F

# 本节的内容是为了检测欠拟合和过拟合，并且给出纠正方法。
# 核心思想，将数据集dataset划分为Trainset（训练集）和Testset（测试集）

train_loader = torch.utils.data.DataLoader(
    dataset.MNIST('../data',train=True,download=True,
        transform=transforms.Compose([
            transforms.Totensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
    ),
batch_size = batch_size,shuffle=True)
