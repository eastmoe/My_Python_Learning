import torch
import time
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

print('---------------------------------------------------------------------------------------------------------------')
# 随机生成3个200*784的正态分布的二维张量用作w，3个长度为200的一维张量用作b
# 这里使用正态分布对w进行初始化。
w1 = torch.randn(200,784,requires_grad=True)  # 注意，第一个维度是out，第二个维度是in
b1 = torch.zeros(200,requires_grad=True)
w2 = torch.randn(200,200,requires_grad=True)
b2 = torch.zeros(200,requires_grad=True)
w3 = torch.randn(10,200,requires_grad=True)  # 是10分类，所以最后的输出有10个
b3 = torch.zeros(10,requires_grad=True)

print('---------------------------------------------------------------------------------------------------------------')
# 利用何恺明博士的初始化函数对w进行初始化。
torch.nn.init.kaiming_normal(w1)
torch.nn.init.kaiming_normal(w2)
torch.nn.init.kaiming_normal(w3)

print('---------------------------------------------------------------------------------------------------------------')
# 定义函数，里面是计算过程。
# 直接计算输出，没有经过激活函数的量被称为logistics
# Relu激活函数：输入的x>=0时，f(x)=x；其它情况，f(x)=0。
def forward(x):
    x = x@w1.t()+b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return  x

print('---------------------------------------------------------------------------------------------------------------')

optimizer = torch.optim.SGD([w1,b1,w2,b2,w3,b3],lr=0.001)  # 设置优化器，优化目标是三组w、b；默认情况下可以将学习sulvlearningrate设置的少一点。
criteon = torch.nn.CrossEntropyLoss()  # 类似crossentropy，包含了softmax+log+nnloss的操作。这里相当于给函数起了一个别名
# 不需要再加softmax，因为那样会放大差距

for epoch in range(epochs):  # epoch指整个数据集，跑完一个epoch就是跑完了整个数据集。
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)  # 对数据张量转置
        logits = forward(data)  # 对数据集直接调用之前的forward函数运算，运算出来的未经过处理的结果就是logits
        loss = criteon(logits,target)  # 调用之前定义的crossentropy的函数别名
        optimizer.zero_grad()  # 将梯度信息清零
        loss.backward()  # 生成梯度信息
        optimizer.step()  # 利用梯度信息用优化器修改权值w、b


print('---------------------------------------------------------------------------------------------------------------')



print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))
