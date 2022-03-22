
import torch
from torch.nn import functional as F



x = torch.arange(44)  # 年份（以1978年为0起）

data = torch.tensor([9.63,9.75,9.87,10.01,10.17,
                     10.3,10.44,10.59,10.75,10.93,
                     11.1,11.27,11.43,11.58,11.72,
                     11.85,11.99,12.11,12.24,12.36,
                     12.48,12.58,12.67,12.76,12.85,
                     12.92,13.0,13.08,13.15,13.21,
                     13.28,13.35,13.41,13.49,13.59,
                     13.67,13.77,13.83,13.92,14.0,
                     14.05,14.1,14.12,14.13])  # 定义人口数据矩阵


a1 = torch.rand(44,requires_grad=True).float()
b1 = torch.rand(44,requires_grad=True).float()
c1 = torch.rand(44,requires_grad=True).float()
a2 = torch.rand(44,requires_grad=True).float()
b2 = torch.rand(44,requires_grad=True).float()
c2 = torch.rand(44,requires_grad=True).float()
a3 = torch.rand(44,requires_grad=True).float()
b3 = torch.rand(44,requires_grad=True).float()
c3 = torch.rand(44,requires_grad=True).float()
a4 = torch.rand(44,requires_grad=True).float()
b4 = torch.rand(44,requires_grad=True).float()
c4 = torch.rand(44,requires_grad=True).float()
a5 = torch.rand(44,requires_grad=True).float()
b5 = torch.rand(44,requires_grad=True).float()
c5 = torch.rand(44,requires_grad=True).float()
a6 = torch.rand(44,requires_grad=True).float()
b6 = torch.rand(44,requires_grad=True).float()
c6 = torch.rand(44,requires_grad=True).float()
# 对参数进行初始化





def forward(x):  # 定义网络函数：一个由二次函数组成的多层网络
    y1 = a1*(x**2)+b1*x+c1
    y1 = F.sigmoid(y1)
    y2 = a2*(y1**2)+b2*y1+c2
    y2 = F.sigmoid(y2)
    y3 = a3*(y2**2)+b3*y2+c3
    y3 = F.sigmoid(y3)
    y4 = a4*(y3**2)+b4*y3+c4
    y4 = F.sigmoid(y4)
    y5 = a5*(y4**2)+b5*y4+c5
    y5 = F.sigmoid(y5)
    y6 = a6*(y5**2)+b6*y5+c6
    return y6

optimizer = torch.optim.SGD([a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6],lr=0.1)  # 优化器
#optimizer = torch.optim.SGD([a1,b1,c1],lr=0.001)  # 优化器

for epoch in range(20):
    for batch in range(50):
        logits = forward(x)  # 计算输入年份x后的输出
        loss = F.mse_loss(logits,data)  # 计算输出与真实值之间的误差
        optimizer.zero_grad()  # 清除原先的梯度信息
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数
    print('第',50*(epoch+1),'次训练：',forward(x),'误差：',loss)


