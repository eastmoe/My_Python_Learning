import time
import torch
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

# 单层感知机误差的偏导：δE/δw(j0)=(O0-t)*o(10)*(1-o(10))*x(0j)
# 有了公式，可以直接输出梯度。
# 多输入感知机，x会被求和然后送到多个σ，而E则接受了多个σ的输出。
# 对于参数w，存在wjk，其中j是与之相连的输入端，k表示与之相连的输出端。
# 输出E=∑(o1i-ti)^2
# 多输入感知机误差的偏导：δE/δw(jk)=(Ok-tk)*ok*(1-ok)*x(0j)
#
print('---------------------------------------------------------------------------------------------------------------')

x = torch.randn(1,10).cuda()  # 生成输入变量x
print('输入变量x：',x)  # 输出x
w = torch.randn(2,10,requires_grad=True).cuda()  # 生成初始权值w，因为这次有10个输入和10个累加器，所以是2*10
print('初始权值w：',w)  # 输出w
w.retain_grad()  # 设置retain_grad()，保留梯度信息
o = torch.sigmoid(x@w.t())  # 计算x和w相乘后经过sigmoid函数的o
print('输出变量o：',o)  # 输出o
t = torch.ones(1,2).cuda()  # 设置预测值t，其实这里写1也行，不过会自动broadcast成长度为2的。
print('预测值t：',t)  # 输出t
loss = F.mse_loss(t,o)  # 计算误差均方根loss
print('误差值loss：',loss)  # 输出loss
loss.backward()  # 计算梯度，loss依据之前的梯度信息对w求梯度
print('loss对w的梯度：',w.grad)  # 输出梯度

print('---------------------------------------------------------------------------------------------------------------')

print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))