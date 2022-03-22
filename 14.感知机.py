import time
import torch
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))
# 单层感知机：y=∑(xi*wi)+b
# 目前使用σ（Sigmoid）函数来表示
# 从与权值相乘，再到累加，再到σ(Sigmoid)函数输出，是一层。
# 输入x的下标表示是第几个节点输入，右上方的数字代表第几层。w的第一个下标表示输入变量，第二个下标表示下一步送进去的节点，上标表示层数。
# o表示经过σ函数的输出，下标表示之前经过了几号节点，上标表示层数。
# 求E(loss)对w的梯度：设E=((O(01)-t)^2)/2，O(01)表示第1层网络第0个节点的输出，t表示预测值。
# δE/δw(j0)=(o0-t)*o0*(1-o0)*x0j，即单层单节点的神经网络仅仅与节点的输出o10和j位置的输入x0j有关。
print('---------------------------------------------------------------------------------------------------------------')

x = torch.randn(1,10).cuda()  # 设输入为二维1*10的张量
print('输入变量x：',x)  # 输出x
w = torch.randn(1,10,requires_grad=True).cuda()  # 生成权值w
w.retain_grad()  # 设置retain_grad()，保留梯度信息
print('参数w：',w)  # 输出w
o = torch.sigmoid(x@w.t())  # 先将w转置，然后x和w做矩阵乘法，之后再通过sigmoid函数，成为输出量o
print('输出o的形状：',o.shape)  # 输出o的形状
print('输出o：',o)  # 输出o
p = torch.ones(1,1).cuda()  # 设置预测值p
print('预测值p:',p)  # 输出p
loss = F.mse_loss(p,o)  # 计算均方差：loss=norm∑((yn-ypn)^2)，yn为y在n处的的值，ypn为y在n处的预测值。
print('均方差loss：',loss)  # 输出loss(loss为长度为一的标量)
loss.backward()  # 计算loss对w的梯度
print('loss对于w的梯度：',w.grad)  # 输出梯度

print('---------------------------------------------------------------------------------------------------------------')

print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))