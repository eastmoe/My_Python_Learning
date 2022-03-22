import time
import torch
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

# 链式法则的基本原理：dy/dx=(dy/du)*(du/dx)
# 设一个两层的感知机，有δE/δw1jk=(δE/δo2k)*(δo3k/δo1k)*(δo1k/δwijk)
print('---------------------------------------------------------------------------------------------------------------')

x = torch.tensor(1.).cuda()  # 生成输入变量x
w1 = torch.tensor(2.,requires_grad=True).cuda()  # 生成初始权值w1
w1.retain_grad()  # 设置retain_grad()，保留梯度信息
b1 = torch.tensor(1.).cuda()  # 生成常量b1
w2 = torch.tensor(2.,requires_grad=True).cuda()  # 生成初始权值w2
w2.retain_grad()  # 设置retain_grad()，保留梯度信息
b2 = torch.tensor(1.).cuda()  # 生成常量b2
w1.retain_grad()  # 设置retain_grad()，保留梯度信息
w2.retain_grad()  # 设置retain_grad()，保留梯度信息
y1 = x*w1+b1  # 构建第一层函数
y2 = y1*w2+b2  # 构建第二层函数
print('输入变量x：',x,'；权值w1：',w1,'；权值w2：',w2,'；常量b1：',b1,'；常量b2：',b2,'；')  # 输出参数与变量值
y2_y1 = torch.autograd.grad(y2,[y1],retain_graph=True)[0]  # 计算dy2/dy1
y1_w1 = torch.autograd.grad(y1,[w1],retain_graph=True)[0]  # 计算dy1/dw1
print('链式法则计算的梯度结果：dy2/dw1=(dy2/dy1)*(dy1/dw1)=',y2_y1*y1_w1)  # 输出结果
y2_w1 = torch.autograd.grad(y2,[w1],retain_graph=True)[0]  # 计算dy2/dw1，为了验证链式法则的准确性。
print('直接计算的梯度结果：dy2/dw1=',y2_w1)  # 输出结果



print('---------------------------------------------------------------------------------------------------------------')

print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))
