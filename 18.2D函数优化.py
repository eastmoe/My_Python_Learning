import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

# 函数f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

def himelblan(x):  # 定义函数，这里采用的是list方式输入，共两个元素x[0]和x[1]，分别代表原式里的x和y。
    z = (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
    return z
x = np.arange(-6,6,0.1)  # 定义x的范围
y = np.arange(-6,6,0.1)  # 定义y的范围
X,Y = np.meshgrid(x,y)  # 生成坐标轴
Z = himelblan([X,Y])  # 定义Z

fig = plt.figure('Himelblan Function')  # 设置图形标题为himelblan
ax = fig.add_subplot(projection='3d')  # 设置投影模式为3D
ax.plot_surface(X,Y,Z)  # 绘制3D图形
ax.view_init(60,-30)  # 转换视角
ax.set_xlabel('x')  # 设置x轴的标签为x
ax.set_ylabel('y')  # 设置y轴的标签为y
ax.set_zlabel('f')  # 设置z轴的标签为f
plt.show()  # 绘图


# 这次是对函数的预测值求梯度，即δpred/δx。其中x是张量，包含了原来的x和y。
x_t = torch.tensor([0.,0.]).cuda()  # 设置x的初始值
x_t.requires_grad=True  # 这里是对x进行优化，所以x需要梯度信息
# 注意，必须先执行 .cuda()，再设置梯度信息属性，不然后面在使用optimizer时会出现can't optimize a non-leaf Tensor错误
optimizer = torch.optim.Adam([x_t],lr=1e-3)   # 用优化器自动完成x’=x-0.001*grad(x)，y’=y-0.001*grad(y)，lr是学习速率。因为x是矩阵
print('计算开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))
for step in range(20000):  # 设置默认计算20000次。
    pred = himelblan(x_t)  # 将x、y送入，得到预测值
    optimizer.zero_grad()  # 将梯度数值清零
    pred.backward()  # 生成梯度信息
    optimizer.step()  # 调用optimizer实现x’=x-0.001*grad(x)，y’=y-0.001*grad(y)的过程
    if step % 1000 == 0:  # 1000次
        print('已经计算的次数：',step,'；[x,y]=',x_t.tolist(),'；f(x,y)=',pred.item())  # 每次间隔100次输出一次结果
    if pred.item() == 0.0:  # 当获得0点后
        print('总的计算次数：', step, '；[x,y]=', x_t.tolist(), '；f(x,y)=', pred.item())  # 输出预测值为0的结果
        break  # 跳出循环

print('---------------------------------------------------------------------------------------------------------------')

print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))