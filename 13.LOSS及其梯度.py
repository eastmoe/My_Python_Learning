import time
import torch
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

# MSE(Mean Square Error)均方差：loss=norm∑((yn-ypn)^2)，yn为y在n处的的值，ypn为y在n处的预测值。
# 使用线性模型的MSE：loss=∑((yn-(xn*w+b))^2)
# 注意MSE（norm）和L2-norm的区别，线性模型的L2-norm=(∑((yn-(xn*w+b))^2))^(1/2)=|∑(yn-(xn*w+b))|，L2-norm会先对误差求差，再平方，再求和，再开平方根。
# 以上的yn、ypn和xw+b都是一个tensor张量。
# MSE（norm）没有L2-norm那样开平方根的过程。
# 求解：torch.norm(y-predict,2).prod(2)。y-predict是误差，2代表L2-norm，因为L2-norm开了平方根，所以后面要加prod(2)来再开个平方以达到MSE。
# 对MSE求导数：dloss/dw=2∑((yn-fw(x))*(df/dw))。


print('---------------------------------------------------------------------------------------------------------------')

# 设y=x*w+b，用pytorch求梯度。
x = torch.ones(1).cuda()  # x初始化为1
print('输入值x初始值：',x)  # 输出输入值x
w = torch.full([1],2).float().cuda()  # w初始化为长度为1，值为2的一维张量。b设为0。
# 转换为float类型是因为只有浮点型张量才能要求标注梯度信息。
w.requires_grad_()  # 标注w需要梯度信息。如果需要目标函数对w求梯度（求导），那么就需要标注。
# 也可以在创建时使用requires_grad=True来标记信息，例如：w = torch.tensor([2.],requires_grad=True)。
print('要估计的w初始值：',w)  # 输出计算值w
predict = torch.ones(1).cuda()  # 预测值predict
print('预测值predict2：',predict)  # 输出预测值predict
mse = F.mse_loss(predict,w*x)  # 计算均方根loss=norm∑((yn-ypn)^2)
print('loss（MSE）的结果：',mse)  # 输出MSE均方根的梯度计算结果。
grad = torch.autograd.grad(mse,[w])  # 函数mse对变量w求梯度。如果要对多个变量求梯度，可以写为[w1,w2,w3......]
print('输出loss对w的梯度：',grad)

print('---------------------------------------------------------------------------------------------------------------')
# 不能对同一个图求两次梯度，所以这里重新定义数据。
x2 = torch.ones(1).cuda()  # x初始化为1
print('同上，输入值x2初始值：',x)  # 输出输入值x2
w2 = torch.full([1],2).float().cuda()  # w2初始化为长度为1，值为2的一维张量。b2设为0。
w2.requires_grad_()  # 标注w2需要梯度信息。如果需要目标函数对w2求梯度（求导），那么就需要标注。
print('同上，要估计的w2初始值：',w2)  # 输出计算值w2
predict2 = torch.ones(1).cuda()  # 预测值predict
print('同上，预测值predict：',predict2)  # 输出预测值predict
mse2 = F.mse_loss(predict2,w2*x2)
print('同上，loss（MSE）的结果：',mse2)  # 输出MSE均方根的梯度计算结果。
mse2.backward()  # 会自动按照图从后往前传播，按需要完成指定变量的梯度计算。计算结果会自动附加在指定变量的.graid属性上，不会再组成list
print('再次输出loss函数（mse）对w的梯度：',w2.grad)
# 注意：w.norm会返回w本身的norm（MSE）均方根值。w.graid.norm是查看w的梯度的均方根。
print('---------------------------------------------------------------------------------------------------------------')

# Softmax激活函数，用于多输入多输出的将不同大小的值统一输入并按照大小转化为概率（值为0~1，和为1）。
# Softmax：pi=e^yi/(∑(e^yi)).
# 每一个输出pi对应每一个输入yi和所有的输入。
# 一般用于将概率最大的值转化为label
# 当输出i与求导变量的j对应时：di/dyi=pi(1-pi)
# 当输出i与求导变量的j不对应时：di/dyj=(-pj)*pi
# 即i=j时偏导数为正，否则为负。

print('---------------------------------------------------------------------------------------------------------------')


a = torch.rand(3).cuda()  # 随机生成一维，长度为3的张量。
a.requires_grad_()  # 给a加上梯度标记
print('张量a：',a)  # 输出a
probability = F.softmax(a,dim=0)  # 对张量a在第一维（0维）上做Softmax变换，结果对应生成张量probability。
print('张量a每个元素对应的概率：',probability)  # 输出probability
#probability.backward(retain_graph=True)  # 保留图意味着这次计算后图的信息不会被清除，可以在接下来再求梯度。
grad_1 = torch.autograd.grad(probability[1],[a],retain_graph=True)  # 同样保留图。
# 注意，无论使用autograd还是backword，最终的loss必须是一维且长度为一的张量。或者是维度为0的标量。
print('对概率probablity的第二个变量求梯度：',grad_1)  # 输出probability第二个元素梯度
grad_2 = torch.autograd.grad(probability[2],[a],retain_graph=True)  # 对概率probablity的第三个变量求梯度：
print('对概率probablity的第三个变量求梯度：',grad_2)  # 输出probability第三个元素梯度
grad_0 = torch.autograd.grad(probability[0],[a])  # 对概率probablity的第一个变量求梯度：
print('对概率probablity的第一个变量求梯度：',grad_0)  # 输出probability第一个元素梯度

print('---------------------------------------------------------------------------------------------------------------')

print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))