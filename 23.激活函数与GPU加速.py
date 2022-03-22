import time
import torch
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

print('---------------------------------------------------------------------------------------------------------------')

# Sigmoid函数，将数据压缩到0~1，缺点是有可能会造成梯度离散。
# tanh，是sigmoid函数的变形，输出范围是-1~1，一般用于RNN（循环神经网络）。
# relu函数，输入小于0不响应（输出为0），大于等于0时直接输出x（f(x)=x）。relu函数在一定程度上解决了sigmoid函数的梯度离散的现象。
# leaky relu函数，当x大于等于0时，输出x；当x小于0时，f(x)=αx，α的值非常小。提出leaky relu是为了解决relu在部分情况下出现梯度离散。

print('---------------------------------------------------------------------------------------------------------------')

class MLP(torch.nn.Module):  # 新建类MLP，从nn.Module继承。
    def __init__(self):  # 定义初始化函数，可以传入参数以手工指定维度
        super(MLP,self).__init__()
        self.model = torch.nn.Sequential(  # 这个nn.Sequential类似于一个容器，可以添加任何继承自nn.Module的
            torch.nn.Linear(784,200),  # 转置矩阵，并设置线性运算
            torch.nn.LeakyReLU(inplace=True),  # 直接使用LeakyReLU替换ReLU即可，还可以设置α的值，默认是0.02。
            torch.nn.Linear(200, 200),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(200, 10),
            torch.nn.LeakyReLU(inplace=True),
        )
        def forward(self,x):
            x = self.model(x)  # 直接使用module.forward函数
            return x

print('---------------------------------------------------------------------------------------------------------------')

# SELU函数：SELU(x)=scale*RELU(x)+min(0,α*(exp(x)-1))，目的是解决relu函数在0点的连续问题。
# 一般情况下，没有什么地方必须使用SELU，不过有部分场景会用到leaky relu
# softplus函数：softplus(x)=(1/β)*log(1+exp(β*x))，也是对relu函数在x=0处经行了平滑的处理。一般不需要关注。

print('---------------------------------------------------------------------------------------------------------------')

# pytorch可以使用.cuda()和.cpu()方法来设置变量和函数运行在gpu上还是cpu上。
# 也可以定义设备，然后在需要运算的地方写上.to方法。

device = torch.device('cuda:0')  # 将device定义为支持cuda的0号设备
net = MLP().to(device)  # 利用.to来指定使用device设备计算
optimizer = torch.optim.SGD(net.parameters(),lr=Learning_Rate)
criteon = torch.nn.CrossEntropyLoss().to(device)  # 利用.to来指定使用device设备计算
for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)
        data = data.to(device)  # 利用.to来指定使用device设备计算
        target =target.cuda()  # 使用传统方式指定计算设备为支持cuda的GPU
# 注意，对类使用.to或.cuda，会将它们搬运到GPU，但是修改依旧同步。
# 对张量使用.to或者.cuda，返回的张量与原来的意义不同，修改不会同步，所以需要覆盖或者建立新变量。



print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))
