import torch
import time
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

print('---------------------------------------------------------------------------------------------------------------')

x = torch.rand(1,784)  # 假设输入图片1*28*28
print('输入变量x的形状：',x.shape)
layer1 = torch.nn.Linear(784,200)  # 直接设置为y=xw+b,其中第一个为输入，第二个为输出
layer2 = torch.nn.Linear(200,200)
layer3 = torch.nn.Linear(200,10)

x = layer1(x)
print('经过第一层后的变量x的形状：',x.shape)
x = layer2(x)
print('经过第二层后的变量x的形状：',x.shape)
x = layer3(x)
print('经过第三层后的变量x：',x.shape)

print('---------------------------------------------------------------------------------------------------------------')

# 当加上激活函数后：
x = torch.rand(1,784)  # 假设输入图片1*28*28
print('输入变量x的形状：',x.shape)
layer1 = torch.nn.Linear(784,200)  # 直接设置为y=xw+b,其中第一个为输入，第二个为输出
layer2 = torch.nn.Linear(200,200)
layer3 = torch.nn.Linear(200,10)

x = layer1(x)
x = F.relu(x,inplace=True)  # 添加inplace参数，可以减少一半的内存消耗
print('经过第一层和Relu函数后的变量x的形状：',x.shape)
x = layer2(x)
x = F.relu(x,inplace=True)
print('经过第二层和Relu函数后的变量x的形状：',x.shape)
x = layer3(x)
x = F.relu(x,inplace=True)
print('经过第三层和Relu函数后的变量x：',x.shape)

print('---------------------------------------------------------------------------------------------------------------')

# 可以从nn.Module继承类。
# 需要写好初始化函数，注明从多少维降低到多少维。
# 把三层或者更多层封装在一起。
# 需要在自己的类中实现forward，不需要实现backword，backword会提供。
print('---------------------------------------------------------------------------------------------------------------')
# 用类来组合多层网络
class MLP(torch.nn.Module):  # 新建类MLP，从nn.Module继承。
    def __init__(self):  # 定义初始化函数，可以传入参数以手工指定维度
        super(MLP,self).__init__()
        self.model = torch.nn.Sequential(  # 这个nn.Sequential类似于一个容器，可以添加任何继承自nn.Module的
            torch.nn.Linear(784,200),  # 转置矩阵，并设置线性运算
            torch.nn.ReLU(inplace=True),  # 使用类中的ReLU方法，不用传输入张量，但是只能在类中使用。
            # 关于nn.ReLU和F.relu():
            # nn是基于类的，你必须先实例化，然后再操作，并且不能访问其中的w和b等参数。只能使用方法来访问。
            # F的relu则可以很方便的管理它们内部的元素。
            # 一般情况下，推荐使用nn.ReLU，除非要进行底层操作或者不想写类。
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 10),
            torch.nn.ReLU(inplace=True),
        )
        def forward(self,x):
            x = self.model(x)  # 直接使用module.forward函数
            return x



print('---------------------------------------------------------------------------------------------------------------')

net = MLP()
optimizer = torch.optim.SGD(net.parameters(),lr=Learning_Rate)  # 优化器，使用继承自nn.Module的方法可以一次性返回所有张量，无需手工写入。
criteon = torch.nn.CrossEntropyLoss()  # cross-entropy交叉熵的计算，包含了softmax+log+nnloss的操作。

for epoch in range(epochs):  # 遍历整个数据集
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)  # 对数据张量转置
        logits = net(data)  # 直接使用之前自己写好并且初始化的类net来运算data数据集，输出的结果维logits
        loss = criteon(logits,target)  # 利用之前定义的cross-entropy交叉熵的别名criteon计算误差
        optimizer.zero_grad()  # 清除当前梯度的值
        optimizer.zero_grad()  # 清除当前梯度的值
        loss.backword()  # 计算梯度信息
        optimizer.step()  # 优化器传递梯度并更新参数（权值）


print('---------------------------------------------------------------------------------------------------------------')





print('---------------------------------------------------------------------------------------------------------------')



print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

