import torch
from torch  import nn
from torch.nn import functional as F
from torch import optim
import torchvision



from utils import plot_image, plot_curve, one_hot
import time

print('起始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))# 输出开始时间

############后期修改##############
device = torch.device('cuda:0')  # 将device定义为支持cuda的0号设备
############后期修改##############

# 以下为第一步，加载数据集

batch_size = 512

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)#不打散

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)#不打散



x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'Image Sample')



#第二步，创建神经网络

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #三层
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x:[b, 1, 28, 28]
        # h1=relu(x*w1+b1)
        x = F.relu(self.fc1(x))
        # h2=relu(h1*w2+b2)
        x = F.relu(self.fc2(x))
        # h3=h2*w3+b3
        x = self.fc3(x)
        return x
############后期修改##############
net = Net().to(device)
#net = Net()
############后期修改##############


# 第三步，优化参数（训练）

#根据下面的差距来进行优化w1、b1、w2、b2、w3、b3
#optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum=0.9)


#记录loss
train_loss = []
print('请输入遍历数据集的次数：')
train_times = input()

train_times = int(train_times)

print('训练开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

for epoch in range(train_times):# 遍历训练循环
    for batch_idx, (x, y) in enumerate(train_loader):# 对一个batch迭代一次

        # 得到x:[b, 1, 28, 28], y:[512],因为网络只能接受二维，所以需要把[， 1， 28， 28]打平为[b, 784]
        x = x.view(x.size(0), 28*28)
        ############后期修改##############
        x = x.to(device)
        ############后期修改##############
        # 经过三层网络，变成[b, 10]
        out = net(x)
        # [b, 10]让output接近one_hot
        y_onehot = one_hot(y)
        ############后期修改##############
        y_onehot = y_onehot.to(device)
        ############后期修改##############
        # 计算差距loss为out和y_onehot之间的误差：loss=mse(out,y_onehot)
        loss = F.mse_loss(out, y_onehot)

        # 对之前的梯度清零
        optimizer.zero_grad()
        # 完成计算梯度：w·=w-lr*grad
        loss.backward()
        # 完成更新梯度
        optimizer.step()

        #记录loss
        train_loss.append(loss.item())

        #每隔10个batch打印一次loss
        if batch_idx % 10 ==0:
            print(epoch, batch_idx, loss.item())

#完成对数据集的多次迭代之后，就可以得到相对合理的参数（w1、b1、w2、b2、w3、b3）

print('训练结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

#打印训练loss曲线
plot_curve(train_loss)


# 第四步，准确度测试


# 定义总的正确数为0
total_correct = 0

for x,y in test_loader:
    # 打平
    x = x.view(x.size(0), 28*28)
    ############后期修改##############
    x = x.to(device)
    y = y.to(device)
    ############后期修改##############

    # out的输出结果是[b, 10]
    out = net(x)
    # 获得预测值，即1维中数字最大的值
    pred = out.argmax(dim = 1)
    ############后期修改##############
    pred = pred.to(device)
    ############后期修改##############
    # 计算正确值，与y作比较，判断是否对等。correct为当前batch里正确的数量
    correct = pred.eq(y).sum().float().item()

    correct = correct

    total_correct += correct

#计算总的数字个数
total_num = len(test_loader.dataset)
#计算准确度
acc = total_correct/total_num

print('手写数字识别测试模型的准确率是：', acc)

# 查看每一个数字的预测结果

# 取一个batch
x, y =next(iter(test_loader))
############后期修改##############
x = x.to(device)
y = y.to(device)
############后期修改##############
out = net(x.view(x.size(0),28*28))
# 把输出值作为预测值
pred = out.argmax(dim = 1)


############后期修改##############
pred = pred.cpu()
x = x.cpu()
############后期修改##############
plot_image(x, pred, 'Every Number Result')


print('结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))# 输出结束时间