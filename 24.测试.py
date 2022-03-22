import time
import torch
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))


# 一般情况下，需要在训练时做测试。
# 如果仅仅只用少数的数据集去训练，有可能会导致其只记住表面的东西，泛化能力较弱。
# 数据量和架构是核心问题。
print('---------------------------------------------------------------------------------------------------------------')

logits = torch.rand(4,10)  # 随机生成输出logits
pred = F.softmax(logits,dim=1)  # 输出经过softmax函数形成预测值
print('预测值的形状：',pred.shape)
pred_label1 = pred.argmax(dim=1)  # 取预测值一维最大值所在位置
print('预测值第二个维度的最大值：',pred_label1)
logits_label1 = logits.argmax(dim=1)  # 取原输出量一维最大值所在位置
print('原输出值第一个维度的最大值：',logits_label1)
# 对softmax处理之后作argmax和对softmax处理之前的logits一样，因为softmax不会改变单调性
label = torch.tensor([9,3,2,4])  # 生成标签张量
correct = torch.eq(pred_label1,label)  # 判断预测准确的数值的位置
print('准确度张量：',correct)
print('准确度：',correct.sum().float().item()/4)  # 判断预测准确的概率
# 一般情况下，不建议每隔一个batch测试一次。
# 可以选择每隔一个epoch（整个数据集）测试一遍。

print('---------------------------------------------------------------------------------------------------------------')
# ..............以上省略了训练的部分................
test_loss =0
correct = 0  # 初始化标量，用来记录数据
for data,target in test_loader:  # 从test——loader提取出数据集
    data = data.view(-1,28*28)  # 将测试数据进行变换
    data,target = data.to(device),target.cuda()  # 搬运到GPU
    logits = net(data)  # 送入网络
    test_loss += criteon(logits,target)  # 一般情况下，可以不计算test loss（测试数据集的误差）
    pred = logits.argmax(dim=1)  # 计算预测值
    correct += pred.eq(target).float().sum().item()  # 计算预测对的数的个数
test_loss /= len(test_loader.dataset)
print('\n测试数据集平均误差：',test_loss,'\n准确度：',correct,'\n',len(test_loader.dataset),'\n')




























print('---------------------------------------------------------------------------------------------------------------')
print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))