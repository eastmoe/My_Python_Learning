import torch
import time

print('程序开始时间：',time.strftime('%Y-%m_%d %H:%M-%S'))  # 记录程序开始时间

print('---------------------------------------------------------------------------------------------------------------')

condition_1 = torch.rand(2,2).cuda()  # 生成2*2的二维张量condition_1，元素服从0-1随机分布，作为状态张量
a = torch.ones(2,2).cuda()  # 生成2*2的二维张量a，元素全1
b = torch.zeros(2,2).cuda()  # 生成2*2的二维张量b，元素全0
print('状态张量condition_1：',condition_1)  #输出状态张量
print('张量a：',a)  # 输出张量a
print('张量b：',b)  # 输出张量b
c = torch.where(condition_1>0.5,a,b).cuda()  # 依据状态张量和条件生成张量c，c的形状与a、b、condition_1相同。
# 当状态张量condition_1指定位置的元素满足>0.5的条件时，新张量c对应的位置会填入a对应位置的值；当不满足时，对应位置会填入b对应位置的值。
print('依据状态张量condition_1和原张量a、b生成的张量c：',c)

print('---------------------------------------------------------------------------------------------------------------')
# gather使用案例
prob= torch.randn(4,10).cuda()  # 假设存在输出二维张量4*10
print('假设输出张量prob：',prob)
idx = prob.topk(dim=1,k=3)  # 取第二个维度划分下的前三个最大的值
print('取维度1的前三大数值构成索引：',idx)
idx_1 = idx[1]  # 索引idx第二个维度的内容
print('索引第二维的值：',idx_1)  # 输出idx第二个维度的内容
idx_1 = idx_1.long()  # 为了符合gather里index的要求，需要把idx转换为LongTensor
label = (torch.arange(10)+100).cuda()  # 生成权值表张量
print('生成的权值表label：',label)  # 输出权值表张量
label = label.expand(4,10)  # 扩展权值表张量到二维4*10
print('对表label进行扩展：',label)  # 输出扩展之后的结果
out = torch.gather(label,dim=1,index=idx_1.long())  # 使用索引idx_1，在权值表张量label里查找指定的元素并输出成张量
print('输出索引结果：',out)

print('---------------------------------------------------------------------------------------------------------------')

print('记录程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

