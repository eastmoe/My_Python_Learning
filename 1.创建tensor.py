import numpy as np  # 以np导入模块numpy
import torch  # 导入深度学习模块torch
import time  # 导入时间模块time


print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S')) #记录程序开始时间

print('---------------------------------------------------------------------------------------------------------------')

a = np.array([2,4.5])  # 用numpy创建维度为1，长度为2的向量
print('numpy生成的向量a：',a)  # 输出numpy生成的a
a_tensor = torch.from_numpy(a)  # 将numpy生成的向量导入为张量a_tensor（从numpy导入的float为double型）
a_tensor = a_tensor.cuda()  # 将a_tensor搬运到GPU
print('张量a：',a_tensor)  # 输出GPU上的张量a

print('---------------------------------------------------------------------------------------------------------------')

b = np.ones([2,4])  # 用numpy创建维度为2*4的矩阵b
print('numpy生成的矩阵b：',b)  # 输出numpy生成的b
b_tensor = torch.from_numpy(b)  # 将numpy生成的矩阵导入为张量b_tensor（从numpy导入的float为double型）
b_tensor = b_tensor.cuda()  # 将b_tensor搬运到GPU
print('张量b：',b_tensor)  # 输出GPU上的张量b

print('---------------------------------------------------------------------------------------------------------------')
# 注意，小写的tensor()接受现有的数据，大写的Tensor()和FloatTensor()一样，接受的是shape，即维度形状，会生成没有初始化的矩阵张量。
c_tensor = torch.tensor([23,6.5])  # 直接利用torch生成1维，长度为2的张量（向量）并给定数据
c_tensor = c_tensor.cuda()  # 把c搬运到GPU
print('张量c：',c_tensor)  # 输出张量c

print('---------------------------------------------------------------------------------------------------------------')

d_tensor = torch.Tensor(3,5)  # 直接生成3*5的矩阵张量d
d_tensor = d_tensor.cuda()  # 搬运d到GPU
print('张量d：',d_tensor)  # 输出张量d

print('---------------------------------------------------------------------------------------------------------------')

e_tensor = torch.tensor([[23,6.5],[4.8,7.1]])  # 直接利用torch生成2*2的矩阵张量e并给定数据
e_tensor = e_tensor.cuda()  # 把e搬运到GPU
print('张量e：',e_tensor)  # 输出张量e

print('---------------------------------------------------------------------------------------------------------------')

f_tensor = torch.empty(6,7,2)  # 申请6*7*2的三维张量内存空间f，但是不初始化
print('未被搬运的未初始化的张量f：',f_tensor)  # 输出未初始化未被搬运的f
f_tensor = f_tensor.cuda()  # 把f搬运到GPU
print('未初始化的张量f：',f_tensor)  # 输出未初始化的f

print('---------------------------------------------------------------------------------------------------------------')
# 生成整形张量时注意torch.IntTensor的大小写
g_tensor = torch.IntTensor(2,4,4)  # 生成2*4*4的三维整形张量g
g_tensor = g_tensor.cuda()  # 把g搬运到GPU
print('整形矩阵张量g：',g_tensor)  # 输出g


print('---------------------------------------------------------------------------------------------------------------')
# 做增强学习，一般都用DoubleTensor类型的数据，因为精度更高。
h_tensor = torch.Tensor(5,5)  # 创建5*5的随机值二维张量h
h_tensor = h_tensor.cuda()  # 把h搬运到GPU
print('张量h：',h_tensor)  # 输出张量h的值
print('张量h的类型:',h_tensor.type())  # 输出张量h的类型
h_tensor = h_tensor.double() # 更改张量和的数值类型为DoubleTensor
print('更改类型后张量h的类型:',h_tensor.type())  # 输出张量h的类型
# 可以直接设置torch的创建的默认tensor数据类型：torch.set_default_tensor_type(torch.DoubleTensor)
# 这样新生成的tensor就默认是DoubleTensor
torch.set_default_tensor_type(torch.DoubleTensor)  # 修改默认数值类型为DoubleTensor
newh_tensor = torch.Tensor(2,2)  # 新建newh测试张量
print('新张量newh的类型',newh_tensor.type())  # 输出新张量newh的数值类型

print('---------------------------------------------------------------------------------------------------------------')

i_tensor = torch.rand(4,4)  # 使用torch.rand（shape），会生成形状维度为shape，数值在0-1之间的张量
i_tensor = i_tensor.cuda()  # 搬运i到GPU
print('使用0-1随机初始化的张量i：',i_tensor)  # 输出张量i
ilike_tensor = torch.rand_like(i_tensor)  # 使用torch.rand_like(a)来生成一个和a形状shape相同，不过内部使用0-1填充的新张量

ilike_tensor = ilike_tensor.cuda()  # 把ilike搬运到GPU
print('仿照i的形状生成的新张量：',ilike_tensor)  # 输出新的张量

print('---------------------------------------------------------------------------------------------------------------')

j_tensor = torch.randint(0,10,[3,5])  # 随机生成一个整数数值在1-10之间，形状为3*5的张量
j_tensor = j_tensor.cuda()  # 将j搬运到gpu
print('随机整数矩阵张量：',j_tensor)  # 输出j

print('---------------------------------------------------------------------------------------------------------------')

k_tensor = torch.randn(3,3)  # 生成一个形状为3*3，数值服从正态分布N(0,1)的张量
k_tensor = k_tensor.cuda()  # 搬运k到gpu
print('正态分布随机张量k：',k_tensor)  # 输出k

rek_tensor = torch.normal(10, 0.5 ,size=(5, 5))  # 生成数值均值为10，方差为0.5，形状为5*5的二维张量
rek_tensor = rek_tensor.cuda()  # 搬运新张量到GPU
print('服从N(10,0.5)的张量：',rek_tensor)  # 输出新的张量

print('---------------------------------------------------------------------------------------------------------------')

l_tensor = torch.full([3,4],12)  # 用整数12填充的方式新建一个3*4的张量
l_tensor = l_tensor.cuda()  # 搬运l到gpu
print('填充型张量l：',l_tensor)  # 输出l的值
l1_tensor = torch.full([],12)  # 生成维度为0的标量12
l1_tensor = l1_tensor.cuda()  # 搬运l1到gpu
print('填充型标量l1：',l1_tensor)  # 输出l1的值
l2_tensor = torch.full([1],12)  # 生成长度和维度都为1的张量12
l2_tensor = l2_tensor.cuda()  # 搬运l2到gpu
print('一维填充型张量l2：',l2_tensor)  # 输出l2的值

print('---------------------------------------------------------------------------------------------------------------')

m_tensor = torch.arange(1,10)  # 从1到10的整数的等差数列张量[1,10)，包括起始的数字，不包括结尾的数字。
m_tensor = m_tensor.cuda()  # 搬运m到GPU
print('等差数列张量m：',m_tensor)  # 输出m
m1_tensor = torch.arange(0,20,2)  # 从0到20的整数的等差数列张量[1,10)，以2为递增的幅度。
m1_tensor = m1_tensor.cuda()  # 搬运m1到GPU
print('2为单位的等差数列张量m1：',m1_tensor)  # 输出m1

print('---------------------------------------------------------------------------------------------------------------')

n_tensor = torch.linspace(0,100,steps=6)  # 计算从0-100的5等分数值并写入张量
n_tensor = n_tensor.cuda()  # 搬运n到GPU
print('等分数值的张量：',n_tensor)
n1_tensor = torch.logspace(0,10,steps=11,base=2)  # 从0到10进行10等分，然后把生成的11个数分别作为2的幂，最后把运算后的数组成张量。
n1_tensor = n1_tensor.cuda()  # 把n1搬运到gpu
print('基于幂的等差数列',n1_tensor)  #输出n1

print('---------------------------------------------------------------------------------------------------------------')

o_tensor = torch.ones(3,5)  # 生成形状为3*5，全部数值为1的二维张量
o_tensor = o_tensor.cuda()  # 搬运到GPU
print('全1张量：',o_tensor)  # 输出张量o
o1_tensor = torch.zeros(3,5)  # 生成形状为3*5，全部数值为1的二维张量
o1_tensor = o1_tensor.cuda()  # 搬运到GPU
print('全0张量：',o1_tensor)  # 输出张量o1
o2_tensor = torch.eye(5,5)  # 生成5阶的单位矩阵张量
o2_tensor = o2_tensor.cuda()  # 搬运到GPU
print('单位矩阵张量：',o2_tensor)  # 输出张量o2
o3_tensor = torch.eye(3)  # 生成3阶的单位矩阵张量
o3_tensor = o3_tensor.cuda()  # 搬运到GPU
print('直接生成的单位矩阵张量：',o3_tensor)  # 输出张量o3


print('---------------------------------------------------------------------------------------------------------------')

p_tensor = torch.randperm(10)  # 生成0-10的整数（不包括10），并且打乱排序后生成张量
p_tensor = p_tensor.cuda()  # 搬运到GPU
print('乱序数字：',p_tensor)  # 输出打散排序的张量

p1 = torch.rand(12,4)  # 生成12*4的矩阵张量p1
p2 = torch.rand(12,6)  # 生成与p1在第一维相等的12*2张量p2
index = torch.randperm(12)  # 生成有12个元素（与p1和p2的第一维相同）的随机数种子index
# 将3个张量都搬至GPU
p1 = p1.cuda()
p2 = p2.cuda()
index = index.cuda()
print('未洗牌的p1：',p1)  #输出原始p1
print('未洗牌的p2：',p2)  #输出原始p2
print('随机种子的一种情况：',index)  #输出index的任意一种情况
print('洗牌后的p1：',p1[index])  #用随机数种子index对p1进行洗牌并输出
print('洗牌后的p2：',p1[index])  #用同一个随机数种子index对p2进行洗牌并输出
# 注意，对相关联的张量进行洗牌操作时，随机数种子index必须一致
print('洗牌后的p1和关联的p2：',p1,p2)  # 输出使用同一个随机数种子进行洗牌后的p1和p2的关联结果


print('---------------------------------------------------------------------------------------------------------------')

print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))  # 记录程序结束时间

