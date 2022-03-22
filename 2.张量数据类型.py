import torch
import time

print('起始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))# 输出开始时间

a = torch.tensor(2.5)#在torch中赋予a值,0维浮点型标量
print('搬运前浮点型标量a的值是：',a)#输出a
print('搬运前浮点型标量a的类型是：',a.type())#输出a的类型
a = a.cuda()#把a搬运到GPU
print('搬运到gpu后浮点型标量a的值是：',a)#输出a
print('搬运到gpu后浮点型标量a的类型是：',a.type())#输出a的类型
print('a的形状',a.shape)
print('a的维度:',a.dim())#等价于lens(x.shape)
print('a占用的内存大小:',a.numel())

print('----------------------------------------------------------------------------------------------------------------')

b = torch.FloatTensor(6)#在torch中给b赋值,随机生成6维(长度为6)向量
print('搬运前6维向量b的值是：',b)#输出b
print('搬运前6维向量b的类型是：',b.type())#输出b的类型
b = b.cuda()#把a搬运到GPU
print('搬运到gpu后6维向量b的值是：',b)#输出b
print('搬运到gpu后6维向量b的类型是：',b.type())#输出b的类型
print('b的形状',b.shape)
print('b的维度:',b.dim())#等价于lens(x.shape)
print('b占用的内存大小:',b.numel())

print('----------------------------------------------------------------------------------------------------------------')

c = torch.FloatTensor(1)#在torch中给c赋值,随机生成1维（长度为1）向量
print('搬运前1维向量c的值是：',c)#输出c
print('搬运前1维向量c的类型是：',c.type())#输出c的类型
c = c.cuda()#把c搬运到GPU
print('搬运到gpu后1维向量c的值是：',c)#输出c
print('搬运到gpu后1维向量c的类型是：',c.type())#输出c的类型
print('c的形状',c.shape)
print('c的维度:',c.dim())#等价于lens(x.shape)
print('c占用的内存大小:',c.numel())


print('----------------------------------------------------------------------------------------------------------------')

#二维的张量普遍用于图像处理，例如100张128*128的图片，就可以化为（100，128*128）
d =torch.randn(2,5)#使用randn随机数(随机正态分布)生成2*5的矩阵张量
#d =torch.FloatTensor(2,5)#同样随机生成2*5的矩阵张量，
print('搬运前张量d的值是：',d)#输出d
print('搬运前张量d的类型是：',d.type())#输出d的类型
d = d.cuda()#把d搬运到GPU
print('搬运到gpu后张量d的值是：',d)#输出d
print('搬运到gpu后张量d的类型是：',d.type())#输出d的类型
print('d的形状',d.shape)
print('d的第一个维度',d.size(0))
print('d的第二个维度',d.size(1))
#print('d的第一个维度',d.shape[0])#同样输出d的第一个维度
#print('d的第二个维度',d.shape[1])#同样输出d的第二个维度
print('d的维度:',d.dim())#等价于lens(x.shape)
print('d占用的内存大小:',d.numel())

print('----------------------------------------------------------------------------------------------------------------')

#三维张量普遍用于语言处理，即120句话，每句话10个单词，每个单词100维编码，即（10，120，100）（单词数，句子数，编码数）
e =torch.rand(2,4,5)#使用rand随机数(随机均匀分布)生成2*4*5的张量
#e =torch.FloatTensor(2,4,5)#同样随机生成2,4,5的张量，
print('搬运前张量e的值是：',e)#输出e
print('搬运前张量e的类型是：',e.type())#输出e的类型
e = e.cuda()#把e搬运到GPU
print('搬运到gpu后张量e的值是：',e)#输出e
print('搬运到gpu后张量e的类型是：',e.type())#输出e的类型
print('e的形状',e.shape)
print('e的第一个维度',e.size(0))
print('e的第二个维度',e.size(1))
print('e的第三个维度',e.size(2))
#print('e的第一个维度',e.shape[0])#同样输出e的第一个维度
#print('e的第二个维度',e.shape[1])#同样输出e的第二个维度
#print('e的第三个维度',e.shape[2])#同样输出e的第二个维度
print('张量e的第0维的元素是：',e[0])#对张量e索引第0维的元素，输出的是一个4*5的张量矩阵
print('把张量的形状转换为python的list：',list(e.shape))#把e。shape返回的内容转换为python的list
print('e的维度:',e.dim())#等价于lens(x.shape)
print('e占用的内存大小:',e.numel())


print('----------------------------------------------------------------------------------------------------------------')

#四维张量可用于卷积神经网络的图片处理，对于100张1920*1080的的三通道RGB彩色图片，可设为（100，3，1920，1080）（数量，颜色通道，高度，宽度）
f=torch.rand(100,3,1920,1080)#使用rand随机均匀分布生成4维张量
f=f.cuda()#将张量f搬运到GPU
print('四维张量f的值',f)#已搬运到GPU后的值
print('f的形状',f.shape)
print('f的维度:',f.dim())#等价于lens(x.shape)
print('f占用的内存大小:',f.numel())

print('----------------------------------------------------------------------------------------------------------------')

print('结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))# 输出结束时间

