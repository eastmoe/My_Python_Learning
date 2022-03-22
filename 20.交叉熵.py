import torch
import time
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

# Classfication问题存在三种误差计算方法，第一种是之前的均方根（MSE），第二种是Cross Entropy Loss，第三种是Hinge Loss。
# 熵，即可以表示为：entropy=-∑(P(i)*logP(i))，即对于当前分布上的每一个i对应的概率和概率的log相乘，并求和。
# 熵越高，表示系统越稳定，惊喜度越低，反之

print('---------------------------------------------------------------------------------------------------------------')

a = torch.full([4],1/4).cuda()  # 生成元素全部为0.25的一维长度为4的张量a，表示概率
print('生成概率张量a',a)  # 输出a
alog2a = a*torch.log2(a)  # 计算a*loga
print('a*loga:',alog2a)  #输出
entropy_a = -alog2a.sum()  # 计算a的熵
print('计算a的熵：',entropy_a)  # 输出
print('')
b = torch.tensor([0.1,0.1,0.1,0.7]).cuda()  # 生成一维长度为4的张量b，表示概率
print('生成概率张量b',b)  # 输出b
blog2b = b*torch.log2(b)  # 计算b*logb
print('b*logb:',blog2b)  #输出
entropy_b = -blog2b.sum()  # 计算b的熵
print('计算b的熵：',entropy_b)  # 输出
print('')
c = torch.tensor([0.001,0.001,0.001,0.997]).cuda()  # 生成一维长度为4的张量c，表示概率
print('生成概率张量c',c)  # 输出c
clog2c = c*torch.log2(c)  # 计算c*logc
print('c*logc:',clog2c)  #输出
entropy_c = -clog2c.sum()  # 计算c的熵
print('计算c的熵：',entropy_c)  # 输出


print('---------------------------------------------------------------------------------------------------------------')

# Cross Entropy与Entropy不同，在Cross Entropy中，H(p,q)=-∑(p(x)*logq(x))=H(p)+Dkl(p|q)
# KL Divergence：Dkl(p|q)是真正用来衡量两个分布的具体关系。重叠的部分越少，Dkl就越大；两个分布越接近，Dkl会越接近0。
# 当P=Q时：cross entropy=entropy
# 当采用（one-hot encoding）0-1编码（即只有0和1两种状态）时，entropy=1log1=0；corss entrop：H(p,q)=Dkl(p|q)。
# 如果用p表示P的概率，那么H(P,Q)=-(y*logp+(1-y)*log(1-p))
# 不使用sigmoid+MSE的原因：1、是因为sigmoid会很容易饱和，从而造成梯度离散。
# 2、# cross-entropy的梯度更大，所以优化速度更快。
# 但是，如果cross-entropy出现错误，那么可以使用MSE。特别是meta-learning，可能会有不错的效果。
#
print('---------------------------------------------------------------------------------------------------------------')

x = torch.randn(1,784).cuda()  # 生成二维1*784张量
w = torch.randn(10,784).cuda()  # 生成二维10*784张量
print('输出x的形状：',x.shape)
print('输出w的形状：',w.shape)
logistics = x@w.t()  # 计算logistics张量。直接计算输出，没有经过激活函数的量被称为logistics
print('logistics的值：',logistics)
y = torch.tensor([3]).cuda()  # 生成真实值
print('真实值：',y)
h = F.cross_entropy(logistics,y)  # 直接计算cross-entropy，这个函数已经把softmax、求log和nullloss集成在了一起，所以必须传入logistics。
print('cross-entropy计算结果：',h)

print('')
print('下面使用手动计算方法计算cross-entropy：')
pred = F.softmax(logistics,dim=1)  # 计算预测概率，使用Softmax激活函数
print('预测值：',pred)
pred_log = torch.log(pred)  # 计算预测值的log值
print('预测值的log：',pred_log)
c_e2 = F.nll_loss(pred_log,y)  # 用nun_loss计算loss
print('手工计算的cross-entropy：',c_e2)



print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))