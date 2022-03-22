import time
import torch

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))
# linear Regrassion: 连续函数：y=xw+b；对于输出概率：y=σ(xw+b)，其中σ是Sigmoid激活函数，最后y的值在0~1之间。
# Logistic Regrassion：将网络理解为f:x->p(y|x;θ)；其中θ=[w,b]
# Logistic Regrassion存在准确性问题：acc.=(∑(predi=yi))/(len(y))，但是不能使用acc来训练因为这样容易发生梯度爆炸和梯度离散。
# Logistic Regrassion本质上仍然是分类问题。
# 使用Softmax激活函数来保持概率为1和放大概率。
# 对于regression问题，最终目的是预测值=y；具体方法是最小化误差(pred,y)
# 对于classification问题，最终目的是最大化基准测试准确性，具体方法是最小化(pθ(y|x),Pт(y|x))，即最小化模型中x和y的分布和生成的x和y的分布之间的差距。








print('---------------------------------------------------------------------------------------------------------------')

print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))
