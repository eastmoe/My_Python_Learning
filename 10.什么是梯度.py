import time
import torch

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

print('----------------------------------------------------------------------------------------------------------------')

# 普通三元函数在点α(x1,y1,z1)的梯度：gradf=Δf=[(df/dx)|x1,(df/dy)|y1,(df/dz)|z1]
# 梯度是一个向量，有大小和方向。沿着梯度的方向，函数增长最快。
# 深度学习中，可以利用梯度来找到极值点，极小值（loss为正）和极大值（loss为负）
# 配合公式Aα+1=Aα-lr*Δf(α)，lr为学习速率（Learning Rate）
# 一般求极值的过程：1、先计算出梯度的向量表达式。2、随机初始化自变量的值，并求出梯度向量的各元素值。3、
print('----------------------------------------------------------------------------------------------------------------')

# 局部最小值和全局最小值比较好区分。
# 注意区分鞍点，鞍点指即在一个维度取得局部最大值，在另一个维度取得局部最小值。这不是真正的极值。
# 影响优化器的因素：初始值、学习速率、动量。
# 选择不恰当的初始值会让你落入局部最小值，并且会延长你的学习路径
# 学习速率设置过大会导致无法找到最值点，引起不断地在极值点周围震荡。
# 动量可以引导走出局部最小值

print('----------------------------------------------------------------------------------------------------------------')



print('----------------------------------------------------------------------------------------------------------------')



print('----------------------------------------------------------------------------------------------------------------')



print('----------------------------------------------------------------------------------------------------------------')



print('----------------------------------------------------------------------------------------------------------------')



print('----------------------------------------------------------------------------------------------------------------')
print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))