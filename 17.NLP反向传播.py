import time
import torch
from torch.nn import functional as F

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

# 假设在多输入输出的神经网络中，j层在k层的前面
# 忽略前面的层，多层多输入的梯度：δE/δwjk=(ok-tk)*ok*(1-ok)*xj0
# 不忽略前面的层，多层多输入的梯度：δE/δwjk=(ok-tk)*ok*(1-ok)*ojj
# 因为 (ok-tk)*ok*(1-ok)只与k层有关，所以将这一部分命名为δk，即δk=(ok-tk)*ok*(1-ok)，δE/δwjk=δk*ojj。一般来说，有k个δk。
# 输出E对第一层参数wij的导数：δE/δwij=oj(1-oj)*(δxj/δwij)*∑((ok-tk)*ok*(1-ok)*wjk)=oj(1-oj)*oi*∑((ok-tk)*ok*(1-ok)*wjk)
# 将(ok-tk)*ok*(1-ok)替换为δk，那么δE/δwij=oj(1-oj)*oi*∑(δk*wjk)
# 即δE/δwij=oi*δj，其中δj=oj(1-oj)*∑(δk*wjk)
# 同理可得δE/δwjk=oj*δk，其中δk=ok(1-ok)*(ok-tk)
# 也可继续向前推，倒数第三层：δii，δE/δwni
#

























print('---------------------------------------------------------------------------------------------------------------')

print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))
