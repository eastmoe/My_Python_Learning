import time
import torch

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

print('----------------------------------------------------------------------------------------------------------------')
# 一般情况下，深度学习是对w和b求梯度。
# 涉及到的：da/dw=0;dw/dw=1;d(a*w)/dw=a;d(w^n)/dw=n*w^(n-1);d(a^w)/dw=a^w*lna;d(log(a)(w))/dw=1/(w*lna);
# d(sinw)/dw=cosw;d(cosw)/dw=-sinw;d(arctanw)/dw=1/(1+w^2);d(arcsinw)/dw=-d(arccosw)/dw=1/((1-w^2)^(1/2)).
# 复合函数求导法则（链导法）：df/dw=(df/du)*(du/dw)

