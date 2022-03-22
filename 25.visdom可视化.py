import time

import numpy as np
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter  # tensorboardX是一个用于pytorch的可视化工具。引入summarywritter来实现这个功能
from visdom import Visdom  # 导入Visdom类，注意大小写。

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))

print('---------------------------------------------------------------------------------------------------------------')
# writter = SummaryWriter()  # 对summarywritter实例化。
# writter.add_scalar('data/scalar1',dummy_s1[0],n_iter)  # dummy_s1[0]是要监听的数据，data/scalar1是起的名字，n_iter是附带的参数
# writter.add_scalars('data/scaler_group',{'xsinx': n_iter * np.sin(n_iter),
#                                          'xcosx': n_iter * np.cos(n_iter),
#                                          'arctanx': np.arctan(n_iter)},n_iter
#                     # 这里是监听多个数据，
# )
# writter.add_image('Image',x,n_iter)  # 显示数据图像
# writter.add_text('Text','text logged at step:'+ str(n_iter),n_iter)  # 显示文本
# for name, param in resent18.named_parameter():
#     writter.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)  # 显示直方图
#     # 因为tensorboard默认使用的是numpy的数据，所以需要保证tensor在CPU中，然后转化为numpy数据，才能使用。
# writter.close()
# tensorboardx的缺点：1、会将数据写进文件，导致占用大量空间
# 2、数据更新太慢，30s才更新一次。
# 3、主题颜色亮黄色，较难修改
print('---------------------------------------------------------------------------------------------------------------')
# visdom是一个web服务器，程序在运行时会把数据丢给visdom，然后visdom再渲染出来。
# visdom开启命令：python -m visdom.server

viz = Visdom()  # 实例化Visdom类为viz
viz.line([0.],[0.],win='train_loss',opts=dict(title='训练误差'))  # 创建一条直线，(0,0)为初始点。win为标识符，即id，
# title为小窗口的名称，可以自行设定。
# 还有env参数，用来指定环境（即大窗口，不指定则会使用main窗口）
#viz.line([loss.item()],[global_step],win='train_loss',update='append') # 第一个[]是y的值，第二个[]是x的值。
# 对于非image数据，传的还是numpy数据。更新方式为附加(append)表示将新的数据挂在已有的点的后面，构成线。
# 这两行命令必须一起使用，才能达到绘制曲线的目的。

print('---------------------------------------------------------------------------------------------------------------')
# 下面绘制多条曲线
viz.line([[0.0,5.0]],[0.],win='test',opts=dict(title='多条曲线',legend=['loaa','acc']))  # 与之前的相同，不过初始值是[y1,y2]和[x]
# legend表示y1和y2曲线的标签，顺序和第一个[]里的一样。
#viz.line([[test_loss,correct/len(test_loader.dataset)]],[global_step],win='test',update='append')
# 这里同样有两种类型的数据，test_loss和correct/len(test_loader.dataset)两个数据，其他的和上面的一样

print('---------------------------------------------------------------------------------------------------------------')
# 利用visdom绘图
#viz.images(data.view(-1,1,28,28),win='x',opts=dict(title='张量输入值x'))
viz.images(torch.rand(512,512),win='x',opts=dict(title='张量输入值x'))
# visdom可以直接使用tensor绘图，无需转换为numpy数据。
#viz.text(str(pred.detach().cpu().numpy()),win='pred',opts=dict(title='预测值文本数据'))
viz.text(str('你好，Pytorch和Visdom。'),win='text1',opts=dict(title='文本数据'))
# 字符串string必须要转化为string才能看

print('---------------------------------------------------------------------------------------------------------------')




print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S'))