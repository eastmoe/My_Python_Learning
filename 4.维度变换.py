import torch
import time

print('程序开始时间：',time.strftime('%Y-%m-%d %H:%M:%S')) #记录程序开始时间
print('---------------------------------------------------------------------------------------------------------------')
# 变换维度的过程中，保证numel()不变，同时需要确认进行变换操作需要符合原理。注意，变换后无法进行维度信息恢复。
# reshape和view功能与用法相同，可互相替代。
a = torch.rand(120,3,512,512)  # 生成4维向量(对应图片集合里的(b,c,h,w))
a = a.cuda()  # 搬移到GPU
print('张量a的形状：',a.shape)  # 输出原本a的形状
a_re1 =a.reshape(120,3,512*512)  # 对a进行维度变换
a_re1 = a_re1.cuda()  # 搬移到GPU
print('Reshape后张量a的形状：',a_re1.shape)  # 输出对a进行维度变换后的形状
a_re2 =a.reshape(120*3*512,512)  # 对a进行维度变换
a_re2 = a_re2.cuda()  # 搬移到GPU
print('另一次Reshape后张量a的形状：',a_re2.shape)  # 输出对a进行维度变换后的形状
a_re3 =a.reshape(120*3,512,512)  # 对a进行维度变换
a_re3 = a_re3.cuda()  # 搬移到GPU
print('另一次Reshape后张量a的形状：',a_re3.shape)  # 输出对a进行维度变换后的形状



print('---------------------------------------------------------------------------------------------------------------')
# unsqueeze的参数：n 指在n+1维之前添加一个维度；-n 指在倒数第n维之后插入一个维度
print('张量a的形状：',a.shape)  # 输出原本a的形状
a_unfold1 = a.unsqueeze(0)  # 在a的前面添加一个维度
a_unfold1 = a_unfold1.cuda()  # 搬运到GPU
print('进行维度展开后的a形状：',a_unfold1.shape)  # 输出进行展开后a的形状
a_unfold2 = a.unsqueeze(-1)  # 在a的最后添加一个维度（-1指原来a的最后一个维度）
a_unfold2 = a_unfold2.cuda()  # 搬运到GPU
print('另一次维度展开后的a形状：',a_unfold2.shape)  # 输出进行展开后a的形状
a_unfold3 = a.unsqueeze(4)  # 在a的第四维的前面添加一个维度
a_unfold3 = a_unfold3.cuda()  # 搬运到GPU
print('另一次维度展开后的a形状：',a_unfold3.shape)  # 输出进行展开后a的形状

print('---------------------------------------------------------------------------------------------------------------')
b = torch.tensor([1,4])
print('张量b的形状：',b.shape)  # 输出原本b的形状
b_unfold1 = b.unsqueeze(0)  # 在b的前面添加一个维度
b_unfold1 = b_unfold1.cuda()  # 搬运到GPU
print('进行维度展开后的b形状：',b_unfold1.shape)  # 输出进行展开后b的形状
print('进行维度展开后的b：',b_unfold1)  # 输出进行展开后的b
b_unfold2 = b_unfold1.unsqueeze(-1)  # 在前面展开的基础上，在b的最后的后面添加一个维度
b_unfold2 = b_unfold2.cuda()  # 搬运到GPU
print('进行二次维度展开后的b形状：',b_unfold2.shape)  # 输出进行二次展开后b的形状
print('进行二次维度展开后的b：',b_unfold2)  # 输出进行二次展开后的b

print('---------------------------------------------------------------------------------------------------------------')

c = torch.rand(32)  # 生成一维，长度为32的张量
c = c.cuda()  # 搬运到GPU
print('c的形状：',c.shape)  # 输出c的形状
c = c.unsqueeze(1).unsqueeze(2).unsqueeze(0)  # 依次展开c，先在第二个维度的前面添加一个维度，变成（32，1），
# 然后在第三个维度的前面添加一个维度，变成（32，1，1），最后在第一个维度的前面添加一个维度，变成（1，32，1，1）
print('扩展后c的形状：',c.shape)  # 输出进行展开后c的形状

print('---------------------------------------------------------------------------------------------------------------')

print('c挤压前的形状：',c.shape)  # 输出进行挤压前c的形状
c_squeeze1 = c.squeeze()  # 对c进行默认挤压，会去除所有为1的维度
c_squeeze1 = c_squeeze1.cuda()  # 搬运到GPU
print('c挤压后的形状：',c_squeeze1.shape)  # 输出进行挤压后c的形状
c_squeeze2 = c.squeeze(0)  # 对c进行挤压，挤压第一维
c_squeeze2 = c_squeeze2.cuda()  # 搬运到GPU
print('另一次c挤压后的形状：',c_squeeze2.shape)  # 输出进行挤压后c的形状
c_squeeze3 = c.squeeze(-1)  # 对c进行挤压，挤压倒数第一维
c_squeeze3 = c_squeeze3.cuda()  # 搬运到GPU
print('另一次c挤压后的形状：',c_squeeze3.shape)  # 输出进行挤压后c的形状


print('---------------------------------------------------------------------------------------------------------------')

print('c的形状：',c.shape)  # 输出c的形状
c_expand1 = c.expand(4,32,14,14)  # 对c进行扩展，将（1，32，1，1）扩展成（4，32，14，14），扩展前后的张量维度必须一致，并且要扩展的张量在要扩展的维度上为1.
c_expand1 = c_expand1.cuda()  # 搬运到GPU
print('扩展后c的形状：',c_expand1.shape)  # 输出扩展后c的形状
c_expand2 = c.expand(-1,-1,14,14)  # 对c进行定向扩展，不用扩展（扩展前后保持不变的维度）填上-1，其它需要扩展的填写需要的值
c_expand2 = c_expand2.cuda()  # 搬运到GPU
print('定向扩展后c的形状：',c_expand2.shape)  # 输出扩展后c的形状

print('---------------------------------------------------------------------------------------------------------------')

print('c的形状：',c.shape)  # 输出c的形状
c_repeat1 = c.repeat(4,10,2,2)  # 用拷贝（复制内存）来扩展维度，repeat的参数代表每一个维度重复拷贝的次数
c_repeat1 = c_repeat1.cuda()  # 搬运到GPU
print('重复扩展后c的形状：',c_repeat1.shape)  # 输出扩展后c的形状
c_repeat2 = c.repeat(4,1,32,32)  # 用拷贝（复制内存）来扩展维度，repeat的参数代表每一个维度重复拷贝的次数
c_repeat2 = c_repeat2.cuda()  # 搬运到GPU
print('重复扩展后c的形状：',c_repeat2.shape)  # 输出扩展后c的形状
c_repeat3 = c.repeat(4,1,14,14)  # 用拷贝（复制内存）来扩展维度，repeat的参数代表每一个维度重复拷贝的次数
c_repeat3 = c_repeat3.cuda()  # 搬运到GPU
print('重复扩展后c的形状：',c_repeat3.shape)  # 输出扩展后c的形状


print('---------------------------------------------------------------------------------------------------------------')
# t 转置只适用于2维矩阵张量
d = torch.rand(2,5)  # 生成2*5的二维矩阵张量
d = d.cuda()  # 搬运到GPU
print('张量d：',d)  # 输出张量d
d_t1 = d.t()  # 对d进行转置
print('转置后的张量d：',d_t1)  # 输出转置后的d

print('---------------------------------------------------------------------------------------------------------------')

e = torch.rand(2,3,5)  #生成2*3*5的三维张量
e = e.cuda()  # 搬运到GPU
print('三维张量e：',e)  #输出e的值
e_tran1 = e.transpose(0,1).contiguous()  # 交换e的第一维和第二维，加上contiguous()函数让数据变得连续。
e_tran1 = e_tran1.cuda()  # 搬运到GPU
print('e交换1、2维度之后：',e_tran1)  # 输出维度变换后的e的值
e_tran2 = e.transpose(0,2).contiguous()  # 交换e的第一维和第三维，加上contiguous()函数让数据变得连续。
e_tran2 = e_tran2.cuda()  # 搬运到GPU
print('e交换1、3维度之后：',e_tran2)  # 输出维度变换后的e的值
e_tran3 = e.transpose(1,2).contiguous()  # 交换e的第二维和第三维，加上contiguous()函数让数据变得连续。
e_tran3 = e_tran3.cuda()  # 搬运到GPU
print('e交换2、3维度之后：',e_tran3)  # 输出维度变换后的e的值

e_tran4 = e.transpose(0,1).contiguous().view(3,2*5).transpose(0,1).contiguous().view(2,5,3).transpose(1,2).contiguous()
# 利用transpose和view对e进行多次变换：（2，3，5）---（3，2，5）---（3，10）---（10，3）---（2，5，3）---（2，3，5）并保留维度信息。
# 注意跟踪维度的先后顺序，防止数据被污染
print('经过多重变换之后的e的形状：',e_tran4.shape)  # 输出变换后的形状，应当与e一致
print('比较e和经过变换后的e是否相等：',torch.all(torch.eq(e,e_tran4)))  # 比较变换后的e和原本的e进行对比，确保完全一致。
# eq用来确保数据一致，all用来确保数据完全一致。


print('---------------------------------------------------------------------------------------------------------------')

# b（图片张数）,h（图片高度像素）,w（图片宽度像素）,c（图片颜色通道）是numpy存储图片的格式，只有这样才能导出图片。
print('图片集a的形状：',a.shape)  # 输出原图片集a的形状
a_np = a.permute(0,2,3,1).contiguous()  # 利用permute进行变换，参数为当前位置需要的维度信息。
# 如（0，2，3，1）则生成的张量维度会分别对应原来的（第一维、第三维、第四维、第二维）
# permute也会打乱内存顺序，因此如果有必要，可以使用contiguous()来确保内存连续。
a_np = a_np.cuda()  # 搬运到GPU
print("变换后a的形状：",a_np.shape)  # 输出变换后的张量形状

print('---------------------------------------------------------------------------------------------------------------')


print('程序结束时间：',time.strftime('%Y-%m-%d %H:%M:%S'))  # 记录程序结束时间

