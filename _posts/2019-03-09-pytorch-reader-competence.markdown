

# torch

#### 1. torch.is_tensor(obj)
```python
# 如果是一个pytorch的tensor类返回True
```
#### 2. torch.is_storage(obj)
```python
# 如果是pytorch的storage类返回True
```
#### 3. torch.set_default_dtype(d)
```python
# 设置tensor分配的数据类型,输入如
# torch.float32 torch.float64
>>> torch.tensor([1.2,3]).dtype
>>> torch.get_default_dtype()
```
#### 4. torch.numel(Tensor)
```python
# 计算tensor的一维长度
a = torch.randn(1,2,3)
torch.numel(a) -> 6
```
#### 5. torch.set_printoptions
```python
打印控制，参数：
precision=4 # 输出小数保留几位
threshold=1000 # 当输出过大输出几个
```

#### 6. torch.tensor
```python
构造函数，该函数一直执行分配与拷贝
data # 需要拷贝的数据源, tuple, ndarray(numpy的)
dtype # 如torch.float64
device # 如torch.device('cuda:0') 或 'cpu'
requires_grad=False # 如果求导需要记录
```

#### 7. torch.as_tensor(data, dtype=None, device=None)
```python
转化成torch的tensor类型，如果输入的data已经是一个相同
数据类型与相同设备的tensor，则无需任何拷贝（指针引用）
否则，一个新的tensor会被分配
如果输入data是一个ndarray并且对应设备为cpu也无需拷贝
```

#### 8. torch.from_numpy(ndarray)
```python
使用numpy的array生成tensor
数据内存是共享的，因此改变其中任意一个另一个也会变
```

#### 9. torch.zeros, ones, empty
```python
*size # 一个元组或列表
out=None # 如果是一个tensor则将其输出至那个tensor
dtype=None # 默认全局类型
layout=torch.strided
device=None
requires_grad=False # 自动求导是否需要记录这个操作
>>> torch.zeros(2,3,device=0)
>>> torch.zeros(a.shape, out=a) # 将张量a清0
```

#### 10. torch.zeros_like, ones_like, empty_like
```python
创建相同形状的0张量，无论如何都是创建，无内存共享
>>> ipt = torch.tensor([1,2,3])
>>> out = torch.zeros_like(ipt) # 创建形状与ipt相同的0张量
device=None # 如果这项是None则其设备由输入tensor决定
requires_grad=False # 自动求导是否需要记录这个操作
```

#### 11. torch.arange
```python
输出一个一维等差数列
start, end, step # 数字
out=None # 如果非None则指定输出的tensor
dtype, device, requires_grad=False
```

#### 12. torch.linspace
```python
输出一个一维度等差数列
start, end, steps # 数字，这里为在start与end之间取steps个点
# 其他与arange一样
```

#### 13. torch.logspace
```python
输出一个一维指数数列
# 在10^start与10^end之间取steps个点
# 其他与linspace一样
```

#### 14. torch.eye
```python
输出单位矩阵(主对角线全是1)
n # 行数
m=None #列数,None表示与行数一样
# 其他与linspace一样
>>> torch.eye(2,3)
```

#### 15. torch.full, full_like
```python
使用固定值生成tensor，输入形式与ones雷同
>>> torch.full([2,2], fill_value=0.2) 
# 用0.2初始化一个2X2矩阵
>>> torch.full_like(a, 0.3)
# 用0.3初始化一个与a同形状的张量，注意这里都是创建，无内存共享 
```

#### 16. torch.cat
```python
拼接操作,参数为(tensor, dim=0, out=None)
该函数除非指定out否则会重新分配内存
>>> x = torch.randn(2, 3)
>>> torch.cat((x, x, x), 0) # 行拼接,输出形状(6,3)
>>> torch.cat((x, x, x), 1) # 列拼接,输出形状(2,9)
```

#### 17. torch.chunk
```python
张量等分，参数为(tensor, chunks, dim=0)
>>> sa = torch.chunk(a,2,dim=1) # 按列将张量a划分成2等分
# 注意，划分的内存是共享的，这个操作不开辟内存
>>> sa[0] # 输出某个分片
```

#### 18. torch.gather
```python
将输入张量按一个index(整数LongTensor)重新排布输出
参数为 (input, dim, index, out=None) out默认值时开辟内存
>>> t = torch.tensor([[1,2],
                      [3,4]])
>>> torch.gather(t, 1, torch.tensor([[0,0],
                                     [1,0]]))
# 注意，这里dim=1表示这个index是按列(第1维)排序的
tensor([[1,1], 
        [4,3]])
```

#### 19. torch.index_select
```python
返回一个新张量(默认开辟内存)，按照index在指定维度选择
参数为 (input, dim, index, out=None)
注意这里的index必须为1维
x=torch.ternsor([1,2]
                [3,4])
torch.index_select(x,0,torch.tensor([0,0,1]))
# tensor([[1,2],
#         [1,2],
#         [3,4]])
torch.index_select(x,1,torch.tensor([0,0,1]))
# tensor([[1,1,2],
#         [3,3,4]])
```

#### 20. torch.masked_select
```python
返回一个新开辟的1维张量，其值为输入在一个mask张量上的被选项
参数为 (input, mask, out=None)
x = torch.tensor([[1,2],
                  [3,4]])
mask = torch.tensor([[1,0],
                     [0,1]],dtype=torch.uint8)
#输出为 tensor([1, 4])
```

#### 21. torch.narrow
```python
输出一个新张量（开辟内存），输出为输入的某维度子集
参数为 (input, dimension, start, length)
x = torch.tensor([[1,2],
                  [3,4],
                  [5,6]])
torch.narrow(x, 0, 0, 2) #从0行开始输出两个行
# tensor([[1,2],
#         [3,4]])
```

#### 22. torch.nonzero
```python
输出一个张量中非零元素的索引(一行一个索引)
参数为 (input, out=None)
x = torch.tensor([[0,1],
                  [2,3]])
torch.nonzero(x)
# tensor([[0,1],
#         [1,0],
#         [1,1]])
```

#### 23. torch.reshape
```python
改变形状，此操作内存共享，不开辟新数据内存！
改变前后的形状必须保证总长度一致
参数 (input, shape) 当shape中出现-1则表示其余元素
```

#### 24. torch.split
```python
将张量分割，内存共享！不开辟新数据内存！
参数 (tensor, split_size_or_sections, dim=0)
x = torch.tensor([[1,2],
                  [3,4],
                  [5,6]])
torch.split(x, [1,2], dim=0)
# (tensor([[1,2]]), tensor([[3,4],[5,6]]))
```

#### 25. torch.squeeze
```python
将一个张量中维度为1的地方去除，该函数共享内存！
参数 (input, dim=None, out=None)
x = torch.zeros(2, 1, 2, 1, 2)
# torch.Size([2, 1, 2, 1, 2])
y = torch.squeeze(x)
# torch.Size([2, 2, 2])
xx = torch.zeros(2, 1, 2, 1, 2)
torch.squeeze(xx,1).size()
# torch.Size([2, 2, 1, 2])
```

#### 26. torch.stack
```python
将一系列张量堆积(增加一个维度)，并按某一维度方向排列
此方法会开辟内存，不共享
参数 (seq, dim=0, out=None)
a=torch.tensor([[1,2],[3,4]])
b=torch.tensor([[5,6],[7,8]])
torch.stack((a,b), dim=0)
# tensor([[[1,2],
#          [3,4]],
#         [[5,6],
#          [7,8]]])  # size(2,2,2)
torch.stack((a,b), dim=1)
# tensor([[[1,2],
#          [5,6]],
#         [[3,4],
#          [7,8]]])
torch.stack((a,b), dim=2)
# tensor([[[1,5],
#          [2,6]],
#         [[3,7],
#          [4,8]]])
a=torch.tensor([1,2])
b=torch.tensor([3,4])
torch.stack((a,b), dim=0)
# tensor([[1,2],
#         [3,4]]) # size(2,2)
torch.stack((a,b), dim=1)
# tensor([[1,3],
#         [2,4]]) # size(2,2)
```

#### 27. torch.t(input)
```python
矩阵转置，输入必须是二维矩阵
注意，这里内存是共享的！
a=torch.tensor([[1,2],[3,4],[5,6]])
torch.t(a)
# tensor([[1,3,5],
#         [2,4,6]])
```

#### 28. torch.take(input, indices)
```python
将输入看作一维，输出也是一维，取一维索引的值
该方法总是开辟数据内存
src = torch.tensor([[4,3,5],[6,7,8]])
torch.take(src,torch.tensor([0,2,5]))
tensor([4,5,8])
```

#### 29. torch.transpose
```python
将两维转置，参数 (input, dim0, dim1)
注意，该方法共享内存!
a=torch.tensor([[1,2],[3,4]])
torch.transpose(a,0,1)
# tensor([[1,3],
#         [2,4]])
```

#### 30. torch.unbind
```python
拆分一个维度，参数 (tensor, dim=0)
注意，该方法共享内存!
a=torch.tensor([[1,2],[3,4]])
torch.unbind(a,0)
# (tensor([1, 2]), tensor([3, 4]))
torch.unbind(a,1)
# (tensor([1, 3]), tensor([2, 4]))
```

#### 31. torch.unsqueeze
```python
增加一个维度，该方法共享内存 ！
参数 (input, dim=0)
a = torch.tensor([1,2,3])
torch.unsqueeze(a, 0)
# tensor([[1, 2, 3]]) # torch.Size([1, 3])
torch.unsqueeze(a, 1)
# tensor([[1],
#         [2],
#         [3]]) # torch.Size([3, 1])
```

#### 32. torch.where
```python
条件选项, 参数 (condition, x, y)
此方法永远开辟内存
x = torch.tensor([[-1.,2.],[0.,3.]])
y = torch.ones(2, 2)
torch.where(x > 0, x, y) # 为真则选x否则选y
# tensor([[1., 2.],
#         [1., 3.]])
```

#### 33. 随机数初始化设置
```python
torch.manual_seed(seed) # 手工设置随机数种子
torch.initial_seed() # 返回当前初始化的种子python.long
torch.get_rng_state() # 返回随机生成器状态
torch.set_rng_state(new_state) # 设置随机生成器状态
```

#### 34. torch.bernoulli
```python
生成伯努利二项随机数，永远开辟内存
输出与输入相同，输入每个元素的值为生成1的概率
a = torch.empty(3, 3).uniform_(0, 1) # 生成一个随机数(0~1)
torch.bernoulli(a)
# tensor([[1., 1., 0.],
#         [1., 1., 0.],
#         [1., 0., 0.]])
```

#### 35. torch.multinomial
```python
随机取数，从一个input的权重数组中取索引，replacement为有无放回
参数为 (input, num_samples, replacement=False)
永远开辟内存产生新张量
weights = torch.tensor([1., 10., 3., 1.])
torch.multinomial(weights, 4)
# tensor([1, 2, 0, 3])
torch.multinomial(weights, 4, replacement=True)
# tensor([0, 2, 1, 1])
```

#### 36. torch.normal
```python
产生正太分布随机数
mean=0.0
std # 输入为tensor表示方差分布，输出形状与std形状一样
std=torch.ones(2,2)
torch.normal(mean=0.5, std=std)
# tensor([[-0.1681,  0.3192],
#         [ 0.3726, -0.7143]])
```

#### 37. torch.rand
```python
产生0~1均匀分布，输入一般为形状
size # 形状元组
out=None # 可选输出，如果非None则指定输出数据空间
dtype=None # None表示默认
device=None
requires_grad=False
torch.rand(2, 3)
# tensor([[ 0.8237,  0.5781,  0.6879],
#         [ 0.3816,  0.7249,  0.0998]])
x = torch.ones(2,2,2)
torch.rand_like(x) # 创建与x形状相同的值
# tensor([[[0.8270, 0.8833],
#          [0.5540, 0.6145]],
#         [[0.4448, 0.4623],
#          [0.9856, 0.5086]]])
```

#### 38. torch.randint
```python
产生整数随机数，
参数(low,high,size..)其他与torch.rand一样
torch.randint(3, 5, (3,3)) # 注意范围[low, high), 右边开区间
# tensor([[4, 3, 3],
#         [3, 4, 4],
#         [3, 4, 4]])
x = torch.ones(2,2)
torch.randint_like(x,3,5) #注意输出数据结构由输入决定
# tensor([[3., 4.],
#         [3., 3.]])
```

#### 39. torch.randn
```python
产生01正太分布，参数与torch.rand一样
torch.randn(2, 3)
# tensor([[ 1.5954,  2.8929, -1.0923],
#         [ 1.1719, -0.4709, -0.1996]])
torch.randn_like # 与rand_like一样
```

#### 40. torch.randperm
```python
产生一个随机排序序列
n # 序列个数
out # 可选，输出张量
dtype, layout, device, requires_grad
torch.randperm(4)
tensor([2, 1, 0, 3])
```

#### 41. torch.save
```python
obj # 要储存的对象
f # 路径对象，可以是字符串
pickle_module # 用来封装元数据或对象, 不用动它
pickle_protocol=2 # 封装协议， 不用动它
## 读写单个张量时
x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, 'D:\\tensor.pt')
y = torch.load('D:\\tensor.pt')
## 读写model时
torch.save(the_model.state_dict(), 'xx.pkl') # 推荐用法
the_model.load_state_dict(torch.load('xx.pkl'))
```

#### 42. torch.load
```python
f # 路径对象，可以是字符串
map_location # 一个函数或设备来指定如何映射到内存
pickle_module # 不用动它
# x = torch.tensor([0, 1, 2, 3, 4])
# torch.save(x, 'D:\\tensor.pt')
cpu = torch.device('cpu')
cuda0 = torch.device('cuda:0')
torch.load('D:\\tensor.pt', map_location=cuda0)
# tensor([0, 1, 2, 3, 4], device='cuda:0')
```

#### 43. torch.get_num_threads
```python
获得用于并行化CPU操作的OpenMP线程数， 输出为int
# 设定操作: torch.set_num_threads(int)
```

#### 44. 生成时阻止梯度
```python
x = torch.zeros(1, requires_grad=True)
## 方案一
with torch.no_grad():
	y = x * 2
y.requires_grad # False
## 方案二
flag = False # 如果这里为True则不阻止梯度传递
with torch.set_grad_enabled(flag):
	y = x * 2
y.requires_grad # False
## 方案三
torch.set_grad_enabled(False) # 如果是True则不阻止梯度传递
y = x * 2
y.requires_grad # False
```

#### 45. torch.abs
```python
取绝对值，为新张量开辟新数据内存
x=torch.tensor([[1,-1],[2,-2]])
torch.abs(x)
# tensor([[1, 1],
#         [2, 2]])
```

#### 46. torch.acos, asin, atan, cos, sin, tan, tanh, sinh, atan2
```python
三角函数，为新张量开辟新数据内存
a = torch.randn(4)
torch.acos(a) 
# tensor([ 1.2294,  2.2004,  1.3690,  1.7298])
```

#### 47. torch.add
```python
点加运算，有两种方案，不过都为新张量开辟新数据内存
a = torch.ones(2,2)
torch.add(a, 20)
# tensor([[21., 21.],
#         [21., 21.]])
a = torch.ones(2,2)
b = torch.ones(2,2)
torch.add(a, 0.1, b) # a+0.1*b
#  这里第二个参数默认为1，因此可以直接torch.add(a,b)
# tensor([[1.1000, 1.1000],
#        [1.1000, 1.1000]])

## 下面测试加法传播
a = torch.tensor([1.,2.,3.])
# tensor([1., 2., 3.])
b = torch.t(torch.tensor([[.1,.2,.3]]))
# tensor([[0.1000],
#         [0.2000],
#         [0.3000]])
torch.add(a,b)
# tensor([[1.1000, 2.1000, 3.1000],
#         [1.2000, 2.2000, 3.2000],
#         [1.3000, 2.3000, 3.3000]])
```

#### 48. torch.addcdiv
```python
点加与点除的综合函数, 为新张量开辟新数据内存
t = torch.tensor([1.,2.,3.])
t1 = torch.tensor([3.,2.,6.])
t2 = torch.tensor([1.,2.,3.])
torch.addcdiv(t, 0.1, t1, t2) # t+0.1*t1/t2
# tensor([1.3000, 2.1000, 3.2000])
# 这里value的默认值为1，支持传播计算
```

#### 49. torch.addcmul
```python
点加与点乘的综合函数, 为新张量开辟新数据内存，支持传播
t = torch.tensor([1.,2.,3.])
t1 = torch.tensor([1.,2.,3.])
t2 = torch.tensor([1.,2.,3.])
torch.addcmul(t, 0.1, t1, t2) # t+0.1*t1*t2
# tensor([1.1000, 2.4000, 3.9000])
```

#### 50. torch.ceil, floor
```python
取上下整函数
a = torch.tensor([[1.1,2.2],[3.3,4.4]])
torch.ceil(a)
# tensor([[2., 3.],
#         [4., 5.]])
```

#### 51. torch.clamp
```python
限定最大值与最小值，为新张量开辟新数据内存
a = torch.randn(4)
# tensor([-0.2781, -1.3871, -0.7118,  0.5874])
torch.clamp(a, min=-0.5, max=0.5)
# tensor([-0.2781, -0.5000, -0.5000,  0.5000])
torch.clamp(a, min=0.5) # 如果仅限定最小值(或最大值)
```

#### 52. torch.div
```python
点除法，为新张量开辟新数据内存，支持传播
torch.div(a, 0.5) # a/0.5
torch.div(a, b) # a/b
```

#### 53. torch.digamma
```python
伽马函数
```

#### 54. torch.erf, erfc, erfinv
```python
误差函数，为新张量开辟新数据内存，支持传播
torch.erf(torch.tensor([0, -1., 10.]))
# tensor([ 0.0000, -0.8427,  1.0000])
# erfc(x)为1-erf(x)
# erfinv(erf(x))=x
```
$$
erf(x)=\frac{2}{\sqrt{\pi}} 
\int_0^xe^{-t^2}dt
$$

#### 55. torch.exp, expm1
```python
自然指数函数，为新张量开辟新数据内存，支持传播
expm1(x) = exp(x)-1
```

#### 56. torch.fmod
```python
取浮点余数函数，为新张量开辟新数据内存，支持传播
torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
# tensor([-1., -0., -1.,  1.,  0.,  1.])
torch.fmod(torch.tensor([1., 2, 3, 4, 5]), 1.5)
# tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])
```

#### 57. torch.frac
```python
取小数部分，为新张量开辟新数据内存，支持传播
torch.frac(torch.tensor([1, 2.5, -3.2]))
# tensor([ 0.0000,  0.5000, -0.2000])
```

#### 58. torch.lerp
```python
线性插值函数，为新张量开辟新数据内存，支持传播
start = torch.tensor([ 1.,  2.,  3.,  4.])
end = torch.tensor([ 10.,  10.,  10.,  10.])
torch.lerp(start, end, 0.5) # start+weight*(end-start)
# tensor([5.5000, 6.0000, 6.5000, 7.0000])
```

#### 59. torch.log, log10, log1p, log2
```python
对数函数，为新张量开辟新数据内存，支持传播
log1p(x) = loge(x+1)
```

#### 60. torch.mul
```python
点乘运算，为新张量开辟新数据内存，支持传播
a = torch.tensor([1,2,3])
torch.mul(a,3)
# tensor([3, 6, 9])
b = torch.tensor([1,2,3])
torch.mul(a,b)
# tensor([1, 4, 9])
a = torch.tensor([[1,2,3]])
b = torch.tensor([[1],
                  [2],
                  [3]])
torch.mul(a,b)
# tensor([[1, 2, 3],
#         [2, 4, 6],
#         [3, 6, 9]])
```

#### 61. torch.neg
```python
取反运算，为新张量开辟新数据内存，支持传播
a = torch.randn(3)
# tensor([ 0.0568, -0.4652, -0.1189])
torch.neg(a)
# tensor([-0.0568,  0.4652,  0.1189])
```

#### 62. torch.pow
```python
指数运算，为新张量开辟新数据内存，支持传播
a = torch.tensor([1,2,3])
torch.pow(a,2)
# tensor([1, 4, 9])
b = torch.tensor([2,1,2])
torch.pow(a,b)
# tensor([1, 2, 9])
torch.pow(3,a)
# tensor([ 3,  9, 27])
```

#### 63. torch.reciprocal
```python
倒数运算，为新张量开辟新数据内存，支持传播
a = torch.tensor([1.,2.,3.])
torch.reciprocal(a)
# tensor([1.0000, 0.5000, 0.3333])
```

#### 64. torch.remainder
```python
取浮点余数函数，为新张量开辟新数据内存，支持传播
与fmod不同的是，这个函数得到的值全是正的
torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
# tensor([1., 0., 1., 1., 0., 1.])
torch.fmod(torch.tensor([1., 2, 3, 4, 5]), 1.5)
# tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])
```

#### 65. torch.round
```python
取最近的整数，为新张量开辟新数据内存，支持传播
a = torch.randn(4)
# tensor([ 0.9920,  0.6077,  0.9734, -1.0362])
torch.round(a)
tensor([ 1.,  1.,  1., -1.])
```

#### 66. torch.rsqrt
```python
求平方根的倒数，为新张量开辟新数据内存，支持传播
a = torch.tensor([4.,9.])
torch.rsqrt(a)
# tensor([0.5000, 0.3333])
```

#### 67. torch.sigmoid
```python
sigmoid函数点操作，为新张量开辟新数据内存，支持传播
sigmoid(x)=1/(1+exp(-x))
a = torch.randn(4) 
# tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
torch.sigmoid(a)
# tensor([ 0.7153,  0.7481,  0.2920,  0.1458])
```

#### 68. torch.sign
```python
符号函数点操作，为新张量开辟新数据内存，支持传播
a = torch.tensor([0.7, -1.2, 0., 2.3])
torch.sign(a)
# tensor([ 1., -1.,  0.,  1.])
```

#### 69. torch.sqrt
```python
平方根操作，为新张量开辟新数据内存，支持传播
a = torch.randn(4)
# tensor([-2.0755,  1.0226,  0.0831,  0.4806])
torch.sqrt(a)
# tensor([    nan,  1.0112,  0.2883,  0.6933])
```

#### 70. torch.trunc
```python
截断整数操作，为新张量开辟新数据内存，支持传播
a = torch.randn(4)
# tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
torch.trunc(a)
# tensor([ 3.,  0., -0., -0.])
```

#### 71. torch.argmax, argmin
```python
求最大(最小)值的索引，为新张量开辟新数据内存，支持传播
a = torch.tensor([[1,2,3],
                  [3,4,5]])
torch.argmax(a, dim=0) # 沿行找
# tensor([1, 1, 1])
torch.argmax(a, dim=1) # 沿列找
# tensor([2, 2])
```

#### 72. torch.cumprod
```python
连乘，为新张量开辟新数据内存，支持传播
yi=x1*x2*x3*...*xi
a = torch.tensor([[1,2,3],
                  [3,4,5]])
torch.cumprod(a, dim=0) # 行连乘
# tensor([[ 1,  2,  3],
#         [ 3,  8, 15]])
torch.cumprod(a, dim=1) # 列连乘
# tensor([[ 1,  2,  6],
#         [ 3, 12, 60]])
```

#### 73. torch.cumsum
```python
连加，为新张量开辟新数据内存，支持传播
yi=x1+x2+x3+...+xi
a = torch.tensor([[1,2,3],
                  [3,4,5]])
torch.cumsum(a, dim=0) # 行连加
# tensor([[1, 2, 3],
#         [4, 6, 8]])
torch.cumsum(a, dim=1) # 列连加
# tensor([[ 1,  3,  6],
#         [ 3,  7, 12]])
```

#### 74. torch.dist
```python
求p范式距，为新张量开辟新数据内存，支持传播
x = torch.tensor([[1.,2.],[3.,4.]])
y = torch.tensor([[2.,1.],[4.,6.]])
torch.dist(x,y,2) # 默认p=2
# tensor(2.6458) # sqrt2(1^2 + 1^2 + 1^2 + 2^2)
torch.dist(x,y,3)
# tensor(2.2240) # sqrt3(1^3 + 1^3 + 1^3 + 2^3)
# 注意是逐元素操作，所有张量都拉成一维
```

#### 75. torch.logsumexp
```python
log(求和(exp))，为新张量开辟新数据内存，支持传播
x = torch.tensor([[1.,2.],
                  [3.,4.]])
torch.logsumexp(x,0)
# tensor([3.1269, 4.1269]) 
# [log(e^1+e^3), log(e^2+e^4)]
torch.logsumexp(x,1)
# tensor([2.3133, 4.3133])
# [log(e^1+e^2), log(e^3+e^4)]
torch.logsumexp(x,1,keepdim=True) # 不降维
# tensor([[2.3133],
#         [4.3133]])
```

#### 76. torch.mean
```python
平均，为新张量开辟新数据内存，支持传播
x = torch.tensor([[1.,2.],
                  [3.,4.]])
torch.mean(x)
# tensor(2.5000) # 拉成一维全部求平均
torch.mean(x,0) # 行逐点平均
# tensor([2., 3.]) 
torch.mean(x,1) # 列逐点平均
# tensor([1.5000, 3.5000])
torch.mean(x,1,keepdim=True) # 不降维
# tensor([[1.5000],
#         [3.5000]])
```

#### 77. torch.median
```python
求中位数，为新张量开辟新数据内存，支持传播
x = torch.tensor([[1.,2.,3.],
                  [3.,4.,5.]])
torch.median(x) # 拉成一维求中位数
# tensor(3.)
torch.median(x, 1) # dim=1，列逐点找
# (tensor([2., 4.]), tensor([1, 1]))
# 返回 (中位数张量, 中位数索引张量)
# 也可以使用keepdim来控制是否降维
```

#### 78. torch.prod
```python
点连乘，为新张量开辟新数据内存，支持传播
x = torch.tensor([[1.,2.,3.],
                  [3.,4.,5.]])
torch.prod(x)
# tensor(360.)
torch.prod(x, 1, keepdim=True) # dim=1，列逐点乘
# tensor([[ 6.],
#         [60.]])
```

#### 79. torch.std
```python

```







