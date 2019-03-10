| [返回主页](index.html) |  [torch](torch.html) | torch.autograd |  [torch.tensor](torch_tensor.html) |

---



# torch.autograd

#### 1. requires_grad
```python
每个tensor都有一个requires_grad标记
这个标记为True,则反向传输的梯度会累加到这个张量的计算图位置上
也就是，如果子节点需要梯度，那么其父节点必然需要梯度
a=torch.tensor([1.], requires_grad=True)
b=a+1
b.requires_grad # True
反之，如果子节点不需要梯度，其父节点必然不需要梯度
如果需要冻结模型的一部分梯度 ：
for param in model.parameters():
    param.requires_grad = False # 将模型的叶节点grad冻结
model.fc = nn.Linear(512, 100) # 进行finetune
# 由于model.fc中的参数需要梯度因此整个模型的输出仍然是需要梯度的
# 注意！x.requires_grad=False 其中的x必须是叶节点才能操作
```

#### 2. 自动求导机制简要
```python
当你进行计算时，自动求导器会记录计算顺序，并产生反向计算图
每个张量的grad_fn指向的函数即为其反向计算的入口
计算图在每次迭代计算时都会重新生成，这样就可以动态调整。
如果使用In-place操作，可以降低内存利用率，但是需要额外冗余计算。
```

#### 3. torch.autograd.backward
```python
计算张量下面所有叶节点的梯度(requires_grad=True的)
计算图默认执行一次直接销毁，因此执行一次前向，仅能执行一次反向。
a = torch.tensor([1.,2.], requires_grad=True)
b = torch.tensor([1.,2.], requires_grad=True)
c = 2*a+3*b+1
grad = torch.tensor([1.,1.])
torch.autograd.backward(c, grad_tensors=grad)
a.grad
# tensor([2., 2.])
b.grad
# tensor([3., 3.])
#####################
c.grad # 非叶节点不储存梯度
torch.autograd.backward(c, grad_tensors=grad) # 继续执行报错
#####################
# 注意，如果输出节点是标量，那么grad_tensors可以不用写，默认传1
# 如果需要保留计算图可以修改retain_graph参数，但不推荐
# 如果保留计算图，则需要在计算前将梯度清0 (由于梯度是累加的)
```

#### 4. torch.autograd.grad
```python
如果要指定求一系列张量的梯度，则用该函数
a = torch.tensor([1.,2.], requires_grad=True)
b = torch.tensor([1.,2.], requires_grad=True)
c = 2*a+3*b+1
d = 2*c+1
grad = torch.tensor([1.,1.])
torch.autograd.grad(d,(a,b,c),grad) # 非叶节点也能输出梯度!
# (tensor([4., 4.]), tensor([6., 6.]), tensor([2., 2.]))
a.grad, b.grad
# (None, None) # 该函数结果不储存在.grad中
torch.autograd.grad(d,a,grad) # 错误，一次前向只能操作一次
```

#### 5. 冻结梯度
```python
## 方案一
x = torch.tensor([1.], requires_grad=True)
with torch.no_grad():
	y = x * 2
y.requires_grad # False
## 方案二
@torch.no_grad()
def doubler(x):
	return x * 2
z = doubler(x)
z.requires_grad # False
## 方案三
flag = False # 如果这里为True则不阻止梯度传递
with torch.set_grad_enabled(flag):
	y = x * 2
y.requires_grad # False
## 方案四
torch.set_grad_enabled(False) # 如果是True则不阻止梯度传递
y = x * 2
y.requires_grad # False
```

#### 6. torch.Tensor.backward
```python
与 torch.autograd.backward 操作相同。
只是传入参数一般只有一个gradient，与grad_tensors作用相同。
一般为输出张量调用out.backward()
```

#### 7. torch.Tensor.detach
```python
生成一个新的并排除在计算图以外的张量(开辟内存)。
a = torch.tensor([1.,2.], requires_grad=True)
b = torch.tensor([1.,2.], requires_grad=True)
c = 2*a+3*b+1
cc = c.detach()
cc.requires_grad # False
cc  # tensor([ 6., 11.])
```

#### 8. torch.Tensor.detach_
```python
将张量排除到计算图以外(共享)。
a = torch.tensor([1.,2.], requires_grad=True)
b = torch.tensor([1.,2.], requires_grad=True)
c = 2*a+3*b+1
cc = c.detach_()
cc is c # True # 这是一个in-place操作
cc.requires_grad # False
cc  # tensor([ 6., 11.])
```

#### 9. torch.Tensor.grad
```python
张量的梯度属性在调用反向传输之前都是None
调用后叶节点将储存.grad属性
```

#### 10. torch.Tensor.is_leaf
```python
是否是叶节点
a = torch.tensor([1.,2.], requires_grad=True)
b = torch.tensor([1.,2.], requires_grad=True)
c = 2*a+3*b+1
c.is_leaf # False
a.is_leaf # True
```

#### 11. torch.Tensor.register_hook
```python
在某张量反向传输被调用到时执行一些动作
def hook_func(grad): # 钩子函数的参数固定为grad
	print(grad)
a = torch.tensor([1.,2.], requires_grad=True)
b = torch.tensor([1.,2.], requires_grad=True)
c = 2*a+3*b+1
d = 2*c+1
hook = c.register_hook(hook_func) # 某张量注册一个钩子函数
d.backward(torch.tensor([1.,1.]))
# tensor([2., 2.]) # 钩子函数被执行
# hook.remove() # 如果保留计算图，可以用这个方法解除钩子
```

#### 12. torch.Tensor.retain_grad
```python
如果需要保留非叶节点的梯度.grad则可以如下操作
a = torch.tensor([1.,2.], requires_grad=True)
b = torch.tensor([1.,2.], requires_grad=True)
c = 2*a+3*b+1
d = 2*c+1
c.retain_grad()
d.backward(torch.tensor([1.,1.]))
c.grad # tensor([2., 2.])
```

#### 13. torch.autograd.Function
```python
如果需要自己定义一个能自动求导的函数
假设我们要定义函数 3*x+2*y+exp(x)+exp(y)

class Func(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, y):
		# 输入都是torch.Tensor
		# ctx是固定的，后面参数自己设计
		expx = torch.exp(x) 
		expy = torch.exp(y)
		# 可以不用torch内置操作,自己调DLL
		result = 3*x+2*y+expx+expy
		# 为反向传播保留输出
		ctx.save_for_backward(expx, expy)
		return result
		
	@staticmethod
	def backward(ctx, grad_output):
		# ctx, grad_output 都是固定的
		# grad_output 为上级传下来的梯度张量
		expx, expy,  = ctx.saved_tensors
		grad_x = grad_output*(3+expx)
		grad_y = grad_output*(2+expy)
		return grad_x, grad_y

myfunc = Func.apply # 使用别名实例化

# 下面进行测试
x = torch.tensor([1.,2.], requires_grad=True)
y = torch.tensor([3.,4.], requires_grad=True)
z = myfunc(x,y)
z.backward(torch.tensor([1.,1.]))
x.grad, y.grad
# (tensor([ 5.7183, 10.3891]), 
#  tensor([22.0855, 56.5981]))
```

#### 14. torch.autograd.gradcheck
```python
通过一些点来检测梯度计算的正确性，
应该是先计算出采样点临近区域的近似梯度来检测
func, inputs # 函数与采样点
eps=1e-06 # 一般不变
atol=1e-05 # 绝对容忍度
rtol=0.001 # 相对容忍度
## 以下为操作方法
iptx = torch.randn(20, requires_grad=True).double()
ipty = torch.randn(20, requires_grad=True).double()
ipt = iptx,ipty
# 必须是 float64 类型的输入, requires_grad必须为True
test = torch.autograd.gradcheck(myfunc, ipt, atol=1e-4)
print(test)  #　没问题的话输出True
```

#### 15. torch.autograd.gradgradcheck
```python
通过一些点来检测梯度的梯度的正确性，
使用方法与gradcheck类似，一般不用这个，因为误差比较大。
test = torch.autograd.gradcheck(myfunc, ipt) # 上述函数无法通过检测
print(test)  #　没问题的话输出True
```

#### 16. torch.autograd.profiler.profile
```python
算法效率分析
x = torch.tensor([1.,2.], requires_grad=True).cuda() 
y = torch.tensor([3.,4.], requires_grad=True).cuda()
grad = torch.tensor([1.,1.]).cuda()
# 如果测试cpu则去掉.cuda()
with torch.autograd.profiler.profile(use_cuda=True) as prof:
	# 如果仅测试cpu则use_cuda=False或不写
    z = myfunc(x,y)    
    z.backward(grad)
print(prof)
```














