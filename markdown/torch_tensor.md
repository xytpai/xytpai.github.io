| [返回主页](index.html) |  [torch](torch.html) | [torch.autograd](torch_autograd.html) | torch.tensor |

---



# torch.tensor

#### 1. new_tensor
```python
生成与输入形状相同类型相同的新张量, 指定数据
开辟新的内存
tensor = torch.ones((2,), dtype=torch.int8)
data = [[0, 1], [2, 3]]
tensor.new_tensor(data)
# tensor([[0, 1],
#         [2, 3]], dtype=torch.int8)
```

#### 2. new_full
```python
生成与输入形状相同类型相同的新张量, 指定填充数据
开辟新的内存
tensor = torch.ones((2,), dtype=torch.float64)
tensor.new_full((3, 4), 3.141592) # 指定size
# tensor([[3.1416, 3.1416, 3.1416, 3.1416],
#         [3.1416, 3.1416, 3.1416, 3.1416],
#         [3.1416, 3.1416, 3.1416, 3.1416]], dtype=torch.float64)
```

#### 3. new_empty, new_ones, new_zeros
```python
生成与输入形状相同类型相同的新张量, 当然设备也相同
开辟新的内存
tensor = torch.ones(())
tensor.new_empty((2, 3)) # 可以指定size
# tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
#         [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
```

#### 4. is_cuda, device
```python
设备属性
a=torch.tensor([1], device=torch.device(0))
a.is_cuda
# True
a.device
# device(type='cuda', index=0)
```

#### 5. (function), (function)_
```python
所有 torch.function 操作几乎都可在tensor直接调用
后面加_的为在原存储空间操作不开辟内存(inplace)
a = torch.tensor([[1,-1],[2,-2]])
b = a.abs()
bb = a.abs_()
b is a # False
bb is a # True
```

#### 6. apply_
```python
使某个函数指针被调用 apply_(callable) → Tensor
这个函数只能在CPU的张量上执行, 效率不高
```

#### 7. byte, char, double
```python
转换数据类型为torch.uint8, int8, float64等等
```

#### 8. cauchy_
```python
填充(inplace操作)张量, 使用Cauchy分布
cauchy_(median=0, sigma=1, *, generator=None) → Tensor
```

#### 9. clone
```python
直接复制一个张量, 所有数据都被负复制
注意, 计算图也会被直接复制(与copy_不同),
因此clone后对输入张量反向传播也是有效的
```

#### 10. cpu
```python
将张量复制到CPU内存
如果本来就是CPU的张量则直接返回原指针
```

#### 11. cuda
```python
将张量复制到GPU显存
如果本来就是相同GPU的张量则直接返回原指针
调用方式: a=a.cuda()
有一个device参数: torch.device 默认为当前设备
```

#### 12. data_ptr
```python
返回数据地址
```

#### 13. dim
```python
返回维度
```

#### 14. element_size
```python
返回一个元素多少字节
a.element_size()
```

#### 15. expand
```python
扩张张量
注意，这里共享原有内存，不进行内存扩张(映射操作)
a = torch.tensor([[1,2],[3,4]])
a.expand(2,2,3) # 输入为期望维度，扩张方式为直接复制
```





