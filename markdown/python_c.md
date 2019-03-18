| [返回主页](index.html) | [Python基础](python.html) | Python扩展 |

---



### 1. Python 调用 DLL 库

如果需要写一个c或cuda代码封装的函数，一般先把它编译到动态链接库中，再通过Python调用, 以下操作在windows上实现, 注意64位Python必须对应x64 <br>
打开vs新建工程，使用x64-Release <br>
配置属性-常规-目标文件扩展名: .dll <br>
配置类型: 动态库(.dll) <br>
创建文件 testc.cpp testc.h <br>
配置属性-C/C++-预编译头-不使用 <br>
<br>
**testc.cpp**
```c
#include <stdio.h>
#include "testc.h"

void my_print() {
	printf("hello\n");
}
int my_add(int a, int b, int c) {
	int res = a + b + c;
	return res;
}
float my_add_f(float a, float b) {
	float res = a + b;
	return res;
}
int my_countzero(int *a, int len) {
	int i, sum = 0;
	for (i = 0; i < len; i++) {
		if (a[i] == 0) sum++;
	}
	return sum;
}
void my_zeros_int(int *a, int len) {
	int i;
	for (i = 0; i < len; i++) {
		a[i] = 0;
	}
}
void my_ones_float(float *a, int len) {
	int i;
	for (i = 0; i < len; i++) {
		a[i] = 1;
	}
}
void mystruct_init(struct MyStruct *a) {
	a->a = 0;
	a->b = 1;
	int i;
	for (i = 0; i < 3; i++) {
		a->c[i] = i;
	}
}
```

**testc.h**
```c
#ifndef TESTC_H
#define TESTC_H

struct MyStruct
{
	int a;
	float b;
	int c[3];
};

extern "C" __declspec(dllexport) void my_print();
extern "C" __declspec(dllexport) int my_add(int a, int b, int c);
extern "C" __declspec(dllexport) float my_add_f(float a, float b);
extern "C" __declspec(dllexport) int my_countzero(int *a, int len);
extern "C" __declspec(dllexport) void my_zeros_int(int *a, int len);
extern "C" __declspec(dllexport) void my_ones_float(float *a, int len);
extern "C" __declspec(dllexport) void mystruct_init(struct MyStruct *a);

#endif // !TESTC_H
```

在工程下右键-重新生成, 在x64-Release下会看到生成文件 <br>
我们拿其中的 .dll .lib 以及 testc.h 文件 <br>
如果使用 C++ 调用, 则先将头文件添加工程，将.lib文件添加资源文件, 将.dll文件放到运行路径 <br>
下面介绍如何用 Python 去执行这个 DLL

```python
# 先将.dll与.h放在一个文件夹并建立test.py
# 注意这里只需要.dll 其中.h为了方便查找函数名
import ctypes
dll = ctypes.CDLL('testc.dll') # 在如这个DLL文件, 只需要DLL
dll.my_print() # 调用里面的函数
a,b,c = 1,2,3
print(dll.my_add(a,b,c)) # 对于int类型,无需变动直传

# 对于输入输出不是int类型的需要先对输入参数以及输出结果定义类型
dll.my_add_f.argtypes = [ctypes.c_float,ctypes.c_float]
dll.my_add_f.restype = ctypes.c_float
# 将数据转化成相应类型
a,b = 1.0,2.0
a = ctypes.c_float(a)
b = ctypes.c_float(b)
# 再输出
out = dll.my_add_f(a,b)
print(out, type(out)) # 3.0 <class 'float'>

# 下面将np数组传入, 统计0个数
# 注意这里用指针仅能得到数据不能得到结构，因此是拉成1维的
dll.my_countzero.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int] # 定义输入
a = np.array([1,0,0,1,0,1]) # 一个np数组(int)
pa = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
print(dll.my_countzero(pa,6)) # 将指针传入

# 下面我们改变np数组(int*)
dll.my_zeros_int.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int]
dll.my_zeros_int(pa,6)
print(a) # [0 0 0 0 0 0]

# 下面我们改变np数组(float*)
dll.my_ones_float.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_int]
a = np.array([[1.,0.,0.],[1.,0.,1.]], dtype=np.float32)
pa = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
dll.my_ones_float(pa,6)
print(a, type(a))
# [[1. 1. 1.]
#  [1. 1. 1.]] <class 'numpy.ndarray'>
```


