| [返回主页](index.html) |

---



# 参考与评述

参考书目《Deep Learning》Lan Goodfellow.
经典的深度学习框架是以计算图&梯度下降方法实现对前馈网络的有监督学习。
这里复现了前馈计算图的梯度计算实现。
实现工程代码地址:   https://github.com/xytpai/Tenshell



# 一、前馈计算图实现

### 1. 前向与梯度计算

- 结果数组 （保存输入节点与计算节点的输出值，能够反映节点在计算方向的拓扑排序）
- 梯度数组 （保存输入节点与计算节点的梯度，能够反映节点在计算方向的拓扑排序）
- 连接图 （反映每个节点的父节点）
- 输出函数集合 （反映每个计算节点如何根据其输入得到输出）
- 梯度函数集合 （反映每个计算节点如何根据输入和它的梯度计算对其任意父节点的梯度）

即可组成一个完整前馈计算图。
我们以下图所示全连接神经网络为例构建计算图（我们将每个神经元看作一个节点）。

```c
(1)\---/(3)\---/(5)\
     X       X      (7)--(8)-->
(2)/---\(4)/---\(6)/
```

其中(1)(2)为输入点，(3)(4)(5)(6)为隐层，(7)为输出层，(8)为计算Loss的输出。这样，除了(1)(2)外其它都是计算节点。我们可以为该网络定义一个包含8个元素的数组，以及计算关系。

**数组及连接图如下：**

```c
//结果数组与梯度数组,我们把输入节点放到计算节点之前
list=[1] [2] [3] [4] [5] [6] [7] [8]
grad=[1] [2] [3] [4] [5] [6] [7] [8]
//父节点集合
Par[1]=null  Par[2]=null
Par[3]=1,2   Par[4]=1,2   //注意由于是前馈网络，Pa(i)=j则i>j
Par[5]=3,4   Par[6]=3,4   
Par[7]=5,6   Par[8]=7
```

**计算节点输出函数集合如下：（以下输出均为标量，输入$x$为向量）**

$$
Fun(i,x)=sigmoid(b_{i}+\sum_j w_{ij}*x_{j}),i\in [3,7] \\
Fun(8,x)=0.5 * (x-E)^2
$$

**计算节点梯度函数集合如下：**
注： dcFun(i,x,p) ，即求节点i在输入x下的输出对其父节点p的输出的偏导数。

$$
dcFun(i,x,p)=sigmoid'(b_{i}+\sum_j {w_{ij}*x_{j}})*w_{ip}, i \in [3,7] \\
dcFun(8,x,p)=(x-E)
$$

先解释下下面所用到的一些方法。

```c
Par[i]->get_output_array(); //返回节点i的所有父节点的输出列表，即i节点输入向量
Par[i]->get_index_array();  //返回节点i的所有父节点的索引列表
L.has_element();            //列表中还有值存在
L.get_next_element();       //返回列表下一个值
put_input_in(list,i,j);     //向list的i到j索引处输入值
```

**前馈实现：**

```c
//前馈计算
put_input_in(list,1,2);
for(i=3;i<=8;i++)
    list[i]=Fun(i,Par[i]->get_output_array());
```

前馈传播之后，为了最小化最终输出list[8] ，即由(8)定义的损失函数输出，我们需要计算每一节点的梯度。
**反馈实现：**

```c
//反馈梯度计算
for(i=1;i<8;i++) grad[i]=0; //清空梯度数组
grad[8]=1;
for(i=8;i>=3;i--) //迭代每个计算节点，累加其各个父节点梯度
{
    input=Par[i]->get_output_array();
	par_array=Par[i]->get_index_array();
	while(par_array.has_element()) //迭代本节点的所有父节点
	{
        par_index=par_array.get_next_element();
        grad[par_index]+=grad[i]*dcFun(i,input,par_index); 
        //这里dcFun(c,x,p)即求节点c在输入x下的输出对其父节点p的输出的偏导数
	}
}
```

以计算（4）节点的输出梯度为例。我们先得到其子节点（5）和（6）的梯度。则
$grad[4] = grad[5]*(5对4输出的偏导) + grad[6]*(6对4输出的偏导)$
如何求5对4输出的偏导？我们由公式推导。
$out5=sigmoid(out3*w_{51}+out4*w_{52}+b_{5})$
$我们令a5=out3*w_{51}+out4*w_{52}+b_{5}$
$\frac{\partial{out5}}{\partial{out4}}=sigmoid'(a5)*w_{52}$
这样我们可以得到grad[4]的表达式。
$grad[4] = grad[5]*(sigmoid'(a5)*w_{52}) + grad[6]*(sigmoid'(a6)*w_{62})$
在反馈计算迭代到（5）（6）节点时，都会累加grad[4]这个值。
我们根据任意节点输出的梯度，以及其输入，就能调整这个节点的一些参数 。

### 2. 参数更新

在上一步中，我们得到了每个节点的输出，以及Loss对每个节点输出的梯度。
参数可以放在如上所示节点的内部，也可以单独作为一个节点的输出。
**如果参数在节点内部：**

$$
\frac{\partial L}{\partial A_W}=
\frac{\partial L}{\partial A_{out}} \frac{\partial A_{out}}{\partial A_W}
$$

由于A的输出对其参数W的导数只由节点A决定，因此这些操作可以在计算所有梯度后并行执行。
**如果参数由单个节点定义：**

```c
X  --A--->
    /     //参数放在W节点输出中，这样A就只是一个计算形式，需要计算材料X与W
   W      //这种形式在如Tensorflow框架中出现
```

这种形式中，我们将W当作一个输入节点，这样，在梯度计算时我们的A将会直接算出Loss对于W输出的梯度。



# 二、全连接神经网络

### 1. MLP前向计算

一个全连接神经网络(MLP)可以当作一个整体作为计算图中的一个计算节点，它有它的依赖，输出方法，以及求父节点梯度的计算方法，权值更新方法。为了方便易用，也可以每一层当作一个计算节点。(PyTorch)
我们还可以将权值放到某个输入节点中，为了区分它和输入，把它定义成变量节点。(Tensorflow)

**Require：** 网络深度l 

**Require：** $W_{i}\ ,\ i\in{1,...,l}    （W_i的每一列表示i层某个神经元的全部权值）$

**Require：** $b_{i}\ ,\ i\in{1,...,l}      （b_i表示i层各个神经元的偏置）$

**Require：** X，程序输入            （X每一行为一个输入样本，行数为多少批样本，应用SGD）

**Require：** Y，目标输出            （Y每一行为一个样本对应标签，行数为多少批样本标签）

H_{0}=X 

**for**  k=1:l  **do** 

$$
A_{k}=b_{k}+H_{k-1}W_{k}  （A_k行数为批次数目，列数为k层神经元数目，这里加法为行行加）\\
H_{k}=f(A_{k})                （对A_k逐元素计算非线性函数）
$$

**end for**

$E=H_{l} $
$L=Loss(E,Y)+\lambda\Omega(\theta)     （\lambda\Omega(\theta)为正则项，用于优化学习）$

### 2. 对向量偏导的定义：

$$
\frac{\partial y_m}{\partial x_n}=
\begin{pmatrix}
	\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & ... &  \frac{\partial y_1}{\partial x_n}\\
	\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & ... &  \frac{\partial y_2}{\partial x_n}\\
	...&...&...&... \\
	\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & ... &  \frac{\partial y_m}{\partial x_n}\\
\end{pmatrix}_{m \times n}
$$

$$
\frac{\partial z}{\partial y_m}=(\frac{\partial z}{\partial y_1},\frac{\partial z}{\partial y_2},...,\frac{\partial z}{\partial y_m})^T
$$

$$
\frac{\partial z}{\partial x_n}=
(\frac{\partial y_m}{\partial x_n})^T(\frac{\partial z}{\partial y_m})=
\begin{pmatrix}
	\frac{\partial y_1}{\partial x_1}\frac{\partial z}{\partial y_1}
	+\frac{\partial y_2}{\partial x_1}\frac{\partial z}{\partial y_2}
	+...\\
	\frac{\partial y_1}{\partial x_2}\frac{\partial z}{\partial y_1}
	+\frac{\partial y_2}{\partial x_2}\frac{\partial z}{\partial y_2}
	+...\\
	...\\
	\frac{\partial y_1}{\partial x_n}\frac{\partial z}{\partial y_1}
	+\frac{\partial y_2}{\partial x_n}\frac{\partial z}{\partial y_2}
	+...
\end{pmatrix}_n=
\begin{pmatrix}
\sum_{k=1}^m{\frac{\partial z}{\partial y_k}\frac{\partial y_k}{\partial x_1}} \\
\sum_{k=1}^m{\frac{\partial z}{\partial y_k}\frac{\partial y_k}{\partial x_2}} \\
... \\
\sum_{k=1}^m{\frac{\partial z}{\partial y_k}\frac{\partial y_k}{\partial x_n}} \\
\end{pmatrix}_n
$$

### 3. 线性单元梯度计算：

已知 $AB=C$ ， $\frac{\partial L}{\partial C}=G$，求 $\frac{\partial L}{\partial A}与\frac{\partial L}{\partial B}$ ：（ $L$对某个矩阵$X$的偏导$G$的形式与$X$一模一样）

$$
\begin{pmatrix}
	a_{11} & a_{12} & a_{13} \\
	a_{21} & a_{22} & a_{23} \\
\end{pmatrix}_{2 \times 3} \times
\begin{pmatrix}
	b_{11} & b_{12} \\
	b_{21} & b_{22} \\
	b_{31} & b_{32} \\
\end{pmatrix}_{3 \times 2}=
\begin{pmatrix}
	a_{11} b_{11}+a_{12} b_{21}+a_{13} b_{31} & a_{11} b_{12}+a_{12} b_{22}+a_{13} b_{32}   \\
	a_{21} b_{11}+a_{22} b_{21}+a_{23} b_{31} & a_{21} b_{12}+a_{22} b_{22}+a_{23} b_{32}   \\
\end{pmatrix}_{2 \times 2}
$$

$$
\frac{\partial L}{\partial A}=
\begin{pmatrix}
	g_{11} b_{11}+g_{12} b_{12} &  g_{11} b_{21}+g_{12} b_{22} & g_{11} b_{31}+g_{12} b_{32}  \\
	g_{21} b_{11}+g_{22} b_{12} &  g_{21} b_{21}+g_{22} b_{22} & g_{21} b_{31}+g_{22} b_{32}  \\
\end{pmatrix}_{2 \times 3}=G \times B^T
$$

$$
\frac{\partial L}{\partial B}=
\begin{pmatrix}
	g_{11} a_{11}+g_{21} a_{21} &  g_{12} a_{11}+g_{22} a_{21}  \\
	g_{11} a_{12}+g_{21} a_{22} &  g_{12} a_{12}+g_{22} a_{22}  \\
	g_{11} a_{13}+g_{21} a_{23} &  g_{12} a_{13}+g_{22} a_{23}  \\
\end{pmatrix}_{3 \times 2}=A^T \times G
$$

对于偏置：已知 $A+b=C,\frac{\partial L}{\partial C}=G$ 求 $\frac{\partial L}{\partial A},\frac{\partial L}{\partial b}$  

$$
\begin{pmatrix}
	a_{11}  &  a_{12} &  a_{13}  \\
	a_{21}  &  a_{22} &  a_{23}  \\
\end{pmatrix}_{2 \times 3}+
\begin{pmatrix}
	b_{1}  &  b_{2} &  b_{3}  \\
\end{pmatrix}
=
\begin{pmatrix}
	a_{11}+b_1  &  a_{12}+b_2 &  a_{13}+b_3  \\
	a_{21}+b_1  &  a_{22}+b_2 &  a_{23}+b_3  \\
\end{pmatrix}_{2 \times 3}
$$

$$
\frac{\partial L}{\partial A}=
\begin{pmatrix}
	g_{11}  &  g_{12} &  g_{13}  \\
	g_{21}  &  g_{22} &  g_{23}  \\
\end{pmatrix}_{2 \times 3}=G
$$

$$
\frac{\partial L}{\partial b}=
\begin{pmatrix}
	g_{11}+g_{21}  &  g_{12}+ g_{22} &   g_{13}+ g_{23}  \\
\end{pmatrix}=G的行和
$$

### 4. 非线性单元梯度计算：

已知 $f(A)=H,\frac{\partial L}{\partial H}=G$ 求 $\frac{\partial L}{\partial A}$

$$
H=
\begin{pmatrix}
	f(a_{11})  &  f(a_{12})  \\
	f(a_{21})  &  f(a_{22})  \\
\end{pmatrix}_{2 \times 2},
\frac{\partial L}{\partial A}=
\begin{pmatrix}
	g_{11}f'(a_{11})  &  g_{12}f'(a_{12})  \\
	g_{21}f'(a_{21})  &  g_{22}f'(a_{22})  \\
\end{pmatrix}_{2 \times 2}=
G⊙f'(A)
$$

### 5. MLP梯度计算：

使用SGD算法，我们的输入是一次一批的，标签也是一次一批。

$G\gets\nabla_{E}L$   （得到矩阵G，E为最后一层的输出）

**for**   $k=l:1$   **do  ：**   

$$
G\gets\nabla_{A_k}L=G⊙f'(a_k) ，（得到对线性单元输出的梯度）\\
\nabla_{b_k}L=G行和，（得到对偏置的梯度） \\
\nabla_{W_k}=G \times H_{k-1}^T ，（得到对权值的梯度）\\
G\gets\nabla_{H_{k-1}}L=W_k^T \times G，（梯度传播到父节点）\\
$$

**end for**

### 6. 优化线性单元：

为了增加代码局部性来提高CPU运算速度，我们优化了计算方式。
与之前的矩阵乘法不同，权值矩阵行数为每个样本输出特征个数，列数与输入特征个数相同。

$$
H=Linear(
\begin{pmatrix}
	a_{00} & a_{01} \\
	a_{10} & a_{11} \\
	a_{20} & a_{21} \\
\end{pmatrix}_{3 \times 2} ,
\begin{pmatrix}
	w_{00} & w_{01} \\
	w_{10} & w_{11} \\
	w_{20} & w_{21} \\
	w_{30} & w_{31} \\
\end{pmatrix}_{4 \times 2},
\begin{pmatrix}
	b_{0} & b_{1} & b_{2} & b_{3} \\
\end{pmatrix} 
) \\
=
\begin{pmatrix}
	a_{00}w_{00}+a_{01}w_{01}+b_0 & a_{00}w_{10}+a_{01}w_{11}+b_1 & a_{00}w_{20}+a_{01}w_{21}+b_2 &
	a_{00}w_{30}+a_{01}w_{31}+b_3\\
	a_{10}w_{00}+a_{11}w_{01}+b_0 & a_{10}w_{10}+a_{11}w_{11}+b_1 & a_{10}w_{20}+a_{11}w_{21}+b_2 &
	a_{10}w_{30}+a_{11}w_{31}+b_3\\
	a_{20}w_{00}+a_{21}w_{01}+b_0 & a_{20}w_{10}+a_{21}w_{11}+b_1 & a_{20}w_{20}+a_{21}w_{21}+b_2 &
	a_{20}w_{30}+a_{21}w_{31}+b_3\\
\end{pmatrix}_{3 \times 4}
$$

$$
\frac {\partial L}{\partial A}=
\begin{pmatrix}
	g_{00}w_{00}+g_{01}w_{10}+g_{02}w_{20}+g_{03}w_{30} &g_{00}w_{01}+g_{01}w_{11}+g_{02}w_{21}+g_{03}w_{31} \\
	g_{10}w_{00}+g_{11}w_{10}+g_{12}w_{20}+g_{13}w_{30} &g_{10}w_{01}+g_{11}w_{11}+g_{12}w_{21}+g_{13}w_{31} \\
	g_{20}w_{00}+g_{21}w_{10}+g_{22}w_{20}+g_{23}w_{30} &g_{20}w_{01}+g_{21}w_{11}+g_{22}w_{21}+g_{23}w_{31} \\
\end{pmatrix}\\
=G \times W
$$

$$
\frac {\partial L}{\partial W}=
\begin{pmatrix}
	g_{00}a_{00}+g_{10}a_{10}+g_{20}a_{20} &g_{00}a_{01}+g_{10}a_{11}+g_{20}a_{21} \\
	g_{01}a_{00}+g_{11}a_{10}+g_{21}a_{20} &g_{01}a_{01}+g_{11}a_{11}+g_{21}a_{21} \\
	g_{02}a_{00}+g_{12}a_{10}+g_{22}a_{20} &g_{02}a_{01}+g_{12}a_{11}+g_{22}a_{21} \\
	g_{03}a_{00}+g_{13}a_{10}+g_{23}a_{20} &g_{03}a_{01}+g_{13}a_{11}+g_{23}a_{21} \\
\end{pmatrix} \\
=G^T \times A
$$

$$
\frac {\partial L}{\partial b}=
\begin{pmatrix}
g_{00}+g_{10}+g_{20} & g_{01}+g_{11}+g_{21} & g_{02}+g_{12}+g_{22} & g_{03}+g_{13}+g_{23} \\
\end{pmatrix}\\
=G列和
$$

### 7. Batch Normalization：

引入BN层的作用是为了加速学习，试想如下网络：

```C
x\
   A-->    //二维坐标输入(x,y)经过一个线性变换得到输出（二分类）
y/
```
如果(x,y)的整个数据集是偏离原点很远的一些点。由于在二分类问题中我们需要找到一条分割不同类点的直线，而初始化的bias(约等于0)表示的直线在靠近原点的地方，因此需要很多步迭代才能达到准确位置。同时，如果这些点非常密集，那么得到的梯度就会非常微小，造成梯度弥散。

- 我们希望将这些点保留相对位置地移动到原点附近（也相当于将原点移动到这些点中心位置）。
因此我们就把每个x减去x的平均值，每个y减去y的平均值。

- 我们还希望让这些点有着归一化的缩放大小（保持相对位置缩放到一个标准窗口内）
因此我们就在让上述处理后的每个x‘,y'除以他们各自的均方差根（强行拉成标准正态分布）

- 还希望在这个归一化标准基础上，增加一点灵活性，以适应非线性函数
因此我们加上一个乘性可学习参数、一个加性可学习参数。

因此，BN层一般都是加在非线性层之前，线性层之后，且BN层之前的线性层可以不要偏置（由于BN含偏置）。
每个神经元输出都套一个BN块，这个BN块搜集一个batch内该神经元所有输出，得到均值和均方差根，并且计算出标准化后的值到下一层。

对于一个batch中神经元A输出的所有x

$$
m=\frac 1n \sum x_i \\
std=\frac 1n \sum (x_i-m)^2 \\
x'_i \gets \frac {x_i-m}{\sqrt {std+eps}} \\
y_i \gets \gamma x'_i +\beta 
$$

其中eps为一个很小值（1e-5）来防止分母为零
以下举例对一个MLP输出矩阵求BN
（每一行一个batch每一列一个特征,每一列就是一个神经元输出）

$$
A=
\begin{pmatrix}
	a_{00} & a_{01} \\
	a_{10} & a_{11} \\
	a_{20} & a_{21} \\
\end{pmatrix} \\
m_0=(a_{00}+a_{10}+a_{20})/3 \\
m_1=(a_{01}+a_{11}+a_{21})/3 \\
std_0=((a_{00}-m_0)^2+(a_{10}-m_0)^2+(a_{20}-m_0)^2)/3 \\
std_1=((a_{01}-m_1)^2+(a_{11}-m_1)^2+(a_{21}-m_1)^2)/3 \\
y_{ij}= \gamma_j a'_{ij}+\beta_j =\gamma_j \frac{ (a_{ij}-m_j)}{\sqrt{std_j+eps} } +\beta_j
$$

下面求梯度，使用链式法则:

$$
\frac {\partial L}{\partial std_j}=\frac{-\gamma_j }{2}(std_j+eps)^{-3/2} \sum_i g_{ij}(a_{ij}-m_j)  \\
\frac {\partial L}{\partial m_j}=\sum_i g_{ij} \gamma_j
\frac {-\sqrt{std_j+eps}-(a_{ij}-m_j)(1/2)(std_j+eps)^{-1/2}\frac{\partial std_j}{\partial m_j}}{std_j+eps} \\
=\frac{-\gamma_j}{\sqrt{std_j+eps}}\sum_i g_{ij}-\frac{\partial L}{\partial std_j}\frac1n \sum_i 2(a_{ij}-m_j) \\
\frac {\partial L}{\partial a_{ij}}=\sum_i g_{ij} \gamma_j
\frac {(1-\frac 1n) \sqrt {std_j+eps}-(a_{ij}-m_j)\frac 12(std_j+eps)^{-1/2}\frac 2n(a_{ij}-m_j)(1-\frac 1n)}{std_j+eps} \\
=\frac 1n \frac{\partial L}{\partial m_j}+\frac {2(a_{ij}-m_j)}{n} \frac{\partial L}{\partial std_j}
+\frac {g_{ij} \gamma_{j}}{\sqrt {std_j+eps}} \\
\frac {\partial L}{\partial \gamma_j}=\sum_i g_{ij}a'_{ij} \\
\frac {\partial L}{\partial \beta_j}=\sum_i g_{ij}
$$



# 三、损失函数

### 1. Softmax单元：

在分类问题中，神经网络的输出层属于One-Hot类型。
比如手写数字识别中只需识别0~9数字，那么神经网络的输出层一共有10个神经元（对应10分类问题）。我们需要将这些输出表示为对应概率，因此和必须=1且每个输出大于0。这就需要通过Softmax层处理。

$$
S_i=\frac{e^i}{\sum_{k=0}^n e^k}
$$

$$
\frac{\partial S_i}{\partial i}=\frac{e^i \sum e^k - e^i e^i}{(\sum e^k)^2}
=\frac{e^i ( \sum e^k - e^i)}{(\sum e^k)^2}=S_i(1-S_i)
$$

$$
\frac{\partial S_i}{\partial j}_{i \neq j}= -\frac{e^i e^j}{(\sum e^k)^2} =
-S_iS_j
$$

已知 $S=softmax(A),\frac{\partial L}{\partial S}=G$ 求 $\frac{\partial L}{\partial A}$：
注意：A的每一行为一个样本，因此Softmax是对每一行进行操作

$$
softmax
\begin{pmatrix}
	a_{11} & a_{12} & a_{13} \\
	a_{21} & a_{22} & a_{23} \\
\end{pmatrix}_{2 \times 3} =
\begin{pmatrix}
	\frac{e^{a11}}{e^{a11}+e^{a12}+e^{a13}} &\frac{e^{a12}}{e^{a11}+e^{a12}+e^{a13}} & \frac{e^{a13}}{e^{a11}+e^{a12}+e^{a13}} \\
	\frac{e^{a21}}{e^{a21}+e^{a22}+e^{a23}} &\frac{e^{a22}}{e^{a21}+e^{a22}+e^{a23}} & \frac{e^{a23}}{e^{a21}+e^{a22}+e^{a23}} \\
\end{pmatrix}_{2 \times 3} =
\begin{pmatrix}
	s_{11} & s_{12} & s_{13} \\
	s_{21} & s_{22} & s_{23} \\
\end{pmatrix}
$$

$$
\frac{\partial L}{\partial A}=
\begin{pmatrix}
	s_{11}(g_{11}(1-s_{11})-g_{12}s_{12}-g_{13}s_{13})  & 
	s_{12}(g_{11}s_{11}-g_{12}(1-s_{12})-g_{13}s_{13})  & 
	... \\
	s_{21}(g_{21}(1-s_{21})-g_{22}s_{22}-g_{23}s_{23})  & 
	s_{21}(g_{21}s_{21}-g_{22}(1-s_{22})-g_{23}s_{23})  & 
	... \\
\end{pmatrix}
$$

### 2. Cross-Entropy单元：

Cross-Entropy是分类问题常用的损失函数，配合Softmax使用，其输入为一个表示不同类别识别概率的向量，以及One-Hot类型的标签向量，输出代价。

$$
L_{Cross-Entropy}=- \frac 1n \sum y \log a + (1-y) \log (1-a)
$$

比如我们得到最终Softmax输出的分类概率为 [0.8 , 0.1 , 0.1] One-Hot标签为[1 , 0 , 0]。
$Loss=-\frac 13 (\log 0.8+\log 0.9+\log 0.9)$
对于一个batch中我们可以这样计算：

$$
L=CrossEntropy(
\begin{pmatrix}
	a_{11} & a_{12} & a_{13} \\
	a_{21} & a_{22} & a_{23} \\
\end{pmatrix},
\begin{pmatrix}
	1 & 0 & 0 \\
	0 & 1 & 0 \\
\end{pmatrix} 
) \\
=-\frac 16(\log a_{11} + \log (1-a_{12}) + \log (1-a_{13}) \\
+  \log (1-a_{21}) + \log a_{22} +  \log (1-a_{23}))
$$

对于Cross-Entropy我们仅需对其输入求导，对标签矩阵无需求导。
求导十分方便，对矩阵A每个位置的求导，仅与该位置的原始数据和Y有关。
$\frac{\partial L}{\partial a_{ij}}=y_{ij}是否为1? 是-\frac 1{na_{ij}}:否\frac 1{n(1-a_{ij})}$

### 3. 分类问题的SCE单元：

$$
L=SCE(
\begin{pmatrix}
	a_{00} & a_{01} & a_{02} \\
	a_{10} & a_{11} & a_{12} \\
\end{pmatrix},
\begin{pmatrix}
	0 \\
	1 \\
\end{pmatrix} 
) \\
=-\frac 12
(\ln \frac {e^{a_{00}}}{e^{a_{00}}+e^{a_{01}}+e^{a_{02}}}  
+ \ln \frac {e^{a_{11}}}{e^{a_{10}}+e^{a_{11}}+e^{a_{12}}})  \\
= \frac 12(\ln s_{00}+\ln s_{11}) \\
= -\frac 12(    a_{00}-\ln (e^{a_{00}}+e^{a_{01}}+e^{a_{02}})+a_{11}-\ln (e^{a_{10}}+e^{a_{11}}+e^{a_{12}})   )
\\
$$

$\frac{\partial L}{\partial a_{ij}}=y_{i}是否为j? 是\frac {s_{ij}-1}{n}:否\frac {s_{ij}}{n}$







# 四、卷积神经网络 

### 1. 单通道图像卷积：

我们用一个例子对图A进行二维卷积运算，图像为单通道，使用一个卷积核。

$$
Conv2d(
\begin{pmatrix}
	a_{00} & a_{01} & a_{02} \\
	a_{10} & a_{11} & a_{12} \\
	a_{20} & a_{21} & a_{22} \\
\end{pmatrix},
\begin{pmatrix}
	w_{00} & w_{01}  \\
	w_{10} & w_{11}  \\
\end{pmatrix},b)\\
=
\begin{pmatrix}
	a_{00}w_{00}+a_{01}w_{01}+a_{10}w_{10}+a_{11}w_{11}+b & a_{01}w_{00}+a_{02}w_{01}+a_{11}w_{10}+a_{12}w_{11}+b\\
	a_{10}w_{00}+a_{11}w_{01}+a_{20}w_{10}+a_{21}w_{11}+b & a_{11}w_{00}+a_{12}w_{01}+a_{21}w_{10}+a_{22}w_{11}+b\\
\end{pmatrix}
$$

如果图A高度Ah宽度Aw，卷积核高度Wh宽度Ww。
输出图高：Ah-Wh+1
输出图宽：Aw-Ww+1

```c
for(i=0;i<Ah-Wh+1;i++)
    for(j=0;j<Aw-Ww+1;j++)
    {
        sum=0;
        for(di=0;di<Wh;di++)
            for(dj=0;dj<Ww;dj++)
                sum+=A[i+di][j+dj]*W[di][dj];
        Out[i][j]=sum;
    }
```

### 2. 单通道卷积梯度计算：

上面例子中，对W的偏导如下：

$$
\frac {\partial L}{\partial W}=
\begin{pmatrix}
	g_{00}a_{00}+g_{01}a_{01}+g_{10}a_{10}+g_{11}a_{11} & g_{00}a_{01}+g_{01}a_{02}+g_{10}a_{11}+g_{11}a_{12}  \\
	g_{00}a_{10}+g_{01}a_{11}+g_{10}a_{20}+g_{11}a_{21} & g_{00}a_{11}+g_{01}a_{12}+g_{10}a_{21}+g_{11}a_{22}  \\
\end{pmatrix} \\
=Conv2d(G,A,0)，（G形状与输出一样）
$$

上面例子中，对A的偏导如下：

$$
\frac {\partial L}{\partial A}=
\begin{pmatrix}
	g_{00}w_{00} & g_{00}w_{01}+g_{01}w_{00} & g_{01}w_{01} \\
	g_{00}w_{10}+g_{10}w_{00} & g_{00}w_{11}+g_{01}w_{10}+g_{10}w_{01}+g_{11}w_{00} & g_{01}w_{11}+g_{11}w_{01}\\
	g_{10}w_{10} & g_{10}w_{11}+g_{11}w_{10} & g_{11}w_{11} \\
\end{pmatrix}
$$

看似十分凌乱，这里运用了一个小技巧：
**先对G做padding变为G'：**
高度方向：上下各增加Wh-1行0。
宽度方向：左右各增加Ww-1列0。
**再对W做元素逆置：**
W矩阵的一维排列：W00,W01,W10,W11 变换成： W11,W10,W01,W00 再整合成原形状矩阵。
最终结果为：
$$
\frac {\partial L}{\partial A}=Conv2d(G',
\begin{pmatrix}
	w_{11} & w_{10}  \\
	w_{01} & w_{00}  \\
\end{pmatrix},0)
$$

来验证一下尺寸是否正确：
Gw=Aw-Ww+1
G'w=Aw-Ww+1+2(Ww-1)=Aw+Ww-1
Aw=G'w-Ww+1

$$
\frac {\partial L}{\partial b}=G各元素和
$$

上面表明，我们同样可以用卷积操作来求卷积核的梯度，十分方便。

### 3. Conv2d-Padding：

如果需要输入图像与输出图像形状一样，可以在进行卷积运算之前将输入图周围补0。
为了尽可能将图像放到中间。
如果Ww为奇数，Whfw=Ww/2（向下取整），图像左右各加Whfw列0，行操作类似。
如果Ww为偶数，Whfw=Ww/2，图像左加Whfw-1列0，右加Whfw列0，行操作类似。
在反传梯度的时候，原来Padding过的位置不需要传梯度。

### 4. 多通道多图卷积：

如果一个图像是三通道的，那么每一个卷积核也应该为三通道（一个卷积核偏置只有一个）。
一个卷积核输出：该卷积核各通道分别卷积图像各通道，各通道输出图叠加，再每个位置加一个偏置。
对于100张2通道4\*4图形成的batch，用6个3\*3卷积核操作的形状如下：

$$
Conv2d( A_{100 \times 2 \times 4 \times 4},W_{6 \times 2 \times 3 \times 3},b_6)=
O_{100 \times 6 \times 2 \times 2}
$$

```c
A1 A2 A3           K11 K12 K13      A^1 A^2 //A的3个通道分别和K1_,K2_的3个通道卷积
          Conv2d                =  
B1 B2 B3           K21 K22 K23      B^1 B^2 //B的3个通道分别和K1_,K2_的3个通道卷积
//A^1这样形成：A1卷积K11+A2卷积K12+A3卷积K13+偏置1（标量）
//A^2这样形成：A1卷积K21+A2卷积K22+A3卷积K23+偏置2（标量）  
//以上加法为矩阵对应位置相加
```

为了使输入与输出图形状一样，对输入图padding，高宽分别补上2\*Whfh与2\*Whfw个0。

### 5. 多通道多图卷积梯度计算：

用一个例子来说明：

```c
//以下每一个元素都是一张图
    I00 I01 I02
I=  I10 I11 I12  //四张三通道图
    I20 I21 I22
    I30 I31 I32

W=  W00 W01 W02  //两个三通道卷积核
    W10 W11 W12
  
    H00 H01
H=  H10 H11  //输出四张两通道图（每一个通道对应一个卷积核输出）
    H20 H21
    H30 H31

    G00 G01  //该矩阵每一个元素都是一张图
G=  G10 G11  //矩阵高度为一个batch的样本数
    G20 G21  //矩阵宽度为一个样本的输出特征图多少，也即该层卷积核数目
    G30 G31
```

下面用类似 **矩阵乘法** 的操作来求多图多通道卷积的梯度：

$$
Conv2dGrad\_I(
\begin{pmatrix}
	G_{00} & G_{01}  \\
	G_{10} & G_{11}  \\
	G_{20} & G_{21}  \\
\end{pmatrix},
\begin{pmatrix}
	W_{00} & W_{01}  & W_{02} \\
	W_{10} & W_{11}  & W_{12} \\
\end{pmatrix}) \\ =
\begin{pmatrix}
	G_{00}W_{00}+G_{01}W_{10} & G_{00}W_{01}+G_{01}W_{11} & G_{00}W_{02}+G_{01}W_{12} \\
	G_{10}W_{00}+G_{11}W_{10} & G_{10}W_{01}+G_{11}W_{11} & G_{10}W_{02}+G_{11}W_{12} \\
	G_{20}W_{00}+G_{21}W_{10} & G_{20}W_{01}+G_{21}W_{11} & G_{20}W_{02}+G_{21}W_{12} \\
\end{pmatrix}) \\ =G \times W \\（乘法为求单图梯度，加法为图对应位置叠加）
$$

$$
Conv2dGrad\_W(
\begin{pmatrix}
	G_{00} & G_{01}  \\
	G_{10} & G_{11}  \\
	G_{20} & G_{21}  \\
\end{pmatrix},
\begin{pmatrix}
	I_{00} & I_{01}  & I_{02} \\
	I_{10} & I_{11}  & I_{12} \\
	I_{20} & I_{21}  & I_{22} \\
\end{pmatrix}) \\ =
\begin{pmatrix}
	G_{00}I_{00}+G_{10}I_{10}++G_{20}I_{20} & 
	G_{00}I_{01}+G_{10}I_{11}++G_{20}I_{21} & G_{00}I_{02}+G_{10}I_{12}++G_{20}I_{22} \\
	G_{01}I_{00}+G_{11}I_{10}++G_{21}I_{20} & 
	G_{01}I_{01}+G_{11}I_{11}++G_{21}I_{21} & G_{01}I_{02}+G_{11}I_{12}++G_{21}I_{22} \\
\end{pmatrix} \\ = G^T \times I \\（乘法为求单卷积核梯度，加法为对应位置叠加）
$$

$$
Conv2dGrad\_b(
\begin{pmatrix}
	G_{00} & G_{01}  \\
	G_{10} & G_{11}  \\
	G_{20} & G_{21}  \\
\end{pmatrix})=
\begin{pmatrix}
	\sum G_{00} + \sum G_{10} + \sum G_{20} \\
         \sum G_{01} + \sum G_{11} + \sum G_{21} \\
\end{pmatrix} \\
（求和符号为所有元素累加）
$$

### 6. 多步长卷积 ：

为了在卷积的同时对图像进行降维，可以指定步长进行卷积。

```c
stride=2; //要指定的步长
//得到输出图的形状
Out_h=(Ah-Wh+1)%stride==0?(Ah-Wh+1)/stride:(Ah-Wh+1)/stride+1; //(Ah-Wh+1)/stride向上取整
Out_w=(Aw-Ww+1)%stride==0?(Aw-Ww+1)/stride:(Aw-Ww+1)/stride+1;
Out_1d=(float*)Out; //变成一维数组操作
index=0;
for(i=0;i<Ah-Wh+1;i+=stride)
    for(j=0;j<Aw-Ww+1;j+=stride) {
        sum=0;
        for(di=0;di<Wh;di++)
            for(dj=0;dj<Ww;dj++)
                sum+=A[i+di][j+dj]*W[di][dj];
        Out_1d[index++]=sum;
    }
```

### 7. 多步长卷积梯度计算 ：

**对卷积核W梯度：**

```c
//CPU优化代码如下
G_1d=(float*)G; //变成一维数组操作
for(i=0;i<Wh;i++)
    for(j=0;j<Ww;j++) {
        sum=0;
        index=0;
        for(di=0;di<Grad_h*stride;di+=stride)
            for(dj=0;dj<Grad_w*stride;dj+=stride)
                sum+=G_1d[index++]*A[i+di][j+dj]; //相当于卷积核G内间隔stride卷积
        Wgrad[i][j]+=sum;
    }
/*如需要并行操作，可以沿用Conv2d操作，具体如下：
1.创造一个Ah-Wh+1行Aw-Ww+1列零矩阵G'
2.将G'用G间隔stride-1行stride-1列填充（G平均分散成G'且G'(0,0)=G(0,0)）
3.对W梯度=Conv2d(A,G',0)*/
```

**对输入图A梯度：**

```c
/*无明显优化算法，统一用Conv2d操作，步骤如下：
1.创造一个Ah-Wh+1行Aw-Ww+1列零矩阵G'
2.将G'用G间隔stride-1行stride-1列填充（G平均分散成G'且G'(0,0)=G(0,0)）
3.对G'进行如下Padding：高度方向上下各增加Wh-1行零，宽度方向左右各增加Ww-1列零，记G''。
4.将卷积核W元素逆置变为W'
5.对A梯度=Conv2d(G'',W',0)*/
//对偏置b的梯度依然是G内所有元素相加
```

### 8. 池化层 ：

将图像切成一块一块，求每块最大值Max或平均值Avg做为输出图的相应点。
MaxPool作用：降维，减少对识别物体的平移敏感度。
AvgPool作用：降维，保持原有平移敏感度。

**MaxPool 前向计算：**

```c
kernel_size=2; //池化方框边长
stride=2;//池化步长
//得到输出图的形状
Out_h=(Ah-kernel_size+1)%stride==0?(Ah-kernel_size+1)/stride:(Ah-kernel_size+1)/stride+1; 
Out_w=(Aw-kernel_size+1)%stride==0?(Aw-kernel_size+1)/stride:(Aw-kernel_size+1)/stride+1;
Out_1d=(float*)Out; //变成一维数组操作
Out_Max_Y=(int*)malloc(Out_h*Out_w*sizeof(int)); //保存最大值坐标方便梯度计算
Out_Max_X=(int*)malloc(Out_h*Out_w*sizeof(int));
index=0;
for(i=0;i<Ah-kernel_size+1;i+=stride)
    for(j=0;j<Aw-kernel_size+1;j+=stride) {
        maxi=i;maxj=j;maxm=A[i][j];
        for(pi=i+1;pi<i+kernel_size;pi++)
            for(pj=j+1;pj<j+kernel_size;pj++)
                if(A[pi][pj]>maxm){
                    maxi=pi;
                    maxj=pj;
                    maxm=A[pi][pj];
                }
        Out_Max_Y[index]=maxi;
        Out_Max_X[index]=maxj;
        Out_1d[index++]=maxm;
    }
```

**MaxPool 梯度计算：**

```c
//梯度矩阵G与输出矩阵大小相同，因此是被降维过的。
//要计算输入矩阵A的梯度
//先得到每块中哪个是最大值
//G中某个位置的值直接传给对应块最大值的位置，该块中其他的梯度不传递值
G_1d=(float*)G; //变成一维度数组操作
index=0;
for(i=0;i<Ah-kernel_size+1;i+=stride)
    for(j=0;j<Aw-kernel_size+1;j+=stride) {
        Agrad[Out_Max_Y[index]][Out_Max_X[index]]+=G_1d[index]; //注意反向梯度计算是累加的
        index++;
    }
free(Out_Max_Y); //如果是动态内存需要回收
free(Out_Max_X);
```

**AvgPool 前向计算：**

```c
kernel_size=2; //池化方框边长
stride=2;//池化步长
//得到输出图的形状
Out_h=(Ah-kernel_size+1)%stride==0?(Ah-kernel_size+1)/stride:(Ah-kernel_size+1)/stride+1; 
Out_w=(Aw-kernel_size+1)%stride==0?(Aw-kernel_size+1)/stride:(Aw-kernel_size+1)/stride+1;
Out_1d=(float*)Out; //变成一维数组操作
Out_Max_Y=(int*)malloc(Out_h*Out_w*sizeof(int)); //保存最大值坐标方便梯度计算
Out_Max_X=(int*)malloc(Out_h*Out_w*sizeof(int));
index=0;
for(i=0;i<Ah-kernel_size+1;i+=stride)
    for(j=0;j<Aw-kernel_size+1;j+=stride) {
        sum=0;
        for(pi=i;pi<i+kernel_size;pi++)
            for(pj=j;pj<j+kernel_size;pj++)
                sum+=A[pi][pj]
        Out_1d[index++]=sum/(kernel_size*kernel_size);
    }
```

**AvgPool 梯度计算：**

```c
//G中某个位置的值，平均分配给对应块各个位置
G_1d=(float*)G; //变成一维数组操作
index=0;
for(i=0;i<Ah-kernel_size+1;i+=stride)
    for(j=0;j<Aw-kernel_size+1;j+=stride) {
        for(pi=i;pi<i+kernel_size;pi++)
            for(pj=0;pj<j+kernel_size;pj++)
                Agrad[pi][pj]+=G_1d[index]/(kernel_size*kernel_size); //平均分配
        index++;
    }
```

### 9. Conv2d-BN-Layer ：

由于Batch Normalization是套在每个神经元输出后面的，一个卷积核即一个神经元，因此操作如下：

```c
A1 A2 //假设输出为三张图，每张图两个通道
B1 B2 //由于A1,B1,C1都是卷积核W1的输出，A2,B2,C2都为卷积核W2的输出
C1 C2 //因此有两个BN节点，分别对应W1与W2
//1.计算(A1,B1,C1)上所有像素的平均值和均方差根，求出对应各像素BN输出
//2.计算(A2,B2,C2)上所有像素的平均值和均方差根，求出对应各像素BN输出
//3.计算梯度参考前面，操作集合确定后方法是一样的。
```



# 五、深度残差网络

如果要训练很深的网络，会面临一个微妙的问题：由于每次梯度传播都是按顺序从最高层到最底层，如果某个很低的层需要很高层的某个信息，那么在此之间所有层都应该学习去保留这些信息，这增加了学习的复杂度。ResNet为了解决低层无法充分利用高层的有用信息(梯度)这一问题。

### 1. 残差网络基本单元 ：

```c
X--A1--relu--A2-- add--relu-->
 \               /
  ---------------
//A1、A2为单层线性单元，relu非线性激活函数，add为对应元素相加
//注：可以用任意深度的非线性块替代上图中A1-relu-A2的位置（最后为线性层）
```

输入X经过一个旁路直接与A2输出叠加。
理论上，一个A1--relu--A2块能够逼近任意函数，姑且称它为万能块。万能块并不好用，比如让它逼近函数H(X)=X十分困难。并且如果要通过万能块传梯度到X，势必会经过一定衰减。
如果任务要求让不加残差网络的万能块逼近H(X)，那么残差网络的万能块转化为逼近H(X)-X。
有如下两个优势：

- 因为X一般都是正的元素（如图像像素，前面经过relu情况），并且H(X)中真正有用的信息必然拥有正的响应值（relu的特性使上层只把梯度传给A2输出为正的神经元，A2输出为负的成为死神经元），因此H(X)-X在有用信息的方向上更接近原点，因此更容易学习。

- 传到X的梯度为万能块与后面层梯度之和，如果万能块贡献很小，那么其传下去的梯度几乎不影响前面各层，该万能层也会被渐渐忽略，即逼近H(X)，也即网络有明显的忽略一些层的能力。如果万能块与后面层贡献相当，那么前面层能学习到两者共同的有用信息。

上述残差网络块需要输入与输出形状一致，事实上很难保证，我们可以通过线性变化让其一致。

### 2. 残差卷积单元 ：

为了将残差结构应用到卷积神经网络上，我们可以做成一个特殊的单元，在应用时叠加这些单元，下述两个单元结构引用了论文 Deep Residual Learning for Image Recognition 中的结构。

**对于较浅的层数<=34：**

```c
Node* ResidualCNNBlock(Node* X,int inchannel,int outchannel,int stride) {
    //输入图X(batch,inchannel,h,w)
    out=Conv2d(X,inchannel,outchannel,3,stride,padding=1,bias=False);//stride步3*3卷积
    out=BatchNorm2d(out);
    out=ReLU(out);
    //(batch,outchannel,h/stride,w/stride)
    out=Conv2d(out,outchannel,outchannel,3,1,padding=1,bias=False);
    out=BatchNorm2d(out);
    //万能块输出(batch,outchannel,h/stride,w/stride)
    if(stride!=1||inchannel!=outchannel) { //输入与输出形状不一致，需要线性变换
        sc_out=Conv2d(X,inchannel,outchannel,1,stride,padding=0,bias=False);
        sc_out=out=BatchNorm2d(sc_out);
        out=Add(out,sc_out);
    } else {
        out=Add(out,X);
    }
    out=ReLU(out); //最后一个非线性映射
    return out; //输出(batch,outchannel,h/stride,w/stride)
}
```

**对于较深的层数：**

```c
//即便输入通道比较多，也可以用比较少的卷积核卷积，来节省计算
Node* ResidualCNNBlock(Node* X,int inchannel,int outchannel,int stride,int knum) {
    //(batch,inchannel,h,w)
    out=Conv2d(X,inchannel,knum,1,1,padding=0,bias=False);//1步1*1卷积降维
    out=BatchNorm2d(out);
    out=ReLU(out);
    //(batch,knum,h,w)
    out=Conv2d(out,knum,knum,3,stride,padding=1,bias=False);//stride步3*3卷积
    out=BatchNorm2d(out);
    out=ReLU(out);
    //(batch,knum,h/stride,w/stride)
    out=Conv2d(out,knum,outchannel,1,1,padding=0,bias=False);//1步1*1卷积升维
    out=BatchNorm2d(out);
    //(batch,outchannel,h/stride,w/stride)
    if(stride!=1||inchannel!=outchannel) { //输入与输出形状不一致，需要线性变换
        sc_out=Conv2d(X,inchannel,outchannel,1,stride,padding=0,bias=False);
        sc_out=out=BatchNorm2d(sc_out);
        out=Add(out,sc_out);
    } else {
        out=Add(out,X);
    }
    out=ReLU(out); //最后一个非线性映射
    return out; //输出(batch,outchannel,h/stride,w/stride)
}
```

### 3. 深度残差卷积网络案例 ：

由于当图像尺寸减半时卷积核数目加倍，我们将相同卷积核数目的残差网络层绑定在一起，设计成一个块：

```c
Node* ResNetBlocks(Node* X,int inchannel,int outchannel,int stride,int depth,int knum) {
    //(batch,inchannel,h,w)
    out=ResidualCNNBlock(X,inchannel,outchannel,stride,knum);
    //(batch,outchannel,h/stride,w/stride)
    for(i=1;i<depth;i++)
        out=ResidualCNNBlock(out,outchannel,outchannel,1,knum);
    return out; //(batch,outchannel,h/stride,w/stride)
}
```

以下为官方的几种设计案例：

```c
//34-Layer:
out=Conv2d_Padding_BN_ReLU(X,Xchannel,kernel_size=7,outchannel=cnl);//预处理
out=ResNetBlocks(out,inchannel=cnl,outchannel=64,stride=2,depth=3);
out=ResNetBlocks(out,inchannel=64,outchannel=128,stride=2,depth=4);
out=ResNetBlocks(out,inchannel=128,outchannel=256,stride=2,depth=6);
out=ResNetBlocks(out,inchannel=256,outchannel=512,stride=2,depth=3);
out=GlobalAvgPool(out); //每张图像尺寸降成1
out=MLP(infeature=512,outfeature=分类数目);
//50-layer,101-layer,152-layer:   (差别仅关注depth参数)
out=Conv2d_Padding_BN_ReLU(X,Xchannel,kernel_size=7,outchannel=cnl);//预处理
out=ResNetBlocks(out,inchannel=cnl,outchannel=256,stride=2,depth=3,knum=64);
out=ResNetBlocks(out,inchannel=256,outchannel=512,stride=2,depth=4,knum=128);//4,4,8
out=ResNetBlocks(out,inchannel=512,outchannel=1024,stride=2,depth=6,knum=256);//6,23,56
out=ResNetBlocks(out,inchannel=1024,outchannel=2048,stride=2,depth=3,knum=512);
out=GlobalAvgPool(out); //每张图像尺寸降成1
out=MLP(infeature=2048,outfeature=分类数目);
```
