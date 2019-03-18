| [返回主页](index.html) | Python基础 | [Python扩展](python_c.html) |

---



# 参考与评述

本文参考书目为《Python基础教程第三版》
从类C语言转而学习Python让我尤为不适应，C语言更能体现硬件底层的细节，而Python相对来说让这些细节变得十分模糊。经过一定的思考后我发现Python的优雅之处恰恰就是屏蔽了这些细节，让编程变得十分简洁高效。Python不像C那样很直白地跟底层接轨，恰恰相反，Python更热衷于跟程序员接轨，因此在学习Python的过程中，不必在意太多被封装了的细节，简单无脑调包才是Python的最大魅力之处。



# 一、常见难点运算符

**1. 整除运算：** ```print(3//2) ```

**2. 求幂运算：** ```print(3**2)``` 相当于$3^2$

**3. 16,8,2进制的表示：** ```print(0xAF,010,0b0101)```

**4. 取绝对值：** ```print(abs(-1.2))```

**5. 浮点数取近似整：** ```print(round(2/3))```

**6. 浮点数向下向上取整：**

```python
import math
print(math.floor(32.9)) #相当于int(32.9)
print(math.ceil(32.1)) #向上取整
```



# 二、部分字符串操作

**1. 跨越多行的字符串：**

```python
print('''asdasd
asdasdas
xxxx''') #用三引号分界，Python3使用的所有字符串都是Unicode字符串
```

**2. 转义符：** 
```print('C:\\nowhere')```使用\作为转义符

**3. 原始字符串（不用转义符实现）：** 
```print(r'C:\nowhere')```

**4. 字符编码：**
```'Hello,world!',encode('UTF-8')```转化成UTF-8编码返回二进制字符
只要用encode就能将某个字符串编码为二进制字符。这些二进制字符再decode以下就能得到原来的字符串。

**5. 可变字符串：** 
python中若定义str='hello'那么str中的字符是不可变的，即便是二进制类型str=b'hello'也是不可变的，如果需要可变二进制字符串，可用如下。

```python
str=bytearray(b'hello')
str[1]=ord(b'u')
pint(bytes(str)) #这样又转回了bytes类型二进制字符串
```

**6. 精简字符串格式化：**
```str="hello %s,%d" % ('world',1)```与C类似
或者 ```str="hello {0},{1}" .format('world',1)```注意这里{i}中的任意i都能重复出现。
或者 ```str="hello {a},{b}" .format(a='world',b=1)```
还可以 ```str="{foo} {0} {bar} {1}".format(1,2,bar=4,foo=3)```

**7. 限定宽度格式化：**
```'haha{num:10}'.format(num=3)```限定num这块字符串宽度为10，注意数字右对齐，字符串左对齐。
如果要同时限定浮点数宽度和精度，则可以用类似```{pi:10.2f}```。（pi位别名）
如果在限定浮点数宽度和精度基础上再用0填充，则可用类似```{pi:010.2f}```。

**8. 格式化对齐：**
```str="{0:<10}\n{0:^10}\n{0:>10}" .format('world')```限定10宽度，分别位左，中，右对齐。

**9.填充字符：**
```'hello {0:%^15}'.format('world!')```输出为'hello %%%%world!%%%%%' 。

**10. 填充字符放在符号与数字之间：**
```print( '{0:10.2f}\n{1:=10.2f}'.format(pi,-pi) )```输出结果如下

```python
      3.14
-     3.14
```

**11. 字符串方法：**
```"hello".center(38)```指定宽度居中对齐。
```"hello".center(38,“*”)```指定宽度居中对齐，并且指定填充符号。
```string.find('xx')```串匹配，返回首匹配项索引，失败返回-1。
```string.find('xx',1)```串匹配，指定起点。
```string.find('xx',1,10)```串匹配，指定起点和终点。
```'+'.join(['qwe','asd','xzc'])```指定字符串连接，输出'qwe+asd+xzc'
```str.lower()```返回小写
```'This is a test'.replace('is','eez')```替换，输出'Theez eez a test'
```'1+2+34+5'.split('+')```拆分返回列表，这里输出['1', '2', '34', '5']
```'   haha   haha   '.strip()```删除开头和末尾空白，这里输出'haha   haha'。
```'&  haha%haha % '.strip('& %')```指定删除开头和末尾的字符，这里输出'haha%haha'。



# 三、列表和元组

**1. 列表和元组：**
列表是可以修改的，元组不可以修改。可以推断元组在内部使用的结构更简单且更高效。
Python中的列表中能包含许多数据类型，而且是不同的数据类型，这与C的数组差异极大。

**2.  列表的嵌套使用：**

```python
edward=['Edward Gumby',42]
john=['Join Smith',50]
database=[edward,john]
#得到 database=[['Edward Gumby', 42], ['Join Smith', 50]]
```

**3. 索引：**
Python中的索引都是以0开始，且可以为负值，相当于一个循环索引（-1表示最后一个元素）。

```python
l=['hello',24,'gg']
l[0] #输出'hello'
l[0][1] #输出'e'
```

**4. 切片：**
对于上述l列表，如果我要一次性取出前两个，那我不需要迭代索引，用切片即可。
l[0:2] #这里 0:2 代表的索引范围是 0 <= i < 2 。当然，如果 ：省去左右部分即取到两端。
如果需要取值步长还可以写为 l[左端点:右端点+1:步长]，如 l[0:10:2]步长即为2。

**5. 某串是否存在与另一串之中：**
str='ssrwxx'
'rw' in str #返回True，这里关键字为in。
当然，对于l=['ssrwxx',256]这种,'rw' in l 就返回False了，由于这个in只对最上一层起作用，因此如果是'ssrwxx' in l 就能返回True了。
我们很容易用这个操作来检查用户名以及密码。

```python
database=[
	['user1','pwd1'],
	['user2','pwd2']
]
username=input('User name:')
pwd=input('pwd:')
if [username,pwd] in database: print ('Access granted')
```

**6. 列表中元素个数、最大值、最小值：**
分别为 len(l),max(l),min(l),其中最大值和最小值必须保证列表中元素类型一致。

**7. 字符串变列表：**
list('Hello') 输出即为 ['H', 'e', 'l', 'l', 'o']

**8. 列表按索引更改：**
这个操作跟C类似，即list[索引]=你想设定的值。

**9. 列表按索引删除：**
del list[索引值] ，非常方便！

**10.切片赋值：**
name[2:]=list('ar') 这样可以将串name第二个字符之后的所有变成ar

```python
name=list('abcdefg')
name[2:]=list('ar') #得到['a', 'b', 'a', 'r']
```

切片操作还能够实现删除、插入，注意 list[1:1]=..为将list[1]之前处插入列表。

**11.列表方法：**
```list.append(input)```附加到列表末尾，单元素。
```list.clear()```就地清空列表。
```b=a.copy()```创造出一个与a一模一样的b，注意b=a只是拷贝了指针。
```[1,2,1,1,2,1].count(1)```计数。
```a.extend(b)```将列表b附加到列表a末尾，多元素，就地执行，a+b效率比这个低。
```[1,22,32,44].index(22)```返回1，即根据内容找索引。
```[1,2,3,6].insert(3,'four')```在第3个元素之前插入'four'。
```[1,2,3,6].pop(0)```删除0号，返回被清除值，如果不指定编号则取出最后一个元素。
```list.remove('xx')```按内容删除。
```list.reverse()```就地逆置。
```list.sort()```就地排序，若要赋值，则用a=sorted(b)。
```x.sort(key=len)```按长度排序。
```x.sort(reverse=True)```递减排序。

**12. 列表转元组：**
```tuple([1,2,3]) = (1,2,3)```

**13. 列表的浅拷贝和深拷贝：**

```python
a=['hello',123]
b=a    #浅拷贝
b=a[:] #深拷贝
```



# 四、字典

**1. 创建字典：**
字典就是一个个键:值对，如```dic={'a':123,'b':'hello','cc':'world'}```
也可以用映射创建字典，```dic=dict([['a','b'],[123,'x']])```
还可以这样，```dic=dict(a=123,b='hahaha')```
当然字典也可以嵌套```{ 'a':{'a1':0,'a2':1},'b':{'b1':'haha','b2':23} }```

**2. 字典的成员资格：**
如果```dic={'a': 123, 'b': 'hahaha'}```则'a' in dic = True,123 in dic=False
可见in只对键起效果。

**3. 字典按键查找值：**

```python
dic={'a':123,'b':'hello','cc':'world'} #那么dic['cc']=world
dic={ 'a':{'a1':0,'a2':1},'b':{'b1':'haha','b2':23} } #那么dic['a']['a1']=0
```

**4. 字典方法：**
```dic.clear() ```就地删除字典。
```a=b.copy()```深拷贝字典。
```{}.fromkeys(['name','age'])```用键创建字典，默认值都是None。
```dic.get('key')```按键查找值，如果查找不成功返回None。
```dic.get('key','err')```按键查找值，如果查找不成功指定输出某字符串。
```dict.items()```返回字典列表视图，如it=dict.items() ,len(it)为字典元素个数，(k,v) in it审查。
```list(dict.items())```将字典所有键值输出到列表。
```dic.keys()```返回字典键的列表视图，与items类似。
```dic.pop('key')```指定键弹出，返回弹出值。
```dic.popitem()```随机弹出，返回弹出键和值。
```dic.setdefault('key','value')```如果key不存在则加入key:value，若存在则不影响。
```dic.update(x)```用x来更新字典dic，如果在dic中存在相同的键则替换，否则添加。



# 五、逻辑控制语句

**1. print的连接与结尾控制：**
```print('a','b','c',sep='_',end='#')```输出为 ```a_b_c#```

**2. 导入时重命名：**
import somemodule的话要用 somemodule.func()
from somemodule import * 则直接可用这个模块下的func()
import somemodule as sm 则用sm.func()

**3. 序列解包：**
```values=1,2,3```对于该元组使用```x,y,z=values```则xyz就按序分配上去了。
```dic={'a':123,'b':'hello','cc':'world'}```对其使用```k,v=dic.popitem()```则键值即可被保存。
如果超出范围可用```a,b,*r=[1,2,3,4]```r的输出结果为[3,4]。
也可```a,*b,r=[1,2,3,4]```b结果为[2,3]，以此类推，也可*a。

**4. 缩进：**
Python中，代码块都是由```:```引导的缩进块分割的。

**5. if条件语句：**

```python
if a>0:
	print('>0') 
	#缩进处都在这个条件中
elif a<0:
	print('<0')
else:
	print('=0')
```

**6. 断言语句assert：**

```python
age=0
assert 0<age<100
#不满足assert条件的话，后面程序奔溃
```

**7. 循环语句：**

```python
while x<=100:
	print(x)
	x+=1
words=['aa','vs','xx']
for word in words:
	print(word)
for number in range(1,101): # 1<=i<101
	print(number)
```

**8. 迭代字典：**

```python
dic={'a':123,'b':'hello','cc':'world'}
for key in dic:
	print(key,'corresponds to',dic[key])
#或者
for key,value in dic.items():
	print(key,'corresponds to',value )
```

**9. 并行迭代：**

```python
names=['anne','bob']
ages=[0,12]
for i in range(len(names)):
	print(names[i],ages[i])
#或者
for name,age in zip(names,ages):
	print(name,age)
list(zip(names,ages))  #输出[('anne', 0), ('bob', 12)]
```

**10. 迭代时获取索引：**

```python
strings=['aa','bb']
for index,string in enumerate(strings):
	strings[index]+='x'

```

**11. break与continue：**
与C一致

**12. 集合推导：**
```[x*x for x in range(10)]```输出[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```[x*x for x in range(10) if x%3==0 ]```输出[0, 9, 36, 81]
```[ (x,y) for x in range(3) for y in range(3) if x<y ] ```输出[(0, 1), (0, 2), (1, 2)]

**13. pass什么都不做：**
类似nop()，使用pass什么都不做

**14. del回收内存：**
del dic即删除dic所包含的所有内容，将内存回收。

**15. 执行存储在字符串中的Python代码：**
```exec("print('hello')")```
一般执行这样的代码需要一个命名空间，不然可能会污染当前环境。

```python
from math import sqrt
scope={}
exec('sqrt=1',scope) #绑定命名空间到字典上
print(sqrt(4)) #2.0
print(scope['sqrt']) #1
```

**16. 执行储存在字符串中的算术：**

```python
scope={}
scope['x']=2
scope['y']=3
print(eval('x*y',scope)) #6
```



# 六、函数

**1. 判断是否是可调用函数：**
```callable(func)```如果func被定义且为函数则返回True否则返回False。

**2. 函数文档：**

```python
def func(x):
	'calculates the square of the number x' #函数文档
	return x*x
func.__doc__ #访问函数文档
help(func)   #访问函数文档以及函数形式
def func(x): #也可以这样的多行文档
    """
    calculates the square 
    of the number x'
    """
    return x*x
```

**3. 可变函数参数：**
对于列表和字典，作为函数参数时，可以改变其值，但如果是字符串这类不可变数据结构则不能够改变。单个变量作为参数也是不可改变的。

```python
def func(x):
	x[0]+='a'
list=['a','b']
func(list)
print(list) #输出['aa', 'b']
```

**4. 函数参数的命名：**

```python
def func(left,right):
	return left+right
a=func(right='r',left='l')
print(a) #输出'lr'
```

**5. 收集参数：**
允许用户提供任意数量的参数。用*号作为前导，意味着收集余下的参数，并放到元组中。

```python
def myPrint(title,*para):
	print(title)
	print(para)
myPrint('hello',1,2,3,'ok')
# hello
# (1, 2, 3, 'ok')
```

**6. 分配参数：**
与收集参数相对应，

```python
def add(x,y):
	return x+y
para=[1,2]
res=add(*para) #一个星号引导列表或元组
print(res) #输出3
para={'x':2,'y':3}
res=add(**para) #两个星号引导字典
print(res) #输出5
```

**7. 全局变量在函数中的声明：**

```python
x=1
def change_global():
	global x #使用global引导
	x=x+1
```



# 七、类

**1. 类的定义：**

```python
class Person:
    id=1
    __subid=12               #以两个下划线打头的变量或者函数都是私有的
    def set_name(self,name): #self作为该类作用域的命名空间，调用时不作形参
        self.name=name
    def get_name(self):
        return self.name
    def greet(self):
        print('Hello! I am {}'.format(self.name))
    def get_subid(self):
        print(self.__subid)
foo=Person()
foo.set_name("Tom")
foo.greet()        #输出Hello! I am Tom
print(foo.id)      #输出1
foo.get_subid()    #输出12
print(foo.__subid) #错误，访问私有变量
```

**2. 实现类的静态成员变量：**

```python
class MemberCounter:
    members=0
    def inc(self):
        MemberCounter.members+=1 #使用 类名.成员变量
a=MemberCounter()
a.inc()
b=MemberCounter()
b.inc()
print(a.members,b.members) #输出(2,2)
```

**3. 继承：**

```python
class Filter:
    def init(self):
        self.blocked=[]
    def filter(self,sequence): #返回列表sequence中与blocked不相交元素
        return [x for x in sequence if x not in self.blocked]
    
class SPFilter(Filter): #继承Filter
    def init(self):     #重写init
        self.blocked=['SP']

ftr=Filter()
ftr.init()
print(ftr.filter(['SP',123,'xx','SP']))  #输出['SP', 123, 'xx', 'SP']
SPftr=SPFilter()
SPftr.init()
print(SPftr.filter(['SP',123,'xx','SP']))#输出[123, 'xx']
print(issubclass(SPFilter,Filter))       #输出True，判别是否是子类
print(SPFilter.__bases__)                #输出父类
print(isinstance(SPftr,SPFilter)   #True，可见SPftr是SPFilter的实例
print(isinstance(SPftr,Filter))    #True，可见SPftr也是其父类的实例
print(SPftr.__class__)    #输出其所属类
```

**4. 抽象基类：**

```python
from abc import ABC,abstractmethod
class Talker(ABC):
    @abstractmethod #修饰器,这里标记为抽象方法
    def talk(self):
        pass
    def hello(self):
        print('hello')
#tk=Talker() #错误,必须实现所有抽象方法才能够实例化
class MTalker(Talker):
    def talk(self):
        print('haha')
tk=MTalker() #由于实现了抽象方法,可以实例化
tk.talk()
tk.hello()
```



# 八、异常

**1. 主动唤起异常：**

```python
raise Exception                #抛出异常
raise Exception('err:hello')   #抛出异常加句子
class SomeError(Exception): pass #自己定义一个异常
raise SomeError('err:hello')   #抛出自定义异常
```

**2. 捕获异常：**

```python
while True:
    try:
        x=int(input('first:'))
        y=int(input('second:'))
        print(x/y)
    except ZeroDivisionError as e:
        print("Second number can't be zero")
        print(e)
        #raise #如果这里加上raise则异常不会被抑制
        #raise Exception('hello') #也可以唤起其他异常
    except ValueError as e:
        print('value is error')
        print(e)
    except: #一网打尽其他所有异常
        print('something wroing happend...')
    else:  
        break  #未发生上述异常则跳出循环
```

**3. 发生异常时执行清理操作：**

```python
x=None
try:
    x=1/0
finally: #这下面加上结束语句,放在else之后(最后)
    print('Cleaning up')
    del x
```

**4. 异常的传递：**
如果不处理函数中发生的异常，它将向上传播到调用函数的地方，如果那里也未得到处理，则继续向上传播，直至到达主程序，如果主程序也没有异常处理，则终止并显示栈跟踪消息。



# 九、魔法方法特性和迭代器

**1. 构造函数：**

```python
class FooBar:
    def __init__(self,value=42): #__init__引导构造函数，传参带默认值
        self.somevar=value
f1=FooBar()
f2=FooBar('asd')
print(f1.somevar) #42
print(f2.somevar) #asd
```

**2. 析构函数：**
```__del__```为析构函数的魔法方法，在对象被撤销(被垃圾回收)前被调用，但鉴于你无法知道准确的调用时间，建议尽可能不要使用它。

**3. 从子类中使用基类的方法：**

```python
class Bird:
    def __init__(self):
        self.info='I am a bird'
    def introduce(self):
        print(self.info)
class SomeBird(Bird):
    def __init__(self):
        super().__init__()    #使用super()调用基类方法
        self.info+=' made by somebird'
b=Bird()
sb=SomeBird()
b.introduce()  #I am a bird
sb.introduce() #I am a bird made by somebird
```

**4. 序列操作的重载：**

```python
class test:
    def __init__(self):
        pass
    def __setitem__(self,key,value): #调用s[i]=x时
        print('set {} {}' .format(key,value))
    def __getitem__(self,key):       #调用s[i]时
        print('get {}'.format(key))
    def __len__(self):               #调用len(s)时
        print('len')
        return 0
    def __delitem__(self,key):       #调用del s[i]时
        print('del {}'.format(key))
t=test()
t[3]     #输出get 3
t['a']   #输出get a
t[4]=5   #输出set 4 5
len(t)   #输出len
del t[3] #输出del 3
```

**5.  Property特性：**
如果需要将获取方法和设置方法绑定到一个类成员变量中，可以用property

```python
class Rect:
    def __init__(self):
        self.width=0
        self.height=0
    def get_size(self):
        return self.width+1,self.height+1 #返回元组
    def set_size(self,size):
        self.width,self.height=size   #size是一个二维元组或列表，这里拆分
    size=property(get_size,set_size)  #用property设置特性
r=Rect()
r.set_size((1,2))
print(r.get_size()) #(2, 3)
r.size=(3,4)
print(r.size)       #(4, 5)

```

**6. 静态方法和类方法：**

```python
class MyClass:
    def __init__(self, info):
        self.info = info
    @classmethod    #类方法，这里用它能实现多个初始化方法
    def create(cls, info):
        a = cls(info)
        return a
    @staticmethod   #静态方法
    def add(a,b):
        return a+b

c1=MyClass('hello')
c2=MyClass.create('world')
print(c1.info)          #hello
print(c2.info)          #world
print(MyClass.add(1,2)) #3
```

**7. 迭代器：**

```python
class Fibs:
    def __init__(self):
        self.a=0
        self.b=1
    def __next__(self): #实现__next__的对象是迭代器,返回下一个元素
        self.a, self.b = self.b, self.a + self.b
        return self.a
    def __iter__(self): #实现__iter__的对象是可迭代的，返回迭代器本身
        return self
fibs=Fibs()
for f in fibs: #输出斐波那契数列中所有<1000的
    if f>1000:
        break
    print(f)
#下面是使用迭代器iter遍历数据
for it in iter([1,2,3]): 
    print(it)
#1
#2
#3
```

**8. 生成器：**

```python
def my_gen():
     n=1
     print("first")
     # yield区域
     yield n    #每次yeild后保存当前状态

     n+=1
     print("second")
     yield n

     n+=1
     print("third")
     yield n

a=my_gen()
print(next(a)) #1
print(next(a)) #2
print(next(a)) #3
for elem in my_gen():
    print(elem) #123
#注意与[x for x in range(10)]区别，这个为生成列表
a=(x for x in range(10)) #返回一个生成器
for elem in a:
    print(elem) #123456789
```



# 十、模块

**1. 调用模块以及模块的测试：**
我首先在一个文件夹内创建test.py(作为主程序)，md1文件夹，md2文件夹。
再在md1文件夹中创建一个自定义模块md1a.py。
以下为test.py内容：

```python
#这里可以加入搜索目录，否则默认当前目录
import sys
sys.path.append('.\\md1')  #我使用相对目录
import md1a
md1a.say_hello() #主函数执行模块里的内容
```

以下为md1a.py内容：

```python
def say_hello():
    print('hello!')
def test():
    '该模块的一个测试'
    say_hello()
    print('this is a test from md1a.py !')
#如果这个模块以main执行那么就调用测试程序
if __name__ == '__main__': test()
```

当我运行test.py时，输出hello!。当我运行md1a.py时输出hello! this is a test from md1a.py!
同时，md1文件夹下会生成 `__pycache__`文件夹，这相当于编译模块的结果，提高执行效率。

**2. 包：**
前面所述的组织形式有点繁琐，为何不直接将md1文件夹直接视作一个大的模块，这里python提供一种叫做包的组织形式，以文件夹为单位。
我们先在工程文件夹下创建test.py作为主程序，package1文件夹作为包。再在包文件夹下创建`__init__.py` 作为包的定义文件（包含它的文件夹都是包），再放上md1a.py和md1b.py作为包中的两个模块。
`__init__.py` 中只需要加一些初始化定义，如`PI=3.14`。
`md1a.py` 加入一行 `def say_helloa(): print('hello a!')` 只要在这个目录中就行。
`md1b.py` 加入一行 `def say_hellob(): print('hello b!')`
主程序中以下内容以调用包

```python
from package1 import md1a,md1b
md1a.say_helloa()
md1b.say_hellob()
import package1.md1a,package1.md1b
package1.md1a.say_helloa()
package1.md1b.say_hellob()
```

**3. 标准库一些受欢迎的模块：**

```python
#-------------------集合-------------------#
set(range10)  #{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
type({1,2,3}) #<class 'set'>
a={1,2,3}
b={2,3,4}
a|b     #a.union(b)
a&b     #a.intersection(b)
a<=b    #a.issubset(b)
a>=b    #a.issuperset(b)
a-b     #a.difference(b)
a^b     #a.symmetric_difference(b)不相交集合
b=a.copy()
a.add(1)
a.remove(1)
a.clear()
3 in b #判断是否在集合内
#-------------------堆-------------------#
from heapq import *
heap=[]          #用一维列表表示堆
heappush(heap,1) #加入堆
heappop(heap)    #每次取出最小值
heap=[8,5,2,4]
heapify(heap)    #转化为合法小根堆[2, 4, 8, 5]
#-------------------双端队列-------------------#
from collections import deque
q=deque(range(3)) #deque([0, 1, 2])
q.append(6)       #deque([0, 1, 2, 6])
q.appendleft(8)   #deque([8, 0, 1, 2, 6])
q.pop()           #6
q.popleft()       #8
q.rotate(-1)      #左移一位,deque([1, 2, 0])
#-------------------时间-------------------#
import time
time.asctime() #'Tue Jul  3 13:38:02 2018'
#-------------------伪随机数-------------------#
from random import *
random()          #返回一个0~1随机实数
getrandbits(10)   #返回n各随机二进制位
uniform(1,4)      #返回a~b的随机实数
choice([1,2,5,8]) #从序列中随机选择一个
a=[1,2,3,4]
shuffle(a)        #就地打乱一个序列
sample(a,2)       #从序列中随机取出n个
#-------------------存储字典或列表-------------------#
import json
data={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5} #一个字典
with open('data.json', 'w') as f: #保存json文件
    json.dump(data, f)
with open('data.json', 'r') as f: #读取json文件
    data = json.load(f)
```



# 十一、文件

**1. 文本文件的读取和写入：**

```python
#写入
f=open('a.txt','w') #默认文本模式打开
f.write('你好qweasd')
f.close()
#读取
f=open('a.txt','r')
print(f.read(4)) #读取四个字符
print(f.read())  #读取剩下字符
f.close()
#读取行
f=open('a.txt','r')
print(f.readlines())#保存到列表
f=open('a.txt','r')
print(f.readline())#读取一行
print(f.readline())
f.close()
#根据文件指针读取写入行
with open('a.txt','r+') as f:
    a=f.readlines()
    print(a)
    f.writelines('haha')
#行迭代
with open('a.txt','r') as f:
    for line in f:
        print(line)
#字符迭代
with open('a.txt','r') as f:
    char=f.read(1)
    while char:
        print(char)
        char=f.read(1)
```



# 十二、C/C++语言扩展

**1. SWIG下载与安装：**
下载连接：http://www.swig.org/download.html 
下载得到swigwin（这个是win系统）的压缩文件。
将这个文件解压后我放在D:\SWIG\swigwin
并在windows环境变量中加入这个目录，这样cmd下直接可以使用swig命令。

**2. 编写C语言扩展框架：**
为了方便操作，我将一个工程存储在一个文件夹下，该文件夹我把它命名位Demo
文件夹里面包含三个基本文件。
`module_main.c` 要用C语言实现的最上层代码源文件。
`module_main.h` 要用C语言实现的最上层代码头文件，模块里就调用头文件里的东西。
`build.py`  我自己写的一个脚本，运行后输入模块名能自动生成动态库。

**module_main.c：**

```python
#include<stdio.h>
#include<math.h>
#include"module_main.h"

float mySqrt(float in)
{
	return sqrt(in);
}
```

**module_main.h：**

```python
#ifndef _MODULE_MAIN_H
#define _MODULE_MAIN_H

float mySqrt(float in);

#endif
```

**build.py：**

```python
import os
module_name=input('module name:')
setup_py=open('setup.py','w')
setup_py.write(
    """
from distutils.core import setup, Extension
module_name='{0}'
user_not_change='warp.c'
module = Extension(
    '_'+module_name,
    sources=[user_not_change,
             'module_main.c'
             ],
    )
setup(
    name        =module_name,
    version     ='0.0',
    author      ='xytpai',
    description = 'Auto',
    ext_modules = [module],
    py_modules = [module_name],
    )
    """.format(module_name)
    )
setup_py.close()
setup_i=open('setup.i','w')
setup_i.write(
    """
%module {0}  
    """.format(module_name)+
    """
%{
#include "module_main.h"
%}  
%include "module_main.h"
    """
    )
setup_i.close()
os.system(r'rd /s /q build')
os.system(r'swig -python -o warp.c setup.i')
os.system(r'python setup.py build')
rmame=module_name+'.py'
os.system('del /q {0}'.format(rmame))
os.system('del /f /s /q setup.i setup.py warp.c')
```

点击build.py输入要的模块名(我自己取了demo)，生成build文件夹。
拷贝出里面的 .pyd和.py文件。

```python
import demo
demo.mySqrt(4) #输出2.0
```