| [返回主页](index.html) |

---

YAML被广泛用于程序配置文件

#### 1. YAML格式

```yaml
age  : true  # age为键后面为值,布尔值小写
name : huang # 使用引号分割, 键值不可以换行写
languages:
 - Ruby # '-'表示数组, 
 - C
 - Python # 自动识别字符串类型
 - 'woshibiaoji' # 强制字符串类型

# 下面为一个二维列表
 - - Ruby # 注意同一个层级竖直对其
 - Perl
 - - c
 - c++

# 下面为层级结构
# 注意！缩进只能用空格,同一层级竖直对齐
CUDNN: 
  BENCHMARK: true
  DETERMINISTIC: false
  ENABLED: true
# 这里CUDNN为一个字典的键
# 旗下的每一个都是键值

DATA_DIR: '' # 代表空字符串
thisisnone: ~ # 代表None
float : 1.2 # 浮点
int : 12 # 整数
ints : 12. # 浮点
str: 'labor''s day' # 如果内部有单引号则需'转义

# 以上为程序配置文件的YAML常见语法
```

#### 2. Python使用

```python
import yaml
f = open('path', encoding='utf-8')
res =  yaml.load(f)
f.close()
# 导入的res为一个字典类型
```


