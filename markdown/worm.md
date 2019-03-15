| [返回主页](index.html) |

---

安装requests库： pip install requests

#### 1. 基本操作

```python
import requests
req = requests.get('http://www.baidu.com')
print(req.status_code) # 200则成功,404或其他标识失败
print(req.text) # 网页内容的字符串形式
print(req.encoding) # 从HTTP的HEADER中猜测出的编码方式
print(req.apparent_encoding) # 从网页内容文本中看出的编码方式
print(req.content) # HTTP响应内容的二进制形式, 可以解析图像

req.encoding = 'utf-8' # 改变编码模式输出
print(req.text)
```

#### 2. 抛出的异常

```python
requests.ConnectionError  # 网络连接异常，如DNS查询失败、拒绝连接
requests.HTTPError        # HTTP错误异常
requests.URLRequired      # URL缺失异常
requests.TooManyRedirects # 超过最大重定向次数,对复杂连接访问
requests.ConnectTimeout   # 连接服务器超时异常,整个过程
requests.Timeout          # 请求URL超时异常,仅指与服务器的连接
```

#### 3. 爬网页通用框架

```python
def getHTMLText(url):
	try:
		req = requests.get(url, timeout=30)
		req.raise_for_status() # 如果状态不是200产生异常
		req.encoding = req.apparent_encoding
		return req.text
	except:
		return 'error'
```

#### 4. HTTP协议相关操作

```python
GET    # 获取URL位置的资源
HEAD   # 获取URL位置资源的响应消息报告，即头部信息
POST   # 请求向URL位置的资源后附加新的数据
PUT    # 请求向URL位置存储一个资源，覆盖原URL位置的资源
PATCH  # 请求局部更新URL位置的资源
DELETE # 请求删除URL位置存储的资源
```

#### 5. request操作

```python
# 使用参数筛选资源
kv = {'key1':'value1', 'key2':'value2'}
req = requests.request('GET','http://..', params=kv)

# 存储数据
req = requests.request('POST','http://..', data=kv)

# 提交json格式
req = requests.request('POST','http://..', json=kv)

# 修改提交数据头
hd = {'user-agent':'Chrome/10'} # 指明我是用Chrome10浏览器访问的
req = requests.request('POST','http://..', headers=kv)
req.request.headers # 查看发的头

# 其他输入字段
cookies: 字典
auth: 元组，支持HTTP认证功能
files: 字典,传输文件时使用的字段
如: 	fs={'file':open('data.xls', 'rb')}
timeout: 超时时间，以秒为单位
proxies: 字典, 设定访问代理服务器，可以增加登入认证
如:
pxs = {'http': 'http://user:pass@10.10.10.1:1234'
	   'https': 'https://10.10.10.1:3211'}
可以有效隐藏爬虫的IP地址信息
allow_redirects: 布尔,默认True,是否允许对url重定向
stream: 布尔默认True, 获取内容立即下载
verify: 布尔默认True, 认证SSL证书开关
cert: 本地SSL证书路径
```

#### 6. Robots协议

```python
在网站根目录下的robots.txt文件，告知哪些页面可被爬虫抓取
user-agent: 告知哪些爬虫
Disallow: 不能访问的资源目录
```

#### 7. 百度关键词提交

```python
# 首先查看百度搜索格式/s?wd=keywords
kv = {'wd': 'Python'}
req = requests.get('http://www.baidu.com/s', params=kv)
req.request.url # 查看发给百度的url是什么
```

#### 8. 网络图片的爬取

```python
path = 'C://abc.jpg'
url = 'http://... jpg'
req = requests.get(url)
req.status_code # 200 则成功
with open(path, 'wb') as f:
	f.write(req.content)
	f.close()
```

#### 9. HTML解析

```python
pip install beautifulsoup4 # 安装
from bs4 import BeautifulSoup

req = requests.get('http://www.baidu.com/s?wd=python')
soup = BeautifulSoup(req.text, 'html.parser') # 解析
print(soup.prettify()) # 显示全部页面

# HTML结构
<p class='title'> content </p>
# 这个p是标签，前面的<>里有标签的属性(用空格区分),content内容,最后标签结尾
# 这个结构是可以归纳为标签树

soup.title # 浏览器左上方显示信息的内容
soup.a # 如果有多个只返回第一个 soup.a.name 返回标签名字
soup.a.parent.name # a的父标签的名字, 字符串
soup.a.attrs # 标签a的属性们,字典,可以使用字典提取方式提取
soup.a.string # 标签中的字符串信息

soup.head.contents # 获取儿子节点信息，返回的是列表
len(soup.head.contents) # 儿子数
soup.head.contents[1] # 输出儿子某个标签里的全部信息
for parent in soup.head.parents:
	print(parent.mame) # 一个个打印祖先节点
# 还有.children .desendants 都是迭代类型

# 同一父亲节点下的平行遍历
soup.head.next_sibling # 下一个兄弟节点
soup.a.previous_sibling # 上一个兄弟节点

```

#### 10. 提取标签内链接

```python
for link in soup.find_all('a'): # 遍历所有a标签
	print(link.get('herf')) # 没有返回None
```

#### 11. find_all

```python
for tag in soup.find_all(True):
	print(tag.name) # 输出所有标签,可重复

import re # 正则表达式
for tag in soup.find_all(re.compile('b')):
	print(tag.name) # 以b开头
	
soup.find_all('p', 'course') # 带有course属性值的p标签
soup.find_all(id='link1') # 属性中id为link1的元素

```

