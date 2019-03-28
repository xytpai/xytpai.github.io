| [返回主页](index.html) |

---

这里使用 Ubuntu18.04 LTS

#### 1. win10中为Ubuntu分配磁盘空间
```python
1. 我的电脑右键 -> 管理 -> 存储.磁盘管理
2. 空出一个分区，操作后这个分区显示未分配
3. 制作系统盘从系统盘中启动
4. 在配置中一路继续（注意这个 alongside windows Boot）
   注意，不要设置免密登入! 否则有可能进不了tty
5. 安装完成后会重启（拔掉安装盘）
```

#### 2. 更新 Ubuntu 所有软件
```python
1. 开始菜单输入 software update 自动更新
   中途可能失败，失败则重复之前步骤
2. sudo apt-get upgrade
3. sudo reboot
   如果网卡有问题关机后断电等一下再启动
```

#### 3. 安装 Nvidia 显卡驱动
```python
1. sudo ubuntu-drivers autoinstall
2. sudo reboot
3. 检查是否安装成功 nvidia-smi
4. 安装最新显卡驱动（由于一般使用较新的软件因此必需）
   http://www.nvidia.com/Download/index.aspx 下载对应的驱动
   开始搜索 software & Updayes 转到附加驱动选择非英伟达显驱应用重启，需连网
   重启后输入: sudo sh NVIDIA..run
   前几个按照默认选项，直到出现一个缺少某些库的提示这时需要install&override
   sudo reboot
   输入 nvidia-smi 可看到最高兼容到的 cuda 版本。
```

#### 4. 安装 CUDA
```python
https://developer.nvidia.com/cuda-toolkit-archive
https://developer.nvidia.com/rdp/cudnn-download
1. 在上面链接下载 CUDA 目前版本10.0 并下载配套CUDNN
   下载CUDNN需要登入，两个加在一起2GB左右
2. sudo sh cuda..run # 安装Cuda
   如果有问题可能是文件出错了再下载一遍
   注意安装时去掉 Driver
3. sudo gedit ~/.bashrc 加入下面两行
   export PATH=/usr/local/cuda-10.0/bin${PATH:+:$PATH}
   LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
5. reboot 使用 nvcc -V 测试
```

#### 5. 安装 CUDNN
```python
1. cp filename.xxx filename.tgz
2. tar -zxvf filename.tgz
3. sudo cp cuda/include/cudnn.h /usr/local/cuda/include
4. sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
5. sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

#### 6. 安装 Pytorch
```python
1. sudo apt-get install python3-pip
2. https://pytorch.org/get-started/locally/ 选择对应版本安装
3. pip3 install numpy
4. python3
   import torch
   x = torch.tensor([1.0])
   x = x.cuda()
   print(x) # 查看cuda库是否可用
   from torch.backends import cudnn
   print(cudnn.is_acceptable(x)) # 查看cudnn库是否可用
5. 如果提示驱动版本太老则去官网下载 
   http://www.nvidia.com/Download/index.aspx
   
```

#### 7. 安装中文输入法
```python
1. sudo apt-get install fcitx-bin
2. sudo apt-get install fcitx-table
3. 开始键查找 language 选择 fcitx
4. sudo reboot
5. 下载搜狗拼音 https://pinyin.sogou.com/linux/
6. 右上角当前输入设置中加入 pinyin
```

#### 8. 安装 VSCode
```python
在 https://code.visualstudio.com/ 中安装deb文件
点击后直接安装
```

#### 9. 安装 Chrome 浏览器
```python
1. sudo wget http://www.linuxidc.com/files/repo/google-chrome.list -P /etc/apt/sources.list.d/
2. wget -q -O - https://dl.google.com/linux/linux_signing_key.pub  | sudo apt-key add -
3. sudo apt update
4. sudo apt install google-chrome-stable
```

#### 10. 安装 git 代码托管
```python
1. sudo apt-get install git
2. git config --global user.name "username"
3. git config --global user.email "email"
4. check: git clone https://github.com/username/project.git
5. write  push.sh:
提交方法如下:
git add .
git commit -m 'personal modify'
git push
```


