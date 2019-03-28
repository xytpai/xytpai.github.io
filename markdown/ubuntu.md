| [返回主页](index.html) |

---

这里使用 Ubuntu18.04 LTS

#### 1. win10中为Ubuntu分配磁盘空间
```python
1. 我的电脑右键 -> 管理 -> 存储.磁盘管理
2. 空出一个分区，操作后这个分区显示未分配
3. 制作系统盘从系统盘中启动
4. 在配置中一路继续（注意这个 alongside windows Boot）
5. 安装完成后会重启（拔掉安装盘）
```

#### 2. 更新 Ubuntu 所有软件
```python
1. 开始菜单输入 software update 自动更新
   中途可能失败，失败则重复之前步骤
2. sudo apt-get upgrade
3. sudo reboot
```

#### 3. 安装 Nvidia 显卡驱动
```python
1. sudo ubuntu-drivers autoinstall
2. sudo reboot
3. 检查是否安装成功 nvidia-smi
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
2. tar -zxvf cudnn-10.0-linux-x64-v7.tgz
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
```



