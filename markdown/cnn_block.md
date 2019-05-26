| [返回主页](index.html) | 正交 Dropout 单元 | [RePr 训练框架](cnn_repr.html) | [实验结果](cnn_experiment.html) |

---



### 1. 正交 Dropout 单元

深度学习就像是在管理一家庞大的公司，公司的首要目标是增加收益（降低损失值）<br>
公司里的员工阶层明确，底层人员直接跟数据打交道，上层人员跟下层人员打交道<br>
公司运行一段时间后，管理者（输出层）直接看到了整体收益状况（损失值）<br>
管理者通过下一层员工的表现（输出）来调整自己的决策（权值）<br>
同时，管理者也有义务告诉下一层的员工，该怎么做才能增加收益（梯度）<br>
依次地，上层员工通过下层员工当前的表现（输出），以及由更上层分派下来的指示（梯度），来调整自己的决策（权值）。这其中的问题是，决策的调整策略是贪心的，很依赖每个员工的初始决策（性格偏好）。如果同一阶层的一些员工一开始就有相似的性格偏好，那他们在之后会对同一问题用同一套调整策略，他们的作用相同。<br>
假设每一个员工都有无限的精力，为了让这个公司更有效率，在运营一段时间后，需要淘汰掉一些作用相同（权值相似）的员工，并用新一批性格迥异的员工来代替他们的位置。<br>

```python
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, outplanes, stride=1, drop=False):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.drop = drop
        self.projection = False
        if (stride > 1) or (inplanes != outplanes):
            self.projection = True
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(planes, planes, kernel_size=3, 
        	stride=stride, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)        
        self.bn_3 = nn.BatchNorm2d(outplanes)
        if self.projection:
            self.conv_prj = nn.Conv2d(inplanes, outplanes, kernel_size=1, 
            	stride=stride, bias=False)
            self.bn_prj = nn.BatchNorm2d(outplanes)
        # 单元其他结构沿用了Resnet的单元架构
        # 不同的是, 增加了drop项
        # 如果被设置为开启, 则滤波器会被选择性输出
        # mask_drop 中被设置为1的滤波器不贡献输出
        if self.drop:
            self.register_buffer('mask_drop', torch.zeros(planes).byte())
            
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.drop:
            # out [b,c,h,w]
            # 将out中被置位的滤波器输出清0, 以达到丢弃的效果
            out[:, self.mask_drop] = 0
            # 为了不让输出随着丢弃数目变化而波动
            # 需要对其进行规范化, 即放大其余滤波器输出的值
            sum_ndrop = torch.sum(~self.mask_drop).float()
            keep_rate = float(self.mask_drop.shape[0]) / sum_ndrop
            out = out * keep_rate
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        if self.projection:
            residual = self.bn_prj(self.conv_prj(x))
        out = self.relu(out + residual)
        return out
    
    def drop_enable(self):
        '''
        该操作选择哪些滤波器被丢弃
        需要先训练一段时间
        然后使用某种方法去评估这些滤波器的相似度
        淘汰掉那些高相似度的滤波器，再训练一段时间
        '''
        assert self.drop, 'Make sure drop=True'
        # 这个函数输入为同层滤波器的指针
        # dp_vector:一维向量，反应每个滤波器的相关度
        # topn: 上述分数值中前多少高的滤波器会被淘汰
        dp_vector, topn = get_correlation(self.conv_2)
        drop_idx = torch.topk(dp_vector, topn, largest=True)[1]
        self.mask_drop[:] = 0
        self.mask_drop[drop_idx] = 1
        

    def drop_disable(self):
        '''
        该操作将那些被丢弃的滤波器进行重新初始化，并将它们激活
        在淘汰掉滤波器后，需要训练一段时间，再重新激活他们
        '''
        assert self.drop, 'Make sure drop=True'
        # 输入同层滤波器的指针、相应BN层的指针、丢弃掩码
        # 这个函数会重新初始化那些被丢弃的滤波器
        init_filters(self.conv_2, self.bn_2, self.mask_drop)
        self.mask_drop[:] = 0
```



### 2. 相关度评价函数

```python
def get_correlation(conv):
    pass
```



### 3. 正交初始化

```python
def init_filters(conv, bn, mask_drop):
    pass
```

