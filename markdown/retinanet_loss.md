| [返回主页](index.html) | [数据集接口](retinanet_dataset.html) | [特征提取器](retinanet_extractor.html) | [检测器](retinanet_detector.html) | [锚生成器](retinanet_anchors.html) | [解析器](retinanet_encoder.html) | 损失接口 | [训练过程](retinanet_train.html) | [推理过程](retinanet_inference.html) | [性能评估](retinanet_eval.html) |

---



### 1. 损失接口 Focal Loss

Focal Loss 加了一个乘项对各类别的损失贡献进行调整，减小易分类样本的贡献。<br>
值得注意的是，假设负例占据大多数，那么迭代一开始需要让正例的模型输出尽可能小。<br>
因此作者将回归输出层偏置进行了特定的初始化。 <br>为什么将0.75分配给了占大多数的负例？因为实验表现比较好。
$$
p_t=\begin{cases}
p & y=1 \\
1-p & otherwise
\end{cases} \\
FL(p_t) = -\frac1{N_p}\sum\alpha_t(1-p_t)^\gamma log(p_t) \\
\alpha_t = \begin{cases}
0.25 & y=1 \\
0.75 & otherwise
\end{cases} , \ \ \gamma=2
$$

```python
def focal_loss_detection(
    feature_cls, feature_reg, 
    targets_cls, targets_reg,
    alpha=0.25, gamma=2,
    factor_cls=1.0, factor_reg=10.0):
    '''
    feature_cls: [b, an, classes] t.float
    feature_reg: [b, an, 4]       t.float
    targets_cls: [b, an]          t.long
    targets_reg: [b, an, 4]       t.float
    '''
    b, an, classes = feature_cls.shape[0:3]
    feature_cls = feature_cls.view(b*an, classes)
    feature_reg = feature_reg.view(b*an, 4)
    targets_cls = targets_cls.view(b*an)
    targets_reg = targets_reg.view(b*an, 4)
    # 计算分类损失
    # 首先进行排除
    mask_cls = targets_cls > -1
    feature_reg = feature_reg[mask_cls]
    targets_reg = targets_reg[mask_cls]
    feature_cls = feature_cls[mask_cls]
    # 得到正负例概率
    p = feature_cls.sigmoid() # [S+-, classes]
    # 拿到标签的OneHot编码
    targets_cls = targets_cls[mask_cls].to(feature_cls.device) #[S+-]
    one_hot = torch.zeros(feature_cls.shape[0], 
            1 + classes).to(feature_cls.device).scatter_(1, targets_cls.view(-1,1), 1) # [S+-, 1+classes]
    one_hot = one_hot[:, 1:] # [S+-, classes]
    # 计算pt
    pt = p*one_hot + (1.0-p)*(1.0-one_hot)
    # 计算乘项
    w = alpha*one_hot + (1.0-alpha)*(1.0-one_hot)
    w = w * torch.pow((1.0-pt), gamma)
    # 计算正例数目
    mask_reg = targets_cls > 0
    num_pos = float(torch.sum(mask_reg))
    # 得到分类损失值
    assert num_pos>0, 'Make sure every image has assigned anchors.'
    loss_cls = torch.sum(-w * pt.log()) / (num_pos)
    # 下面计算回归损失
    # 选中所有正例
    feature_reg = feature_reg[mask_reg]
    targets_reg = targets_reg[mask_reg].to(feature_cls.device)
    # 计算回归损失值
    loss_reg = F.smooth_l1_loss(feature_reg, targets_reg, reduction='sum')
    loss = (factor_cls*loss_cls + factor_reg*loss_reg) / num_pos
    return loss
```





---

### 参考文献

Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002 <br>
代码: https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
