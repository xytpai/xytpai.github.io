| [返回主页](index.html) | [特征提取器](retinanet_extractor.html) | [检测器](retinanet_detector.html) | 锚生成器 | [损失接口](retinanet_loss.html)

---



### 1. 全局参数

```python
# 定义最小层级的锚框大小
A_HW = [
    [28.0, 28.0],
    [19.8, 39.6],
    [39.6, 19.8]
]
# 一共多少个尺度层级
# 这些层级在FPN的不同级上预测
# 层级间大小差2倍
SCALES        = 4
# 输入图像大小, 本实验限制为正方形
# 边长限定为 2^n+1
# 这样能保证平移尺度不变并存在绝对中心
IMG_SIZE      = 257 # 2^n+1
# 最开始的步级
FIRST_STRIDE  = 8
# 生成锚
# YXYX 的排布为锚的 ymin, xmin, ymax, xmax
# YXHW 的排布为锚的 y, x, height, width
# 锚生整合排列 [sum_scale(Hi*Wi*an), 4]
ANCHORS_YXYX, ANCHORS_YXHW = gen_anchors(
    A_HW, SCALES, IMG_SIZE, FIRST_STRIDE
)
# ANCHORS_YXYX 与 ANCHORS_YXHW 为全局参数以减少计算量
```



### 2. 锚生成函数

```python
def gen_anchors(a_hw, scales, img_size, first_stride):
    '''return: anchros_yxyx, anchors_yxhw
    anchors_yxyx:   [an_alls, 4]  torch.float32  ymin, xmin, ymax, xmax
    anchors_yxhw:   [an_alls, 4]
    '''
    anchors_yxyx = []
    anchors_yxhw = []
    stride = first_stride
    an = len(a_hw)
    for scale_id in range(scales):
        fsz = ((img_size-1) // first_stride) // pow(2, scale_id) + 1
        anchors_yxyx_i = torch.zeros(fsz, fsz, an, 4)
        anchors_yxhw_i = torch.zeros(fsz, fsz, an, 4)
        for h in range(fsz):
            for w in range(fsz):
                a_y, a_x = h * float(stride), w * float(stride)
                scale = float(stride//first_stride)
                for a_i in range(an):
                    a_h, a_w = scale*a_hw[a_i][0], scale*a_hw[a_i][1]
                    a_h_2, a_w_2 = a_h/2.0, a_w/2.0
                    a_ymin, a_ymax = a_y - a_h_2, a_y + a_h_2
                    a_xmin, a_xmax = a_x - a_w_2, a_x + a_w_2
                    anchors_yxyx_i[h, w, a_i, :] = torch.Tensor([a_ymin, a_xmin, a_ymax, a_xmax])
                    anchors_yxhw_i[h, w, a_i, :] = torch.Tensor([a_y, a_x, a_h, a_w])
        stride *= 2
        anchors_yxyx_i = anchors_yxyx_i.view(fsz*fsz*an, 4)
        anchors_yxhw_i = anchors_yxhw_i.view(fsz*fsz*an, 4)
        anchors_yxyx.append(anchors_yxyx_i)
        anchors_yxhw.append(anchors_yxhw_i)
    return torch.cat(anchors_yxyx, dim=0), torch.cat(anchors_yxhw, dim=0)
```



### 3. 锚测试函数

生成锚的图像左右各一张，左边为 YXYX 生成，右边为 YXHW 生成。

```python
if __name__ == '__main__':
    print(ANCHORS_YXYX.shape)
    print(ANCHORS_YXHW.shape)
    img_1 = torch.zeros(IMG_SIZE, IMG_SIZE)
    img_2 = torch.zeros(IMG_SIZE, IMG_SIZE)

    for n in range(ANCHORS_YXYX.shape[0]):
        ymin, xmin, ymax, xmax = ANCHORS_YXYX[n]
        y, x, h, w = ANCHORS_YXHW[n]
        
        ymin = torch.clamp(ymin, min=0, max=IMG_SIZE-1)
        xmin = torch.clamp(xmin, min=0, max=IMG_SIZE-1)
        ymax = torch.clamp(ymax, min=0, max=IMG_SIZE-1)
        xmax = torch.clamp(xmax, min=0, max=IMG_SIZE-1)
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax) 
        img_1[ymin, xmin:xmax] = 1.0
        img_1[ymax, xmin:xmax] = 1.0
        img_1[ymin:ymax, xmin] = 1.0
        img_1[ymin:ymax, xmax] = 1.0

        _ymin = y - h/2.0
        _xmin = x - w/2.0
        _ymax = y + h/2.0
        _xmax = x + w/2.0
        _ymin = torch.clamp(_ymin, min=0, max=IMG_SIZE-1)
        _xmin = torch.clamp(_xmin, min=0, max=IMG_SIZE-1)
        _ymax = torch.clamp(_ymax, min=0, max=IMG_SIZE-1)
        _xmax = torch.clamp(_xmax, min=0, max=IMG_SIZE-1)
        _ymin, _xmin, _ymax, _xmax = int(_ymin), int(_xmin), int(_ymax), int(_xmax) 
        img_2[_ymin, _xmin:_xmax] = 1.0
        img_2[_ymax, _xmin:_xmax] = 1.0
        img_2[_ymin:_ymax, _xmin] = 1.0
        img_2[_ymin:_ymax, _xmax] = 1.0

    plt.subplot(1,2,1)
    plt.imshow(img_1.numpy())
    plt.subplot(1,2,2)
    plt.imshow(img_2.numpy())
    plt.show()
```





---

### 参考文献

Faster R-CNN: https://arxiv.org/abs/1506.01497
