| [返回主页](index.html) | [特征提取器](retinanet_extractor.html) | [检测器](retinanet_detector.html) | 锚生成器 | [解析器](retinanet_encoder.html) |  [损失接口](retinanet_loss.html) | [训练过程](retinanet_train.html) | [推理过程](retinanet_inference.html) | [性能评估](retinanet_eval.html)

---



### 1. 锚生成函数

```python
def gen_anchors(a_hw, scales, img_size, first_stride):
    '''return: anchros_yxyx, anchors_yxhw
    anchors_yxyx:  [sum_scale(Hi*Wi*an), 4]  t.float32  ymin, xmin, ymax, xmax
    anchors_yxhw:  [sum_scale(Hi*Wi*an), 4]  t.float32  y, x, height, width
    
    Example:
    
    # 定义最小层级的锚框大小
	a_hw = [
    	[28.0, 28.0],
    	[19.8, 39.6],
    	[39.6, 19.8]
	]
	
	# 一共多少个尺度层级
	# 这些层级在FPN的不同级上预测
	# 层级间大小差2倍
	scales        = 4
	
	# 输入图像大小
	# 2^n+1最佳
	# 奇数尺寸能保证有几个框在正中心
	img_size      = (257, 257) # H,W
	
	# 最开始的步级
	first_stride  = 8
    '''
    anchors_yxyx = []
    anchors_yxhw = []
    stride = first_stride
    an = len(a_hw)
    for scale_id in range(scales):
        fsz_h = (img_size[0]-1) // (first_stride * pow(2, scale_id)) + 1
        fsz_w = (img_size[1]-1) // (first_stride * pow(2, scale_id)) + 1
        anchors_yxyx_i = torch.zeros(fsz_h, fsz_w, an, 4)
        anchors_yxhw_i = torch.zeros(fsz_h, fsz_w, an, 4)
        for h in range(fsz_h):
            for w in range(fsz_w):
                a_y, a_x = h * float(stride), w * float(stride)
                scale = float(stride//first_stride)
                for a_i in range(an):
                    a_h, a_w = scale*a_hw[a_i][0], scale*a_hw[a_i][1]
                    a_h_2, a_w_2 = a_h/2.0, a_w/2.0
                    a_ymin, a_ymax = a_y - a_h_2, a_y + a_h_2
                    a_xmin, a_xmax = a_x - a_w_2, a_x + a_w_2
                    anchors_yxyx_i[h, w, a_i, :] = \
                        torch.Tensor([a_ymin, a_xmin, a_ymax, a_xmax])
                    anchors_yxhw_i[h, w, a_i, :] = \
                        torch.Tensor([a_y, a_x, a_h, a_w])
        stride *= 2
        anchors_yxyx_i = anchors_yxyx_i.view(fsz_h*fsz_w*an, 4)
        anchors_yxhw_i = anchors_yxhw_i.view(fsz_h*fsz_w*an, 4)
        anchors_yxyx.append(anchors_yxyx_i)
        anchors_yxhw.append(anchors_yxhw_i)
    return torch.cat(anchors_yxyx, dim=0), torch.cat(anchors_yxhw, dim=0)
```



### 2. 测试

生成锚的图像左右各一张，左边为 YXYX 生成，右边为 YXHW 生成。

```python
if __name__ == '__main__':
    # 生成锚
	# YXYX 的排布为锚的 ymin, xmin, ymax, xmax
	# YXHW 的排布为锚的 y, x, height, width
	# 锚生整合排列 [sum_scale(Hi*Wi*an), 4]
    A_HW = [
    	[14.0, 14.0],
    	# [19.8, 39.6],
    	# [39.6, 19.8]
	]
    SCALES        = 2
    IMG_SIZE    = (80,300)
    FIRST_STRIDE  = 32
    ANCHORS_YXYX, ANCHORS_YXHW = gen_anchors(A_HW, SCALES, IMG_SIZE, FIRST_STRIDE)
    print(ANCHORS_YXYX.shape)
    print(ANCHORS_YXHW.shape)
    img_1 = torch.zeros(IMG_SIZE[0], IMG_SIZE[1])
    img_2 = torch.zeros(IMG_SIZE[0], IMG_SIZE[1])

    for n in range(ANCHORS_YXYX.shape[0]):
        ymin, xmin, ymax, xmax = ANCHORS_YXYX[n]
        y, x, h, w = ANCHORS_YXHW[n]

        ymin = torch.clamp(ymin, min=0, max=IMG_SIZE[0]-1)
        xmin = torch.clamp(xmin, min=0, max=IMG_SIZE[1]-1)
        ymax = torch.clamp(ymax, min=0, max=IMG_SIZE[0]-1)
        xmax = torch.clamp(xmax, min=0, max=IMG_SIZE[1]-1)
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax) 
        img_1[ymin, xmin:xmax] = 1.0
        img_1[ymax, xmin:xmax] = 1.0
        img_1[ymin:ymax, xmin] = 1.0
        img_1[ymin:ymax, xmax] = 1.0

        _ymin = y - h/2.0
        _xmin = x - w/2.0
        _ymax = y + h/2.0
        _xmax = x + w/2.0
        _ymin = torch.clamp(_ymin, min=0, max=IMG_SIZE[0]-1)
        _xmin = torch.clamp(_xmin, min=0, max=IMG_SIZE[1]-1)
        _ymax = torch.clamp(_ymax, min=0, max=IMG_SIZE[0]-1)
        _xmax = torch.clamp(_xmax, min=0, max=IMG_SIZE[1]-1)
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
