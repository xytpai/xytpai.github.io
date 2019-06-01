| [返回主页](index.html) | [数据集接口](retinanet_dataset.html) | [特征提取器](retinanet_extractor.html) | [检测器](retinanet_detector.html) | [锚生成器](retinanet_anchors.html) | 解析器 |  [损失接口](retinanet_loss.html) | [训练过程](retinanet_train.html) | [性能评估](retinanet_eval.html) |

---



### 1. IOU计算函数

```python
def box_iou(box1, box2, eps=1e-10):
    '''
    box1:   [n,4] t.float32
    box2:   [m,4] t.float32
    return: [n,m] t.float32
    4:      y_min,x_min,y_max,x_max
    '''
    tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
    br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
    hw = (br-tl+eps).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    area1 = (box1[:,2]-box1[:,0]+eps) * (box1[:,3]-box1[:,1]+eps)  # [n,]
    area2 = (box2[:,2]-box2[:,0]+eps) * (box2[:,3]-box2[:,1]+eps)  # [m,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou
```



### 2. 数据解析器

网络输出、标签、锚框三者的对接接口。<br>
训练时，标签与锚框编码成一个张量，这个张量与网络输出即可送入损失接口计算。<br>
推理时，网络输出与锚框解码成一个张量，这个张量之后经过非极大抑制后得到输出目标框。

```python
from anchors import gen_anchors # 锚生成器


class Encoder:
    def __init__(self, train_iou_th=(0.3, 0.5), train_size=(257, 257)):
        # TODO: 对锚框进行规划
        # train_size 仅对训练时有效
        # 训练时图像大小固定
        # 好处是固定的大小速度更快
        self.a_hw = [
            [28.0, 28.0],
            [19.8, 39.6],
            [39.6, 19.8]
        ]
        self.scales = 4
        self.first_stride = 8
        # 以下仅在训练时起作用
        self.train_size = train_size
        # 定义IOU阈值
        # 小于th[0]为背景
        # 大于th[1]为前景
        self.train_iou_th = train_iou_th
        # =================
        self.train_anchors_yxyx, self.train_anchors_yxhw = \
            gen_anchors(self.a_hw, self.scales, self.train_size, self.first_stride)
        self.train_an = self.train_anchors_yxyx.shape[0]

    def encode(self, label_class, label_box):
        # 标签与锚框编码成输出张量
        # label_class: ([N1],[N2],...[Nb])       t.long
        # label_box:   ([N1,4],[N2,4],...[Nb,4]) t.float32 
        # 这里 label_class 如果为 0 则认为是背景
        # 这里 label_box 的第二维表示: ymin, xmin, ymax, xmax
        # 本实验认为标签来自内存上的标签，而非显卡上的
        # 编码后再进显存与网络输出计算Loss
        label_class_out = []
        label_box_out   = []
        _ay_ax = self.train_anchors_yxhw[:, :2]
        _ah_aw = self.train_anchors_yxhw[:, 2:]
        for b in range(len(label_class)):
            # 类别编号为-1表示计算时舍弃
            label_class_out_b = torch.full((self.train_an,), -1).long()
            label_box_out_b = torch.zeros(self.train_an, 4)
            # 计算IOU矩阵, 行代表每一个锚框，列代表每一个标签框
            iou = box_iou(self.train_anchors_yxyx, label_box[b]) # [an, Nb]
            if (iou.shape[1] <= 0):  # 如果这张图中没有可用的框
                label_class_out.append(label_class_out_b)
                label_box_out.append(label_box_out_b)
                continue
            # 得到正反例掩码
            iou_pos_mask = iou > self.train_iou_th[1] # [an, Nb]
            iou_neg_mask = iou < self.train_iou_th[0] # [an, Nb]
            # 得到对每一个锚框来说最匹配的标签框
            label_select = torch.argmax(iou, dim=1) # [an]
            # 得到对每一个标签框来说最匹配的锚框
            anchors_select = torch.argmax(iou, dim=0) # [Nb]
            # 对于某一锚框，只要有一个标签框与其IOU大于阈值就为正例
            anchors_pos_mask = torch.max(iou_pos_mask, dim=1)[0].byte() # [an]
            # 对于某一锚框，所有标签框与其IOU全部小于阈值就为反例
            anchors_neg_mask = torch.min(iou_neg_mask, dim=1)[0].byte() # [an]
            # 先对反例分配背景类别
            label_class_out_b[anchors_neg_mask] = 0
            # 为了不出现无正例情况, 与标签框最匹配的锚框当作正例
            # 注意这一步分配先于后面的分配
            label_class_out_b[anchors_select] = label_class[b]
            # 再对正例分配相应类别
            label_select_1 = label_select[anchors_pos_mask]
            label_class_out_b[anchors_pos_mask] = label_class[b][label_select_1]
            # 下面计算需要回归的值
            # 每个锚框对应的网络输出为4个向量 f1,f2,f3,f4
            # f1 -> (lb_y - a_y) / a_h
            # f2 -> (lb_x - a_x) / a_w
            # f3 -> log(lb_h / a_h)
            # f4 -> log(lb_w / a_w)
            # 需要对两套选择方案各计算一次
            lb_yxyx_2 = label_box[b] # [Nb, 4]
            ay_ax = _ay_ax[anchors_select]
            ah_aw = _ah_aw[anchors_select]
            lb_ymin_xmin_2, lb_ymax_xmax_2 = lb_yxyx_2[:, :2], lb_yxyx_2[:, 2:]
            lbh_lbw_2 = lb_ymax_xmax_2 - lb_ymin_xmin_2
            lby_lbx_2 = (lb_ymin_xmin_2 + lb_ymax_xmax_2)/2.0
            f1_f2_2 = (lby_lbx_2 - ay_ax) / ah_aw
            f3_f4_2 = (lbh_lbw_2 / ah_aw).log()
            label_box_out_b[anchors_select] = torch.cat([f1_f2_2, f3_f4_2], dim=1)
            # 需要对两套选择方案各计算一次
            lb_yxyx_1 = label_box[b][label_select_1] # [S, 4]
            ay_ax = _ay_ax[anchors_pos_mask]
            ah_aw = _ah_aw[anchors_pos_mask]
            lb_ymin_xmin_1, lb_ymax_xmax_1 = lb_yxyx_1[:, :2], lb_yxyx_1[:, 2:]
            lbh_lbw_1 = lb_ymax_xmax_1 - lb_ymin_xmin_1
            lby_lbx_1 = (lb_ymin_xmin_1 + lb_ymax_xmax_1)/2.0
            f1_f2_1 = (lby_lbx_1 - ay_ax) / ah_aw
            f3_f4_1 = (lbh_lbw_1 / ah_aw).log()
            label_box_out_b[anchors_pos_mask] = torch.cat([f1_f2_1, f3_f4_1], dim=1)
            # 加入列表
            label_class_out.append(label_class_out_b)
            label_box_out.append(label_box_out_b)
        # 堆积为 class_b:[b, an]  box_b:[b, an, 4]
        class_targets = torch.stack(label_class_out, dim=0)
        box_targets = torch.stack(label_box_out, dim=0)
        return class_targets, box_targets

    def decode(self, cls_out, reg_out, img_size):
        # cls_out:[b, an, classes]  reg_out:[b, an, 4]
        # 推理时对网络输出解码成目标框
        # 推理时不固定图像大小，因此需要 img_size=(H,W)
        # 解码输出: 
        #   cls_i_preds:[b, an]  t.long  响应最高的类别
        #   cls_p_preds:[b, an]  t.float 响应最高类别的置信度
        #   reg_preds:[b, an, 4] t.float 框回归值:ymin,xmin,ymax,xmax
        cls_p_preds, cls_i_preds = torch.max(cls_out.sigmoid(), dim=2)
        cls_i_preds = cls_i_preds + 1
        anchors_yxyx, anchors_yxhw = \
            gen_anchors(self.a_hw, self.scales, img_size, self.first_stride)
        ay_ax = anchors_yxhw[:, :2]
        ah_aw = anchors_yxhw[:, 2:]
        reg_preds = []
        for b in range(cls_out.shape[0]):
            # 下面计算解码的值
            # 每个锚框对应的网络输出回归向量 f1,f2,f3,f4
            # y = f1 * a_h + a_y
            # x = f2 * a_w + a_x
            # h = f3.exp() * a_h
            # w = f4.exp() * a_w
            # ymin, xmin = y-h/2, x-w/2
            # ymax, xmax = y+h/2, x+w/2
            f1_f2 = reg_out[b, :, :2]
            f3_f4 = reg_out[b, :, 2:]
            y_x = f1_f2 * ah_aw + ay_ax
            h_w = f3_f4.exp() * ah_aw
            ymin_xmin = y_x - h_w/2.0
            ymax_xmax = y_x + h_w/2.0
            ymin_xmin_ymax_xmax = torch.cat([ymin_xmin, ymax_xmax], dim=1)
            reg_preds.append(ymin_xmin_ymax_xmax)
        # 压成 reg_preds:[b, an, 4] t.float  ymin,xmin,ymax,xmax
        reg_preds = torch.stack(reg_preds, dim=0)
        return cls_i_preds, cls_p_preds, reg_preds
```





---

### 参考文献

代码: https://github.com/kuangliu/pytorch-retinanet/blob/master/encoder.py
