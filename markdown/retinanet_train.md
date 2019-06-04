| [返回主页](index.html) | [数据集接口](retinanet_dataset.html) | [特征提取器](retinanet_extractor.html) | [检测器](retinanet_detector.html) | [锚生成器](retinanet_anchors.html) | [解析器](retinanet_encoder.html) | [损失接口](retinanet_loss.html) | [评估接口](retinanet_eval.html) | 训练流程 |

---



### 1. 特征提取器预训练

这个训练是为了初始化特征提取器的参数，并且确定 BatchNorm 层的均值与方差。

```python
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder 
from extractor import Extractor



# ===============
# TODO: 确定各参数
# ===============
load = False  # 是否使用之前的参数
save = True   # 是否储存新参数
epoch_num = [150000, 300000, 550000] # 迭代步数区间
# 150k-batch:lr=0.1; 150k-batch:lr=0.01; 250k-batch:lr=0.001
step_save = 200    # 多少迭代步后储存
step_eval = 50     # 多少迭代步后评估并记录
lr = 0.1           # 初始学习率
lr_decay = 0.1     # 每一个epoch后权重衰减比例
nbatch_train = 128 # 训练batch大小
nbatch_eval  = 128 # 评估batch大小，由于评估与训练占显存量不一样
size = 224         # 使用多少大小的图像输入
device = [8,9]     # 定义(多)GPU编号列表, 第一个为主设备
root_train = '/home1/xyt/dataset/ILSVRC2012/train'
root_eval  = '/home1/xyt/dataset/ILSVRC2012/val'
# ===============



# 定义数据集增强或转换步骤
transform_train = transforms.Compose([
    transforms.Resize(480),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size, scale=(0.53, 1.0)),
    transforms.ColorJitter(brightness=0.03,contrast=0.03,saturation=0.03,hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_eval = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



# 定义数据集路径
dataset_train = ImageFolder(root_train, transform=transform_train)
dataset_eval = ImageFolder(root_eval, transform=transform_eval)
loader_train = torch.utils.data.DataLoader(dataset_train, 
    batch_size=nbatch_train, shuffle=True, num_workers=0)
loader_eval = torch.utils.data.DataLoader(dataset_eval, 
    batch_size=nbatch_eval, shuffle=True, num_workers=0) # 需要打乱



# 准备网络
net = Extractor()
device_n = len(device)
if device_n > 1: # 多GPU
    net = nn.DataParallel(net, device_ids=device)
    net = net.cuda(device[0]) # 先将网络参数传到主设备
else: # Single-GPU
    net = net.cuda(device[0])
device_out = 'cuda:%d' % (device[0])
log_train_acc = [] # train准确率记录
log_eval_acc = []  # eval准确率记录
if load:
    net.load_state_dict(torch.load('net_e.pkl', map_location=device_out))
    log_train_acc = list(np.load('log_train_acc.npy'))
    log_eval_acc  = list(np.load('log_eval_acc.npy'))
criterion = nn.CrossEntropyLoss()



# 主循环
step_id = 0 # 记录目前的步数
break_flag = False
for epoch_id in range(len(epoch_num)):
    while True:
        # Train
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, 
            weight_decay=0.0001, momentum=0.9)
        for i, (inputs, labels) in enumerate(loader_train):
            # Forward and Backward and Optimize
            if device_n < 2: # 多GPU会自动进行cuda()操作
                inputs = inputs.cuda(device[0])
            labels = labels.cuda(device[0]) # 标签只能进主设备
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            Y_pred = torch.max(outputs,1)[1].cpu()
            labels = labels.cpu()
            acc = float(sum(np.array(Y_pred==labels)))/len(labels)
            print('step:%d,loss:%f,acc:%f' % (step_id, loss, acc))
            loss.backward()
            optimizer.step()
            step_id += 1 # 每完成一部step都会自增
            # Eval
            if (step_id%step_eval == (step_eval-1)):
                log_train_acc.append(float(acc))
                net.eval()
                for i, (inputs, labels) in enumerate(loader_eval):
                    if device_n < 2:
                        inputs = inputs.cuda(device[0])
                    labels = labels.cuda(device[0])
                    outputs = net(inputs)
                    Y_pred = torch.max(outputs,1)[1].cpu()
                    labels = labels.cpu()
                    acc = float(sum(np.array(Y_pred==labels)))/len(labels)
                    log_eval_acc.append(float(acc))
                    net.train()
                    # 随机采一次后直接跳出
                    # 因此尽量扩大测试batch大小
                    break
            # Save
            if (step_id%step_save == (step_save-1)) and save:
                torch.save(net.state_dict(),'net_e.pkl')
                if len(log_train_acc)>0:
                    np.save('log_train_acc.npy', log_train_acc)
                    np.save('log_eval_acc.npy', log_eval_acc)
            # Break inner
            if step_id >= epoch_num[epoch_id]:
                break_flag = True
                break
        # Break outer
        if break_flag:
            break_flag = False
            break
    # 衰减学习率
    lr *= lr_decay
```



### 2. 检测器训练

```python
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import Dataset_Detection
from extractor import Extractor
from detector import Detector
from encoder import Encoder
from loss import focal_loss_detection



# define parameters
load = False
save = True
pretrain = True
freeze_bn = False
num_epochs = 200
nbatch = 30
lr = 0.01
step_save = 100
root = '/home1/xyt/dataset/VOC2012/JPEGImages'
list_file = 'data/voc_trainval.txt'
iou_th = (0.3, 0.5)
size = 257
GPU_DEVICE = 7



# get models
torch.cuda.set_device(GPU_DEVICE)
net_e = Extractor().cuda()
net_d = Detector().cuda()
encoder = Encoder(train_iou_th=iou_th, train_size=(size, size))



# define dataset
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
dataset = Dataset_Detection(root, list_file, size=size, train=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=nbatch, shuffle=True,
                                     num_workers=0, collate_fn=dataset.collate_fn)



# load
log = []
device_out = 'cuda:%d' % (GPU_DEVICE)
if load:
    net_e.load_state_dict(torch.load('net_e.pkl', map_location=device_out))
    net_d.load_state_dict(torch.load('net_d.pkl', map_location=device_out))
    log = list(np.load('log.npy'))
else:
    if pretrain:
        # net_e.load_state_dict(torch.load('net_e_pretrain.pkl', 
        #	map_location=device_out))
        net_e.load_state_dict({k.replace('module.',''): \
            v for k,v in torch.load('net_e_pretrain.pkl', map_location=device_out).items()}) # 预训练模型使用多GPU
if freeze_bn:
    net_e.freeze_bn()
opt_e = torch.optim.SGD(net_e.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
opt_d = torch.optim.SGD(net_d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)



# main loop
for epoch in range(num_epochs):
    for i, (img, bbox, label) in enumerate(loader):
        # get inputs
        img = img.cuda()
        # zero grad
        opt_e.zero_grad()
        opt_d.zero_grad()
        # get output and loss
        out = net_e(img, classify=False)
        cls_out, reg_out = net_d(out)
        cls_targets, reg_targets = encoder.encode(label, bbox)
        cls_targets, reg_targets = cls_targets.cuda(), reg_targets.cuda()
        loss = focal_loss_detection(cls_out, reg_out, 
                    cls_targets, reg_targets)
        # opt
        loss.backward()
        opt_e.step()
        opt_d.step()
        # print
        print('epc:%d,step:%d,loss:%f' % (epoch,i,loss))
        # save
        log.append(float(loss))
        if (i%step_save == (step_save-1)) and save:
                torch.save(net_e.state_dict(),'net_e.pkl')
                torch.save(net_d.state_dict(),'net_d.pkl')
                np.save('log.npy', log)
```



### 3. 查看检测效果

```python
import torch
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import cv2, math
from extractor import Extractor
from detector import Detector
from encoder import Encoder
from dataset import show_bbox



# TODO: define parameter
img_path = 'img/0.jpg'



# define network
net_e = Extractor()
net_d  = Detector()
encoder = Encoder()



# load
net_e.load_state_dict(torch.load('net_e.pkl', map_location='cpu'))
net_d.load_state_dict(torch.load('net_d.pkl', map_location='cpu'))
net_e.eval()
net_d.eval()



# prepare img
img = cv2.imread(img_path)
height = img.shape[0]
width = img.shape[1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.reshape((1, height, width, 3)).transpose((0, 3, 1, 2))  # [1, 3, H, W]



# TODO: get output
x = torch.from_numpy(img).float().div(255)
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
x[0] = normalize(x[0])
out = net_e(x, classify=False)
cls_out, reg_out = net_d(out)
cls_i_preds, cls_p_preds, reg_preds = encoder.decode(cls_out, reg_out, (height, width))



print(cls_i_preds[0].shape)
print(cls_p_preds[0].shape)
print(reg_preds[0].shape)
VOC_LABEL_NAMES = (
    'background',#0
    'aeroplane',#1
    'bicycle',#2
    'bird',#3
    'boat',#4
    'bottle',#5
    'bus',#6
    'car',#7
    'cat',#8
    'chair',#9
    'cow',#10
    'diningtable',#11
    'dog',#12
    'horse',#13
    'motorbike',#14
    'person',#15
    'pottedplant',#16
    'sheep',#17
    'sofa',#18
    'train',#19
    'tvmonitor'#20
    )
show_bbox(img[0]/255.0, reg_preds[0], cls_i_preds[0], VOC_LABEL_NAMES)
```


