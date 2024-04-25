# yolov5_vehicle_reid

yolov5+reid实现的车辆重识别

# 车辆重识别

车辆重识别数据集采用veri,格式与markt1501类似。

本项目是对之前行人重识别基础上修改的，所以代码很多地方和之前的项目是差不多的。**麻烦点个star鼓励一下**

yolov5行人重识别参考资料：

CSDN：yolov5行人重识别[https://blog.csdn.net/z240626191s/article/details/129221510]

github: yolov5行人重识别代码[https://github.com/YINYIPENG-EN/yolov5_reid.git]

ps:arrow_right:**该训练reid项目vehicle_search与_search项目是独立的！！**训练完reid后，把训练好的权重放到 vehicle_search/weights下，**切换**到vehicle_search_search项目中在去进行reid识别【不然有时候会报can't import xxx】。

**参数说明：**

--config_file: 配置文件路径，默认configs/softmax_triplet.yml

--weights: Reid pretrained weight path

--neck:  If train with BNNeck, options: **bnneck** or no

--test_neck:  BNNeck to be used for test, before or after BNNneck options: **before** or **after**

--model_name: Name of backbone.

--pretrain_choice: Imagenet

--IF_WITH_CENTER: us center loss, True or False.

--resume:resume train

# 环境说明

torch >= 1.7.0

torchvision >=0.8.0

opencv-python  4.7.0.72

opencv-python-headless   4.7.0.72

numpy 1.21.6

matplotlib  3.4.3

loguru  0.5.3

pytorch-ignite=0.4.11

:fountain_pen:

**配置文件的修改：**

(注意：项目中有两个配置文件，一个是config下的defaults.py配置文件，一个是configs下的yml配置文件，**一般配置yml文件即可**，当两个配置文件参数名相同的时候以yml文件为主，这个需要注意一下)



**configs文件**:

以**softmax_triplet.yml**为例：

```
SOLVER:
  OPTIMIZER_NAME: 'Adam' # 优化器
  MAX_EPOCHS: 120  # 总epochs
  BASE_LR: 0.00035
  IMS_PER_BATCH: 8  # batch
TEST:
  IMS_PER_BATCH: 4 # test batch
  RE_RANKING: 'no'
  WEIGHT: "path"  # test weight path
  FEAT_NORM: 'yes'
OUTPUT_DIR: "/logs" # model save path
```



#  训练预权重下载

将 **r50_ibn_2.pth，resnet50-19c8e357.pth**放在yolov5_vehicle_reid/weights下

链接：https://pan.baidu.com/s/1QYvFE6rDSmxNl4VBNBar-A 
提取码：yypn



# 训练

```shell
python tools/train.py --weights 【预权重路径】
```

## 中断后的继续训练或微调训练

如果训练意外终止，或者希望继续训练，可以适用本功能。只需要传入--resume参数即可

```
python tools/train.py --weights 【your weight path】 --resume
```

## 冻结训练

新增冻结训练，加快网络前期训练速度。

训练中只需传入：--freeze    --freeze_epoch 20即可，其中--freeze表示是否开启冻结训练，--freeze_epoch是冻结训练后的epoch，这里默认为20，那么网络会在前20个epoch冻结训练，从21个epoch开始解冻训练

示例如下：

```
python tools/train.py --weights your weigt path --freeze --freeze_epoch 20
```

# 测试

输入以下命令即可快速开启测试，获得测试结果

【此脚本是针对训练后的模型单独获得测试结果，例如mAP、Rank等指标】

```
python tools/test.py --weights weights/ReID_resnet50_ibn_a.pth

```

# 说明

开发不易，**本项目部分功能有偿提供**。联系方式可进入CSDN博客链接扫描本末二维码添加，或直接微信搜索:y24065939s。

CSDN:车辆重识别[https://blog.csdn.net/z240626191s/article/details/133840737?spm=1001.2014.3001.5501]

**1.训练核心代码**

有偿训练代码有两种：含**tensorboard**与不含**tensorboard**（价格不一样，与旧版本相比均支持继续训练）

tensorboard包含(loss、acc、mAP、Rank、lr)曲线的可视化。

2024.4.24新增tensorboard内容：根据距离矩阵记录困难样本

**2.Yolov5 reid Gui**

本项目vehicle_search中的**无Gui部分检测为免费提供**，**GUI部分为有偿使用**，vehicle_search详细使用可进入person_search中的readme中查看

