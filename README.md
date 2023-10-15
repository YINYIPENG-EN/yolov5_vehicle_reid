# yolov5_vehicle_reid
yolov5+reid实现的车辆重识别
# 车辆重识别

车辆重识别数据集采用veri,格式与markt1501类似。

本项目是对之前行人重识别基础上修改的，所以代码很多地方和之前的项目是差不多的。**麻烦点个star鼓励一下**

yolov5行人重识别参考资料：

CSDN：https://blog.csdn.net/z240626191s/article/details/129221510

github: https://github.com/YINYIPENG-EN/yolov5_reid.git

ps:arrow_right:**该训练reid项目vehicle_search与_search项目是独立的！！**训练完reid后，把训练好的权重放到 vehicle_search/weights下，**切换**到vehicle_search_search项目中在去进行reid识别【不然有时候会报can't import xxx】。

**参数说明：**

--config_file: 配置文件路径，默认configs/softmax_triplet.yml

--weights: Reid pretrained weight path

--neck:  If train with BNNeck, options: **bnneck** or no

--test_neck:  BNNeck to be used for test, before or after BNNneck options: **before** or **after**

--model_name: Name of backbone.

--pretrain_choice: Imagenet

--IF_WITH_CENTER: us center loss, True or False.

# 环境说明

torch >= 1.7.0

torchvision >=0.8.0

opencv-python  4.7.0.72
opencv-python-headless   4.7.0.72
numpy 1.21.6
matplotlib  3.4.3

loguru  0.5.3

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



#  训练预权重下载：

将 **r50_ibn_2.pth，resnet50-19c8e357.pth**放在yolov5_vehicle_reid/weights下

链接：https://pan.baidu.com/s/1QYvFE6rDSmxNl4VBNBar-A 
提取码：yypn



# 训练

```shell
python tools/train.py --weights 【预权重路径】--config_file configs/softmax_triplet.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('veri')" DATASETS.ROOT_DIR "(r'./data')
```

