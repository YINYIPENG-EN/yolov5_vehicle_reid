import torch
import torch.nn as nn
import torchvision.transforms as T
from data.datasets.dataset_loader import read_image
from data.transforms.transforms import RandomErasing
from modeling import Baseline

model_path = '../person_search/weights/ReID_resnet50_ibn_a.pth'
model = Baseline(num_classes=751, last_stride=1, model_path=model_path, neck='no', neck_feat='after', model_name='resnet50_ibn_a', pretrain_choice='imagenet')
# img = Image.open('../data/Market1501/bounding_box_train/0002_c1s1_000451_03.jpg')

normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([
            T.Resize([256, 128]),
            # T.RandomHorizontalFlip(p=0.5),
            # T.Pad(10),
            # T.RandomCrop([256, 128]),
            T.ToTensor(),
            # normalize_transform,
            # RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
img_path, pid, camid = '../data/Market1501/bounding_box_train/0002_c1s1_000451_03.jpg', 0, 0
img = read_image(img_path)
img_trans = transform(img)
img_trans = img_trans.unsqueeze(0)
print(img_trans.shape)
cls_score, global_feat = model(img_trans)
print(torch.max(nn.Softmax(dim=1)(cls_score)))
print('cls_score shape: ', cls_score.shape)  # shape 751
print('global_feat shape is: ', global_feat.shape)  # shape 2048
