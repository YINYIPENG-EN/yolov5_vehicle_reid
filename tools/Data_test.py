import torchvision.transforms as T
from data.transforms.transforms import RandomErasing
from data.datasets.dataset_loader import read_image

normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([
            T.Resize([256, 128]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
img_path, pid, camid = '../data/Market1501/bounding_box_train/0002_c1s1_000451_03.jpg', 0, 0
img = read_image(img_path)

img_trans = transform(img)
img_trans = img_trans.permute((1, 2, 0))
img_trans = img_trans.numpy()
import cv2
import numpy as np
img_trans = img_trans * [0.229, 0.224, 0.225]
img_trans = img_trans + [0.485, 0.456, 0.406]
img_trans = cv2.cvtColor(img_trans.astype(np.float32), cv2.COLOR_RGB2BGR)
cv2.imshow("img", img_trans)
cv2.waitKey(0)
