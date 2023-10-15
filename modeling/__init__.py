# encoding: utf-8
from .baseline import Baseline


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50': model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH,
    # cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    # num_classes:751
    # LAST_STRIDE:1
    # weights:pretrained weight
    # neck:bnneck
    # tesk_neck:after
    # model_name: model name, resnet50, se_resnet, et al.
    # pretrain_choice: imagenet
    model = Baseline(num_classes, cfg.LAST_STRIDE, cfg.weights, cfg.neck, cfg.test_neck, cfg.model_name, cfg.pretrain_choice)
    return model
