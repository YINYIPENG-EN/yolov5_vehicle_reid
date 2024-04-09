# encoding: utf-8
import argparse
import os
import sys
import torch
from torch.backends import cudnn
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from loguru import logger


def train(cfg, args):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)   # 加载数据集
    # prepare model  模型初始化
    model = build_model(args, num_classes)

    if not args.IF_WITH_CENTER:
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)  # triplet 三元组损失函数
        optimizer = make_optimizer(cfg, model)  # 优化器
        loss_func = make_loss(cfg, num_classes)  # modified by gu
        if args.resume:
            path_to_optimizer = args.weights.replace('model', 'optimizer')
            optimizer_dict = torch.load(path_to_optimizer)
            optimizer.load_state_dict(optimizer_dict['optimizer_state'])
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = optimizer_dict['lr']
        if args.pretrain_choice == 'imagenet':
            start_epoch = 0 if not args.resume else optimizer_dict['epoch']
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(args.pretrain_choice))
        if args.resume:
            scheduler.load_state_dict(optimizer_dict['scheduler'])
        logger.info('ready train...')
        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,  # modify for using self trained model
            loss_func,
            num_query,
            start_epoch,  # add for using self trained model
            args
        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(args.IF_WITH_CENTER))


def main():
    parser = argparse.ArgumentParser(description="Yolo v5 with vehicle ReID Baseline Training")
    parser.add_argument(
        "--config_file", type=str, default=r"configs/softmax_triplet.yml", help="path to config file"
    )
    parser.add_argument('--LAST_STRIDE', type=int, default=1, help='last stride')
    parser.add_argument('--weights', type=str, default='weights/r50_ibn_2.pth', help='pretrained weight path')
    parser.add_argument('--neck', type=str, default='bnneck', help='If train with BNNeck, options: bnneck or no')
    parser.add_argument('--test_neck', type=str, default='after', help='Which feature of BNNeck to be used for test, '
                                                                       'before or after BNNneck, options: before or '
                                                                       'after')
    parser.add_argument('--model_name', type=str, default='resnet50_ibn_a', help='Name of backbone')
    # Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
    parser.add_argument('--pretrain_choice', type=str, default='imagenet')
    parser.add_argument('--IF_WITH_CENTER', action='store_true', default=False, help="If train loss include center "
                                                                                    "loss, options: 'yes' or 'no'. "
                                                                                    "Loss with center loss has "
                                                                                    "different optimizer "
                                                                                    "configuration")
    parser.add_argument('--resume', action='store_true', help='resume train')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("yolov5 vehicle reid_baseline")
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg, args)


if __name__ == '__main__':
    main()
