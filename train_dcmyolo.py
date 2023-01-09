# coding=utf-8
# ================================================================
#
#   File name   : train_dcmyolo.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2022/10/26 13:26
#   Description : 执行训练
#
# ================================================================
import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dcmyolo.model.yolo_body import YoloBody
from dcmyolo.utils.utils_training import (ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from dcmyolo.utils.loss import YOLOLoss
from dcmyolo.utils.callbacks import LossHistory, EvalCallback
from dcmyolo.utils.dataloader import YoloDataset, yolo_dataset_collate
from dcmyolo.utils.utils_data import download_weights, get_anchors, get_classes, show_config
from dcmyolo.utils.utils_fit import fit_one_epoch
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dcmyolo train script")
    parser.add_argument('--classes_path', type=str, default='dcmyolo/model_data/coco_classes.txt', help="类别标签文件路径")
    parser.add_argument('--anchors_path', type=str, default='dcmyolo/model_data/coco_anchors.txt', help="anchors文件路径")
    parser.add_argument('--train_annotation_path', type=str, default='data/coco/train.txt', help="存放训练集图片路径和标签的txt")
    parser.add_argument('--val_annotation_path', type=str, default='data/coco/val.txt', help="存放验证图片路径和标签的txt")
    parser.add_argument('--phi', type=str, default='s', help="所使用的YoloV5的版本。n、s、m、l、x")
    # ---------------------------------------------------------------------#
    # --backbone_model_dir参数
    # 如果有backbone的预训练模型，可以backbone预训练模型目录，当model_path不存在的时候不加载整个模型的权值。
    # 只写到模型文件的上一级目录即可，文件名会根据phi自动计算（前提是从百度网盘下载的模型文件名没改）
    # ---------------------------------------------------------------------#
    parser.add_argument('--backbone_model_dir', type=str, default='dcmyolo/model_data/', help="backbone的预训练模型，写到上一级目录即可")
    parser.add_argument('--model_path', type=str, default='dcmyolo/model_data/pretrained.pth', help="yolov5预训练模型的路径")
    parser.add_argument('--save_period', type=int, default=10, help="多少个epoch保存一次权值")
    parser.add_argument('--save_dir', type=str, default='logs_wangzhe', help="权值与日志文件保存的文件夹")
    parser.add_argument('--input_shape', nargs='+', type=int, default=[640, 640], help="输入的shape大小，一定要是32的倍数")
    parser.add_argument('--use_fp16', action='store_true', help="是否使用混合精度训练")
    #------------------------------------------------------------------#
    #   mosaic              马赛克数据增强。
    #   mosaic_prob         每个step有多少概率使用mosaic数据增强，默认50%。
    #
    #   mixup               是否使用mixup数据增强，仅在mosaic=True时有效。
    #                       只会对mosaic增强后的图片进行mixup的处理。
    #   mixup_prob          有多少概率在mosaic后使用mixup数据增强，默认50%。
    #                       总的mixup概率为mosaic_prob * mixup_prob。
    #
    #   special_aug_ratio   参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #                       当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
    #                       默认为前70%个epoch，100个世代会开启70个世代。
    #------------------------------------------------------------------#
    parser.add_argument('--use_mosaic', action='store_true', help="是否使用马赛克数据增强")
    parser.add_argument('--mosaic_prob', type=float, default=0.5, help="每个step有多少概率使用mosaic数据增强")
    parser.add_argument('--use_mixup', action='store_true', help="是否使用mixup数据增强，仅在mosaic=True时有效")
    parser.add_argument('--mixup_prob', type=float, default=0.5, help="有多少概率在mosaic后使用mixup数据增强")
    parser.add_argument('--special_aug_ratio', type=float, default=0.7, help="当mosaic=True时，会在该范围内开启mosaic")
    parser.add_argument('--epoch', type=int, default=100, help="总迭代次数")
    parser.add_argument('--batch_size', type=int, default=128, help="每批次取多少张图片")
    parser.add_argument('--label_smoothing', type=float, default=0, help="是否开启标签平滑")
    parser.add_argument('--init_lr', type=float, default=1e-2, help="初始学习率")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="最小学习率")
    parser.add_argument('--optimizer_type', type=str, default="sgd", help="使用到的优化器种类，可选的有adam、sgd")
    parser.add_argument('--momentum', type=float, default=0.937, help="优化器内部使用到的momentum参数")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="权值衰减，可防止过拟合")
    parser.add_argument('--lr_decay_type', type=str, default="step", help="使用到的学习率下降方式，可选的有step、cos")
    parser.add_argument('--eval_flag', action='store_true', help="是否在训练时进行评估，评估对象为验证集")
    parser.add_argument('--eval_period', type=int, default=10, help="代表多少个epoch评估一次")
    parser.add_argument('--num_workers', type=int, default=4, help="多少个线程读取数据")
    args = parser.parse_args()

    use_cuda            = torch.cuda.is_available()

    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16            = args.use_fp16

    # ---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    # ---------------------------------------------------------------------#
    classes_path    = args.classes_path

    # ---------------------------------------------------------------------#
    #   anchors_path    代表先验框对应的txt文件。
    #   anchors_mask    用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    anchors_path    = args.anchors_path
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # ---------------------------------------------------------------------#
    # 如果有backbone的预训练模型，可以backbone预训练模型目录，当model_path不存在的时候不加载整个模型的权值。
    # 只写到模型文件的上一级目录即可，文件名会根据phi自动计算（前提是从百度网盘下载的模型文件名没改）
    # ---------------------------------------------------------------------#
    backbone_model_dir = args.backbone_model_dir

    # ---------------------------------------------------------------------#
    # 预训练模型或者上次训练中断后保存的模型，当model_path不存在的时候不加载整个模型的权值。
    # 该模型的参数会覆盖backbone_model_dir模型的参数
    # 预训练模型下载:
    # ---------------------------------------------------------------------#
    model_path = args.model_path

    #------------------------------------------------------#
    #   input_shape     输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape     = args.input_shape

    #------------------------------------------------------#
    #   phi             所使用的YoloV5的版本。n、s、m、l、x
    #------------------------------------------------------#
    phi             = args.phi

    #------------------------------------------------------------------#
    #   mosaic              马赛克数据增强。
    #   mosaic_prob         每个step有多少概率使用mosaic数据增强，默认50%。
    #
    #   mixup               是否使用mixup数据增强，仅在mosaic=True时有效。
    #                       只会对mosaic增强后的图片进行mixup的处理。
    #   mixup_prob          有多少概率在mosaic后使用mixup数据增强，默认50%。
    #                       总的mixup概率为mosaic_prob * mixup_prob。
    #
    #   special_aug_ratio   参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #                       当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
    #                       默认为前70%个epoch，100个世代会开启70个世代。
    #------------------------------------------------------------------#
    mosaic              = args.use_mosaic
    mosaic_prob         = args.mosaic_prob
    mixup               = args.use_mixup
    mixup_prob          = args.mixup_prob
    special_aug_ratio   = args.special_aug_ratio
    # ------------------------------------------------------------------#
    #   label_smoothing     标签平滑。一般0.01以下。如0.01、0.005。
    # ------------------------------------------------------------------#
    label_smoothing     = args.label_smoothing

    # ------------------------------------------------------------------#
    #   训练总轮数
    # ------------------------------------------------------------------#
    epoch = args.epoch

    # ------------------------------------------------------------------#
    #   batch_size
    # ------------------------------------------------------------------#
    batch_size = args.batch_size

    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    init_lr             = args.init_lr
    min_lr              = args.min_lr
    if min_lr <= 0:
        min_lr              = init_lr * 0.01

    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3 weight_decay = 0
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2 weight_decay = 5e-4
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type      = args.optimizer_type
    momentum            = args.momentum
    weight_decay        = args.weight_decay

    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = args.lr_decay_type

    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = args.save_period

    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = args.save_dir

    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    #------------------------------------------------------------------#
    eval_flag           = args.eval_flag
    eval_period         = args.eval_period

    # ------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers         = args.num_workers

    # ------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    # ------------------------------------------------------#
    train_annotation_path   = args.train_annotation_path
    val_annotation_path     = args.val_annotation_path

    device          = torch.device('cuda' if use_cuda else 'cpu')

    #------------------------------------------------------#
    #   获取classes和anchor
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes, phi, backbone_model_dir=backbone_model_dir)
    weights_init(model)
    if os.path.exists(model_path):
        print('发现预训练模型: ', model_path)
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    else:
        print('model_path: ', model_path, '不存在，不使用预训练模型')

    #----------------------#
    #   获得损失函数
    #----------------------#
    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, use_cuda, anchors_mask, label_smoothing)
    #----------------------#
    #   记录Loss
    #----------------------#
    time_str        = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if use_cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
            
    # ----------------------------#
    #   权值平滑
    # ----------------------------#
    ema = ModelEMA(model_train)
    
    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    random.shuffle(train_lines)
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    random.shuffle(val_lines)
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    show_config(
        classes_path=classes_path, anchors_path=anchors_path, anchors_mask=anchors_mask, model_path=model_path,
        input_shape=input_shape, epoch=epoch, batch_size=batch_size, init_lr=init_lr, min_lr=min_lr,
        optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period,
        save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    optimizer = {
        'adam'  : optim.Adam(model.parameters(), init_lr, betas=(momentum, 0.999)),
        'sgd'   : optim.SGD(model.parameters(), init_lr, momentum=momentum, nesterov=True)
    }[optimizer_type]

    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr, min_lr, epoch)

    # ---------------------------------------#
    #   判断每一个世代的长度
    # ---------------------------------------#
    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # if ema:
    #     ema.updates     = epoch_step * init_Epoch

    #---------------------------------------#
    #   构建数据集加载器。
    #---------------------------------------#
    train_dataset   = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=epoch, \
                                    mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
    val_dataset     = YoloDataset(val_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=epoch, \
                                    mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

    gen             = DataLoader(train_dataset, shuffle=True, batch_size = batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val         = DataLoader(val_dataset  , shuffle=True, batch_size = batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)

    #----------------------#
    #   记录eval的map曲线
    #----------------------#
    eval_callback   = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, use_cuda, \
                                    eval_flag=eval_flag, period=eval_period)

    #---------------------------------------#
    #   开始模型训练
    #---------------------------------------#
    for i in range(epoch):
        gen.dataset.epoch_now       = i
        gen_val.dataset.epoch_now   = i
        set_optimizer_lr(optimizer, lr_scheduler_func, i)
        fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, i, epoch_step,
                      epoch_step_val, gen, gen_val, epoch, use_cuda, fp16, scaler, save_period, save_dir)

