# coding=utf-8
# ================================================================
#
#   File name   : loss.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2022/11/22 15:05 
#   Description :
#
# ================================================================

import torch
from torch.nn import Module
import numpy as np
import math


class YOLOLoss(Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothing=0):
        super(YOLOLoss, self).__init__()
        # -----------------------------------------------------------#
        #   20x20的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   40x40的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   80x80的特征层对应的anchor是[10,13],[16,30],[33,23]
        # -----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.threshold = 4

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # ----------------------------------------------------#
        #   求出预测框左上角右下角
        # ----------------------------------------------------#
        print("max:", torch.max(b1[...,0]), torch.max(b2[...,0]))
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # ----------------------------------------------------#
        #   求出真实框左上角右下角
        # ----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # ----------------------------------------------------#
        #   求真实框和预测框所有的iou
        # ----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        # ----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        # ----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # ----------------------------------------------------#
        #   计算对角线距离
        # ----------------------------------------------------#
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, box1, box2, xywh=False, giou=False, diou=False, ciou=False, eiou=False, eps=1e-7):
        """
        实现各种IoU
        Parameters
        ----------
        box1        shape(b, c, h, w,4)
        box2        shape(b, c, h, w,4)
        xywh        是否使用中心点和wh，如果是False，输入就是左上右下四个坐标
        GIoU        是否GIoU
        DIoU        是否DIoU
        CIoU        是否CIoU
        EIoU        是否EIoU
        eps         防止除零的小量

        Returns
        -------

        """
        # 获取边界框的坐标
        if xywh:
            # 将 xywh 转换成 xyxy
            b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
            b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
            b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
            b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

        else:
            # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

        # 区域交集
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # 区域并集
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        # 计算iou
        iou = inter / union

        if giou or diou or ciou or eiou:
            # 计算最小外接矩形的wh
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

            if ciou or diou or eiou:
                # 计算最小外接矩形角线的平方
                c2 = cw ** 2 + ch ** 2 + eps
                # 计算最小外接矩形中点距离的平方
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                if diou:
                    # 输出DIoU
                    return iou - rho2 / c2
                elif ciou:
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    # 输出CIoU
                    return iou - (rho2 / c2 + v * alpha)
                elif eiou:
                    rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                    rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                    cw2 = cw ** 2 + eps
                    ch2 = ch ** 2 + eps
                    # 输出EIoU
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
            else:
                c_area = cw * ch + eps  # convex area
                # 输出GIoU
                return iou - (c_area - union) / c_area
        else:
            # 输出IoU
            return iou

    # ---------------------------------------------------#
    #   平滑标签
    # ---------------------------------------------------#
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None, y_true=None):
        # ----------------------------------------------------#
        #   l               代表使用的是第几个有效特征层
        #   input的shape为  bs, 3*(5+num_classes), 20, 20
        #                   bs, 3*(5+num_classes), 40, 40
        #                   bs, 3*(5+num_classes), 80, 80
        #   targets         真实框的标签情况 [batch_size, num_gt, 5]
        # ----------------------------------------------------#
        # --------------------------------#
        #   获得图片数量，特征层的高和宽
        #   20, 20
        # --------------------------------#
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        # -----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   [640, 640] 高的步长为640 / 20 = 32，宽的步长为640 / 20 = 32
        #   如果特征层为20x20的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为40x40的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为80x80的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        # -----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        # -------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   bs, 3 * (5+num_classes), 20, 20 => bs, 3, 5 + num_classes, 20, 20 => batch_size, 3, 20, 20, 5 + num_classes

        #   batch_size, 3, 20, 20, 5 + num_classes
        #   batch_size, 3, 40, 40, 5 + num_classes
        #   batch_size, 3, 80, 80, 5 + num_classes
        # -----------------------------------------------#
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()

        # -----------------------------------------------#
        #   先验框的中心位置的调整参数
        # -----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # -----------------------------------------------#
        #   先验框的宽高调整参数
        # -----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # -----------------------------------------------#
        #   获得置信度，是否有物体
        # -----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        # -----------------------------------------------#
        #   种类置信度
        # -----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # ---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        # ----------------------------------------------------------------#
        print("max x:", torch.max(x))
        pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)
        print("max pred_boxes:", torch.max(pred_boxes[...,0]))
        if self.cuda:
            y_true = y_true.type_as(x)

        loss = 0
        n = torch.sum(y_true[..., 4] == 1)
        if n != 0:
            # ---------------------------------------------------------------#
            #   计算预测结果和真实结果的giou，计算对应有真实框的先验框的giou损失
            #                         loss_cls计算对应有真实框的先验框的分类损失
            # ----------------------------------------------------------------#
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - giou)[y_true[..., 4] == 1])
            loss_cls = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1],
                                               self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1],
                                                                  self.label_smoothing, self.num_classes)))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
            # -----------------------------------------------------------#
            #   计算置信度的loss
            #   也就意味着先验框对应的预测框预测的更准确
            #   它才是用来预测这个物体的。
            # -----------------------------------------------------------#
            tobj = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        else:
            tobj = torch.zeros_like(y_true[..., 4])
        loss_conf = torch.mean(self.BCELoss(conf, tobj))

        loss += loss_conf * self.balance[l] * self.obj_ratio
        # if n != 0:
        #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss

    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_pred_boxes(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(targets)

        # -----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        # -----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # -------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        # -------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)
        return pred_boxes
