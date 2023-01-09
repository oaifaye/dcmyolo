# coding=utf-8
# ================================================================
#
#   File name   : CSPdarknet.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2022/10/26 13:26
#   Description : 将labelme的json文件转换成voc格式
#
# ================================================================
import warnings

import torch
import torch.nn as nn
import os


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)), 
                self.cv2(x)
            )
            , dim=1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        
class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth, backbone_model_path):
        super().__init__()
        self.stem       = Conv(3, base_channels, 6, 2, 2)
        self.dark2 = nn.Sequential(
            # 64, 320, 320 -> 128, 160, 160
            Conv(base_channels, base_channels * 2, 3, 2),
            # 128, 160, 160 -> 128, 160, 160
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        
        self.dark3 = nn.Sequential(
            # 128, 160, 160 -> 256, 80, 80
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            # 256, 80, 80 -> 256, 80, 80
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        self.dark4 = nn.Sequential(
            # 512, 80, 80 -> 512, 40, 40
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            # 512, 40, 40 -> 512, 40, 40
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        
        self.dark5 = nn.Sequential(
            # 512, 40, 40 -> 1024, 20, 20
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            # 1024, 20, 20 -> 1024, 20, 20
            C3(base_channels * 16, base_channels * 16, base_depth),
            # 1024, 20, 20 -> 1024, 20, 20
            SPPF(base_channels * 16, base_channels * 16),
        )

        if os.path.exists(backbone_model_path):
            checkpoint = torch.load(backbone_model_path, map_location="cpu")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from ", backbone_model_path)
        else:
            print("cspdarknet weights not exist ", backbone_model_path)

    def forward(self, x):

        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        #   经过第一个卷积层，将channel扩充到base_channels
        # -----------------------------------------------#
        x = self.stem(x)

        # -----------------------------------------------#
        #   Conv + CSP_1
        #   完成卷积之后，64, 320, 320 -> 128, 160, 160
        #   完成CSPlayer之后，128, 160, 160 -> 128, 160, 160
        # -----------------------------------------------#
        x = self.dark2(x)

        # -----------------------------------------------#
        #   Conv + CSP_2
        #   完成卷积之后，128, 160, 160 -> 256, 80, 80
        #   完成CSPlayer之后，256, 80, 80 -> 256, 80, 80
        #                   在这里引出有效特征层256, 80, 80
        #                   进行加强特征提取网络FPN的构建
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x

        # -----------------------------------------------#
        #   Conv + CSP_3
        #   完成卷积之后，256, 80, 80 -> 512, 40, 40
        #   完成CSPlayer之后，512, 40, 40 -> 512, 40, 40
        #                   在这里引出有效特征层512， 40, 40, 512
        #                   进行加强特征提取网络FPN的构建
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x

        # -----------------------------------------------#
        #   Conv + CSP_4 + SPPF
        #   完成卷积之后，512, 40, 40 -> 1024, 20, 20
        #   完成SPP之后，1024, 20, 20 -> 1024, 20, 20
        #   完成CSPlayer之后，1024, 20, 20 -> 1024, 20, 20
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
