# -*- coding=utf-8 -*-
import glob
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from dcmyolo.utils.kmeans import kmeans, avg_iou
import cv2
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument('--txt_path', type=str, default='data/wangzhe/train.txt', help="标注文件txt")
parser.add_argument('--anchors_path', type=str, default='dcmyolo/model_data/wangzhe_anchors.txt', help="anchors文件txt")
parser.add_argument('--clusters', type=int, default=9, help="聚类的数目,一般情况下是9")
parser.add_argument('--input_size', type=int, default=640, help="模型中图像的输入尺寸")
args = parser.parse_args()

# 根文件夹
TXT_PATH = args.txt_path
# 聚类的数目
CLUSTERS = args.clusters
# 模型中图像的输入尺寸，默认是一样的
SIZE = args.input_size
anchors_path = args.anchors_path

# 加载YOLO格式的标注数据
def load_dataset(path):
    with open(path, 'r') as txt:
        lines = txt.readlines()
    dataset = []
    for line in lines:
        items = line.replace("\n", '').split(" ")
        img_path = items[0]
        if not os.path.join(img_path):
            raise "该图片不存在：" + img_path
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        for item in items[1:len(items)]:
            # print('item:', item)
            xyxy = item.split(',')
            roi_with = float(int(xyxy[2]) - int(xyxy[0])) / w
            roi_height = float(int(xyxy[3]) - int(xyxy[1])) / h
            dataset.append([roi_with, roi_height])
    return np.array(dataset)

data = load_dataset(TXT_PATH)
out = kmeans(data, k=CLUSTERS)

print(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print('=' * 20)
box_list = []
for o0, o1 in zip(out[:, 0], out[:, 1]):
    area = o0 * o1 * SIZE
    box_list.append([o0, o1])
box_list = sorted(box_list, key=lambda x: x[0]*x[1])

boxes = []
for o0, o1 in box_list:
    boxes.append(str(int(round(o0 * SIZE, 0))))
    boxes.append(str(int(round(o1 * SIZE, 0))))

print("anchor:", ','.join(boxes))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))

with open(anchors_path, 'w', encoding='utf-8') as f:
    f.write(','.join(boxes))
