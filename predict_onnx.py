# coding=utf-8
# ================================================================
#
#   File name   : export.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2022/11/24 15:17 
#   Description :
#
# ================================================================
import argparse
from dcmyolo.utils.utils_predict import YOLO
import onnxruntime
import cv2
import numpy as np
import time
from PIL import Image
import torch
from dcmyolo.utils.utils_data import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)


def export_onnx(model_path, classes_path, anchors_path, onnx_save_path, simplify=True, input_shape=[640, 640],
                phi='s', append_nms=True):
    """
    pth导出成onnx
    Parameters
    ----------
    model_path      pth模型的路径
    classes_path    分类标签文件
    anchors_path    anchors文件
    onnx_save_path  onnx保存路径
    simplify        是否使用onnxsim简化模型
    input_shape     输入图片的大小
    phi             所使用的YoloV5的版本。n、s、m、l、x
    append_nms      是否添加nms

    Returns
    -------

    """
    yolo = YOLO(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path, input_shape=input_shape,
                phi=phi, letterbox_image=False, need_detect_box=True)
    yolo.convert_to_onnx(onnx_save_path, simplify, append_nms)

def predict_image(onnx_model, class_names, image, input_shape=(640, 640), iou_threshold=0.3, score_threshold=0.5):
    letterbox_image = False
    org_h, org_w = np.array(np.shape(image)[0:2])
    img_org = np.asarray(image, dtype=np.uint8)
    img_org = cv2.cvtColor(img_org, cv2.COLOR_RGB2BGR)
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data = resize_image(image, input_shape, letterbox_image)
    # image_data.show()
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    output = ['gather101']
    results = onnx_model.run(output, {"images": image_data,
                                      "max_output_boxes_per_class": [20],
                                      "iou_threshold": [1 - iou_threshold],
                                      "score_threshold": [score_threshold],
                                      "slice101_starts": [2], "slice101_ends": [3],
                                      "slice101_axes": [1], "slice101_steps": [1],
                                      "squeeze101_axes": [1]
                                      })
    # img_org = cv2.imread(img_path)
    # 只有一个框的时候，会比多个框少一维
    if len(results[0].shape) == 2:
        results = results[0]
    else:
        results = results[0][0]
    scale_x = org_w / input_shape[0]
    scale_y = org_h / input_shape[1]
    for item in results:
        cx, cy, w, h, obj_cnf, cls = item
        x1 = (cx - w / 2) * scale_x
        y1 = (cy - h / 2) * scale_y
        x2 = (cx + w / 2) * scale_x
        y2 = (cy + h / 2) * scale_y
        img_org = cv2.rectangle(img_org, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        img_org = cv2.putText(img_org, class_names[int(cls)] + '-' + str(obj_cnf)[0:4], (int(x1), int(y1)),
                              cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
    return img_org


def predict_image_path(model_path, classes_path, img_path, input_shape=(640, 640),
                       iou_threshold=0.3, score_threshold=0.5):
    onnx_model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
    image = Image.open(img_path)
    class_names, num_classes = get_classes(classes_path)
    img_org = predict_image(onnx_model, class_names, image, input_shape=input_shape,
                            iou_threshold=iou_threshold, score_threshold=score_threshold)
    cv2.imshow('1', img_org)
    cv2.waitKey(-1)

def predict_video(model_path, classes_path, video_path, video_save_path, input_shape=(640, 640),
                  iou_threshold=0.3, score_threshold=0.5):
    onnx_model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
    class_names, num_classes = get_classes(classes_path)
    capture = cv2.VideoCapture(video_path)
    video_fps = int(round(capture.get(cv2.CAP_PROP_FPS)))
    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = predict_image(onnx_model, class_names, frame, input_shape=input_shape,
                              iou_threshold=iou_threshold, score_threshold=score_threshold)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if video_save_path != "":
            out.write(frame)

    print("Video Detection Done!")
    capture.release()
    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="onnx script")
    parser.add_argument('--operation_type', type=str, default='', help="操作类型export_onnx / predict_image / predict_video")
    parser.add_argument('--model_path', type=str, default='', help="pth模型的路径")
    parser.add_argument('--classes_path', type=str, default='', help="分类标签文件")
    parser.add_argument('--anchors_path', type=str, default='', help="anchors文件")
    parser.add_argument('--onnx_path', type=str, default='', help="onnx保存路径")
    parser.add_argument('--video_path', type=str, default='', help="视频时才会用到，视频的路径")
    parser.add_argument('--video_save_path', type=str, default='', help="视频时才会用到，视频检测之后的保存路径")
    parser.add_argument('--phi', type=str, default='', help="所使用的YoloV5的版本。n、s、m、l、x")
    parser.add_argument('--no_simplify', action='store_false', help="不使用onnxsim简化模型")
    parser.add_argument('--input_shape', nargs='+', type=int, default=[640, 640], help="输入的shape大小，一定要是32的倍数")
    parser.add_argument('--append_nms', action='store_true', help="添加nms")
    parser.add_argument('--iou_threshold', type=float, default=0.2, help="两个bbox的iou超过这个值会被认为是同一物体")
    parser.add_argument('--score_threshold', type=float, default=0.5, help="检测物体的概率小于这个值将会被舍弃")
    args = parser.parse_args()
    operation_type = args.operation_type
    model_path = args.model_path
    classes_path = args.classes_path
    anchors_path = args.anchors_path
    onnx_path = args.onnx_path
    no_simplify = args.no_simplify
    input_shape = args.input_shape
    append_nms = args.append_nms
    video_path = args.video_path
    video_save_path = args.video_save_path
    iou_threshold = args.iou_threshold
    score_threshold = args.score_threshold
    phi = args.phi
    if operation_type == 'export_onnx':
        export_onnx(model_path, classes_path, anchors_path, onnx_path, simplify=(not no_simplify),
                    input_shape=input_shape, phi='s', append_nms=append_nms)
    elif operation_type == 'predict_image':
        while True:
            img_path = input('Input image filename:')
            predict_image_path(onnx_path, classes_path, img_path, input_shape=input_shape,
                               iou_threshold=iou_threshold, score_threshold=score_threshold)
    elif operation_type == 'predict_video':
        predict_video(onnx_path, classes_path, video_path, video_save_path,
                      iou_threshold=iou_threshold, score_threshold=score_threshold)
    else:
        raise AssertionError("Please specify the correct mode: 'export_onnx', 'predict_image', 'predict_video'.")

    # 导出onnx脚本： python predict_onnx.py --operation_type export_onnx --model_path dcmyolo/model_data/wangzhe_best_weights.pth --classes_path dcmyolo/model_data/wangzhe_classes.txt --anchors_path dcmyolo/model_data/coco_anchors.txt --onnx_path dcmyolo/model_data/wangzhe_best_weights.onnx --append_nms
    # 检测图片：     python predict_onnx.py --operation_type predict_image --onnx_path dcmyolo/model_data/wangzhe_best_weights.onnx --classes_path dcmyolo/model_data/wangzhe_classes.txt
    # 检测视频：     python predict_onnx.py --operation_type predict_video --onnx_path dcmyolo/model_data/wangzhe_best_weights.onnx --classes_path dcmyolo/model_data/wangzhe_classes.txt --video_path data/video/wangzhe1.mp4 --video_save_path data/video/wangzhe1_out1.mp4