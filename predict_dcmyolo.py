# coding=utf-8
# ================================================================
#
#   File name   : predict_dcmyolo.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2022/10/26 13:26
#   Description : 推理demo，包括检测图片、视频
#
# ================================================================
import time
import argparse
import cv2
import numpy as np
from PIL import Image
from dcmyolo.utils.utils_predict import YOLO


def predict_image():
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()


def predict_video():
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
        frame = np.array(yolo.detect_image(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.imshow("video",frame)
        # c= cv2.waitKey(1) & 0xff
        if video_save_path != "":
            out.write(frame)

        # if c==27:
        #     capture.release()
        #     break

    print("Video Detection Done!")
    capture.release()
    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()


def heatmap():
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            yolo.detect_heatmap(image, heatmap_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dcmyolo predict script")
    parser.add_argument('--operation_type', type=str, default='', help="操作类型 predict_image / predict_video  / heatmap")
    parser.add_argument('--model_path', type=str, default='dcmyolo/model_data/wangzhe_best_weights.pth', help="pth模型的路径")
    parser.add_argument('--classes_path', type=str, default='dcmyolo/model_data/wangzhe_classes.txt', help="分类标签文件")
    parser.add_argument('--anchors_path', type=str, default='dcmyolo/model_data/coco_anchors.txt', help="anchors文件")
    parser.add_argument('--video_path', type=str, default='', help="视频时才会用到，视频的路径")
    parser.add_argument('--video_save_path', type=str, default='', help="视频时才会用到，视频检测之后的保存路径")
    parser.add_argument('--heatmap_save_path', type=str, default='', help="heatmap保存路径")
    args = parser.parse_args()

    operation_type = args.operation_type
    model_path = args.model_path
    classes_path = args.classes_path
    anchors_path = args.anchors_path
    video_path = args.video_path
    video_save_path = args.video_save_path
    heatmap_save_path = args.heatmap_save_path

    yolo = YOLO(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path)

    if operation_type == 'predict_image':
        predict_image()
    elif operation_type == 'predict_video':
        predict_video()
    elif operation_type == 'heatmap':
        heatmap()
    else:
        raise AssertionError("Please specify the correct mode: 'predict_image', 'predict_video', 'heatmap'.")

    # 检测图片 python predict_dcmyolo.py --operation_type predict_image --model_path dcmyolo/model_data/wangzhe_best_weights.pth --classes_path dcmyolo/model_data/wangzhe_classes.txt --anchors_path dcmyolo/model_data/coco_anchors.txt
    # 检测视频 python predict_dcmyolo.py --operation_type predict_video --model_path dcmyolo/model_data/wangzhe_best_weights.pth --classes_path dcmyolo/model_data/wangzhe_classes.txt --anchors_path dcmyolo/model_data/coco_anchors.txt --video_path data/video/wangzhe1.mp4 --video_save_path_path data/video/wangzhe1_out.mp4
    # heatmap  python predict_dcmyolo.py --operation_type heatmap --model_path dcmyolo/model_data/wangzhe_best_weights.pth --classes_path dcmyolo/model_data/wangzhe_classes.txt --anchors_path dcmyolo/model_data/coco_anchors.txt --heatmap_save_path data/heatmap.jpg