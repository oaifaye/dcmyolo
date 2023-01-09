import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from dcmyolo.model.yolo_body import YoloBody
from dcmyolo.utils.utils_data import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from dcmyolo.utils.utils_bbox import DecodeBox
import onnx
import cv2
from PIL import Image


'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, model_path="dcmyolo/model_data/best_epoch_weights.pth", classes_path=None, anchors_path='dcmyolo/model_data/coco_anchors.txt',
                 input_shape=[640, 640], phi='s', confidence=0.2, nms_iou=0.3, letterbox_image=False, need_detect_box=False):
        self.model_path =model_path
        self.classes_path =classes_path
        self.anchors_path =anchors_path
        self.model_path =model_path
        self.phi = phi
        self.confidence =confidence
        self.nms_iou =nms_iou
        self.letterbox_image =letterbox_image
        self.need_detect_box =need_detect_box
        self.input_shape =input_shape
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.cuda = torch.cuda.is_available()
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path,
                    input_shape=input_shape, phi=phi, confidence=confidence, nms_iou=nms_iou,
                    letterbox_image=letterbox_image, need_detect_box=need_detect_box)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi, self.anchors,
                               (self.input_shape[0], self.input_shape[1]), need_detect_box=self.need_detect_box)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

        # torch.save(self.net.state_dict(), 'model_data/yolo_export.pth')

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # image_data.show()
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)

            if results[0] is None:
                return image

            top_label   = np.array(results[0][:, 6], dtype='int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='dcmyolo/model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#

        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            if score < self.confidence:
                continue
            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score      = np.max(sigmoid(sub_output[..., 4]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, model_path, simplify, append_nms):
        import onnx
        self.generate(onnx=True)
        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["boxes", "scores", "boxes_scores"]
        print(f'Starting export with onnx {onnx.__version__}.')
        # print('model:', self.net)
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        if append_nms:
            print('append_nms save as {}'.format(model_path))
            model = self.onnx_append_nms(model_path)
            onnx.checker.check_model(model)
            onnx.save(model, model_path)

        print('Onnx model save as {}'.format(model_path))

    def onnx_append_nms(self, model_path, unused_node=[]):
        from onnx import defs, checker, helper, numpy_helper, mapping
        from onnx import ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorProto, \
            OperatorSetIdProto
        from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model, make_node

        # dynamic_batch = False
        model_onnx = onnx.load(model_path)
        graph = model_onnx.graph
        ngraph = GraphProto()
        ngraph.name = graph.name
        ngraph.input.extend([i for i in graph.input if i.name not in unused_node])
        ngraph.initializer.extend([i for i in graph.initializer if i.name not in unused_node])
        ngraph.value_info.extend([i for i in graph.value_info if i.name not in unused_node])
        ngraph.node.extend([i for i in graph.node if i.name not in unused_node])
        output_info = [i for i in graph.output]
        ngraph.output.extend(output_info)

        '''
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
        '''
        nms = make_node(
            'NonMaxSuppression',
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
            name='batch_nms',
        )

        ngraph.node.append(nms)
        slice = onnx.helper.make_node(
            "Slice",
            inputs=["selected_indices", "slice101_starts", "slice101_ends", "slice101_axes", "slice101_steps"],
            outputs=["slice101"],
        )
        ngraph.node.append(slice)

        squeeze = onnx.helper.make_node(
            "Squeeze",
            inputs=["slice101"],
            outputs=["squeeze101"],
        )
        ngraph.node.append(squeeze)

        gather = onnx.helper.make_node(
            "Gather",
            inputs=["boxes_scores", "squeeze101"],
            outputs=["gather101"],
            axis=1,
        )
        ngraph.node.append(gather)

        ngraph.input.extend([helper.make_tensor_value_info('max_output_boxes_per_class', TensorProto.INT64, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [1])])
        # ngraph.input.extend([helper.make_tensor_value_info('slice101_starts', TensorProto.INT64, [1])])
        # ngraph.input.extend([helper.make_tensor_value_info('slice101_ends', TensorProto.INT64, [1])])
        # ngraph.input.extend([helper.make_tensor_value_info('slice101_axes', TensorProto.INT64, [1])])
        # ngraph.input.extend([helper.make_tensor_value_info('slice101_steps', TensorProto.INT64, [1])])
        # ngraph.input.extend([helper.make_tensor_value_info('squeeze101_axes', TensorProto.INT64, [1])])

        ngraph.output.extend([helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [-1, 3])])
        ngraph.output.extend([helper.make_tensor_value_info('gather101', TensorProto.FLOAT, [-1])])

        # bias_values = np.random.uniform(-10, 10, size=[1]).astype("float32")

        # "slice101_starts": [2], "slice101_ends": [3],
        # "slice101_axes": [1], "slice101_steps": [1],
        # "squeeze101_axes": [1]
        initializer_slice101_starts = make_tensor(name="slice101_starts", data_type=onnx.TensorProto.INT64, dims=[1], vals=[2])
        initializer_slice101_ends = make_tensor(name="slice101_ends", data_type=onnx.TensorProto.INT64, dims=[1], vals=[3])
        initializer_slice101_axes = make_tensor(name="slice101_axes", data_type=onnx.TensorProto.INT64, dims=[1], vals=[1])
        initializer_slice101_steps = make_tensor(name="slice101_steps", data_type=onnx.TensorProto.INT64, dims=[1], vals=[1])
        # initializer_squeeze101_axes = make_tensor(name="squeeze101_axes", data_type=onnx.TensorProto.INT64, dims=[1], vals=[1])
        ngraph.initializer.extend([initializer_slice101_starts, initializer_slice101_ends, initializer_slice101_axes,
                                   initializer_slice101_steps])

        model_attrs = dict(
            ir_version=model_onnx.ir_version,
            opset_imports=model_onnx.opset_import,
            producer_version=model_onnx.producer_version,
            model_version=model_onnx.model_version
        )

        model = make_model(ngraph, **model_attrs)

        return model

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
