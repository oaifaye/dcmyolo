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
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path"        : 'model_data/ep4990-loss0.059-val_loss0.085.pth',
        "model_path"        : 'dcmyolo/model_data/best_epoch_weights.pth',
        "classes_path"      : 'dcmyolo/model_data/wangzhe_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'dcmyolo/model_data/coco_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------------#
        #   phi             所使用的YoloV5的版本。n、s、m、l、x
        #------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : torch.cuda.is_available(),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
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

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi, self.anchors,
                               (self.input_shape[0], self.input_shape[1]), need_detect_box=True)
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
    def detect_image(self, image, crop = False, count = False):
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
        image_data.show()
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

            # model_path = 'model_data/models.onnx'
            # input_layer_names = ["images"]
            # output_layer_names = ["output1", "output2"]
            # print(f'Starting export with onnx {onnx.__version__}.')
            # print('model:', self.net)
            # torch.onnx.export(self.net,
            #                   images,
            #                   f=model_path,
            #                   verbose=False,
            #                   opset_version=12,
            #                   training=torch.onnx.TrainingMode.EVAL,
            #                   do_constant_folding=True,
            #                   input_names=input_layer_names,
            #                   output_names=output_layer_names,
            #                   dynamic_axes=None)
            #
            # # Checks
            # model_onnx = onnx.load(model_path)  # load onnx model
            # onnx.checker.check_model(model_onnx)  # check onnx model


            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
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
        print('model:', self.net)
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

        dynamic_batch = False
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

        # ngraph.value_info.extend([nms])
        # if dynamic_batch:
        #     num_detection = make_tensor_value_info('num_detections', TensorProto.INT32, ["-1", 1])
        #     nmsed_box = make_tensor_value_info('nmsed_boxes', TensorProto.FLOAT, ["-1", 200, 4])
        #     nmsed_score = make_tensor_value_info('nmsed_scores', TensorProto.FLOAT, ["-1", 200, 1])
        #     nmsed_class = make_tensor_value_info('nmsed_classes', TensorProto.FLOAT, ["-1", 200, 1])
        # else:
        #     num_detection = make_tensor_value_info('num_detections', TensorProto.INT32, [1, 1])
        #     nmsed_box = make_tensor_value_info('nmsed_boxes', TensorProto.FLOAT, [1, 200, 4])
        #     nmsed_score = make_tensor_value_info('nmsed_scores', TensorProto.FLOAT, [1, 200, 1])
        #     nmsed_class = make_tensor_value_info('nmsed_classes', TensorProto.FLOAT, [1, 200, 1])

        # ngraph.output.extend(
        #     nms,
        #     inputs=[
        #         boxes,
        #         scores,
        #         max_output_boxes_per_class,
        #         iou_threshold,
        #         score_threshold,
        #     ],
        #     outputs=[selected_indices],
        #     name="NonMaxSuppression",
        # )
        # selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [2, 3])
        # iou_threshold = helper.make_tensor_value_info('iou_threshold', TensorProto.INT64, [-1])

        ngraph.input.extend([helper.make_tensor_value_info('max_output_boxes_per_class', TensorProto.INT64, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('slice101_starts', TensorProto.INT64, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('slice101_ends', TensorProto.INT64, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('slice101_axes', TensorProto.INT64, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('slice101_steps', TensorProto.INT64, [1])])
        ngraph.input.extend([helper.make_tensor_value_info('squeeze101_axes', TensorProto.INT64, [1])])


        ngraph.output.extend([helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [-1, 3])])
        ngraph.output.extend([helper.make_tensor_value_info('gather101', TensorProto.FLOAT, [-1])])

        model_attrs = dict(
            ir_version=model_onnx.ir_version,
            opset_imports=model_onnx.opset_import,
            producer_version=model_onnx.producer_version,
            model_version=model_onnx.model_version
        )

        model = make_model(ngraph, **model_attrs)

        return model

    def run_onnx(self, model_path, img_path):
        import onnxruntime
        onnx_model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
        # output = ["selected_indices", 'output1', 'output2', 'gather101']
        output = ['gather101']
        image = Image.open(img_path)
        input_shape = 640
        letterbox_image = False
        org_h, org_w = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (input_shape, input_shape), letterbox_image)
        image_data.show()
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        results = onnx_model.run(output, {"images": image_data,
                                          "max_output_boxes_per_class": [20],
                                          "iou_threshold": [0.7],
                                          "score_threshold": [0.5],
                                          "slice101_starts": [2], "slice101_ends": [3],
                                          "slice101_axes": [1], "slice101_steps": [1],
                                          "squeeze101_axes": [1]
                                          })

        # class_names, num_classes = get_classes('model_data/toukui_classes.txt')
        # anchors, num_anchors = get_anchors('model_data/yolo_anchors.txt')
        img_org = cv2.imread(img_path)
        # img_org = cv2.resize(img_org, (input_shape, input_shape))
        # for item in results[3][0]:
        # 只有一个框的时候，会比多个框少一维
        if len(results[0].shape) == 2:
            results = results[0]
        else:
            results = results[0][0]
        scale_x = org_w / input_shape
        scale_y = org_h / input_shape
        for item in results:
            cx, cy, w, h, obj_cnf, cls = item
            # w, h = [10, 10]
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y
            img_org = cv2.rectangle(img_org, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            img_org = cv2.putText(img_org, str(cls) + '-' + str(obj_cnf)[0:4], (int(x1), int(y1)),
                                  cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 2)
        # cv2.imwrite(target_path, img_org)
        cv2.imshow('1', img_org)
        cv2.waitKey(-1)

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
