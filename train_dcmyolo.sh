for i in $(ps -ax |grep train_dcmyolo |awk '{print $1}')
do
 id=`echo $i |awk -F"/" '{print $1}'`
 kill -9  $id
done

nohup python -u train_dcmyolo.py \
	--classes_path dcmyolo/model_data/wangzhe_classes.txt \
	--anchors_path dcmyolo/model_data/coco_anchors.txt \
	--train_annotation_path	data/wangzhe/train.txt \
	--val_annotation_path data/wangzhe/val.txt \
	--phi s \
	--backbone_model_dir dcmyolo/model_data \
	--model_path dcmyolo/model_data/yolov5_s_v6.1.pth \
	--input_shape 640 640 \
	--batch_size 4 \
	--epoch 1000 \
	--save_period 100 \
	>  log_train_dcmyolo.log &
tail -f log_train_dcmyolo.log
