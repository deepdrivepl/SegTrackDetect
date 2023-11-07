# python ../yolov7-ZeF20/export.py --weights ../yolov7-configs/logs/yolov7-tiny/256x256-300/weights/best.pt --device 0 --grid --img-size 160 256
# cp ../yolov7-configs/logs/yolov7-tiny/256x256-300/weights/best.torchscript.pt trained_models/yolov7-tiny-300-best.torchscript.pt


python ../3rdparty/yolov7-ZeF20/export.py --weights ../logs/yolov7-tiny/256x256-new-trainval-split-300-fixed-cropped-ds/weights/best.pt --device 0 --grid --img-size 160 256
cp ../logs/yolov7-tiny/256x256-new-trainval-split-300-fixed-cropped-ds/weights/best.torchscript.pt trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-best.torchscript.pt
