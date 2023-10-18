python ../yolov7-ZeF20/export.py --weights ../yolov7-configs/logs/yolov7-tiny/256x256-300/weights/best.pt --device 0 --grid --img-size 160 256
cp ../yolov7-configs/logs/yolov7-tiny/256x256-300/weights/best.torchscript.pt trained_models/yolov7-tiny-300-best.torchscript.pt
