python inference_yolov7.py --det_ckpt trained_models/yolov7-w6-640-cd-crops-best.torchscript.pt --mode det
python inference_yolov7.py --det_ckpt trained_models/yolov7-w6-640-cd-crops-best.torchscript.pt --mode roi
python inference_track.py --det_ckpt trained_models/yolov7-w6-640-cd-crops-best.torchscript.pt --second_nms
python inference_track_roi.py --det_ckpt trained_models/yolov7-w6-640-cd-crops-best.torchscript.pt --second_nms
python inference_yolov7.py --det_ckpt trained_models/yolov7-w6-640-cd-crops-best.torchscript.pt --mode sw --second_nms
python inference_yolov7.py --det_ckpt trained_models/yolov7-w6-640-cd-crops-best.torchscript.pt --mode roi --second_nms