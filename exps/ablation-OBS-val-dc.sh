# # DC
# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NONE \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable  --debug

# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th09 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.9


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th08 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.8


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th07 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.7


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th06 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.6


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th05 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.5 --debug


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th04 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.4


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th03 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.3


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th02 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.2



# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-NMS-th01 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --obs_disable --second_nms --second_nms_iou_th 0.1


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th09 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.9


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th08 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.8


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th07 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.7


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th06 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.6


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th05 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.5 --debug


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th04 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.4


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th03 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.3


# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th02 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.2



# python inference_sequence.py --out_dir ablation/OBS-iou/val/DC-OBS-th01 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --second_nms --obs_iou_th 0.1



python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS-th05-sanityx2 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.5 --second_nms



python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th09 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.9


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th08 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.8


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th07 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.7


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th06 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.6


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th05 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.5 --debug


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th04 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.4


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th03 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.3


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th02 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.2



python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/val/DC-OBS_no_NMS-th01 \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge  --obs_iou_th 0.1