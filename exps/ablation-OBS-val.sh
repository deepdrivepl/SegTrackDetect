# SDS with resize
# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NONE \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable

# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th09 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.9


# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th08 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.8


# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th07 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.7


# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th06 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.6


# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th05 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.5


# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th04 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.4


# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th03 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.3


# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th02 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.2



# python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-NMS-th01 \
# --ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.1


python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th09 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize --obs_iou_th 0.9


python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th08 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize --obs_iou_th 0.8


python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th07 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize  --obs_iou_th 0.7


python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th06 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize  --obs_iou_th 0.6


python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th05 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize  --obs_iou_th 0.5


python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th04 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize  --obs_iou_th 0.4


python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th03 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize  --obs_iou_th 0.3


python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th02 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize  --obs_iou_th 0.2



python inference_sequence.py --out_dir ablation/OBS-iou/val/SDS-OBS_no_NMS-th01 \
--ds SeaDronesSee --flist data/SeaDronesSee/val-ours.txt --name val-ours \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize  --obs_iou_th 0.1