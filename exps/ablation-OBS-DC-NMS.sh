# OBS only

python inference_sequence.py --out_dir ablation/OBS-iou/Zebra-OBSonly-th01 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.1


python inference_sequence.py --out_dir ablation/OBS-iou/SDS-OBSonly-th01 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize --obs_iou_th 0.1

python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBSonly-th05 \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge --obs_iou_th 0.5 







# DC no resize

# # sanity check
# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th05-sanity-check \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.5 





# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th09 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable --second_nms --second_nms_iou_th 0.9



# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NONE \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th08 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable --second_nms --second_nms_iou_th 0.8


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th07 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable --second_nms --second_nms_iou_th 0.7


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th06 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable --second_nms --second_nms_iou_th 0.6

# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th05 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable --second_nms --second_nms_iou_th 0.5


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th04 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge ---obs_disable --second_nms --second_nms_iou_th 0.4


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th03 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable --second_nms --second_nms_iou_th 0.3

# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th02 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable --second_nms --second_nms_iou_th 0.2

# python inference_sequence.py --out_dir ablation/OBS-iou/DC-NMS-th01 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --obs_disable --second_nms --second_nms_iou_th 0.1


# # MTSD
# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-disabled \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --obs_disable --dilate --k_size 7

# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th09 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.9


# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th08 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.8


# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th07 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.7


# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th06 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.6


# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th05 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.5


# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th04 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.4


# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th03 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.3


# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th02 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.2


# python inference_sequence.py --out_dir ablation/OBS-iou/MTSD-OBS-th01 \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --mode 'roi' --merge --allow_resize --second_nms --dilate --k_size 7 --obs_iou_th 0.1
