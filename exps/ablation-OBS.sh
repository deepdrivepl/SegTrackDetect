# # # SDS with resize
# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NONE \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th09 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.9


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th08 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.8


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th07 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.7


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th06 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.6


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th05 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.5


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th04 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.4


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th03 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.3


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th02 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.2



# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-NMS-th01 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_disable --second_nms --second_nms_iou_th 0.1


# # DC no resize
# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-disabled \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_disable


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th05 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.5



# Zebra
# # TEST if anything changed since OBS
# python inference_sequence.py --out_dir ablation/OBS-iou/Zebra-OBS-th01-sanity-check \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --second_nms --obs_iou_th 0.1


# python inference_sequence.py --out_dir ablation/OBS-iou/Zebra-NONE \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --obs_disable --dilate --k_size 7


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th09 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.9


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th08 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.8


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th07 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.7


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th06 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.6


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th05 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.5


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th04 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.4


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th03 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.3


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th02 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.2


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS-th01 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.1


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th09 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.9


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NONE_no_dil \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --obs_disable


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th08 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.8 


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th07 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.7 


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th06 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.6 


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th05 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.5 


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th04 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.4 


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th03 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.3 


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th02 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.2 


python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS+NMS_no_dil-th01 \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'roi' --merge --allow_resize --second_nms  --obs_iou_th 0.1 


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th09 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.9


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th08 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.8


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th07 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.7


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th06 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.6


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th05 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.5


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th04 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.4


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th03 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.3


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th02 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.2


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-OBS_no_NMS_no_dil-th01 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --obs_iou_th 0.1


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th09 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.9


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th08 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.8


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th07 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.7


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th06 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.6


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th05 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.5


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th04 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.4


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th03 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.3


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th02 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.2


# python inference_sequence_before_resize_fix.py --out_dir ablation/OBS-iou/Zebra-NMS_no_dil-th01 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize  --second_nms --obs_disable --second_nms_iou_th 0.1


# # DC cd
# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th09 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.9


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th08 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.8


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th07 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.7


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th06 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.6


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th04 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.4


# python inference_sequence.py --out_dir ablation/OBS-iou/SDS-OBS-th01-NMS-disabled \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --obs_iou_th 0.1


# python inference_sequence.py --out_dir ablation/OBS-iou/Zebra-OBS-th02-NMS-disabled \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.2


# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th03 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.3

# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th02 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.2

# python inference_sequence.py --out_dir ablation/OBS-iou/DC-OBS-th01 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --second_nms --obs_iou_th 0.1


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
