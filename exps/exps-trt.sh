# python inference_sequence_trt.py --out_dir ablation/embedded-trt85/trt-fp32 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS1_trt \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300_trt \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.1 --dilate --k_size 7


# python inference_sequence_trt.py --out_dir ablation/embedded-trt85/trt-fp16 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS1_trt_fp16 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300_trt_fp16 \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.1 --dilate --k_size 7 


# python inference_sequence_trt.py --out_dir ablation/embedded-trt85/trt-int8 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS1_trt_int8 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300_trt_int8 \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.1 --dilate --k_size 7 



python inference_sequence.py --out_dir ablation/embedded-trt85/ts-fp32 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_300 \
--mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.1 --dilate --k_size 7 #--debug