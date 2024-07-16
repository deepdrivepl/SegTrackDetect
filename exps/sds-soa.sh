# # # sanity check / best table + new code
# # python inference_sequence.py --out_dir ablation/soa-sds/ss_sds1-mulscales100_d7_nms65_obs07 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet-SDS1 \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge  --allow_resize --dilate --k_size 7 --second_nms --obs_iou_th 0.7


# # # sanity check / best table + new code
# # python inference_sequence.py --out_dir ablation/soa-sds/st_sds1-mulscales100_d7_nms65_obs07 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet-SDS1 \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --dilate --k_size 7 --second_nms --obs_iou_th 0.7


# # python inference_sequence.py --out_dir ablation/soa-sds/st_sds1-mulscales100_d7_nms00_obs07 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet-SDS1 \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --dilate --k_size 7 --obs_iou_th 0.7


# # python inference_sequence.py --out_dir ablation/soa-sds/st_sds1-mulscales100_d7_nms00_obs05 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet-SDS1 \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --dilate --k_size 7 --obs_iou_th 0.5


# # python inference_sequence.py --out_dir ablation/soa-sds/st_sds1-mulscales100_d0_nms00_obs05 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet-SDS1 \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5


# # python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales100_d0_nms00_obs05 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_small \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5

# # # dets
# # python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales_d0_nms00_obs05 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_small \
# # --det_model yolov7_tiny_SDS_crops_mul_scales \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5


# # python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales100best_d0_nms00_obs05 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_small \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100_best \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5

# # python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales300_d0_nms00_obs05 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_small \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_300 \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5

# # python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales100bigger_d0_nms00_obs05 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_small \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100_bigger \
# # --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5


# # st, no NMS, mulscales 300 - fixed
# # dilate or not
# python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales300_d7_nms00_obs05 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_small \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300 \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5 --dilate --k_size 7 

# # different OBS th
# python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales300_d7_nms00_obs04 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_small \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300 \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.4 --dilate --k_size 7 

# python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales300_d7_nms00_obs03 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_small \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300 \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.3 --dilate --k_size 7 

# python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales300_d7_nms00_obs02 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_small \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300 \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.2 --dilate --k_size 7 

# python inference_sequence.py --out_dir ablation/soa-sds/st_sdss-mulscales300_d7_nms00_obs01 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_small \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300 \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.1 --dilate --k_size 7 


# dilate SDS1
python inference_sequence.py --out_dir ablation/soa-sds/st_sds1-mulscales300_d7_nms00_obs01 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_300 \
--mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.1 --dilate --k_size 7 

# nodil SDS1
python inference_sequence.py --out_dir ablation/soa-sds/st_sds1-mulscales300_d0_nms00_obs01 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_300 \
--mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.1


# higher ths
python inference_sequence.py --out_dir ablation/soa-sds/st_sds1-mulscales300_d7_nms00_obs05 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_300 \
--mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5 --dilate --k_size 7 

# nodil SDS1
python inference_sequence.py --out_dir ablation/soa-sds/st_sds1-mulscales300_d0_nms00_obs05 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_300 \
--mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5

# different seg models - later


# python inference_sequence_full_frame.py --out_dir ablation/soa-sds/st_sdss-mulscales300_d0_nms00_obs05_ff \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_small \
# --det_model yolov7_tiny_SDS_crops_mul_scales_300 \
# --mode 'roi_track' --binary_trk --merge  --allow_resize --obs_iou_th 0.5