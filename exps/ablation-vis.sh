# python inference_sequence.py --out_dir ablation/ROI-RES-VS-SW/DC-resize \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --allow_resize #--debug

# python inference_sequence.py --out_dir ablation/ROI-RES-VS-SW/DC-sw \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge

# python inference_sequence.py --out_dir ablation/ROI-RES-VS-SW/SDS-resize \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize

# python inference_sequence.py --out_dir ablation/ROI-RES-VS-SW/SDS-sw \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge


python inference_sequence_vis.py --out_dir ablation/ROI-RES-VS-SW/vis-fix/DC-resize-d7 \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.5 --debug --vis_conf_th 0.1

python inference_sequence_vis.py --out_dir ablation/ROI-RES-VS-SW/vis-fix/DC-sw-d7 \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --debug --vis_conf_th 0.1

# python inference_sequence_vis.py --out_dir ablation/ROI-RES-VS-SW/vis-fix/SDS-resize-d7 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --obs_iou_th 0.5 --debug --vis_conf_th 0.1

# python inference_sequence_vis.py --out_dir ablation/ROI-RES-VS-SW/vis-fix/SDS-sw-d7 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --debug --vis_conf_th 0.1


# vis 
# python inference_sequence_vis.py --out_dir ablation/ROI-RES-VS-SW/vis/DC-resize-d7 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --allow_resize  --dilate --k_size 7 --debug --vis_conf_th 0.2

# python inference_sequence_vis.py --out_dir ablation/OBS-iou/vis/SDS-NMS-th01 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --second_nms --obs_disable --vis_conf_th 0.1 --debug



# python inference_sequence.py --out_dir ablation/ROI-RES-VS-SW/SDS-resize-dilate \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7

# python inference_sequence.py --out_dir ablation/ROI-RES-VS-SW/SDS-sw-dilate \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --dilate --k_size 7


# # FIXES - move det_window, move sw, extend sw, a bit better filtering, but still sucks
# python inference_sequence_vis.py --out_dir ablation/fix-res-sw/move-roi-extend-sw-move-sw-fix-filter-full/DC-val-sw \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge  --dilate --obs_iou_th 0.5 --k_size 7 --debug --vis_conf_th 0.1

# python inference_sequence_vis.py --out_dir ablation/fix-res-sw/move-roi-extend-sw-move-sw-fix-filter-full/DC-val-resize \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --allow_resize --dilate --obs_iou_th 0.5 --k_size 7 --debug --vis_conf_th 0.1