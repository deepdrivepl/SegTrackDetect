# # python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi-small \
# # --ds DroneCrowd --split val \
# # --roi_model unet_DC1_small \
# # --det_model yolov7_tiny_DC_crops_004 \
# # --mode 'roi' --merge --obs_iou_th 0.7 --debug

# python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi_track-small \
# --ds DroneCrowd --split val \
# --roi_model unet_DC1_small \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --obs_iou_th 0.7 --debug --binary_trk

# # python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi-tiny \
# # --ds DroneCrowd --split val \
# # --roi_model unet_DC1_tiny \
# # --det_model yolov7_tiny_DC_crops_004 \
# # --mode 'roi' --merge --obs_iou_th 0.7 --debug

# python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi_track-tiny \
# --ds DroneCrowd --split val \
# --roi_model unet_DC1_tiny \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --obs_iou_th 0.7 --debug --binary_trk


# # python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi-medium \
# # --ds DroneCrowd --split val \
# # --roi_model unet_DC1_medium \
# # --det_model yolov7_tiny_DC_crops_004 \
# # --mode 'roi' --merge --obs_iou_th 0.7 --debug

# python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi_track-medium \
# --ds DroneCrowd --split val \
# --roi_model unet_DC1_medium \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --obs_iou_th 0.7 --debug --binary_trk


# python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi-medium-th05 \
# --ds DroneCrowd --split val \
# --roi_model unet_DC1_medium \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --obs_iou_th 0.5

# python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi_track-medium-th05 \
# --ds DroneCrowd --split val \
# --roi_model unet_DC1_medium \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --obs_iou_th 0.5 --binary_trk



# python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi-small_dil \
# --ds DroneCrowd --split val \
# --roi_model unet_DC1_small \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --obs_iou_th 0.7 --dilate --k_size 7

# python inference_sequence.py --out_dir ablation/soa-dc/seg-res/noext-roi_track-small_dil \
# --ds DroneCrowd --split val \
# --roi_model unet_DC1_small \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk --dilate --k_size 7


python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/seg-res/beforefixres-roi-small_nodil \
--ds DroneCrowd --split val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7 

python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/seg-res/beforefixres-roi_track-small_nodil \
--ds DroneCrowd --split val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk