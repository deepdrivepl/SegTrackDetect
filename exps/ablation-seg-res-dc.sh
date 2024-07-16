# # # # SDS with resize
# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi-tiny \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_tiny \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 --debug


# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi-small \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_small \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 #--debug


# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi-medium \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_medium \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 #--debug


# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi-large \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_large \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 #--debug


# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi_track_bintrk-tiny \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_tiny \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk


# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi_track_bintrk-small \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_small \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk


# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi_track_bintrk-medium \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_medium \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk


# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi_track_bintrk-large \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_large \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk


# python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-track_bintrk \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk



# SDS - vis
# python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article/SDS-roi-tiny \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 --debug


# python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article/SDS-roi_track_bintrk-tiny \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk --debug


python inference_sequence_vis_seg_res_dc.py --out_dir ablation/SEG-RES/vis-dc2/DC-roi \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge --second_nms --obs_iou_th 0.5 --debug


python inference_sequence_vis_seg_res_dc.py --out_dir ablation/SEG-RES/vis-dc2/DC-roi_track \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi_track' --merge --second_nms --obs_iou_th 0.5 --debug --binary_trk


python inference_sequence_vis_seg_res_dc.py --out_dir ablation/SEG-RES/vis-dc2/DC-track \
--ds DroneCrowd --split val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'track' --merge --second_nms --obs_iou_th 0.5 --debug --binary_trk