# # # # # SDS with resize
# # # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi-tiny \
# # # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # # --roi_model unet_SDS_tiny \
# # # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 --debug


# # # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi-small \
# # # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # # --roi_model unet_SDS_small \
# # # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 #--debug


# # # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi-medium \
# # # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # # --roi_model unet_SDS_medium \
# # # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 #--debug


# # # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi-large \
# # # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # # --roi_model unet_SDS_large \
# # # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 #--debug


# # # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi_track_bintrk-tiny \
# # # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # # --roi_model unet_SDS_tiny \
# # # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk


# # # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi_track_bintrk-small \
# # # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # # --roi_model unet_SDS_small \
# # # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk


# # # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi_track_bintrk-medium \
# # # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # # --roi_model unet_SDS_medium \
# # # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk


# # # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-roi_track_bintrk-large \
# # # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # # --roi_model unet_SDS_large \
# # # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk


# # python inference_sequence.py --out_dir ablation/SEG-RES/vis/SDS-track_bintrk \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk



# # SDS - vis
# # python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article/SDS-roi-tiny \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_tiny \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 --debug


# # python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article/SDS-roi_track_bintrk-tiny \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_tiny \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk --debug


# # python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article/SDS-roi-medium \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_medium \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 --debug



# # python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article/SDS-roi_track_bintrk-medium \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_medium \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk --debug


# # python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article/SDS-roi-large \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_large \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 --debug


# # python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article/SDS-roi_track_bintrk-large \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_large \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk --debug


# python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article2/SDS-track \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk --debug


# # python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article2/SDS-roi-small \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --roi_model unet_SDS_small \
# # --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# # --mode 'roi' --merge --allow_resize --second_nms --obs_iou_th 0.1 --debug


# python inference_sequence_vis_seg_res.py --out_dir ablation/SEG-RES/vis-article2/SDS-roi_track_bintrk-small \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_small \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi_track' --merge --allow_resize --second_nms --obs_iou_th 0.1 --binary_trk --debug


# python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/seg-NMS \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --second_nms --second_nms_iou_th 0.1 --obs_disable --debug


# python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/seg-NMS065 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --second_nms --second_nms_iou_th 0.65 --obs_disable --debug

# python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/seg_track-NMS065 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi_track' --merge --allow_resize --second_nms --second_nms_iou_th 0.65 --obs_disable --debug --binary_trk


# python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/seg_track-OBS065 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi_track' --merge --allow_resize --obs_iou_th 0.65 --debug --binary_trk


# python inference_sequence_vis.py --out_dir ablation/OBS-iou/vis/SDS-NMS-th02 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --second_nms --obs_disable --vis_conf_th 0.1 --debug

python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/new/seg-nms065 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet_SDS_tiny \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize --obs_disable --second_nms --vis_conf_th 0.1 --debug

python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/new/seg_track-nms065 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet_SDS_tiny \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi_track' --merge --allow_resize --obs_disable --second_nms --binary_trk --vis_conf_th 0.1 --debug

python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/new/seg_track-obs065 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet_SDS_tiny \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi_track' --merge --allow_resize --obs_iou_th 0.65 --binary_trk --debug --vis_conf_th 0.14


# python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/seg-NMS03 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi' --merge --allow_resize --second_nms --second_nms_iou_th 0.3 --obs_disable --debug

# python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/seg_track-NMS03 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi_track' --merge --allow_resize --second_nms --second_nms_iou_th 0.3 --obs_disable --debug --binary_trk


# python inference_sequence_before_resize_fix_vis.py --out_dir ablation/intro/seg_track-OBS03 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet_SDS_tiny \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --mode 'roi_track' --merge --allow_resize --obs_iou_th 0.3 --debug --binary_trk



