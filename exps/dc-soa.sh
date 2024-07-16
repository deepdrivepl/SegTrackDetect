# yolov7 model

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/models/001-512x512 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms


# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/models/001-320x512 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_rect \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/models/003-512x512 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_003 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/models/004-512x512 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/models/005-512x512 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_005 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/models/006-512x512 \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_006 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms


# different sw strategies, fixed model (004)
# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/det-windows/before_resize/seg \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/det-windows/before_resize/seg_track \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms --binary_trk


# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize/seg \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize/seg_track \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms --binary_trk


# python inference_sequence_ul.py --out_dir ablation/soa-dc/det-windows/fixes_resize_ul/seg \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence_ul.py --out_dir ablation/soa-dc/det-windows/fixes_resize_ul/seg_track \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms --binary_trk

# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize_noext/seg \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize_noext/seg_track \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms --binary_trk


# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize_noextx2/seg \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms

# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize_noextx2/seg_track \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms --binary_trk



# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize_noextx2-deb/seg \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms --debug

# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize_noextx2-deb/seg_track \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.5 --second_nms --binary_trk --debug


# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize_noextx2_noNMS/seg \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5

# python inference_sequence.py --out_dir ablation/soa-dc/det-windows/fixes_resize_noextx2_noNMS/seg_track \
# --ds DroneCrowd --split val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.5 --binary_trk


# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/seg-noNMS-obs05-beforeres \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.5

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/seg_track-noNMS-obs05-beforeres \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.5 --binary_trk


# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg-noNMS-obs07-beforeres-nodil-segsmall \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet_DC1_small \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --obs_iou_th 0.7

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg_track-noNMS-obs07-beforeres-nodil-segsmall \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet_DC1_small \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk


# python inference_sequence.py --out_dir ablation/soa-dc/test/seg-noNMS-obs07-noext-nodil-segsmall \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet_DC1_small \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --obs_iou_th 0.7

# python inference_sequence.py --out_dir ablation/soa-dc/test/seg_track-noNMS-obs07-noext-nodil-segsmall \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet_DC1_small \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk


# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg-noNMS-obs07-beforeres \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.7

# python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg_track-noNMS-obs07-beforeres \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.7 --binary_trk



# python inference_sequence.py --out_dir ablation/soa-dc/test/seg-noNMS-obs07-noext \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi' --merge --dilate --k_size 7 --obs_iou_th 0.7

# python inference_sequence.py --out_dir ablation/soa-dc/test/seg_track-noNMS-obs07-noext \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_004 \
# --mode 'roi_track' --merge --dilate --k_size 7 --obs_iou_th 0.7 --binary_trk


# sanity check / best table
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/sanity-check-seg \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi' --merge --dilate --k_size 7 --second_nms 


# sanity check / best table + track
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/sanity-check-seg_track \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--mode 'roi_track' --merge --dilate --k_size 7 --second_nms --binary_trk


# IoU th NMS -> 0.65 (windows) + seg
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg_track-noNMS-obs07-beforeagg-nodil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk


# IoU th NMS -> 0.65 (windows) + second nms + seg_track
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg_track-NMS-obs07-beforeagg-nodil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk --second_nms

# IoU th NMS -> 0.65 (windows) + second nms + seg
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg-NMS-obs07-beforeagg-nodil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7 --second_nms


# IoU th NMS -> 0.65 (windows) + seg_track
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg-noNMS-obs07-beforeagg-nodil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7


# IoU th NMS -> 0.65 (windows) + seg + dil
python inference_sequence.py --out_dir ablation/soa-dc/test/seg_track-noNMS-obs07-afteragg-dil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk --dilate --k_size 7

# IoU th NMS -> 0.65 (windows) + seg
python inference_sequence.py --out_dir ablation/soa-dc/test/seg_track-noNMS-obs07-afteragg-nodil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk


# IoU th NMS -> 0.65 (windows) + second nms + seg_track
python inference_sequence.py --out_dir ablation/soa-dc/test/seg_track-NMS-obs07-afteragg-nodil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk --second_nms


# IoU th NMS -> 0.65 (windows) + seg + dil
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg_track-noNMS-obs07-beforeagg-dil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk --dilate --k_size 7



# IoU th NMS -> 0.65 (windows) + seg + dil + second NMS
python inference_sequence.py --out_dir ablation/soa-dc/test/seg_track-NMS-obs07-afteragg-dil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi_track' --merge --obs_iou_th 0.7 --binary_trk --dilate --k_size 7 --second_nms


# IoU th NMS -> 0.65 (windows) + second nms + seg
python inference_sequence.py --out_dir ablation/soa-dc/test/seg-NMS-obs07-afteragg-nodil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7 --second_nms


# IoU th NMS -> 0.65 (windows) + seg_track
python inference_sequence.py --out_dir ablation/soa-dc/test/seg-noNMS-obs07-afteragg-nodil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7


# IoU th NMS -> 0.65 (windows) + seg_track + dil
python inference_sequence.py --out_dir ablation/soa-dc/test/seg-noNMS-obs07-afteragg-dil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7 --dilate --k_size 7


# IoU th NMS -> 0.65 (windows) + seg_track + dil + second NMS
python inference_sequence.py --out_dir ablation/soa-dc/test/seg-NMS-obs07-afteragg-dil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7 --dilate --k_size 7 --second_nms


# IoU th NMS -> 0.65 (windows) + seg_track + dil
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg-noNMS-obs07-beforeagg-dil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7 --dilate --k_size 7



# IoU th NMS -> 0.65 (windows) + seg_track + dil + second NMS
python inference_sequence_before_resize_fix.py --out_dir ablation/soa-dc/test/seg-NMS-obs07-beforeagg-dil-segsmall-old_det_th \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet_DC1_small \
--det_model yolov7_tiny_DC_crops_004 \
--mode 'roi' --merge --obs_iou_th 0.7 --dilate --k_size 7 --second_nms




