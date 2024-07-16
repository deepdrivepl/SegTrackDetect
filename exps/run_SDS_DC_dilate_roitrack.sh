# # python inference_sequence.py --out_dir tests/new_ds_metrics/DroneCrowd/roi_track \
# # --ds DroneCrowd --split test \
# # --second_nms --mode 'roi_track' --merge \
# # --roi_model unet-DC0 \
# # --det_model yolov7_tiny_DC


# # python inference_sequence.py --out_dir tests/new_ds_metrics/SeaDronesSee/roi-dc0 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --second_nms --mode 'roi' --merge \
# # --roi_model unet-SDS0 \
# # --det_model yolov7_tiny_SDS


# # python inference_sequence.py --out_dir tests/new_ds_metrics/SeaDronesSee/roi-dc1 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --second_nms --mode 'roi' --merge \
# # --roi_model unet-SDS1 \
# # --det_model yolov7_tiny_SDS

# s

# # python inference_sequence.py --out_dir tests/new_ds_metrics/SeaDronesSee/roi-dc2 \
# # --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# # --second_nms --mode 'roi' --merge \
# # --roi_model unet-SDS2 \
# # --det_model yolov7_tiny_SDS



# # python inference_sequence.py --out_dir tests/new_ds_metrics/DroneCrowd/roi-dc0 \
# # --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# # --second_nms --mode 'roi' --merge \
# # --roi_model unet-DC0 \
# # --det_model yolov7_tiny_DC


# # python inference_sequence.py --out_dir tests/new_ds_metrics/DroneCrowd/roi-dc1 \
# # --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# # --second_nms --mode 'roi' --merge \
# # --roi_model unet-DC1 \
# # --det_model yolov7_tiny_DC


# # python inference_sequence.py --out_dir tests/new_ds_metrics/DroneCrowd/roi-dc2 \
# # --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# # --second_nms --mode 'roi' --merge \
# # --roi_model unet-DC2 \
# # --det_model yolov7_tiny_DC

# python inference_sequence.py --out_dir debug-errors/roi-track-fixes-resize-binary-trk-vis/Zebra \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi_track' --merge  --binary_trk --debug

# python inference_sequence.py --out_dir debug-errors/roi-track-fixes-resize-binary-trk-vis/SDS \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --second_nms --mode 'roi_track' --merge \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS \
#  --binary_trk --debug


# # python inference_sequence.py --out_dir debug-errors/roi-fixes-resize-dilate/MTSD \
# # --ds MTSD --split val \
# # --det_model yolov4 --roi_model u2net \
# # --second_nms --mode 'roi_track' --merge --debug --dilate


# python inference_sequence.py --out_dir debug-errors/roi-track-fixes-resize-binary-trk-vis/DC \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --second_nms --mode 'roi_track' --merge \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC \
#  --binary_trk --debug



python inference_sequence.py --out_dir debug-errors/roi-track-binary-fixes-resize-dilate7-padding10-letterbox-thresholded-mask-check-mot-coords/SDS \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--second_nms --mode 'roi_track' --merge --debug --dilate --k_size 7 \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS \
 --binary_trk --debug


