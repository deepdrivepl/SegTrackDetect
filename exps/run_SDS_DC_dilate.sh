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

# python inference_sequence.py --out_dir debug-errors/roi-fixes-resize-dilate/Zebra \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi' --merge --debug --dilate

# python inference_sequence.py --out_dir debug-errors/roi-fixes-resize-dilate/SDS \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS \
# --debug --dilate


# python inference_sequence.py --out_dir debug-errors/roi-fixes-resize-dilate/MTSD \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --second_nms --mode 'roi' --merge --debug --dilate


# python inference_sequence.py --out_dir debug-errors/roi-fixes-resize-dilate/DC \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC \
# --debug --dilate



# python inference_sequence.py --out_dir compare-dets/DC/v7tiny-512x512 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC \
#  --dilate


# python inference_sequence.py --out_dir compare-dets/DC/v7tiny-512x512-crops \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
#  --dilate


# python inference_sequence.py --out_dir compare-dets/DC/v7tiny-640x640-crops \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops_bigger \
#  --dilate

# python inference_sequence.py --out_dir compare-dets/DC/v7-640x640-vis \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-DC1 \
# --det_model yolov7_DC \
# --dilate --debug


# python inference_sequence.py --out_dir compare-dets/DC/v7e6e-1280x1280 \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-DC1 \
# --det_model yolov7e6e_DC \
#  --dilate


# python inference_sequence.py --out_dir compare-dets/SDS/v7tiny-512x512 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS \
#  --dilate


# python inference_sequence.py --out_dir compare-dets/SDS/v7tiny-512x512-crops \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops \
#  --dilate


# python inference_sequence.py --out_dir compare-dets/SDS/v7tiny-640x640-crops \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_bigger \
# --dilate

# python inference_sequence.py --out_dir compare-dets/SDS/v7-640x640 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-SDS1 \
# --det_model yolov7_SDS \
# --dilate


# python inference_sequence.py --out_dir compare-dets/SDS/v7e6e-1280x1280-vis \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --second_nms --mode 'roi' --merge \
# --roi_model unet-SDS1 \
# --det_model yolov7e6e_DC \
# --dilate --debug --vis_conf_th 0.1


