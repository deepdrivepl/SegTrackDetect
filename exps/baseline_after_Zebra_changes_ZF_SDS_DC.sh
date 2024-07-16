# # ZebraFish
# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-track-p10-d7-letterbox-bintrk/Zebra \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi_track' --merge --dilate --k_size 7 --debug --binary_trk

# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/Zebra \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi' --merge --dilate --k_size 7 --debug


# # SDS
# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/SDS \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS \
# --second_nms --mode 'roi' --merge --dilate --k_size 7 --debug

# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/SDS-crops \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops \
# --second_nms --mode 'roi' --merge --dilate --k_size 7 --debug


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-track-p10-d7-letterbox-bintrk/SDS \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS \
# --second_nms --mode 'roi_track' --merge --dilate --k_size 7 --debug --binary_trk

# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-track-p10-d7-letterbox-bintrk/SDS-crops-incmot \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops \
# --second_nms --mode 'roi_track' --merge --dilate --k_size 7 --debug --binary_trk


# # DC
# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/DC \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC \
# --second_nms --mode 'roi' --merge --dilate --k_size 7 --debug

# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/DC-crops \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --second_nms --mode 'roi' --merge --dilate --k_size 7 --debug


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-track-p10-d7-letterbox-bintrk/DC \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC \
# --second_nms --mode 'roi_track' --merge --dilate --k_size 7 --debug --binary_trk

# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-track-p10-d7-letterbox-bintrk/DC-crops \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --second_nms --mode 'roi_track' --merge --dilate --k_size 7 --debug --binary_trk


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/MTSD-article-nms \
# --ds MTSD --split val \
# --det_model yolov4 --roi_model u2net \
# --second_nms --mode 'roi' --merge --dilate --k_size 7 --debug


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/SDS-crops-mul-scales-100 \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1 \
# --det_model yolov7_tiny_SDS_crops_mul_scales_100 \
# --second_nms --mode 'roi' --merge --dilate --k_size 7 # --debug


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d0-letterbox/SDS-unet-masks0-crops-mul-scales \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS0-masks \
# --det_model yolov7_tiny_SDS_crops_mul_scales \
# --second_nms --mode 'roi' --merge


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d0-letterbox/SDS-unet-masks1-crops-mul-scales \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1-masks \
# --det_model yolov7_tiny_SDS_crops_mul_scales \
# --second_nms --mode 'roi' --merge


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d0-letterbox/SDS-unet-masks2-crops-mul-scales \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS2-masks \
# --det_model yolov7_tiny_SDS_crops_mul_scales \
# --second_nms --mode 'roi' --merge



# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/SDS-unet-masks0-crops-mul-scales \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS0-masks \
# --det_model yolov7_tiny_SDS_crops_mul_scales \
# --second_nms --mode 'roi' --merge --dilate --k_size 7


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/SDS-unet-masks1-crops-mul-scales \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS1-masks \
# --det_model yolov7_tiny_SDS_crops_mul_scales \
# --second_nms --mode 'roi' --merge --dilate --k_size 7


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/SDS-unet-masks2-crops-mul-scales \
# --ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
# --roi_model unet-SDS2-masks \
# --det_model yolov7_tiny_SDS_crops_mul_scales \
# --second_nms --mode 'roi' --merge --dilate --k_size 7


# python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d7-letterbox/DC-crops-noresize-debug \
# --ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
# --roi_model unet-DC1 \
# --det_model yolov7_tiny_DC_crops \
# --second_nms --mode 'roi' --merge  --dilate --k_size 7 --debug


python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d0-letterbox/DC-crops-noresize \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--second_nms --mode 'roi' --merge

python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-p10-d0-letterbox/DC-noresize \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC \
--second_nms --mode 'roi' --merge

python inference_sequence.py --out_dir debug-errors/after-ZF-changes/roi-track-p10-d0-letterbox/DC-crops-noresize \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--second_nms --mode 'roi_track' --merge --debug

python inference_sequence.py --out_dir debug-errors/after-ZF-changes/track-p10-d0-letterbox/DC-crops-noresize \
--ds DroneCrowd --flist data/DroneCrowd/test_minus_val.txt --name test_minus_val \
--roi_model unet-DC1 \
--det_model yolov7_tiny_DC_crops \
--second_nms --mode 'track' --merge --debug