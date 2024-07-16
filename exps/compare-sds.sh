python inference_sequence.py --out_dir ablation/compare-SDS/004-tiny-100ep-last \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize

python inference_sequence.py --out_dir ablation/compare-SDS/004-tiny-100ep-best \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100_best \
--mode 'roi' --merge --allow_resize

python inference_sequence.py --out_dir ablation/compare-SDS/006-tiny-300ep \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_300 \
--mode 'roi' --merge --allow_resize



python inference_sequence.py --out_dir ablation/compare-SDS/007-tiny-100ep-bigger \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100_bigger \
--mode 'roi' --merge --allow_resize


python inference_sequence.py --out_dir ablation/compare-SDS/005-v7 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_SDS_crops_mul_scales \
--mode 'roi' --merge --allow_resize