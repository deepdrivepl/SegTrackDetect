
python inference_sequence.py --out_dir ablation/SAM-vs-BBOXES/SDS-bboxes0 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS0 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize



python inference_sequence.py --out_dir ablation/SAM-vs-BBOXES/SDS-bboxes1 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize



python inference_sequence.py --out_dir ablation/SAM-vs-BBOXES/SDS-bboxes2 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS2 \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize




python inference_sequence.py --out_dir ablation/SAM-vs-BBOXES/SDS-masks0 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS0-masks \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize



python inference_sequence.py --out_dir ablation/SAM-vs-BBOXES/SDS-masks1 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS1-masks \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize



python inference_sequence.py --out_dir ablation/SAM-vs-BBOXES/SDS-masks2 \
--ds SeaDronesSee --flist data/SeaDronesSee/test_dev.txt --name test_dev \
--roi_model unet-SDS2-masks \
--det_model yolov7_tiny_SDS_crops_mul_scales_100 \
--mode 'roi' --merge --allow_resize