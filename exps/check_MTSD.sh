# python inference_sequence.py --out_dir test-MTSD/MTSD-dil \
# --ds MTSD --split val --det_model yolov4 --roi_model u2net \
# --second_nms --mode 'roi' --merge --dilate --k_size 7


python inference_sequence.py --out_dir test-MTSD/MTSD-nodil \
--ds MTSD --split val --det_model yolov4 --roi_model u2net \
--second_nms --mode 'roi' --merge


python inference_sequence.py --out_dir test-MTSD/MTSD-nodil-naive \
--ds MTSD --split val --det_model yolov4 --roi_model u2net \
--second_nms --mode 'roi' --merge --bbox_type naive


python inference_sequence.py --out_dir test-MTSD/MTSD-nodil-noobs \
--ds MTSD --split val --det_model yolov4 --roi_model u2net \
--second_nms --mode 'roi' --merge --obs_type none