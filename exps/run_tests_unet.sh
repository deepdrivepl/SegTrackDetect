# python inference_track_roi.py --out_dir unet-results-new-trainval-split --use_threshold
# python inference_track_roi.py --out_dir unet-results-new-trainval-split 
# python inference_track_roi.py --out_dir unet-results --weights "trained_models/unet-r18_005_best_model_loss.pt" 
# python inference_track_roi.py --out_dir unet-results-new-trainval-split/semseg --weights "trained_models/unet-r18_007-new-trainval-split-semseg_best-model-loss.pt"
# python inference_track_roi.py --out_dir unet-results-new-trainval-split/semseg --weights "trained_models/unet-r18_007-new-trainval-split-semseg_best-model-loss.pt" --use_threshold
# python inference_track_roi.py --out_dir debug/unet-old-new-semseg-compare-th127 --weights "trained_models/unet-r18_005_best_model_loss.pt"
# python inference_track_roi.py --out_dir debug/unet-old-new-semseg-compare-th127 --weights "trained_models/unet-r18_006-new-trainval-split_best-model-loss.pt"
# python inference_track_roi.py --out_dir debug/unet-old-new-semseg-compare-th127 --weights "trained_models/unet-r18_007-new-trainval-split-semseg_best-model-loss.pt"

# python inference_track_roi.py --out_dir debug/final-test-old-no-thx2 --weights "trained_models/unet-r18_005_best_model_loss.pt"

# python inference_sequence.py --out_dir debug/final-test-old-no-thx2-bboxes-included

# python inference_sequence.py --out_dir debug/test-vis --flist test_list --debug

# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added/track --flist val_list --debug --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added/roi --flist val_list --debug --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added/roi_track --flist val_list --debug --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added/sw --flist val_list --debug --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added/sd --flist val_list --debug --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-less-empty/track --flist val_list --debug --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-less-empty-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-less-empty/roi --flist val_list --debug --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-less-empty-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-less-empty/roi_track --flist val_list --debug --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-less-empty-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-less-empty/sw --flist val_list --debug --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-less-empty-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-less-empty/sd --flist val_list --debug --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-less-empty-best.torchscript.pt'


# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate/track --flist val_list --debug --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate/roi --flist val_list --debug --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate/roi_track --flist val_list --debug --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate/sw --flist val_list --debug --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate/sd --flist val_list --debug --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS/track-fix-2 --flist val_list --debug --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS/roi-fix-2 --flist val_list --debug --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online/track --flist val_list --debug --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online/roi --flist val_list --debug --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online/roi_track --flist val_list --debug --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online/sw --flist val_list --debug --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online/sd --flist val_list --debug --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


# python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'



# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'



# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'conf' 
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'conf'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'conf'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'conf'


# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-iou/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'iou' 
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-iou/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'iou'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-iou/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'iou'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-iou/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'iou'


# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-area/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'area' 
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-area/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'area'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-area/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'area'
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-area/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'area'


# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-06-iou-obs/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all'  --obs_iou_th 0.6
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-06-iou-obs/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --obs_iou_th 0.6
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-06-iou-obs/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --obs_iou_th 0.6
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-06-iou-obs/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --obs_iou_th 0.6
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-06-iou-obs/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --obs_iou_th 0.6


# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-05-iou-obs/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all'  --obs_iou_th 0.5
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-05-iou-obs/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --obs_iou_th 0.5
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-05-iou-obs/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --obs_iou_th 0.5
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-05-iou-obs/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --obs_iou_th 0.5
# python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-05-iou-obs/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --obs_iou_th 0.5



# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-same-window-check/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' 
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-same-window-check/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-same-window-check/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-same-window-check/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all'
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-same-window-check/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all'

# max_age = 10
# python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-max-age-10/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --max_age 10
python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-max-age-10/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --max_age 10


# max_age = 1
python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all-max-age-1/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --max_age 1
python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-all--max-age-1/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' --obs_type 'all' --max_age 1