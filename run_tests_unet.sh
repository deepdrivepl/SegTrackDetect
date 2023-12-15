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


python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence_same_window_check.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-same-win/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'



python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/sw --flist val_list --second_nms --mode 'sw' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
python inference_sequence_conf_iou_area.py --out_dir metrics/val-fixed-seq-empty-added-test-rotate-IBS-online-single-matrix-conf-area-iou/sd --flist val_list --second_nms --mode 'sd' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'