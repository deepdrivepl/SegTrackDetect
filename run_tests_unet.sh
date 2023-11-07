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

python inference_sequence.py --out_dir metrics/val/roi_track --flist val_list --debug --second_nms --mode 'roi_track' --merge
python inference_sequence.py --out_dir metrics/val/roi --flist val_list --debug --second_nms --mode 'roi' --merge
python inference_sequence.py --out_dir metrics/val/track --flist val_list --debug --second_nms --mode 'track' --merge
python inference_sequence.py --out_dir metrics/val/sw --flist val_list --debug --second_nms --mode 'sw' --merge
python inference_sequence.py --out_dir metrics/val/sd --flist val_list --debug --second_nms --mode 'sd' --merge