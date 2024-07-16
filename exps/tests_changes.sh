# # python inference_sequence.py --out_dir tests/rm-sw-single-det/track --flist val_list --second_nms --mode 'track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# # python inference_sequence.py --out_dir tests/rm-sw-single-det/roi --flist val_list --second_nms --mode 'roi' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'
# # python inference_sequence.py --out_dir tests/rm-sw-single-det/roi_track --flist val_list --second_nms --mode 'roi_track' --merge --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


# # ############################ DET weights #############################################################

# # python inference_sequence.py --out_dir tests/det-models/yolov7-tiny-300-best --flist val_list \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-300-best.torchscript.pt'


# # python inference_sequence.py --out_dir tests/det-models/yolov7-tiny-new-trainval-split-300-best --flist val_list \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-best.torchscript.pt'


# # python inference_sequence.py --out_dir tests/det-models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-best --flist val_list \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-best.torchscript.pt'


# # python inference_sequence.py --out_dir tests/det-models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best --flist val_list \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt'


# # python inference_sequence.py --out_dir tests/det-models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-less-empty-best --flist val_list \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-less-empty-best.torchscript.pt'

# # ############################ ROI weights #############################################################

# # python inference_sequence.py --out_dir tests/roi-models/unet-r18_005_best_model_loss --flist val_list \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' \
# # --roi_weights 'trained_models/unet-r18_005_best_model_loss.pt'


# # python inference_sequence.py --out_dir tests/roi-models/unet-r18_006-new-trainval-split_best-model-loss --flist val_list \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' \
# # --roi_weights 'trained_models/unet-r18_006-new-trainval-split_best-model-loss.pt'


# # python inference_sequence.py --out_dir tests/roi-models/unet-r18_007-new-trainval-split-semseg_best-model-loss --flist val_list \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' \
# # --roi_weights 'trained_models/unet-r18_007-new-trainval-split-semseg_best-model-loss.pt'


# # # python inference_sequence_ds_new.py --out_dir tests/tests2 \
# # # --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# # # --second_nms --mode 'roi_track' --merge \
# # # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' \
# # # --roi_weights 'trained_models/unet-r18_007-new-trainval-split-semseg_best-model-loss.pt'


# # python inference_sequence_ds_new.py --out_dir tests/tests2 \
# # --ds ZebraFish --split 'test' \
# # --second_nms --mode 'roi_track' --merge \
# # --det_weights 'trained_models/yolov7-tiny-new-trainval-split-300-fixed-cropped-ds-empty-added-best.torchscript.pt' \
# # --roi_weights 'trained_models/unet-r18_007-new-trainval-split-semseg_best-model-loss.pt'


# # python inference_sequence_ds_new.py --out_dir tests/test --split test \
# # --second_nms --mode 'track' --merge


# # python inference_sequence_ds_new.py --out_dir tests/preproc/normalize_no_letterbox_v2 --split val --second_nms --mode 'roi_track' --merge #--debug

# # python inference_sequence.py --out_dir tests/preproc/normalize_no_letterbox_v3 --flist data/3DZeF20/train.txt --name train --second_nms --mode 'roi_track' --merge #--debug


# # python inference_sequence.py --out_dir tests/preproc/normalize_no_letterbox_v4 --split train --second_nms --mode 'roi_track' --merge #--debug


# # python inference_sequence.py --out_dir tests/preproc/normalize_no_letterbox_v5 --flist data/3DZeF20/val.txt --name val --second_nms --mode 'roi_track' --merge #--debug


# # python inference_sequence.py --out_dir tests/preproc/normalize_no_letterbox_v6 --split test --second_nms --mode 'roi_track' --merge #--debug


# # # python inference_sequence.py --out_dir tests/mtsd/train1 --ds MTSD --split train --second_nms --mode 'roi_track' --merge --debug
# # python inference_sequence.py --out_dir tests/mtsd/val3 --ds MTSD --split val --second_nms --mode 'roi_track' --merge --debug --det_model yolov4 --roi_model u2net
# # # python inference_sequence.py --out_dir tests/mtsd/test1 --ds MTSD --split test --second_nms --mode 'roi_track' --merge --debug

# # # python inference_sequence.py --out_dir tests/zebra/val1 --flist data/3DZeF20/val.txt --name val --second_nms --mode 'roi_track' --merge --debug

# # python inference_sequence.py --ds SeaDronesSee --out_dir tests/SDS/t1 --flist data/SeaDronesSee/train.txt --name train --second_nms --mode 'roi_track' --merge #--debug
# # python inference_sequence.py --ds SeaDronesSee --out_dir tests/SDS/v1 --flist data/SeaDronesSee/val.txt --name val --second_nms --mode 'roi_track' --merge #--debug
# # python inference_sequence.py --ds SeaDronesSee --out_dir tests/SDS/tst1 --flist data/SeaDronesSee/test.txt --name test --second_nms --mode 'roi_track' --merge #--debug

# # python inference_sequence.py --out_dir tests/SDS/t2 --ds SeaDronesSee --split train --second_nms --mode 'roi_track' --merge --debug
# # python inference_sequence.py --out_dir tests/SDS/v2 --ds SeaDronesSee --split val --second_nms --mode 'roi_track' --merge --debug 
# # python inference_sequence.py --out_dir tests/SDS/tst2 --ds SeaDronesSee --split test --second_nms --mode 'roi_track' --merge --debug

# # python inference_sequence.py --out_dir tests/SDS/o1 --ds SeaDronesSee --flist data/SeaDronesSee/some-files.txt --name train --second_nms --mode 'roi_track' --merge --debug

# # python inference_sequence.py --out_dir tests/vis/SDS --ds SeaDronesSee --flist data/SeaDronesSee/some-files.txt --name some-files \
# # --second_nms --mode 'roi_track' --merge --debug

# # python inference_sequence.py --out_dir tests/vis2/DroneCrowd --ds DroneCrowd --flist data/DroneCrowd/some-files.txt --name some-files \
# # --second_nms --mode 'roi_track' --merge --debug

# # python inference_sequence.py --out_dir tests/vis2/MTSD --ds MTSD --flist data/MTSD/some-files.txt --name some-files \
# # --second_nms --mode 'roi_track' --merge --debug --det_model yolov4 --roi_model u2net --debug


# python inference_sequence.py --out_dir tests/vis3/Zebra --flist data/3DZeF20/val.txt --name val --second_nms --mode 'roi_track' --merge --debug



# python inference_sequence.py --out_dir tests/changes_names/roi_track \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi_track' --merge

# python inference_sequence.py --out_dir tests/changes_names/track \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'track' --merge


# python inference_sequence.py --out_dir tests/changes_names/roi \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi' --merge



# python inference_sequence.py --out_dir tests/changes_names/roi_track \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi_track' --merge

# python inference_sequence.py --out_dir tests/changes_names/track \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'track' --merge


# python inference_sequence.py --out_dir tests/resize/roi4 \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi' --merge --debug

# python inference_sequence.py --out_dir tests/resize/roi \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --second_nms --mode 'roi' --merge # --debug --dilate --k_size 3


python inference_sequence.py --out_dir tests/resize/roi_track \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--second_nms --mode 'roi_track' --merge # --debug --dilate --k_size 3


python inference_sequence.py --out_dir tests/resize/track \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--second_nms --mode 'track' --merge # --debug --dilate --k_size 3


# python inference_sequence.py --out_dir tests/splitted-vis --ds ZebraFish --flist data/3DZeF20/val.txt --name val --second_nms --mode 'roi_track' --merge --debug