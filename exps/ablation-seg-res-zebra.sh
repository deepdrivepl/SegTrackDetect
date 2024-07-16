# python inference_sequence_vis_seg_res_dc.py --out_dir ablation/SEG-RES/vis-zebra/zebra-roi_track \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi_track' --merge --allow_resize --dilate --k_size 7 --second_nms --obs_iou_th 0.1 --debug --binary_trk


# python inference_sequence_vis_seg_res_dc.py --out_dir ablation/SEG-RES/vis-zebra/zebra-roi \
# --ds ZebraFish --flist data/3DZeF20/val.txt --name val \
# --mode 'roi' --merge --allow_resize --dilate --k_size 7 --second_nms --obs_iou_th 0.1 --debug





python inference_sequence_vis_seg_res_dc.py --out_dir ablation/SEG-RES/vis-zebra/zebra-track \
--ds ZebraFish --flist data/3DZeF20/val.txt --name val \
--mode 'track' --merge --allow_resize --dilate --k_size 7 --second_nms --obs_iou_th 0.1 --debug --binary_trk
