import argparse
import os
import importlib
import json

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets import ROIDataset, WindowDetectionDataset
from datasets import DATASETS
from drawing import make_vis

from detector.aggregation import xyxy2xywh
from detector.obs import OBS_SORT_TYPES

from rois import ROIModule
from detector import Detector
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # models
    parser.add_argument('--roi_model', type=str, default="SDS_large")
    parser.add_argument('--det_model', type=str, default="SDS")
    parser.add_argument('--tracker', type=str, default="sort")

    # dataset
    parser.add_argument('--ds', type=str, default="SeaDronesSee", choices=DATASETS.keys())
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--flist', type=str, help='If provided, infer images listed in flist.txt; if not, infer split images.')
    parser.add_argument('--name', type=str, help='Name for img list provided in flist.txt')

    # ROI
    parser.add_argument('--bbox_type', type=str, default='sorted', choices=['all', 'naive', 'sorted']) # TODO one fixed method
    parser.add_argument('--allow_resize', default=False, action='store_true')
    
    # general
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--out_dir', type=str, default='detections')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--vis_conf_th', type=float, default=0.3)
        
    parser.add_argument('--obs_type', type=str, choices=['iou', 'conf', 'area', 'all', 'none'], default='all') # one fixed method
    parser.add_argument('--obs_iou_th', type=float, default=0.7)
    args = parser.parse_args()
    
    
    # create out_dir and save args to json
    os.makedirs(args.out_dir, exist_ok=False)
    with open(os.path.join(args.out_dir, "args.json"), 'w', encoding='utf-8') as f:
        info = {**vars(args)}
        json.dump(info, f, ensure_ascii=False, indent=4)

    if args.debug:
        windows_dir = f'{args.out_dir}/vis-windows'
        detections_dir = f'{args.out_dir}/vis-detections'
        os.makedirs(windows_dir, exist_ok=True); os.makedirs(detections_dir, exist_ok=True)
    
    # get dataset
    flist = args.flist if args.flist is None else [x.rstrip() for x in open(args.flist)]
    ds = (DATASETS[args.ds])(split=args.split, flist=flist, name=args.name)
    seq2images = ds.get_seq2imgs() if ds.get_sequences() is not None else {1: ds.get_images()}
    
    if ds.get_sequences() is None:
        print("Non-sequential data found; falling back to ROI mode.")
        exit(1) # TODO run second script (img only)


    # get models
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 and not args.cpu else 'cpu'
    detector = Detector(args.det_model, device)
    roi_extractor = ROIModule(
        tracker_name = args.tracker,
        estimator_name = args.roi_model,
        is_sequence = True if ds.get_sequences() is not None else False,
        device = device,
        bbox_type = args.bbox_type,
        allow_resize = args.allow_resize
    )

    filter_fn = OBS_SORT_TYPES[args.obs_type]
    
    # inference
    annotations = []
    for seq_name, seq_flist in tqdm(seq2images.items()):
        seq_flist = sorted(seq_flist)
        
        dataset = ROIDataset(seq_flist, ds, roi_extractor.estimator.input_size, roi_extractor.estimator.preprocess)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        with torch.no_grad():
            for i, (img, metadata) in tqdm(enumerate(dataloader)):

                H_orig, W_orig = metadata['coco']['height'].item(), metadata['coco']['width'].item()
                original_shape = (H_orig, W_orig)

                # get detection windows from ROI
                det_bboxes = roi_extractor.get_fused_roi(
                    frame_id = i,
                    img_tensor = img,
                    orig_shape = original_shape,
                    det_shape = detector.input_size,  
                )
                

                det_dataset =  WindowDetectionDataset(metadata['image_path'][0], ds, det_bboxes, detector.input_size)
                det_dataloader = DataLoader(det_dataset, batch_size=len(det_dataset) if len(det_dataset)>0 else 1, shuffle=False, num_workers=4) # all windows in a single batch

                img_det_list, img_win_list = [], []
                for j, (img_det, det_metadata) in enumerate(det_dataloader):
                    detections = detector.get_detections(img_det)
                    img_dets, img_wins = detector.postprocess_detections(detections, det_metadata)
                    img_det_list+=img_dets
                    img_win_list+=img_wins

                # Concatenate lists once after loop
                win_out = torch.cat(img_win_list).to(device)
                img_out = torch.cat(img_det_list).to(device)

                # Remove NaNs from both img_out and win_out
                valid_mask = ~torch.any(img_out.isnan(), dim=1)
                win_out = win_out[valid_mask]
                img_out = img_out[valid_mask]

                # Overlapping Box Suppression
                win_out, img_out = filter_fn(win_out, img_out, th=args.obs_iou_th)

                roi_extractor.predictor.update_tracker_state(img_out.detach().cpu().numpy()[:, :-1])
                                                    
                if args.debug:
                    frame = cv2.imread(metadata['image_path'][0])
                    frame, frame_dets = make_vis(frame, seg_mask_fullres, mot_mask, seg_bboxes, mot_bboxes, det_bboxes, img_out, ds.classes, ds.colors, args.vis_conf_th)
                    out_fname_wins = f"{windows_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                    out_fname_dets = f"{detections_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                    os.makedirs(os.path.dirname(out_fname_wins), exist_ok=True); os.makedirs(os.path.dirname(out_fname_dets), exist_ok=True)
                    cv2.imwrite(out_fname_wins, frame); cv2.imwrite(out_fname_dets, frame_dets)
                
                img_out[:,:4] = xyxy2xywh(img_out[:,:4])
                for p in img_out.tolist():
                    annotations.append(
                        {
                            "id": len(annotations), 
                            "image_id": int(metadata['coco']['id'].item()),
                            "category_id": int(p[-1]),
                            "bbox": [round(x, 3) for x in p[:4]],
                            "area": p[2] * p[3],
                            "score": round(p[4], 5),
                            "iscrowd": 0,
                        }
                    )

        
    with open(os.path.join(args.out_dir, f'results-{args.split if args.flist is None else args.name}.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
