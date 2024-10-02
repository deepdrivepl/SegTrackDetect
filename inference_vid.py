import argparse
import os
import json
import time

from glob import glob
from tqdm import tqdm
from statistics import mean
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import WindowDetectionDataset, ROIDataset, DATASETS
from drawing import make_vis

from rois import ROIModule
from detector import Detector
from detector.aggregation import xyxy2xywh
from detector.obs import OBS_SORT_TYPES



        
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
        
    # OBS
    parser.add_argument('--obs_type', type=str, choices=['iou', 'conf', 'area', 'all', 'none'], default='all') # one fixed method
    parser.add_argument('--obs_iou_th', type=float, default=0.7)
    args = parser.parse_args()
    

    # create out_dir and save args to json
    os.makedirs(args.out_dir, exist_ok=False)
    with open(os.path.join(args.out_dir, "args.json"), 'w', encoding='utf-8') as f:
        info = {**vars(args)}
        json.dump(info, f, ensure_ascii=False, indent=4)

    if args.debug:
        debug_dir = f'{args.out_dir}/vis'
        os.makedirs(debug_dir, exist_ok=True)
    

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

    # save configs
    detector_config = detector.get_config_dict()
    roi_extractor_config = roi_extractor.get_config_dict()
    with open(os.path.join(args.out_dir, "configs.json"), 'w', encoding='utf-8') as f:
        config = {**roi_extractor_config, **detector_config}
        json.dump(config, f, ensure_ascii=False, indent=4)

    filter_fn = OBS_SORT_TYPES[args.obs_type]

    
    # inference
    annotations = []
    all_images = 0
    total_times = []

    times = defaultdict(list)

    for seq_name, seq_flist in tqdm(seq2images.items()):
        seq_flist = sorted(seq_flist)

        roi_extractor.reset_predictor() # new tracker for each sequence 
        
        dataset = ROIDataset(seq_flist, ds, roi_extractor.estimator.input_size, roi_extractor.estimator.preprocess)
        all_images += len(dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        with torch.no_grad():
            for i, (img, metadata) in tqdm(enumerate(dataloader)):

                start_batch = time.time()

                img_roi = dataset.roi_transform(img)
                original_shape = metadata['coco']['height'].item(), metadata['coco']['width'].item()

                # get detection windows from ROI
                det_bboxes = roi_extractor.get_fused_roi(
                    frame_id = i,
                    img_tensor = img_roi,
                    orig_shape = original_shape,
                    det_shape = detector.input_size,  
                )
                times['roi'].append(time.time()-start_batch)
                

                t1 = time.time()
                det_dataset =  WindowDetectionDataset(img.to(device), metadata['image_path'][0], ds, det_bboxes, detector.input_size)
                img_det, det_metadata = det_dataset.get_batch()
                times['det_get_batch'].append(time.time()-t1)


                t1 = time.time()
                detections = detector.get_detections(img_det)
                times['det_infer'].append(time.time()-t1)

                t1 = time.time()
                img_det, img_win = detector.postprocess_detections(detections, det_metadata)


                # Remove NaNs from both img_det and img_win
                valid_mask = ~torch.any(img_det.isnan(), dim=1)
                img_win = img_win[valid_mask]
                img_det = img_det[valid_mask]
                times['det_postproc'].append(time.time()-t1)

                t1 = time.time()
                # Overlapping Box Suppression
                img_win, img_det = filter_fn(img_win, img_det, th=args.obs_iou_th)
                times['obs'].append(time.time()-t1)

                t1 = time.time()
                roi_extractor.predictor.update_tracker_state(img_det.detach().cpu().numpy()[:, :-1])
                                                    
                if args.debug:
                    frame = cv2.imread(metadata['image_path'][0])
                    estim_mask, pred_mask = roi_extractor.get_masks(frame.shape[:2])
                    frame = make_vis(frame, estim_mask, pred_mask, img_win, img_det, ds.classes, ds.colors, args.vis_conf_th)
                    out_fname = f"{debug_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                    cv2.imwrite(out_fname, frame)
                
                img_det[:,:4] = xyxy2xywh(img_det[:,:4])
                for p in img_det.tolist():
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

                end_batch = time.time()
                times['save_dets'].append(time.time()-t1)
                times['total'].append(end_batch-start_batch)
        # break

    times = {k: sum(v)/all_images for k,v in times.items()}
    times['fps'] = 1/times['total']
    times = pd.DataFrame(times, index=[0])
    times.to_csv(os.path.join(args.out_dir, 'times.csv'), index=False)

        
    with open(os.path.join(args.out_dir, f'results-{args.split if args.flist is None else args.name}.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
