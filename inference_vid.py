import argparse
import os
import json
import time

from glob import glob
from tqdm import tqdm
from statistics import mean

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import ROIDataset, WindowDetectionDataset, DATASETS
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

    obs_times, saving_times = [],[]
    
    # inference
    annotations = []
    all_images = 0
    total_times = []
    win_times = []
    for seq_name, seq_flist in tqdm(seq2images.items()):
        seq_flist = sorted(seq_flist)

        roi_extractor.reset_predictor() # new tracker for each sequence 
        
        dataset = ROIDataset(seq_flist, ds, roi_extractor.estimator.input_size, roi_extractor.estimator.preprocess)
        all_images += len(dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        with torch.no_grad():
            for i, (img, metadata) in tqdm(enumerate(dataloader)):

                start_batch = time.time()

                original_shape = metadata['coco']['height'].item(), metadata['coco']['width'].item()

                # get detection windows from ROI
                det_bboxes = roi_extractor.get_fused_roi(
                    frame_id = i,
                    img_tensor = img,
                    orig_shape = original_shape,
                    det_shape = detector.input_size,  
                )
                
                det_dataset =  WindowDetectionDataset(metadata['image_path'][0], ds, det_bboxes, detector.input_size)
                det_dataloader = DataLoader(det_dataset, batch_size=len(det_dataset) if len(det_dataset)>0 else 1, shuffle=False, num_workers=2) # all windows in a single batch

                img_det_list, img_win_list = [], []
                for j, (img_det, det_metadata) in enumerate(det_dataloader):
                    detections = detector.get_detections(img_det)
                    img_dets, img_wins = detector.postprocess_detections(detections, det_metadata)
                    img_det_list+=img_dets
                    img_win_list+=img_wins

                    win_times+=det_metadata['time'].numpy().tolist()


                # Concatenate lists once after loop
                img_win = torch.cat(img_win_list).to(device)
                img_det = torch.cat(img_det_list).to(device)

                # Remove NaNs from both img_det and img_win
                valid_mask = ~torch.any(img_det.isnan(), dim=1)
                img_win = img_win[valid_mask]
                img_det = img_det[valid_mask]

                # Overlapping Box Suppression
                t1 = time.time()
                img_win, img_det = filter_fn(img_win, img_det, th=args.obs_iou_th)
                obs_times.append(time.time()-t1)

                roi_extractor.predictor.update_tracker_state(img_det.detach().cpu().numpy()[:, :-1])
                                                    
                if args.debug:
                    frame = cv2.imread(metadata['image_path'][0])
                    estim_mask, pred_mask = roi_extractor.get_masks(frame.shape[:2])
                    frame = make_vis(frame, estim_mask, pred_mask, img_win, img_det, ds.classes, ds.colors, args.vis_conf_th)
                    out_fname = f"{debug_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                    cv2.imwrite(out_fname, frame)
                
                t1 = time.time()
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
                saving_times.append(time.time()-t1)

                end_batch = time.time()
                total_times.append(end_batch-start_batch)
        break


    t_coords, t_det_wins, t_pred, t_estim = roi_extractor.get_execution_times(all_images)
    t_pred_pred, t_pred_mask, t_pred_update = t_pred
    t_estim_infer, t_estim_postproc = t_estim
    t_det_infer, t_det_postproc, t_det_nms, t_det_to_orig = detector.get_execution_times(all_images)
    t_det_preproc = sum(win_times)/all_images

    times = pd.DataFrame.from_dict({
        "estim_infer": [t_estim_infer*1000],
        "estim_postproc": [t_estim_postproc*1000],
        "pred_pred": [t_pred_pred*1000],
        "pred_mask": [t_pred_mask*1000],
        "pred_update": [t_pred_update*1000],
        "fusion_roi_coords": [t_coords*1000],
        "fusion_det_windows": [t_det_wins*1000],
        "det_preproc": [t_det_preproc*1000],
        "det_infer": [t_det_infer*1000],
        "det_nms": [(t_det_postproc+t_det_nms)*1000],
        "det_postprocessing": [t_det_to_orig*1000],
        "obs": [(sum(obs_times)/all_images)*1000],
        "saving_dets": [(sum(saving_times)/all_images)*1000],
        "total_times [s]": sum(total_times),
        "num_frames": all_images,
        "avg_time_per_img [s]": sum(total_times)/all_images,
        "FPS": all_images/sum(total_times)
    })
    print(times)

    times.to_csv(os.path.join(args.out_dir, 'times-ms.csv'), index=False)

        
    with open(os.path.join(args.out_dir, f'results-{args.split if args.flist is None else args.name}.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
