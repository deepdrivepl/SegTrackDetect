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

from data_loader import SingleDetectionDataset, ROIDataset, WindowDetectionDataset

from configs import DATASETS, DET_MODELS, ROI_MODELS, TRACKERS

from utils.bboxes import getDetectionBboxes, getSlidingWindowBBoxes, NMS, non_max_suppression,scale_coords, xyxy2xywh, findBboxes, IBS
from utils.general import create_directory, save_args, load_model
from utils.drawing import make_vis
                    
                
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # configs
    parser.add_argument('--roi_model', type=str, default="unet", choices=ROI_MODELS.keys())
    parser.add_argument('--det_model', type=str, default="yolov7_tiny", choices=DET_MODELS.keys())
    parser.add_argument('--tracker', type=str, default="sort", choices=TRACKERS.keys())
    parser.add_argument('--ds', type=str, default="ZeF20", choices=DATASETS.keys())
    parser.add_argument('--roi_weights', type=str, help="overwrite ROI weights from config")
    parser.add_argument('--flist', type=str, default='test_list', choices=['test_list', 'val_list', 'train_list'])
    # ROI
    parser.add_argument('--dilate', default=False, action='store_true')
    parser.add_argument('--k_size', type=int, default=7)
    parser.add_argument('--iter', type=int, default=2)
    parser.add_argument('--bbox_type', type=str, default='sorted', choices=['all', 'naive', 'sorted'])
    # NMS
    parser.add_argument('--second_nms', default=False, action='store_true')
    parser.add_argument('--merge', default=False, action='store_true')
    parser.add_argument('--redundant', default=False, action='store_true')
    parser.add_argument('--max_det', type=int, default=500)
    parser.add_argument('--agnostic', default=False, action='store_true')
    # general
    parser.add_argument('--mode', type=str, default='roi_track', choices=['roi', 'track', 'roi_track', 'sw', 'sd'])
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--out_dir', type=str, default='detections')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--vis_conf_th', type=float, default=0.3)
    # tracker
    parser.add_argument('--frame_delay', type=int, default=3)
    
    # sliding window
    parser.add_argument('--overlap_frac', type=float, default=0.05)
    args = parser.parse_args()
    
    
    # save args
    os.makedirs(args.out_dir, exist_ok=False)
    out_dir = args.out_dir
    save_args(out_dir, args)
    
    
    # get models
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 and not args.cpu else 'cpu'
    
    cfg_det = DET_MODELS[args.det_model]
    net_det = load_model(cfg_det, device)

    cfg_roi = ROI_MODELS[args.roi_model]
    net_roi = load_model(cfg_roi, device, weights=args.roi_weights if args.roi_weights is not None else None)
    
    cfg_trk = TRACKERS[args.tracker]
    trk_class = getattr(importlib.import_module(cfg_trk['module_name']), cfg_trk['class_name'])
    
    # get dataset
    cfg_ds = DATASETS[args.ds]
    flist = sorted([os.path.join(cfg_ds['root_dir'], x.rstrip()) for x in open(cfg_ds[args.flist])])
    # flist = [x for x in flist if 'imgT' in x]
    
    unique_sequences = sorted(list(set([x.split(os.sep)[cfg_ds["seq_pos"]] for x in flist])))
    print(unique_sequences)
    
    # inference
    for unique_sequence in unique_sequences:
        seq_flist = sorted([x for x in flist if x.split(os.sep)[cfg_ds["seq_pos"]]==unique_sequence])#[:200] # TEMP

        if 'roi' in args.mode:
            dataset = ROIDataset(seq_flist, cfg_roi["in_size"], cfg_roi["transform"])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        else:
            dataset = SingleDetectionDataset(seq_flist, cfg_det["in_size"])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        tracker = (trk_class)(**cfg_trk['args'])

        annotations = []
        with torch.no_grad():
            for i, (img, metadata) in tqdm(enumerate(dataloader)):

                H_orig, W_orig = metadata['image_h'].item(), metadata['image_w'].item()
                original_shape = (H_orig, W_orig)

                roi_bboxes, trk_bboxes, bboxes_det = np.empty((0,4)), np.empty((0,4)), np.empty((0,4))
                d0_fullres = None
                
                if args.mode == 'sw':
                    bboxes_det = getSlidingWindowBBoxes(H_orig, W_orig, args.overlap_frac, det_size=cfg_det["in_size"])
                else: # track, roi, roi_track
                    if args.mode == 'track' and i < args.frame_delay: # single det (for frame_delay frames) to initialize the tracker
                        print(i, args.frame_delay)
                        out = net_det(img.to(device))
                        out = cfg_det["postprocess"](out)
                        out = non_max_suppression(
                            out, 
                            conf_thres = cfg_det['conf_thresh'], 
                            iou_thres = cfg_det['iou_thresh'],
                            multi_label = True,
                            labels = [],
                            merge = args.merge,
                            agnostic = args.agnostic
                        )[0]
                        scale_coords(
                            (metadata['unpadded_h'][0].item(), metadata['unpadded_w'][0].item()),
                            out[:, :4],
                            (metadata['image_h'][0].item(), metadata['image_w'][0].item())
                        )
                        out = out.detach().cpu().numpy()
                        trks = tracker.get_pred_locations()
                        tracker.update(out[:, :-1], trks)
                        make_vis(d0_fullres, roi_bboxes, trk_bboxes, bboxes_det, out, metadata, out_dir, i,  args.vis_conf_th)

                        out[:,:4] = xyxy2xywh(out[:,:4])
                        for p in out.tolist():
                            annotations.append(
                                {
                                    "image_id": i,
                                    "category_id": int(p[-1]),
                                    "bbox": [round(x, 3) for x in p[:4]],
                                    "score": round(p[4], 5),
                                }
                            )
                        continue # do not run the window detection, just track for frame_delay frames 

                    elif 'roi' in args.mode: # predict ROIs
                        d0 = net_roi(img.to(device))
                        d0_fullres, d0 = cfg_roi["postprocess"](d0, original_shape, cfg_roi["sigmoid_included"], cfg_roi["thresh"])

                        if args.dilate:
                            kernel = np.ones((args.k_size, args.k_size), np.uint8)
                            d0 = cv2.dilate(d0, kernel, iterations = args.iter)

                        roi_bboxes = findBboxes(d0, original_shape, d0.shape)

                    if 'track' in args.mode:
                        trks = tracker.get_pred_locations()
                        if i >= args.frame_delay:
                            trk_bboxes = trks[:,:-1]

                    merged_bboxes = np.concatenate((roi_bboxes, trk_bboxes), axis=0)

                    bboxes_det = getDetectionBboxes(
                        merged_bboxes, 
                        H_orig, W_orig, 
                        det_size=cfg_det['in_size'], 
                        bbox_type=args.bbox_type
                    )
                    bboxes_det = [x for x in bboxes_det if (x[2]-x[0]>0 and x[3]-x[1]>0)]

                det_dataset =  WindowDetectionDataset(metadata['image_path'][0], bboxes_det, cfg_det['in_size'])
                det_dataloader = DataLoader(det_dataset, batch_size=len(det_dataset) if len(det_dataset)>0 else 1, shuffle=False, num_workers=4) # all windows in a single batch

                img_out = None
                for j, (img_det, det_metadata) in enumerate(det_dataloader):
                    
                    out = net_det(img_det.to(device))
                    out = cfg_det["postprocess"](out)
                    out = non_max_suppression(
                        out, 
                        conf_thres = cfg_det['conf_thresh'], 
                        iou_thres = cfg_det['iou_thresh'],
                        multi_label = True,
                        labels = [],
                        merge = args.merge,
                        agnostic = args.agnostic
                    )
                    for si, pred in enumerate(out):
                        if len(pred) == 0:
                            continue

                        pred[:,:4] = pred[:,:4] + det_metadata['translate'][si].to(device) # to the coordinates of the original image
                        
                        # out[si] = IBS(det_metadata['bbox'][si].to(device), pred, H_orig, W_orig, th=10) # it filers correct detections !!!!
                        
                    if img_out is None:
                        img_out = torch.cat(out)
                    else:
                        img_out = torch.cat((img_out, out), 0)

                # if img_out is None:
                #     tracker.update(np.empty((0, 5)), trks)
                #     continue

                if args.second_nms:
                    img_out = NMS(img_out, iou_thres=cfg_det["iou_thresh"], redundant=args.redundant, merge=args.merge, max_det=args.max_det, agnostic=args.agnostic)

                if 'track' in args.mode:
                    if img_out is None:
                        tracker.update(np.empty((0, 5)), trks)
                        continue

                    tracker.update(img_out.detach().cpu().numpy()[:, :-1], trks)
                
                if args.debug:
                    make_vis(d0_fullres, roi_bboxes, trk_bboxes, bboxes_det, img_out, metadata, out_dir, i,  args.vis_conf_th)
                
                img_out[:,:4] = xyxy2xywh(img_out[:,:4])
                for p in img_out.tolist():
                    annotations.append(
                        {
                            "image_id": i,
                            "category_id": int(p[-1]),
                            "bbox": [round(x, 3) for x in p[:4]],
                            "score": round(p[4], 5),
                        }
                    )
        
        with open(os.path.join(out_dir, f'results-{unique_sequence}.json'), 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=4)