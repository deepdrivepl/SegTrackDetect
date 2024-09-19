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

from configs import DET_MODELS
from datasets import DATASETS

from utils.bboxes import non_max_suppression,scale_coords, xyxy2xywh, rot90points
from utils.general import save_args, load_model
from utils.drawing import make_vis
from utils.obs import OBS_SORT_TYPES
from rois import ROIModule
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # configs
    parser.add_argument('--roi_model', type=str, default="unet")
    parser.add_argument('--det_model', type=str, default="yolov7_tiny")
    parser.add_argument('--tracker', type=str, default="sort")

    # dataset
    parser.add_argument('--ds', type=str, default="ZebraFish", choices=DATASETS.keys())
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--flist', type=str, help='If provided, infer images listed in flist.txt; if not, infer split images.')
    parser.add_argument('--name', type=str, help='Name for img list provided in flist.txt')

    # ROI
    parser.add_argument('--bbox_type', type=str, default='sorted', choices=['all', 'naive', 'sorted']) # TODO one fixed method
    parser.add_argument('--allow_resize', default=False, action='store_true')

    # NMS
    parser.add_argument('--second_nms', default=False, action='store_true') 
    parser.add_argument('--second_nms_iou_th', type=float)
    parser.add_argument('--merge', default=False, action='store_true')
    parser.add_argument('--redundant', default=False, action='store_true')
    parser.add_argument('--max_det', type=int, default=500)
    parser.add_argument('--agnostic', default=False, action='store_true')
    
    # general
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--out_dir', type=str, default='detections')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--vis_conf_th', type=float, default=0.3)
        
    parser.add_argument('--obs_type', type=str, choices=['iou', 'conf', 'area', 'all', 'none'], default='all') # one fixed method
    parser.add_argument('--obs_iou_th', type=float, default=0.7)

    args = parser.parse_args()
    
    filter_fn = OBS_SORT_TYPES[args.obs_type]
    
    # save args
    os.makedirs(args.out_dir, exist_ok=False)
    out_dir = args.out_dir
    save_args(out_dir, args)

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
    
    cfg_det = DET_MODELS[args.det_model]
    net_det = load_model(cfg_det, device, weights=None)

    roi_extractor = ROIModule(
        tracker_name = args.tracker,
        estimator_name = args.roi_model,
        is_sequence = True if ds.get_sequences() is not None else False,
        device = device,
        bbox_type = args.bbox_type,
        allow_resize = args.allow_resize
    )
    
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
                    det_shape = cfg_det['in_size'],  
                )
                

                det_dataset =  WindowDetectionDataset(metadata['image_path'][0], ds, det_bboxes, cfg_det['in_size'])
                det_dataloader = DataLoader(det_dataset, batch_size=len(det_dataset) if len(det_dataset)>0 else 1, shuffle=False, num_workers=4) # all windows in a single batch

                img_out = torch.empty((0, 6))
                win_out = torch.empty((0, 4))

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
                            
                        win_out = torch.cat((
                            win_out,
                            det_metadata['bbox'][si].repeat(len(pred),1)
                        ))

                        if det_metadata['resize'][si].item():

                            pred[:,:4] = scale_coords(det_metadata['unpadded_shape'][si], pred[:, :4], det_metadata['crop_shape'][si])
                            # pred[:,:4] = scale_coords(det_metadata['det_shape'][si], pred[:,:4], det_metadata['crop_shape'][si])

                        if det_metadata['rotation'][si].item():
                            h_window, w_window = det_metadata['roi_shape'][si]
                            xmin_, ymax_ = rot90points(pred[:,0], pred[:,1], [w_window.item(),h_window.item()])
                            xmax_, ymin_ = rot90points(pred[:,2], pred[:,3], [w_window.item(),h_window.item()])
                            pred[:,0] = xmin_
                            pred[:,1] = ymin_
                            pred[:,2] = xmax_
                            pred[:,3] = ymax_

                        pred[:,:4] = pred[:,:4] + det_metadata['translate'][si].to(device) # to the coordinates of the original image
          
  
                    if not img_out.numel():
                        img_out = torch.cat(out)
                    else:
                        img_out = torch.cat((img_out, out), 0)

                win_out = win_out.to(device)
                img_out = img_out.to(device)
                    
                win_out = win_out[~torch.any(img_out.isnan(),dim=1)]
                img_out = img_out[~torch.any(img_out.isnan(),dim=1)]

                # OBS
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

        
    with open(os.path.join(out_dir, f'results-{args.split if args.flist is None else args.name}.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
