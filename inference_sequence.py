import argparse
import os
import importlib
import json
import time

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from data_loader import SingleDetectionDataset, ROIDataset, WindowDetectionDataset

from configs import DATASETS, DET_MODELS, ROI_MODELS, TRACKERS

from utils.bboxes import getDetectionBboxes, getSlidingWindowBBoxes, NMS, non_max_suppression,scale_coords, xyxy2xywh, findBboxes, IBS, rot90points,box_iou
from utils.general import create_directory, save_args, load_model
from utils.drawing import make_vis



def filter_dets(windows, detections, th=0.7):
    filtered_detections, filtered_windows = torch.empty((0,6)), torch.empty((0,4))
    
    unique_windows = torch.unique(windows, dim=0)
    for w, unique_window in enumerate(unique_windows):
        ind_win = torch.unique(torch.where(windows==unique_window)[0])
        ind_notwin = torch.unique(torch.where(windows!=unique_window)[0])
        window_dets = detections[ind_win,:]
        other_dets = detections[ind_notwin,:]
        
        intersection_maxs = torch.min(other_dets[:, None, 2:4], unique_window.unsqueeze(0)[:, 2:4]) # xmax, ymax
        intersection_mins = torch.max(other_dets[:, None, :2], unique_window.unsqueeze(0)[:, :2]) # xmin, ymin
        intersections = torch.flatten(torch.cat((intersection_mins, intersection_maxs), dim=2), start_dim=0, end_dim=1)
        intersections[(intersections[:,2] - intersections[:,0] < 0) | (intersections[:,3] - intersections[:,1] < 0)] = 0

        ious = box_iou(intersections, window_dets[:,:4])
        to_del = torch.where(ious > th)[1]
        dets_filtered = window_dets[[x for x in range(window_dets.shape[0]) if x not in to_del],:]
        filtered_detections = torch.cat((filtered_detections, dets_filtered.to(filtered_detections.device)))
        filtered_detections = torch.unique(filtered_detections, dim=0)
        
        filtered_windows = torch.cat((filtered_windows.to(unique_window.device), unique_window.repeat(dets_filtered.shape[0], 1)))
        
        # avoid filtering "same detection" twice (same objects)
        to_del_orig = []
        for r,row in enumerate(detections):
            if row.tolist() in to_del.tolist():
                to_del_orig.append(r)
                
        detections = detections[[x for x in range(detections.shape[0]) if x not in to_del_orig],:]
        windows = windows[[x for x in range(windows.shape[0]) if x not in to_del_orig],:]
    return filtered_windows, filtered_detections   



def filter_dets_single_matrix(windows, bboxes, th=0.7):
    unique_windows = torch.unique(windows, dim=0)
    
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # check later
    # # detections from window = 0
    # for i, unique_window in enumerate(unique_windows):
    #     win_ind = torch.unique(torch.where(windows==unique_window)[0])
    #     intersections[win_ind, i, :] = 0
    # # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]

    to_del = torch.hstack((to_del, ious.unsqueeze(-1)))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered


def filter_dets_single_matrix_all(windows, bboxes, th=0.6):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # check later
    # # detections from window = 0
    # for i, unique_window in enumerate(unique_windows):
    #     win_ind = torch.unique(torch.where(windows==unique_window)[0])
    #     intersections[win_ind, i, :] = 0
    # # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    confs, areas = [], []
    for i in det_ind:
        xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
        confs.append(conf.item())
        areas.append((xmax-xmin)*(ymax-ymin))
        
    confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    ious = normalize(ious.unsqueeze(-1))
    mean = torch.mean(torch.stack([ious,confs, areas]), 0)

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered


def filter_dets_single_matrix_iou(windows, bboxes, th=0.7):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # check later
    # # detections from window = 0
    # for i, unique_window in enumerate(unique_windows):
    #     win_ind = torch.unique(torch.where(windows==unique_window)[0])
    #     intersections[win_ind, i, :] = 0
    # # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    # confs, areas = [], []
    # for i in det_ind:
    #     xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
    #     confs.append(conf.item())
    #     areas.append((xmax-xmin)*(ymax-ymin))
        
    # confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    # areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    ious = normalize(ious.unsqueeze(-1))
    mean = ious

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered



def filter_dets_single_matrix_area(windows, bboxes, th=0.7):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # check later
    # # detections from window = 0
    # for i, unique_window in enumerate(unique_windows):
    #     win_ind = torch.unique(torch.where(windows==unique_window)[0])
    #     intersections[win_ind, i, :] = 0
    # # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    confs, areas = [], []
    for i in det_ind:
        xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
        confs.append(conf.item())
        areas.append((xmax-xmin)*(ymax-ymin))
        
    # confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    # ious = normalize(ious.unsqueeze(-1))
    mean = areas #torch.mean(torch.stack([ious,confs, areas]), 0)

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered



def filter_dets_single_matrix_conf(windows, bboxes, th=0.7):
    
    def normalize(input_tensor):
        input_tensor -= input_tensor.min(0, keepdim=True)[0]
        input_tensor /= input_tensor.max(0, keepdim=True)[0]
        return input_tensor
    
    unique_windows = torch.unique(windows, dim=0)
    
    intersection_maxs = torch.min(bboxes[:, None, 2:4], unique_windows[:, 2:4]) # xmax, ymax
    intersection_mins = torch.max(bboxes[:, None, :2], unique_windows[:, :2]) # xmin, ymin
    intersections = torch.cat((intersection_mins, intersection_maxs), dim=2)
    intersections[(intersections[:,:,2] - intersections[:,:,0] < 0) | (intersections[:,:,3] - intersections[:,:,1] < 0)] = 0 # no common area
    
    # check later
    # # detections from window = 0
    # for i, unique_window in enumerate(unique_windows):
    #     win_ind = torch.unique(torch.where(windows==unique_window)[0])
    #     intersections[win_ind, i, :] = 0
    # # print(intersections.shape, bboxes.shape)
    
    ious = torch.empty((len(bboxes), len(unique_windows), len(bboxes)))
    for i in range(intersections.shape[1]):
        ious[:,i,...] = box_iou(intersections[:,i,:], bboxes[:,:4])
        
    for i in range(bboxes.shape[0]): # a detection cannot be removed because of itself
        ious[i,:,i] = 0
    
    #print(ious)
    to_del = torch.nonzero(ious > th)
    ious = ious[ious > th]
    if not ious.numel():
        return windows, bboxes
    
    det_ind = to_del[:,2].to(int)
    confs, areas = [], []
    for i in det_ind:
        xmin,ymin,xmax,ymax,conf = bboxes[i,:-1]
        confs.append(conf.item())
        areas.append((xmax-xmin)*(ymax-ymin))
        
    confs = 1 - normalize(torch.tensor(confs).unsqueeze(-1))
    # areas = 1 - normalize(torch.tensor(areas).unsqueeze(-1))
    # ious = normalize(ious.unsqueeze(-1))
    mean = confs #torch.mean(torch.stack([ious,confs, areas]), 0)

    to_del = torch.hstack((to_del, mean))
    to_del = to_del[to_del[:, -1].sort(descending=True)[1]] # (N, 4)
    #print(to_del)
    
    to_del_ids = []
    for i in range(to_del.shape[0]):
        if to_del[i, 0].item() in to_del_ids:
            #print(f'Cause already deleted {to_del[i, 0]}')
            continue
        if to_del[i, 2].item() in to_del_ids:
            #print(f'Detection already deleted {to_del[i, 2]}')
            continue
        to_del_ids.append(int(to_del[i, 2].item()))
        #print(f'Delete {to_del[i, 2]} because of {to_del[i, 0]}, iou: {to_del[i, 3]}')
    bboxes_filtered = bboxes[[x for x in range(bboxes.shape[0]) if x not in to_del_ids],:]
    windows_filtered = windows[[x for x in range(windows.shape[0]) if x not in to_del_ids],:]
    # print(bboxes_filtered)
    return windows_filtered, bboxes_filtered
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # configs
    parser.add_argument('--roi_model', type=str, default="unet", choices=ROI_MODELS.keys())
    parser.add_argument('--det_model', type=str, default="yolov7_tiny", choices=DET_MODELS.keys())
    parser.add_argument('--tracker', type=str, default="sort", choices=TRACKERS.keys())
    parser.add_argument('--ds', type=str, default="ZeF20", choices=DATASETS.keys())
    parser.add_argument('--roi_weights', type=str, help="overwrite ROI weights from config")
    parser.add_argument('--det_weights', type=str, help="overwrite DT weights from config")
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
    parser.add_argument('--obs_type', type=str, choices=['iou', 'conf', 'area', 'all', 'none'], default='iou')
    parser.add_argument('--obs_iou_th', type=float, default=0.7)
    args = parser.parse_args()
    
    obs_mapping = {
        'iou': filter_dets_single_matrix_iou, 
        'conf': filter_dets_single_matrix_conf, 
        'area': filter_dets_single_matrix_area, 
        'all': filter_dets_single_matrix_all, 
        'none': filter_dets
    }
    filter_fn = obs_mapping[args.obs_type]
    
    # save args
    os.makedirs(args.out_dir, exist_ok=False)
    out_dir = args.out_dir
    save_args(out_dir, args)
    
    
    # get models
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 and not args.cpu else 'cpu'
    
    cfg_det = DET_MODELS[args.det_model]
    net_det = load_model(cfg_det, device, weights=args.det_weights if args.det_weights is not None else None)

    cfg_roi = ROI_MODELS[args.roi_model]
    net_roi = load_model(cfg_roi, device, weights=args.roi_weights if args.roi_weights is not None else None)
    
    cfg_trk = TRACKERS[args.tracker]
    trk_class = getattr(importlib.import_module(cfg_trk['module_name']), cfg_trk['class_name'])
    
    # get dataset
    cfg_ds = DATASETS[args.ds]
    flist = sorted([os.path.join(cfg_ds['root_dir'], x.rstrip()) for x in open(cfg_ds[args.flist])])
    unique_sequences = sorted(list(set([f'{x.split(os.sep)[cfg_ds["seq_pos"]]}/{x.split(os.sep)[cfg_ds["sec_seq_pos"]]}' for x in flist])))
    # unique_sequences = [x for x in unique_sequences if '03' in x and 'imgT' in x] # and 'imgF' in x]
    print(unique_sequences)
    
    # inference
    for unique_sequence in unique_sequences:
        seq1, seq2 = unique_sequence.split(os.sep)
        seq_flist = sorted([x for x in flist if x.split(os.sep)[cfg_ds["seq_pos"]]==seq1 and x.split(os.sep)[cfg_ds["sec_seq_pos"]]==seq2])#[:200] # TEMP

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

                start = time.time()
                H_orig, W_orig = metadata['image_h'].item(), metadata['image_w'].item()
                original_shape = (H_orig, W_orig)

                roi_bboxes, trk_bboxes, bboxes_det = np.empty((0,4)), np.empty((0,4)), np.empty((0,4))
                d0_fullres = None
                
                if args.mode == 'sd':
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
                    out_ = out.copy()
                    out_[:,:4] = xyxy2xywh(out[:,:4])
                    end = time.time()

                    for p in out_.tolist():
                        annotations.append(
                            {
                                "image_id": i,
                                "image_path": metadata['image_path'][0],
                                "category_id": int(p[-1]),
                                "bbox": [round(x, 3) for x in p[:4]],
                                "score": round(p[4], 5),
                                "inference_time": end-start,
                            }
                        )
                    make_vis(d0_fullres, roi_bboxes, trk_bboxes, bboxes_det, out, metadata, out_dir, i,  args.vis_conf_th)
                    continue # do not run the window detection
                    
                elif args.mode == 'sw':
                    start = time.time()
                    bboxes_det = getSlidingWindowBBoxes(H_orig, W_orig, args.overlap_frac, det_size=cfg_det["in_size"])
                    
                else: # track, roi, roi_track
                    if args.mode == 'track' and i < args.frame_delay: # single det (for frame_delay frames) to initialize the tracker
                        print(i, args.frame_delay)
                        out = net_det(img.to(device))
                        out = cfg_det["postprocess"](out)
                        # print(out.shape)
                        # print(out[:,:10,:])
                        # exit()
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
                        
                        out_ = out.copy()
                        out_[:,:4] = xyxy2xywh(out[:,:4])
                        end = time.time()
                        
                        for p in out_.tolist():
                            annotations.append(
                                {
                                    "image_id": i,
                                    "image_path": metadata['image_path'][0],
                                    "category_id": int(p[-1]),
                                    "bbox": [round(x, 3) for x in p[:4]],
                                    "score": round(p[4], 5),
                                    "inference_time": end-start,
                                }
                            )
                        make_vis(d0_fullres, roi_bboxes, trk_bboxes, bboxes_det, out, metadata, out_dir, i,  args.vis_conf_th)
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
                    bboxes_roi = np.array([x[1] for x in bboxes_det]).astype(np.int32)
                    bboxes_det = np.array([x[0] for x in bboxes_det]).astype(np.int32)
                      
                    indices = np.nonzero(((bboxes_det[:,2]-bboxes_det[:,0]) > 0) & ((bboxes_det[:,3]-bboxes_det[:,1]) > 0))[0]
                    # [x for x in bboxes_det if (x[2]-x[0]>0 and x[3]-x[1]>0)] # TODO restore this line later
                    bboxes_roi = bboxes_roi[indices, :]
                    bboxes_det = bboxes_det[indices, :]
                det_dataset =  WindowDetectionDataset(metadata['image_path'][0], bboxes_det, cfg_det['in_size'])
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

                        # pred[:,:4] = pred[:,:4]*det_metadata['resize'][si].to(device)
                        if det_metadata['rotation'][si].item():
                            h_window, w_window = det_metadata['shape'][si]
                            xmin_, ymax_ = rot90points(pred[:,0], pred[:,1], [w_window.item(),h_window.item()])
                            xmax_, ymin_ = rot90points(pred[:,2], pred[:,3], [w_window.item(),h_window.item()])
                            pred[:,0] = xmin_
                            pred[:,1] = ymin_
                            pred[:,2] = xmax_
                            pred[:,3] = ymax_
                        pred[:,:4] = pred[:,:4] + det_metadata['translate'][si].to(device) # to the coordinates of the original image
                        # out[si] = IBS(det_metadata['bbox'][si].to(device), pred, H_orig, W_orig, th=10) # it filers correct detections !!!!
                    # print(det_metadata['bbox'])    
                    if not img_out.numel():
                        img_out = torch.cat(out)
                    else:
                        img_out = torch.cat((img_out, out), 0)

                if args.second_nms and img_out is not None:
                    img_out, win_out = NMS(img_out, win_out.to(device), iou_thres=cfg_det["iou_thresh"], redundant=args.redundant, merge=args.merge, max_det=args.max_det, agnostic=args.agnostic)
                    
                # OBS
                # win_out, img_out = filter_dets(win_out, img_out, th=0.7)
                win_out, img_out = filter_fn(win_out, img_out, th=args.obs_iou_th) #filter_dets_single_matrix_all(win_out, img_out, th=0.7)
                    
                if 'track' in args.mode:
                    tracker.update(img_out.detach().cpu().numpy()[:, :-1], trks)

                end = time.time()
                
                if args.debug:
                    make_vis(d0_fullres, roi_bboxes, trk_bboxes, bboxes_det, img_out.detach().cpu().numpy(), metadata, out_dir, i,  args.vis_conf_th)
                
                img_out[:,:4] = xyxy2xywh(img_out[:,:4])
                for p, w in zip(img_out.tolist(), win_out.tolist()):
                    annotations.append(
                        {
                            "image_id": i,
                            "image_path": metadata['image_path'][0],
                            "category_id": int(p[-1]),
                            "bbox": [round(x, 3) for x in p[:4]],
                            "window_bbox": list(map(int, w)),
                            "score": round(p[4], 5),
                            "inference_time": end-start,
                        }
                    )
        
        with open(os.path.join(out_dir, f'results-{seq1}-{seq2}.json'), 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=4)