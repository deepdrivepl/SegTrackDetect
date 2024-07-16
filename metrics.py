import json
import os
import contextlib
import warnings
warnings.filterwarnings(action='ignore')

from tqdm import tqdm
from glob import glob

import numpy as np
import pandas as pd

from tabulate import tabulate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import ultralytics.utils.metrics as umetrics
import torch
from time import sleep


th = 0.01 #0.2 #0.01

sds_gt = 'data/SeaDronesSee/test_dev.json'
# sds_gt = 'val-ours.json'
names_sds = ['method','AP', 'AP50', 'AP75', 'APu', 'APvt', 'APt', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARu', 'ARvt', 'ARt', 'ARs', 'ARm', 'ARl']
max_dets_sds = [1,10,100]
iou_th_sds = None

dc_gt = 'test_minus_val.json'
# dc_gt = 'val.json'
names_dc = ['method', 'AP', 'AP50', 'AP75', 'APu', 'APvt', 'APt', 'APs', 'APm', 'APl', 'AR500', 'ARu', 'ARvt', 'ARt', 'ARs', 'ARm', 'ARl']
max_dets_dc = [500]
iou_th_dc = 0.5



def get_gt(gt_path, agnostic=False):
    print('loading gt')
    gt = json.load(open(gt_path))
    unique_ids = sorted(list(set([x['id'] for x in gt['images']])))
    
    gt_objects = []
    for img_id in tqdm(unique_ids):
        img_annos = [x for x in gt['annotations'] if x['image_id']==img_id]

        img_bboxes = []

        for img_anno in img_annos:
            xmin,ymin,w,h = img_anno['bbox']
            xmax,ymax = xmin+w, ymin+h
            cls = img_anno['category_id']  if not agnostic else 0
            img_bboxes.append([xmin,ymin,xmax,ymax,cls])
        gt_objects.append(img_bboxes)

    nc = len(gt['categories']) if not agnostic else 1
    return gt_objects, unique_ids, nc

def get_stats(gt_objects, unique_ids, nc, res_path, conf=0.1, iou_thres=0.5, agnostic=False):
    pred = json.load(open(res_path))
    pred_objects = []
    for img_id in unique_ids:
        img_annos = [x for x in pred if x['image_id']==img_id]

        img_bboxes = []

        for img_anno in img_annos:
            xmin,ymin,w,h = img_anno['bbox']
            xmax,ymax = xmin+w, ymin+h
            cls = img_anno['category_id'] if not agnostic else 0
            score = img_anno['score']
            img_bboxes.append([xmin,ymin,xmax,ymax,score,cls])
        pred_objects.append(img_bboxes)
    assert len(pred_objects)==len(gt_objects)
    
    CM = umetrics.ConfusionMatrix(nc, conf, iou_thres)
    for i in range(len(gt_objects)):

        img_gt = torch.tensor(gt_objects[i])
        img_pred = torch.tensor(pred_objects[i])
        if not img_pred.numel():
            img_pred = torch.empty((0,6))

        if img_gt.numel():
            gt_bboxes = img_gt[:, :-1]
            gt_cls = img_gt[:,-1]
        else:
            gt_bboxes = torch.empty((0,4))
            gt_cls = torch.empty((0))

        CM.process_batch(img_pred, gt_bboxes, gt_cls)

    tp, fp, fn = CM.tp_fp_fn()    
    tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f1 = 2*(p*r)/(p+r)
    
    return tp,fp,fn,p,r,f1
    
    
def get_metrics_df(det_paths, gt_path, iou_th, max_dets, names, csv_path, conf=0.1, iou_thres=0.5):
    metrics = pd.DataFrame()
    # conf_ths = np.arange(0.1, 1.0, 0.1)
    conf_ths = np.arange(0.1, 0.2, 0.1)
    gt_objects, unique_ids, nc = get_gt(gt_path)
    for det_path in tqdm(det_paths):

        det = json.load(open(det_path))
        # print(len(det))
        det = [x for x in det if x['score'] >= th] if th is not None else det
        # print(len(det))
        # print(max_dets, iou_th)

        th_path = det_path.replace('.json', f'-mapped.json')
        with open(th_path, 'w', encoding='utf-8') as f:
            json.dump(det, f, ensure_ascii=False, indent=4)

        # method, ds = os.path.dirname(det_path).split(os.sep)[-2:] # SDS-OBS-th06
        # method = ds.split('-')[1]
        # iouth = ds.split('-')[-1].replace('th0', '0.')
        method = os.path.dirname(det_path).split(os.sep)[-1] # SDS-OBS-th06
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            if method in df.method.tolist():
                print(f'skipping {method}')
                continue
        print(f'processing {method}')
        # iouth = ds.split('-')[-1].replace('th0', '0.')
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            anno = COCO(gt_path)
            pred = anno.loadRes(th_path)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox', iouThr=iou_th, maxDets=max_dets)
            imgIds = sorted(anno.getImgIds())
            eval.params.imgIds = imgIds


            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
            all_params = eval.eval
            # p,r = get_pr(all_params)
        
        for conf_th in tqdm(conf_ths):
            tp,fp,fn,p,r,f1 = get_stats(gt_objects, unique_ids, nc, th_path, conf=conf_th, iou_thres=iou_thres)

        
            stats = {k:v for k,v in zip(names+['TP', 'FP', 'FN', 'P','R', 'F1'], [method] + [x*100 for x in eval.stats.tolist()]+[tp,fp,fn]+[_*100 for _ in [p,r,f1]])}
            all_metrics = [stats]
        
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                metrics = pd.concat([df, pd.DataFrame(all_metrics)], ignore_index=True)
                metrics.to_csv(csv_path, index=False)
            else:
                metrics = pd.DataFrame(all_metrics)
                metrics.to_csv(csv_path, index=False)
            
            print(metrics)

    return metrics


dc_obs = sorted(glob('ablation/soa-sds/**/results-test_dev.json', recursive=True)) #, reverse=True)
print(len(dc_obs))

# print(len(dc_obs), dc_obs[0])
dc_csv = 'SDS-test.csv'

metrics_obs_dc = get_metrics_df(dc_obs, sds_gt, iou_th_sds, max_dets_sds, names_sds, dc_csv)
# sleep(1200)
# metrics_obs_dc = get_metrics_df(dc_obs, dc_gt, iou_th_sds, max_dets_sds, names_sds, dc_csv)