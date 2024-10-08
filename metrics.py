import json
import argparse
import os
import warnings
warnings.filterwarnings(action='ignore')

from tqdm import tqdm
from glob import glob

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--gt_path', type=str, default='data/SeaDronesSee/test_dev.json')
    parser.add_argument('--th', type=float, default=0.01)
    parser.add_argument('--csv', type=str, default='checks/metrics.csv')
    parser.add_argument('--dc', default=False, action='store_true')
    args = parser.parse_args()


    names = ['name', 'AP', 'AP50', 'AP75', 'APu', 'APvt', 'APt', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARu', 'ARvt', 'ARt', 'ARs', 'ARm', 'ARl']
    max_dets = [1,10,100]
    iou_th = None
    if args.dc:
        names = ['name', 'AP', 'AP50', 'AP75', 'APu', 'APvt', 'APt', 'APs', 'APm', 'APl', 'AR500', 'ARu', 'ARvt', 'ARt', 'ARs', 'ARm', 'ARl']
        max_dets = [500]
        iou_th = 0.5


    detections = glob(f'{args.dir}/results*.json')
    detections = [x for x in detections if 'mapped' not in x]
    assert len(detections)==1
    detections = detections[0]
    th = 0.01

    if not os.path.isfile(detections):
        print(f'{detections} does not exist.')
        exit(1)

    name = args.dir.split(os.sep)[-1]
    det = json.load(open(detections))
    print(len(det))
    det = [x for x in det if x['score'] >= args.th] if args.th is not None else det
    print(len(det))

    th_path = detections.replace('.json', f'-mapped.json')
    with open(th_path, 'w', encoding='utf-8') as f:
        json.dump(det, f, ensure_ascii=False, indent=4)

    anno = COCO(args.gt_path)
    pred = anno.loadRes(th_path) 
    print(f'Running validation with iouThr={iou_th}, maxDets={max_dets}, names={names}')
    eval = COCOeval(anno, pred, 'bbox', iouThr=iou_th, maxDets=max_dets)

    imgIds = sorted(anno.getImgIds())
    eval.params.imgIds = imgIds

    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]

    stats = {k:v for k,v in zip(names, [name] + [x*100 for x in eval.stats.tolist()])}
    all_metrics = [stats]

    if os.path.isfile(args.csv):
        metrics = pd.read_csv(args.csv)
        metrics = pd.concat([metrics, pd.DataFrame(all_metrics)], ignore_index=True)
    else:
        metrics = pd.DataFrame(all_metrics)

    metrics.to_csv(args.csv, index=False)

