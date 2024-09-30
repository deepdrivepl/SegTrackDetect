import math 
import time

import cv2
import numpy as np

import torch
import kornia
from torchvision import transforms as T
from collections import OrderedDict

import torch.nn.functional as F
from torchvision.ops import boxes


def getSlidingWindowBBoxes(win_bbox, det_size, max_H, max_W, overlap_px = 20):
    h,w = det_size

    xmin, ymin, xmax, ymax = win_bbox
    win_h, win_w = ymax-ymin, xmax-xmin

    n_x = math.ceil(win_w / (w-overlap_px))
    n_y = math.ceil(win_h / (h-overlap_px))

    XS = np.array([xmin+(w-overlap_px)*n for n in range(n_x)])
    YS = np.array([ymin+(h-overlap_px)*n for n in range(n_y)])

    max_xmax = XS[-1]+w
    max_ymax = YS[-1]+h

    bboxes = []
    for _xmin in XS:
        for _ymin in YS:
            _xmin,_ymin = [int(_) for _ in [_xmin,_ymin]]
            # _xmax,_ymax = min(_xmin+w, max_W), min(_ymin+h, max_H)
            _xmax,_ymax = min(_xmin+w, xmax), min(_ymin+h, ymax)
            bboxes.append([_xmin, _ymin, _xmax, _ymax])
    full_bbox = [min([x[0] for x in bboxes]), min([x[1] for x in bboxes]), max([x[2] for x in bboxes]),  max([x[3] for x in bboxes])]
    return bboxes, full_bbox



def find_bounding_boxes(binary_mask, original_shape, current_shape):
    orig_height, orig_width = original_shape
    curr_height, curr_width = current_shape

    scale_x = orig_width / curr_width
    scale_y = orig_height / curr_height

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return empty array if no contours
    if not contours:
        return np.empty((0, 4))

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        xmin = max(0, int(x * scale_x))
        ymin = max(0, int(y * scale_y))
        xmax = min(orig_width, int((x + w) * scale_x))
        ymax = min(orig_height, int((y + h) * scale_y))

        bounding_boxes.append([xmin, ymin, xmax, ymax])

    return np.array(bounding_boxes)


def get_detection_windows(bboxes, orig_shape, det_size=(960, 960), bbox_type='naive', allow_resize=True):
    max_H, max_W = orig_shape
    if bbox_type == 'naive':
        bboxes = getDetectionBboxesNaive(bboxes, max_H, max_W, det_size=det_size, allow_resize=allow_resize)
    elif bbox_type == 'all':
        bboxes =  getDetectionBboxesAll(bboxes, max_H, max_W, det_size=det_size, allow_resize=allow_resize)
    elif bbox_type == 'sorted':
        bboxes =  getDetectionBboxesSorted(bboxes, max_H, max_W, det_size=det_size, allow_resize=allow_resize)
    else:
        raise NotImplementedError
    return bboxes


def isInsidePoint(bbox, point):
    xmin, ymin, xmax, ymax = bbox
    x, y = point
    if xmin<=x<=xmax and ymin<=y<=ymax:
        return True
    
def isInsideBbox(inner_bbox, outer_bbox):
    xmin,ymin,xmax,ymax = inner_bbox
    p1, p2, p3, p4 = (xmin,ymin), (xmax,ymin), (xmax, ymax), (xmin, ymax)
    return all([isInsidePoint(outer_bbox, point) for point in [p1,p2,p3,p4]])


def WindowIoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def same_orientation(h_roi, w_roi, h_window, w_window):
    if h_roi >= w_roi and h_window >= w_window:
        return True
    if h_roi < w_roi and h_window < w_window:
        return True
    return False


def rotate(bbox_roi, det_size):
    h_window, w_window = det_size
    xmin, ymin, xmax, ymax = bbox_roi
    h_roi, w_roi = ymax-ymin, xmax-xmin
    
    if same_orientation(h_roi, w_roi, h_window, w_window):
        return False
    return True
    
    
# should return crop coordinates - resizing handled in a datataloader
def getDetectionBbox(bbox, max_H, max_W, det_size, padding=20, allow_resize=False):

    if rotate(bbox, det_size):
        w,h = det_size
    else:
        h,w = det_size
        
    xmin, ymin, xmax, ymax = bbox
    h_roi, w_roi = ymax-ymin, xmax-xmin

    if (h_roi > h or w_roi > w) and not allow_resize:
        crop_bboxes, full_bbox = getSlidingWindowBBoxes(bbox, det_size, max_H, max_W)
        return crop_bboxes, [full_bbox]

    if h_roi > h: # padding - keep some space between returned roi and the detection window 
        h = h_roi + padding 
    if w_roi > w:
        w = w_roi + padding


    xc = (xmax+xmin)//2
    yc = (ymin+ymax)//2

    xmin = max(xc - w//2, 0)
    ymin = max(yc - h//2, 0)

    xmax = min(xmin+w, max_W)
    ymax = min(ymin+h, max_H)


    crop_bbox = [xmin, ymin, xmax, ymax]
    return [crop_bbox], [crop_bbox]
                
        
def getDetectionBboxesAll(bboxes, max_H, max_W, det_size, allow_resize):
    det_bboxes = []
    for bbox in bboxes:
        det_bboxes += getDetectionBbox(bbox,  max_H, max_W, det_size=det_size, allow_resize=allow_resize)[0]
    return det_bboxes

# fix filtering 
def getDetectionBboxesNaive(rois, max_H, max_W, det_size, allow_resize):
    det_windows = []
    full_det_windows = []
    
    for i, roi in enumerate(rois):
        if any([isInsideBbox(roi, full_det_window) for full_det_window in full_det_windows]):
            continue
            
        det_win, full_det_win = getDetectionBbox(roi,  max_H, max_W, det_size=det_size, allow_resize=allow_resize)
        det_windows += det_win
        full_det_windows += full_det_win
    return det_windows


def getDetectionBboxesSorted(rois, max_H, max_W, det_size, allow_resize):
    det_windows = []
    for i, roi in enumerate(rois):
        det_windows += getDetectionBbox(roi, max_H, max_W, det_size=det_size, allow_resize=allow_resize)[1]


    hits = {i: 0 for i in range(len(det_windows))}
    for i, det_window in enumerate(det_windows):
        hits[i]+=sum([isInsideBbox(roi, det_window) for roi in rois])

    # hits - how many rois inside each window, sort - det_windows with many rois first
    hits = OrderedDict(sorted(hits.items(), key=lambda item: item[1], reverse=True))
    rois = [rois[i] for i in hits.keys()] # rois with windows encapsulating many other rois first

    final_det_windows = getDetectionBboxesNaive(rois, max_H, max_W, det_size=det_size, allow_resize=allow_resize)
    return final_det_windows 
