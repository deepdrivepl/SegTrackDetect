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

    return np.array(bounding_boxes).astype(np.int32)


def same_orientation(h_roi, w_roi, h_window, w_window):
    if h_roi >= w_roi and h_window >= w_window:
        return True
    if h_roi < w_roi and h_window < w_window:
        return True
    return False


def rotate(bbox_roi, det_shape):
    h_window, w_window = det_shape
    xmin, ymin, xmax, ymax = bbox_roi
    h_roi, w_roi = ymax-ymin, xmax-xmin
    
    if same_orientation(h_roi, w_roi, h_window, w_window):
        return False
    return True


def get_sliding_windows(roi_bbox, img_shape, det_shape, overlap_px = 20):
    h,w = det_shape

    xmin, ymin, xmax, ymax = roi_bbox
    roi_h, roi_w = ymax-ymin, xmax-xmin

    n_x = math.ceil(roi_w / (w-overlap_px))
    n_y = math.ceil(roi_h / (h-overlap_px))

    XS = np.array([xmin+(w-overlap_px)*n for n in range(n_x)])
    YS = np.array([ymin+(h-overlap_px)*n for n in range(n_y)])

    max_xmax = XS[-1]+w
    max_ymax = YS[-1]+h

    bboxes = []
    for _xmin in XS:
        for _ymin in YS:
            _xmin,_ymin = [int(_) for _ in [_xmin,_ymin]]
            # _xmax,_ymax = min(_xmin+w, img_shape[1]), min(_ymin+h, img_shape[0])
            _xmax,_ymax = min(_xmin+w, xmax), min(_ymin+h, ymax)
            bboxes.append([_xmin, _ymin, _xmax, _ymax])
    full_bbox = [min([x[0] for x in bboxes]), min([x[1] for x in bboxes]), max([x[2] for x in bboxes]),  max([x[3] for x in bboxes])]
    return bboxes, full_bbox


# should return crop coordinates - resizing handled in a datataloader
def get_detection_window(roi_bbox, img_shape, det_shape, padding=20, allow_resize=False):
    rot = rotate(roi_bbox, det_shape)
    if rot:
        w,h = det_shape
    else:
        h,w = det_shape
    ar = h/w 
        
    xmin, ymin, xmax, ymax = roi_bbox
    h_roi, w_roi = ymax-ymin, xmax-xmin

    # sliding-window within ROI region
    needs_sliding = (h_roi > h or w_roi > w) and not allow_resize
    if needs_sliding:
        crop_bboxes, full_bbox = get_sliding_windows(roi_bbox, img_shape, det_shape) # check if empty
        return crop_bboxes, [full_bbox]

    # Apply padding if needed
    h_roi = max(h_roi + padding, h)
    w_roi = max(w_roi + padding, w)

    # Adjust ROI size to maintain aspect ratio
    if h_roi / w_roi != ar:
        if h_roi / w_roi > ar:  # ROI is taller, adjust width
            w_roi = h_roi / ar
        else:  # ROI is wider, adjust height
            h_roi = w_roi * ar

    xc = (xmax+xmin)//2
    yc = (ymin+ymax)//2

    xmin = max(xc - w_roi//2, 0)
    ymin = max(yc - h_roi//2, 0)

    xmax = xmin+w_roi
    ymax = ymin+h_roi

    if xmax > img_shape[1]:
        dx = xmax-img_shape[1]
        xmin = max(xmin - dx, 0)
        xmax = max(xmax - dx, 0)
    if ymax > img_shape[0]:
        dy = ymax-img_shape[0]
        ymin = max(ymin - dy, 0)
        ymax = max(ymax - dy, 0)

    crop_bbox = [xmin, ymin, xmax, ymax]
    return [crop_bbox], [crop_bbox]



def get_detection_windows(rois, img_shape, det_shape=(960, 960), bbox_type='naive', allow_resize=True):

    crop_windows, full_windows = [], []
    for roi in rois:
        crop, full = get_detection_window(roi, img_shape, det_shape=det_shape, allow_resize=allow_resize)
        crop_windows+=crop
        full_windows+=full

    if bbox_type == 'all':
        det_windows = crop_windows
    elif bbox_type == 'naive':
        det_windows = filter_detection_windows_naive(rois, crop_windows, full_windows, img_shape, det_shape=det_shape, allow_resize=allow_resize)
    elif bbox_type == 'sorted':
        det_windows =  filter_detection_windows_sorted(rois, crop_windows, full_windows, img_shape, det_shape=det_shape, allow_resize=allow_resize)
    else:
        raise NotImplementedError
    return np.array(det_windows).astype(np.int32)


    
def is_bbox_inside_bbox(inner_bbox, outer_bbox):
    def is_point_inside_bbox(point, bbox):
        xmin, ymin, xmax, ymax = bbox
        x, y = point
        if xmin<=x<=xmax and ymin<=y<=ymax:
            return True

    xmin,ymin,xmax,ymax = inner_bbox
    p1, p2, p3, p4 = (xmin,ymin), (xmax,ymin), (xmax, ymax), (xmin, ymax)
    return all([is_point_inside_bbox(point, outer_bbox) for point in [p1,p2,p3,p4]])



# fix filtering 
def filter_detection_windows_naive(rois, crop_windows, full_windows, img_shape, det_shape, allow_resize):
    
    final_crop_windows, final_full_windows = [],[]
    for roi, crop_window, full_window in zip(rois, crop_windows, full_windows):
        if any([is_bbox_inside_bbox(roi, final_full_window) for final_full_window in final_full_windows]):
            continue
            
        final_crop_windows.append(crop_window)
        final_full_windows.append(full_window)

    return final_crop_windows


def filter_detection_windows_sorted(rois, crop_windows, full_windows, img_shape, det_shape, allow_resize):
    hits = {i: 0 for i in range(len(full_windows))}
    for i, full_window in enumerate(full_windows):
        hits[i]+=sum([is_bbox_inside_bbox(roi, full_window) for roi in rois])

    # hits - how many rois inside each window, sort - det_windows with many rois first
    hits = OrderedDict(sorted(hits.items(), key=lambda item: item[1], reverse=True))
    rois = [rois[i] for i in hits.keys()]
    crop_windows = [crop_windows[i] for i in hits.keys()]
    full_windows = [full_windows[i] for i in hits.keys()]

    final_crop_windows = filter_detection_windows_naive(rois, crop_windows, full_windows, img_shape, det_shape=det_shape, allow_resize=allow_resize)
    return final_crop_windows 
