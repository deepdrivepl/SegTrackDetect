import math 
import time

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.ops as ops



def findBboxes(label, original_shape, current_shape):
    H,W = original_shape
    _H,_W = current_shape
    label[label>0] = 255
    contours = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    if len(contours) == 0:
        return np.empty((0,4))
    
    bboxes = []
    for i, cntr in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cntr)
        xmin, ymin = (x/_W)*W, (y/_H)*H
        xmax, ymax = ((x+w)/_W)*W, ((y+h)/_H)*H

        xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
        xmax, ymax = min(W, int(xmax)), min(H, int(ymax))
        
        bboxes.append([xmin,ymin,xmax,ymax])
    return np.array(bboxes)


def getDetectionBboxes(bboxes, max_H, max_W, det_size=(960, 960), bbox_type='naive'):
    if bbox_type == 'naive':
        bboxes = getDetectionBboxesNaive(bboxes, max_H, max_W, det_size=det_size)
    elif bbox_type == 'all':
        bboxes =  getDetectionBboxesAll(bboxes, max_H, max_W, det_size=det_size)
    elif bbox_type == 'sorted':
        bboxes =  getDetectionBboxesSorted(bboxes, max_H, max_W, det_size=det_size)
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


def rot90points(x, y, hw):
    return y, hw[1] - 1 - x
    
    
# should return crop coordinates - resizing handled in a datataloader
def getDetectionBbox(bbox, max_H, max_W, det_size, padding=10):
    if rotate(bbox, det_size):
        w,h = det_size
    else:
        h,w = det_size
        
    xmin, ymin, xmax, ymax = bbox
    h_roi, w_roi = ymax-ymin, xmax-xmin

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
    return crop_bbox
                
        
def getDetectionBboxesAll(bboxes, max_H, max_W, det_size):
    det_bboxes = []
    for bbox in bboxes:
        det_bboxes.append(getDetectionBbox(bbox,  max_H, max_W, det_size=det_size))
    return det_bboxes
    

def getDetectionBboxesNaive(bboxes, max_H, max_W, det_size):
    det_bboxes = []
    for bbox in bboxes:
        if any([isInsideBbox(bbox, det_bbox) for det_bbox in det_bboxes]):
            continue
        # for det_bbox in det_bboxes:
        #     if WindowIoU(bbox, det_bbox,max_H, max_W) > 0.2:
        #         print("> 0.2")
        # if any([WindowIoU(bbox, det_bbox, max_H, max_W) > 0.7 for det_bbox in det_bboxes]):
        #     continue
            
        det_bboxes.append(getDetectionBbox(bbox,  max_H, max_W, det_size=det_size))
    return det_bboxes


def getDetectionBboxesSorted(bboxes, max_H, max_W, det_size):
    _det_bboxes = [getDetectionBbox(bbox, max_H, max_W, det_size=det_size) for bbox in bboxes]
    hits = {i: 0 for i in range(len(_det_bboxes))}
    for i, det_bbox in enumerate(_det_bboxes):
        hits[i]+=sum([isInsideBbox(bbox, det_bbox) for bbox in bboxes])
    # print(hits)
    if all([x==1 for x in hits.values()]):
        return _det_bboxes
    elif any([x==len(bboxes) for x in hits.values()]):
        fnd = list(hits.keys())[list(hits.values()).index(len(bboxes))]
        # print(fnd)
        return [_det_bboxes[fnd]]
    else:
        hits = dict(sorted(hits.items(), key=lambda item: item[1], reverse=True))
        bboxes = [bboxes[i] for i in hits.keys()]
        return getDetectionBboxesNaive(bboxes, max_H, max_W, det_size=det_size)



# based on https://github.com/WongKinYiu/yolov7
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=64):
    
    shape = img.shape[:2]  # orig hw
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # if not scaleup:  
    r = min(r, 1.0) # only scale down, do not scale up (for better test mAP)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    img_in = np.full((*new_shape, 3), color).astype(np.uint8)
    img_in[:new_unpad[1],:new_unpad[0],...] = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # print(img_in.shape, new_unpad)
    return img_in, new_unpad[::-1]
    


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)



def NMS(prediction, windows, iou_thres,  redundant=True, merge=False, max_det=500, agnostic=False):
    # https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/utils/general.py
    # output = [torch.zeros(0, 6)] * len(prediction)
    x = prediction.clone()
        
    # Batched NMS
    boxes, scores, c = x[:, :4], x[:, 4], x[:, 5] * (0 if agnostic else 1)

    i = ops.batched_nms(boxes, scores, c, iou_thres)
    # i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]
    if merge and (1 < x.shape[0] < 3E3):  # Merge NMS (boxes merged using weighted mean)
        # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        # iou = bbox_iou(torch.unsqueeze(boxes[i], 0), boxes,DIoU=True) > iou_thres
        iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        weights = iou * scores[None]  # box weights
        x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        if redundant:
            i = i[iou.sum(1) > 1]  # require redundancy

        # output[xi] = x[i]
    return x[i], windows[i]  #output



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y



def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, merge=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output



def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    
    
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:,0]  # x min
    y[:, 1] = x[:,1]  # y min
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y