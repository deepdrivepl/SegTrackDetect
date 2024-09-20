import cv2
import numpy as np

import torch
from torchvision import transforms as T


def yolo_transform(h,w):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def yolo_postprocess(output):
    return output[0]


ZeF20 = dict(
    weights = "weights/yolov7t-ZebraFish.pt",
    in_size = (160,256),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    multi_label = True,
    labels = [],
    merge = False,
    agnostic = False,
    transform = yolo_transform,
    postprocess = yolo_postprocess,
)


SeaDronesSee = dict(
    weights = "weights/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    multi_label = True,
    labels = [],
    merge = True,
    agnostic = False,
    transform = yolo_transform,
    postprocess = yolo_postprocess,
)



DroneCrowd = dict(
    weights = "weights/DC/004-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-50ep-best.torchscript.pt",
    in_size = (512,512),
    conf_thresh = 0.01,
    iou_thresh = 0.65,
    multi_label = True,
    labels = [],
    merge = False,
    agnostic = False,
    transform = yolo_transform,
    postprocess = yolo_postprocess,
)

MTSD = dict(
    weights = "weights/yolov4_MTSD.pt",
    in_size = (960,960),
    conf_thresh = 0.005,
    iou_thresh = 0.45,
    multi_label = True,
    labels = [],
    merge = False,
    agnostic = False,
    transform = T.ToTensor(),
    postprocess = yolo_postprocess,
)
