import cv2
import numpy as np

import torch
from torchvision import transforms as T



def yolo_transform(h,w):
    """
    Create a transformation pipeline for YOLO models.

    Args:
        h (int): Desired height for resizing the image.
        w (int): Desired width for resizing the image.

    Returns:
        torchvision.transforms.Compose: A composition of transformations including:
            - Conversion to tensor.
            - Resizing to the specified height and width.
            - Normalization with ImageNet statistics (mean and standard deviation).
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h, w), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def yolo_postprocess(output):
    """
    Post-process the YOLO model output.

    Args:
        output (torch.Tensor): The output tensor from the YOLO model.

    Returns:
        torch.Tensor: The post-processed output, which typically involves 
                      selecting the first element of the model's output.
    """
    return output[0]




"""
Dictionaries that contain the configurations for various YOLO detection models. 
All weights are torch.jit.ScriptModules to avoid model dependencies.

Keys:
- weights: Path to the pre-trained model weights (torch.jit.ScriptModule).
- in_size: Image dimensions for resizing during preprocessing.
- conf_thresh: Non-Maximum Suppression minimum confidence for detection.
- iou_thresh: Non-Maximum Suppression IoU threshold.
- multi_label: Non-Maximum Suppression support for multi-label detection.
- labels: Labels for autolabeling in Non-Maximum Suppression.
- merge: Whether to merge overlapping bounding boxes using weighted mean in Non-Maximum Suppression.
- agnostic: Class agnostic Non-Maximum Suppression.
- transform: Preprocessing function to be applied to the input image.
- postprocess: Post-processing function applied to the model outputs.
"""


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
    weights = "weights/004-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-50ep-best.torchscript.pt",
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
