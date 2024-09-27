import cv2
import numpy as np

import torch
import kornia
from .common import estimator_transform




def unet_postprocess(output, ori_shape, sigmoid_included=False, thresh=None, dilate=False, k_size=3, iter=1):
    # Apply sigmoid if not included in model
    if not sigmoid_included:
        output = torch.sigmoid(output)
    
    # Apply threshold if provided
    if thresh is not None:
        output = (output > thresh).float()
    if dilate:
        kernel = torch.ones((k_size, k_size), device=output.device)
        output = kornia.morphology.dilation(output, kernel)

    return output


ZeF20 = dict(
    weights = "weights/unetR18-ZebraFish.pt",
    in_size = (160,256),
    thresh = None,
    args = None,
    transform = estimator_transform,
    sigmoid_included = False,
    postprocess = unet_postprocess,
    dilate = True, # TODO - best metrics (padding 10, dilate 7, letterbox)
    k_size = 7,
    iter = 1

)

DroneCrowd_tiny = dict(
    weights = "weights/DroneCrowd-001-R18-96x160-best-loss.pt",
    in_size = (96,160),
    thresh = 0.5,
    args = None,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
    dilate = False, 
    k_size = 7,
    iter = 1
)

DroneCrowd_small = dict(
    weights = "weights/DroneCrowd-001-R18-192x320-best-loss.pt",
    in_size = (192,320),
    thresh = 0.5,
    args = None,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
    dilate = False, 
    k_size = 7,
    iter = 1
)

DroneCrowd_medium = dict(
    weights = "weights/DroneCrowd-001-R18-384x640-best-loss.pt",
    in_size = (384,640),
    thresh = 0.5,
    args = None,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
    dilate = False, 
    k_size = 7,
    iter = 1
)


SeaDronesSee_tiny = dict(
    weights = "weights/SeaDronesSee-000-R18-64x96-best-loss.pt",
    in_size = (64,96),
    thresh = 0.5,
    args = None,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
    dilate = True, 
    k_size = 7,
    iter = 1
)

SeaDronesSee_small = dict(
    weights = "weights/SeaDronesSee-000-R18-128x192-best-loss.pt",
    in_size = (128,192),
    thresh = 0.5,
    args = None,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
    dilate = True, 
    k_size = 7,
    iter = 1
)

SeaDronesSee_medium = dict(
    weights = "weights/SeaDronesSee-000-R18-224x384-best-loss.pt",
    in_size = (224,384),
    thresh = 0.5,
    args = None,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
    dilate = True, 
    k_size = 7,
    iter = 1
)

SeaDronesSee_large = dict(
    weights = "weights/SeaDronesSee-000-R18-448x768-best-loss.pt",
    in_size = (448,768),
    thresh = 0.5,
    args = None,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
    dilate = True, 
    k_size = 7,
    iter = 1
)