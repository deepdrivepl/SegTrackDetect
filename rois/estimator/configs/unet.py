import cv2
import numpy as np

import torch
import kornia
from .common import estimator_transform




def unet_postprocess(output, ori_shape, sigmoid_included=False, thresh=None, dilate=False, k_size=3, iter=1):
    """
    Postprocesses the output of the UNet model to generate a binary mask.

    This function applies sigmoid activation if not included in the model, 
    followed by thresholding and optional dilation.

    Args:
        output (torch.Tensor): The output tensor from the UNet model, 
            expected to be of shape (1, C, H, W).
        ori_shape (tuple): The original shape of the input image as (height, width).
        sigmoid_included (bool, optional): Indicates if the model output includes 
            a sigmoid activation. If False, sigmoid will be applied. Default is False.
        thresh (float, optional): Threshold value for binarizing the output. 
            Pixels with values above this threshold will be set to 1.0, 
            and others to 0.0. Default is None, meaning no thresholding will occur.
        dilate (bool, optional): If True, applies dilation to the binary mask 
            to enhance the mask features. Default is False.
        k_size (int, optional): The size of the dilation kernel. Default is 3.
        iter (int, optional): The number of iterations for dilation. Default is 1.

    Returns:
        torch.Tensor: The postprocessed binary mask. Shape (H, W)
    """
    # Apply sigmoid if not included in model
    if not sigmoid_included:
        output = torch.sigmoid(output)
    
    # Apply threshold if provided
    if thresh is not None:
        output = (output > thresh).float()
    if dilate:
        kernel = torch.ones((k_size, k_size), device=output.device)
        output = kornia.morphology.dilation(output, kernel)

    return output[0,0,...]



"""
Dictionaries that contain the configurations for various UNet ROI estimation models. 
Weights are torch.jit.ScriptModules to avoid model dependencies.

Keys:
- weights: Path to the pre-trained model weights (torch.jit.ScriptModule).
- in_size: Image dimensions for resizing during preprocessing.
- thresh: Theshold value used to binarize the output mask during postprocessing.
- sigmoid_included: Indicates if the model output includes a sigmoid activation.
- dilate: If True, applies dilation to the binary mask to enhance the features.
- k_size: The size of the dilation kernel.
- iter: The number of iterations for dilation.
- transform: Preprocessing function to be applied to the input image.
- postprocess: Post-processing function applied to the model outputs.
"""

ZeF20 = dict(
    weights = "weights/unetR18-ZebraFish.pt",
    in_size = (160,256),
    thresh = 0.5,
    args = None,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = unet_postprocess,
    dilate = True, 
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