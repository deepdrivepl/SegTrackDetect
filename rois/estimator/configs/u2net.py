import cv2
import numpy as np

import torch
import kornia

from .common import estimator_transform


def u2net_postprocess(output, ori_shape, sigmoid_included=True, thresh=0.5, dilate=False, k_size=3, iter=1):
    """
    Postprocesses the output of the U2Net model to generate a binary mask.

    This function applies sigmoid activation if it is not included in the model, 
    followed by thresholding to binarize the output. If requested, dilation is 
    applied to refine the binary mask.

    Args:
        output (tuple of torch.Tensor): Tuple of output tensors from the U2Net model, 
            each tuple element is expected to be of shape (1, C, H, W).
        ori_shape (tuple): The original shape of the input image as 
            (height, width).
        sigmoid_included (bool, optional): Indicates if the model output includes 
            a sigmoid activation. If False, the sigmoid will be applied. Default is True.
        thresh (float, optional): Threshold value for binarizing the output. 
            Pixels with values above this threshold will be set to 1.0, and others 
            will be set to 0.0. Default is 0.5.
        dilate (bool, optional): If True, applies dilation to the binary mask 
            to enhance the features. Default is False.
        k_size (int, optional): The size of the dilation kernel. Default is 3.
        iter (int, optional): The number of iterations for dilation. Default is 1.

    Returns:
        torch.Tensor: The postprocessed binary mask after applying 
        thresholding and optional dilation. Shape (H, W)
    """
    output = output[0] 

    if not sigmoid_included:
        output = torch.sigmoid(output)
    
    if thresh is not None:
        output = (output > thresh).float()

    if dilate:
        kernel = torch.ones((k_size, k_size), device=output.device)
        output = kornia.morphology.dilation(output, kernel)
    return output[0,0,...]




"""
Dictionaries that contain the configurations for various U2Net ROI estimation models. 
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

MTSD = dict(
    weights = "weights/u2netp_MTSD.pt",
    in_size = (576,576),
    thresh = 0.1,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = u2net_postprocess,
    dilate = False, 
    k_size = 7,
    iter = 1
)