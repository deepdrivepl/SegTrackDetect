import math 

import cv2
import numpy as np

import torch
import torchvision.ops as ops



def rot90points(x, y, hw):
    """Rotate points 90 degrees.

    Args:
        x (Tensor): x-coordinates of the points.
        y (Tensor): y-coordinates of the points.
        hw (Tensor): Height and width of the image.

    Returns:
        Tuple[Tensor, Tensor]: Rotated x and y coordinates.
    """
    return y, hw[1] - 1 - x



def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coordinates from img1_shape to img0_shape.

    Args:
        img1_shape (Tuple[int, int]): Shape of the image with the predictions (height, width).
        coords (Tensor): Coordinates to rescale in (x1, y1, x2, y2) format.
        img0_shape (Tuple[int, int]): Original image shape (height, width).
        ratio_pad (Optional[Tuple[Tuple[float, float], Tuple[float, float]]]): 
            Gain and padding for the scaling process.

    Returns:
        Tensor: Rescaled coordinates.
    """
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
    """Clip bounding boxes to the image shape.

    Clipping is performed to ensure that the bounding boxes do not exceed the dimensions
    of the image. The coordinates are clamped to the image width and height.

    Args:
        boxes (Tensor): A tensor of shape (N, 4) containing bounding boxes in 
                        (x1, y1, x2, y2) format, where (x1, y1) is the top-left corner 
                        and (x2, y2) is the bottom-right corner.
        img_shape (tuple): A tuple (height, width) representing the dimensions of the image.

    Returns:
        None: The input tensor is modified in place.
    """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    
    
    
def xyxy2xywh(x):
    """Convert bounding boxes from (x1, y1, x2, y2) to (x, y, w, h) format.

    The conversion changes the representation of the bounding boxes where:
    - (x, y) is the top-left corner,
    - (w, h) are the width and height.

    Args:
        x (Tensor or ndarray): An array of shape (N, 4) containing bounding boxes in 
                               (x1, y1, x2, y2) format.

    Returns:
        Tensor or ndarray: An array of the same shape containing bounding boxes in 
                           (x, y, w, h) format, where:
                           - x: x-coordinate of the top-left corner
                           - y: y-coordinate of the top-left corner
                           - w: width of the box
                           - h: height of the box
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:,0]  # x min
    y[:, 1] = x[:,1]  # y min
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y