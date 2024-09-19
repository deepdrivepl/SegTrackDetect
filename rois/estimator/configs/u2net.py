import cv2
import numpy as np

import torch

from .common import estimator_transform


def u2net_postprocess(output, ori_shape, sigmoid_included=True, thresh=0.5, dilate=False, k_size=3, iter=1):
    output = output[0] 
    output = torch.squeeze(output)
    if not sigmoid_included:
        output = output.sigmoid()
    if thresh:
        output = torch.where(output > thresh, 1.0, 0.0)
    output = 255 * output.detach().cpu().numpy().astype(np.uint8)
    if dilate:
        kernel = np.ones((k_size, k_size), np.uint8)
        output = cv2.dilate(output, kernel, iterations = iter)
    # output_fullres = cv2.resize(output, (ori_shape[1], ori_shape[0]))
    return output


MTSD = dict(
    weights = "weights/u2netp_MTSD.pt",
    in_size = (576,576),
    thresh = 0.5,
    transform = estimator_transform,
    sigmoid_included = True,
    postprocess = u2net_postprocess,
    dilate = False, 
    k_size = 7,
    iter = 1
)