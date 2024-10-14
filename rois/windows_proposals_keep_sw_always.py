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




def get_roi_bounding_boxes(binary_mask, original_shape, current_shape):
    """Extracts bounding boxes around regions of interest (ROI) from a binary mask.

    Args:
        binary_mask (np.ndarray): Binary mask containing the ROI regions.
        original_shape (tuple): Original image dimensions as (height, width).
        current_shape (tuple): Current image dimensions as (height, width).

    Returns:
        np.ndarray: An array of bounding boxes, each represented by (xmin, ymin, xmax, ymax).
    """
    orig_height, orig_width = original_shape
    curr_height, curr_width = current_shape

    scale_x = orig_width / curr_width
    scale_y = orig_height / curr_height

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return empty array if no contours
    if not contours:
        return np.empty((0, 4))

    # Get bounding boxes for all contours
    bboxes = np.array([cv2.boundingRect(contour) for contour in contours])

    # Unpack the bounding box data
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # Scale bounding boxes
    xmin = np.maximum(0, (x * scale_x))
    ymin = np.maximum(0, (y * scale_y))
    xmax = np.minimum(orig_width, ((x + w) * scale_x))
    ymax = np.minimum(orig_height, ((y + h) * scale_y))

    # Stack into (N, 4) array
    bounding_boxes = np.stack([xmin, ymin, xmax, ymax], axis=1)

    return bounding_boxes.astype(np.int32)


def get_sliding_windows_vect(roi_bboxes, img_shape, det_shape, overlap_px=20):
    """Generates sliding window bounding boxes over the regions of interest.
    Used for datasets that don't allow resize and ROIs larger than the input size 
    of the detector.

    Args:
        roi_bboxes (np.ndarray): Array of bounding boxes representing the ROIs.
        img_shape (tuple): Original image dimensions as (height, width).
        det_shape (tuple): Detection window dimensions as (height, width).
        overlap_px (int, optional): Overlap between sliding windows in pixels. Defaults to 20.

    Returns:
        np.ndarray: Crop bounding boxes arrays for the sliding windows.
    """
    h, w = det_shape
    all_bboxes = []
    
    for roi_bbox in roi_bboxes:
        xmin, ymin, xmax, ymax = roi_bbox
        roi_h, roi_w = ymax - ymin, xmax - xmin

        n_x = math.ceil(roi_w / (w - overlap_px))
        n_y = math.ceil(roi_h / (h - overlap_px))

        # Generate the start positions for the sliding windows
        XS = xmin + (w - overlap_px) * np.arange(n_x)
        YS = ymin + (h - overlap_px) * np.arange(n_y)

        # Create a meshgrid of xmin and ymin coordinates
        X_grid, Y_grid = np.meshgrid(XS, YS, indexing='ij')

        # Calculate the bounding box coordinates
        X_max = np.minimum(X_grid + w, img_shape[1])
        Y_max = np.minimum(Y_grid + h, img_shape[0])

        # Stack to form the bounding boxes
        bboxes = np.stack((X_grid, Y_grid, X_max, Y_max), axis=-1).reshape(-1, 4)

        all_bboxes.append(bboxes)

    # Concatenate all bboxes from each ROI
    all_bboxes = np.vstack(all_bboxes).astype(np.int32)

    return all_bboxes

# def get_sliding_windows_vect(roi_bboxes, img_shape, det_shape, overlap_px=20):
#     """Generates sliding window bounding boxes over the regions of interest.
#     Used for datasets that don't allow resize and ROIs larger than the input size 
#     of the detector.

#     Args:
#         roi_bboxes (np.ndarray): Array of bounding boxes representing the ROIs.
#         img_shape (tuple): Original image dimensions as (height, width).
#         det_shape (tuple): Detection window dimensions as (height, width).
#         overlap_px (int, optional): Overlap between sliding windows in pixels. Defaults to 20.

#     Returns:
#         np.ndarray: Crop bounding boxes arrays for the sliding windows.
#     """
#     h, w = det_shape

#     # Extract xmin, ymin, xmax, ymax from the roi_bboxes array
#     xmin, ymin, xmax, ymax = roi_bboxes[:, 0], roi_bboxes[:, 1], roi_bboxes[:, 2], roi_bboxes[:, 3]

#     # Calculate ROI heights and widths
#     roi_h = ymax - ymin
#     roi_w = xmax - xmin

#     # Calculate number of sliding windows in x and y directions
#     n_x = np.ceil(roi_w / (w - overlap_px)).astype(int)
#     n_y = np.ceil(roi_h / (h - overlap_px)).astype(int)

#     # Compute sliding window positions for all ROIs
#     XS = [np.arange(xmin[i], xmin[i] + (w - overlap_px) * n_x[i], w - overlap_px) for i in range(len(roi_bboxes))]
#     YS = [np.arange(ymin[i], ymin[i] + (h - overlap_px) * n_y[i], h - overlap_px) for i in range(len(roi_bboxes))]

#     # Calculate bounding boxes for each sliding window
#     bboxes_list = []
#     for i in range(len(roi_bboxes)):
#         X_mesh, Y_mesh = np.meshgrid(XS[i], YS[i])
#         _xmin = X_mesh.flatten().astype(int)
#         _ymin = Y_mesh.flatten().astype(int)
#         _xmax = np.minimum(_xmin + w, img_shape[1]).astype(int)
#         _ymax = np.minimum(_ymin + h, img_shape[0]).astype(int)
        
#         bboxes = np.stack([_xmin, _ymin, _xmax, _ymax], axis=1)
#         bboxes_list.append(bboxes)

#     # Stack all bounding boxes together for all ROIs
#     crop_bboxes = np.vstack(bboxes_list)

#     return crop_bboxes



def needs_rotation(roi_bboxes, det_shape):
    """
    Determines whether the detection window needs to be rotated to match the orientation 
    of each region of interest (ROI). If the orientation of the detection window differs 
    from that of the ROI, rotation is required to ensure alignment with the model's expected 
    detection window configuration.

    Args:
        roi_bboxes (np.ndarray): An array of bounding boxes representing the regions of interest (ROIs). 
                                 Each bounding box is expected to have the format [xmin, ymin, xmax, ymax].
        det_shape (tuple): A tuple representing the dimensions of the detector input size as (height, width).

    Returns:
        np.ndarray: A boolean array where each element indicates whether the corresponding detection window 
                    requires rotation (True for rotation, False otherwise).
    """
    h_window, w_window = det_shape

    # Calculate heights and widths for ROI bboxes
    h_roi = roi_bboxes[:, 3] - roi_bboxes[:, 1]
    w_roi = roi_bboxes[:, 2] - roi_bboxes[:, 0]

    # Check same orientation
    same_orientation = ((h_roi >= w_roi) & (h_window >= w_window)) | ((h_roi < w_roi) & (h_window < w_window))

    return ~same_orientation



def get_detection_window_vect_simplified(roi_bboxes, img_shape, det_shape, padding=20, allow_resize=True):
    """Calculates detection windows (either sliding or non-sliding) for ROIs.

    Args:
        roi_bboxes (np.ndarray): Array of bounding boxes representing the ROIs.
        img_shape (tuple): Original image dimensions as (height, width).
        det_shape (tuple): Detection window dimensions as (height, width).
        padding (int, optional): Padding added to ROIs. Defaults to 20.
        allow_resize (bool, optional): If True, allows resizing of the detection window. Defaults to True.

    Returns:
        tuple: Crop and full bounding boxes arrays for the detection windows.
    """

    # Apply rotation logic based on the input ROI and detection window
    rotation_mask = needs_rotation(roi_bboxes, det_shape)
    heights = np.where(rotation_mask, det_shape[1], det_shape[0])  # Swap dimensions if rotation is needed
    widths  = np.where(rotation_mask, det_shape[0], det_shape[1])
    aspect_ratios = heights / widths
    
    # Calculate ROI heights, widths, and centers (xc, yc)
    heights_roi = roi_bboxes[:, 3] - roi_bboxes[:, 1]
    widths_roi = roi_bboxes[:, 2] - roi_bboxes[:, 0]


    # Find ROIs that require the sliding-window 
    sliding_mask = ~allow_resize & ((heights_roi > heights) | (widths_roi > widths))
    sliding_ids, non_sliding_ids = np.where(sliding_mask)[0], np.where(~sliding_mask)[0]
    

    sliding_crop_bboxes, non_sliding_crop_bboxes = np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.int32)
    sliding_rois, non_sliding_rois = roi_bboxes[sliding_ids, ...], roi_bboxes[non_sliding_ids, ...]
    if len(sliding_ids) > 0:
        sliding_crop_bboxes = get_sliding_windows_vect(sliding_rois, img_shape, det_shape, overlap_px = 20)


    if len(non_sliding_ids) > 0:
        xc = (non_sliding_rois[:, 0] + non_sliding_rois[:, 2]) // 2
        yc = (non_sliding_rois[:, 1] + non_sliding_rois[:, 3]) // 2

        # Start with det_shape dimensions, apply padding, and maintain aspect ratio
        heights = np.maximum(heights_roi + padding, heights)[non_sliding_ids]
        widths = np.maximum(widths_roi + padding, widths)[non_sliding_ids]
        heights_roi, widths_roi = heights_roi[non_sliding_ids], widths_roi[non_sliding_ids]

        # Adjust dimensions to keep aspect ratio consistent
        widths  = np.where(heights / widths > aspect_ratios[non_sliding_ids], heights / aspect_ratios, widths)
        heights = np.where(heights / widths < aspect_ratios[non_sliding_ids], widths * aspect_ratios, heights)

        # Calculate xmin, ymin, xmax, ymax for the crop bounding boxes
        xmin = xc - widths // 2
        ymin = yc - heights // 2
        xmax = xmin + widths
        ymax = ymin + heights

        # Ensure bounding boxes stay within image dimensions by translating them if needed
        img_w, img_h = img_shape[1], img_shape[0]

        # Correct boxes that exceed the right or bottom image boundary
        xmax_exceeds = xmax > img_w
        ymax_exceeds = ymax > img_h
        xmin = np.where(xmax_exceeds, xmin - (xmax - img_w), xmin)
        xmax = np.minimum(xmax, img_w)
        ymin = np.where(ymax_exceeds, ymin - (ymax - img_h), ymin)
        ymax = np.minimum(ymax, img_h)

        # Correct boxes that fall below the left or top image boundary
        xmin = np.maximum(xmin, 0)
        ymin = np.maximum(ymin, 0)
        xmax = xmin + widths
        ymax = ymin + heights

        # Stack the final bounding box coordinates
        non_sliding_crop_bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1)

    return sliding_crop_bboxes, non_sliding_crop_bboxes, sliding_rois, non_sliding_rois



def get_detection_windows(rois, img_shape, det_shape=(960, 960), bbox_type='naive', allow_resize=True):
    """
    Generates detection windows based on the given regions of interest (ROIs) and image dimensions. 
    It allows for different methods of filtering the detection windows based on the specified 
    bounding box type and resizing behavior.

    Args:
        rois (np.ndarray): Array of bounding boxes representing the regions of interest (ROIs), 
                           with each box formatted as [xmin, ymin, xmax, ymax].
        img_shape (tuple): Dimensions of the original image as (height, width).
        det_shape (tuple, optional): Dimensions of the detection window (height, width). 
                                     Default is (960, 960).
        bbox_type (str, optional): Type of bounding box filtering to apply. Choices are:
                                   'all' (use all windows), 'naive' (filter naively), 
                                   and 'sorted' (filter based on sorted ROIs). 
                                   Default is 'naive'.
        allow_resize (bool, optional): Whether to allow resizing of windows if they don't match 
                                       the desired shape. Default is True.

    Returns:
        np.ndarray: Array of detection windows in the format [xmin, ymin, xmax, ymax].
    """
    sliding_windows, non_sliding_windows, sliding_rois, non_sliding_rois = get_detection_window_vect_simplified(rois, img_shape, det_shape, allow_resize=allow_resize)
    if bbox_type == 'all':
        det_windows = np.concatenate((sliding_windows, non_sliding_windows), axis=0)
    elif bbox_type == 'naive':
        filtered_windows = filter_detection_windows_naive(non_sliding_rois, non_sliding_windows, sliding_windows, img_shape, det_shape=det_shape, allow_resize=allow_resize)
        det_windows = np.concatenate((sliding_windows, filtered_windows), axis=0)
    elif bbox_type == 'sorted':
        filtered_windows =  filter_detection_windows_sorted(non_sliding_rois, non_sliding_windows, sliding_windows, img_shape, det_shape=det_shape, allow_resize=allow_resize)
        det_windows = np.concatenate((sliding_windows, filtered_windows), axis=0)

    else:
        raise NotImplementedError
    return det_windows


    
def is_bbox_inside_bbox(inner_bbox, outer_bbox):
    """
    Checks if a bounding box (inner_bbox) is fully contained within another bounding box (outer_bbox).

    Args:
        inner_bbox (np.ndarray): The inner bounding box in the format [xmin, ymin, xmax, ymax].
        outer_bbox (np.ndarray): The outer bounding box in the format [xmin, ymin, xmax, ymax].

    Returns:
        bool: True if inner_bbox is completely inside outer_bbox, otherwise False.
    """
    # Check if a bounding box (inner_bbox) is fully inside another bounding box (outer_bbox)
    xmin_inner, ymin_inner, xmax_inner, ymax_inner = inner_bbox
    xmin_outer, ymin_outer, xmax_outer, ymax_outer = outer_bbox
    return (xmin_outer <= xmin_inner <= xmax_outer and
            ymin_outer <= ymin_inner <= ymax_outer and
            xmax_outer >= xmax_inner and ymax_outer >= ymax_inner)



def filter_detection_windows_naive(rois, windows, final_windows, img_shape, det_shape, allow_resize):
    """
    Filters detection windows by naively removing any windows that are fully covered by 
    previously processed windows.

    Args:
        rois (np.ndarray): Array of ROIs as bounding boxes [xmin, ymin, xmax, ymax].
        crop_windows (np.ndarray): Array of cropped windows associated with ROIs.
        full_windows (np.ndarray): Array of full windows associated with ROIs.
        img_shape (tuple): Dimensions of the original image (height, width).
        det_shape (tuple): Dimensions of the detection window (height, width).
        allow_resize (bool): Flag indicating whether resizing is allowed.

    Returns:
        np.ndarray: Array of filtered crop windows where redundant windows are removed.
    """

    for i, roi in enumerate(rois):
        if any(is_bbox_inside_bbox(roi, window) for window in final_windows):
            continue

        final_windows = np.concatenate((final_windows, np.expand_dims(windows[i], 0)), axis=0)

    return final_windows



def filter_detection_windows_sorted(rois, windows, final_windows, img_shape, det_shape, allow_resize):
    """
    Filters detection windows by sorting them based on how many ROIs are contained within 
    each window. Windows that cover more ROIs are processed first.

    Args:
        rois (np.ndarray): Array of ROIs as bounding boxes [xmin, ymin, xmax, ymax].
        crop_windows (np.ndarray): Array of cropped windows associated with ROIs.
        full_windows (np.ndarray): Array of full windows associated with ROIs.
        img_shape (tuple): Dimensions of the original image (height, width).
        det_shape (tuple): Dimensions of the detection window (height, width).
        allow_resize (bool): Flag indicating whether resizing is allowed.

    Returns:
        np.ndarray: Array of filtered crop windows sorted by coverage of multiple ROIs.
    """

    hits = np.zeros(len(windows), dtype=int)

    for i, window in enumerate(windows):
        hits[i] = np.sum(np.array([is_bbox_inside_bbox(roi, window) for roi in rois]))

    # Get indices of sorted hits
    sorted_indices = np.argsort(-hits)

    # Sort ROIs, crop_windows, and full_windows based on hits
    rois_sorted = rois[sorted_indices]
    windows_sorted = windows[sorted_indices]

    final_crop_windows = filter_detection_windows_naive(rois_sorted, windows_sorted, final_windows, img_shape, det_shape, allow_resize)
    return final_crop_windows
