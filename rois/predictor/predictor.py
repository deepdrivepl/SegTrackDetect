import importlib

import torch
import cv2
import numpy as np

from .configs import PREDICTOR_MODELS


    
class Predictor:
    """
    A class for predicting regions of interest (ROIs) using a specified tracker.

    This class initializes a tracker based on the configuration provided in the
    `PREDICTOR_MODELS` dictionary and manages the prediction of bounding boxes 
    and corresponding masks.

    Attributes:
        config (dict): The configuration dictionary for the selected tracker.
        tracker (object): An instance of the tracker class initialized with the provided arguments.
        frame_delay (int): The frame delay for the tracker.

    Args:
        tracker_name (str): The name of the tracker to be used. This must be a key in 
            the `PREDICTOR_MODELS` dictionary.

    """

    def __init__(self, tracker_name):
        assert tracker_name in PREDICTOR_MODELS.keys(), f'{tracker_name} not in PREDICTOR_MODELS.keys()'

        self.config = PREDICTOR_MODELS[tracker_name]
        trk_class = getattr(importlib.import_module(self.config['module_name']), self.config['class_name'])
        self.tracker = tracker = (trk_class)(**self.config['args'])
        self.frame_delay = self.config['frame_delay']


    def get_predicted_roi(self, frame_id, orig_shape, estim_shape):
        """
        Get the predicted region of interest (ROI) mask for the current frame.

        This method retrieves the predicted bounding boxes from the tracker, scales 
        them to the estimated shape, and creates a binary mask representing the 
        predicted regions.

        Args:
            frame_id (int): The current frame index.
            orig_shape (tuple): The original shape of the image as (height, width).
            estim_shape (tuple): The estimated shape of the image as (height, width).

        Returns:
            torch.Tensor: A binary mask with the predicted regions marked as 1.
        """
        self.predicted_bboxes = self.tracker.get_pred_locations()
        predicted_bboxes = torch.tensor(self.predicted_bboxes)

        H_orig, W_orig = orig_shape
        H_est, W_est = estim_shape
        predicted_mask = torch.zeros((H_est, W_est), dtype=torch.float)
        if frame_id >= self.frame_delay:

            mot_bboxes = predicted_bboxes[:, :-1]

            scale_x = W_est / W_orig
            scale_y = H_est / H_orig

            mot_bboxes[:, 0] = torch.clamp(mot_bboxes[:, 0] * scale_x, 0, W_est - 1)  # xmin
            mot_bboxes[:, 1] = torch.clamp(mot_bboxes[:, 1] * scale_y, 0, H_est - 1)  # ymin
            mot_bboxes[:, 2] = torch.clamp(mot_bboxes[:, 2] * scale_x, 0, W_est - 1)  # xmax
            mot_bboxes[:, 3] = torch.clamp(mot_bboxes[:, 3] * scale_y, 0, H_est - 1)  # ymax

            # Filter valid boxes
            valid_mask = (mot_bboxes[:, 2] - mot_bboxes[:, 0] > 0) & (mot_bboxes[:, 3] - mot_bboxes[:, 1] > 0)
            mot_bboxes = mot_bboxes[valid_mask]

            # Vectorized mask creation
            for xmin, ymin, xmax, ymax in mot_bboxes.int():
                predicted_mask[ymin:ymax+1, xmin:xmax+1] = 1

        return predicted_mask


    def update_tracker_state(self, detections_array):
        """
        Update the tracker state with new detections.

        This method updates the tracker with the latest detections and 
        the previously predicted bounding boxes.

        Args:
            detections_array (torch.Tensor): A tensor containing the new detections 
                to be processed by the tracker.
        """
        self.tracker.update(detections_array, self.predicted_bboxes)