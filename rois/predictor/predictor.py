import importlib
import os
import time

from statistics import mean

import torch
import cv2
import numpy as np

from .configs import PREDICTOR_MODELS


    
class Predictor:


    def __init__(self, tracker_name):
        assert tracker_name in PREDICTOR_MODELS.keys(), f'{tracker_name} not in PREDICTOR_MODELS.keys()'

        self.config = PREDICTOR_MODELS[tracker_name]
        trk_class = getattr(importlib.import_module(self.config['module_name']), self.config['class_name'])
        self.tracker = tracker = (trk_class)(**self.config['args'])
        self.frame_delay = self.config['frame_delay']

        self.prediction_times = []
        self.update_times = []
        self.mask_creation_times = []


    def get_predicted_roi(self, frame_id, orig_shape, estim_shape):

        t1 = time.time()
        self.predicted_bboxes = self.tracker.get_pred_locations()
        predicted_bboxes = torch.tensor(self.predicted_bboxes)
        self.prediction_times.append(time.time()-t1)

        t2 = time.time()
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

        self.mask_creation_times.append(time.time() - t2)

        return predicted_mask


    def update_tracker_state(self, detections_array):
        t1 = time.time()
        self.tracker.update(detections_array, self.predicted_bboxes)
        self.update_times.append(time.time()-t1)


    def get_execution_times(self, num_images):
        return sum(self.prediction_times)/num_images, sum(self.mask_creation_times)/num_images, sum(self.update_times)/num_images