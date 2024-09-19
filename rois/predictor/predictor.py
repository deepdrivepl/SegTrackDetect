import importlib
import os

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

    
    def get_predicted_roi(self, frame_id, orig_shape, estim_shape):


        self.predicted_bboxes = self.tracker.get_pred_locations()

        H,W = orig_shape
        predicted_mask = np.zeros((H, W), dtype=np.uint8)
        if frame_id >= self.frame_delay:

            mot_bboxes = self.predicted_bboxes[:,:-1]

            mot_bboxes[:,0] = np.where(mot_bboxes[:,0] < 0, 0, mot_bboxes[:,0])
            mot_bboxes[:,1] = np.where(mot_bboxes[:,1] < 0, 0, mot_bboxes[:,1])
            mot_bboxes[:,2] = np.where(mot_bboxes[:,2] >= W, W-1, mot_bboxes[:,2])
            mot_bboxes[:,3] = np.where(mot_bboxes[:,3] >= H, H-1, mot_bboxes[:,3])

            indices = np.nonzero(((mot_bboxes[:,2]-mot_bboxes[:,0]) > 0) & ((mot_bboxes[:,3]-mot_bboxes[:,1]) > 0))
            mot_bboxes = mot_bboxes[indices[0], :]

            for mot_bbox in mot_bboxes:
                xmin,ymin,xmax,ymax = map(int, mot_bbox[:4])
                predicted_mask[ymin:ymax+1, xmin:xmax+1] = 255
        predicted_mask = cv2.resize(predicted_mask, (estim_shape[1], estim_shape[0]))

        return predicted_mask


    def update_tracker_state(self, detections_array):
        self.tracker.update(detections_array, self.predicted_bboxes)