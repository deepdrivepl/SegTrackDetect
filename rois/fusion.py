import cv2
import numpy as np
import time

import torch

from statistics import mean

from .estimator import Estimator
from .predictor import Predictor
from .windows_proposals import get_roi_bounding_boxes, get_detection_windows


class ROIModule:

    def __init__(self, tracker_name, estimator_name, is_sequence=True, device='cuda',
        bbox_type='sorted', allow_resize=True):

        self.tracker_name = tracker_name
        self.predictor = Predictor(tracker_name)
        self.estimator = Estimator(estimator_name, device=device)
        self.is_sequence = is_sequence
        self.bbox_type = bbox_type
        self.allow_resize = allow_resize

        self.rois_coorinates_times = []
        self.detection_windows_times = []


    def get_fused_roi(self, frame_id, img_tensor, orig_shape, det_shape):

        self.estimated_mask = self.estimator.get_estimated_roi(img_tensor, orig_shape)[0,0,...] # shape [B,1,H,W]
        estimated_shape = self.estimated_mask.shape[-2:]

        if self.is_sequence:
            self.predicted_mask = self.predictor.get_predicted_roi(frame_id, orig_shape, estimated_shape)
            fused_mask = torch.logical_or(self.estimated_mask.cpu(), self.predicted_mask).float()
        else:
            fused_mask = self.estimated_mask


        t1 = time.time()
        fused_mask = (fused_mask.numpy() * 255).astype(np.uint8) # convert to numpy for cv2.findContours()
        fused_bboxes = get_roi_bounding_boxes(fused_mask, orig_shape, estimated_shape)
        self.rois_coorinates_times.append(time.time()-t1)

        t2 = time.time()
        detection_windows = get_detection_windows(
            fused_bboxes, 
            img_shape=orig_shape, 
            det_shape=det_shape, 
            bbox_type=self.bbox_type,
            allow_resize=self.allow_resize,
        )

        if len(detection_windows) > 0:
            indices = np.nonzero(((detection_windows[:,2]-detection_windows[:,0]) > 0) & ((detection_windows[:,3]-detection_windows[:,1]) > 0))
            detection_windows = detection_windows[indices[0], :]
        self.detection_windows_times.append(time.time()-t2)
        return detection_windows


    def reset_predictor(self):
        self.predictor = Predictor(self.tracker_name)


    def get_masks(self, shape):
        estimated_mask = cv2.resize((self.estimated_mask.cpu().numpy()*255).astype(np.uint8), (shape[1], shape[0]))
        predicted_mask = cv2.resize((self.predicted_mask.cpu().numpy()*255).astype(np.uint8), (shape[1], shape[0]))
        return estimated_mask, predicted_mask


    def get_config_dict(self):
        estimator_config = {k:v for k,v in self.estimator.config.items() if k not in ['transform', 'postprocess']}
        predictor_config = self.predictor.config
        return {'ROI_estimator': estimator_config, 'ROI_predictor': predictor_config}


    def get_execution_times(self, num_images):
        return sum(self.rois_coorinates_times)/num_images, sum(self.detection_windows_times)/num_images, self.predictor.get_execution_times(num_images), self.estimator.get_execution_times(num_images)

