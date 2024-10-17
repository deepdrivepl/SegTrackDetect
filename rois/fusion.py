import cv2
import numpy as np
import time

import torch

from statistics import mean

from .estimator import Estimator
from .predictor import Predictor
from .windows_proposals import get_roi_bounding_boxes, get_detection_windows


class ROIModule:
    """
    ROIModule manages Region of Interest (ROI) estimation and prediction for image sequences, 
    ROI fusion, and detection window proposals.

    Attributes:
        tracker_name (str): Name of the tracker model.
        predictor (Predictor): An instance of the Predictor class for ROI prediction.
        estimator (Estimator): An instance of the Estimator class for ROI estimation.
        is_sequence (bool): If True, indicates a sequence of frames is processed. If True, the ROI Prediction Module will be disabled.
        bbox_type (str): The bounding box type ('sorted' by default).
        allow_resize (bool): Whether resizing of bounding boxes is allowed.
        rois_coordinates_times (list): List of times taken to calculate ROI bounding boxes.
        detection_windows_times (list): List of times taken to calculate detection windows.
    
    Args:
        tracker_name (str): The name of the tracker model to use.
        estimator_name (str): The name of the estimator model to use.
        is_sequence (bool, optional): Whether the input is a sequence of frames. Defaults to True.
        device (str, optional): Device for computations ('cuda' or 'cpu'). Defaults to 'cuda'.
        bbox_type (str, optional): Type of detection windows bounding boxes to generate. Defaults to 'sorted'.
        allow_resize (bool, optional): If True, allows resizing of detection bounding boxes. Defaults to True. If False, a sliding-widow within large ROIs will be used.
    """

    def __init__(self, tracker_name, estimator_name, is_sequence=True, device='cuda',
        bbox_type='sorted', allow_resize=True):

        self.tracker_name = tracker_name
        self.predictor = Predictor(tracker_name) if is_sequence else None
        self.estimator = Estimator(estimator_name, device=device)
        self.is_sequence = is_sequence
        self.bbox_type = bbox_type
        self.allow_resize = allow_resize

        self.rois_coorinates_times = []
        self.detection_windows_times = []


    def get_fused_roi(self, frame_id, img_tensor, orig_shape, det_shape):
        """
        Computes the fused ROI from both estimated and predicted masks, 
        and generates detection windows from the resulting bounding boxes.

        Args:
            frame_id (int): The frame identifier for prediction.
            img_tensor (torch.Tensor): The input image as a tensor.
            orig_shape (tuple): Original image dimensions.
            det_shape (tuple): Dimensions of the detection window.

        Returns:
            numpy.ndarray: Array of detection windows in the format (xmin, ymin, xmax, ymax).
        """
        self.estimated_mask = self.estimator.get_estimated_roi(img_tensor, orig_shape) # shape [H,W]
        estimated_shape = self.estimated_mask.shape[-2:]

        if self.is_sequence:
            self.predicted_mask = self.predictor.get_predicted_roi(frame_id, orig_shape, estimated_shape)
        else:
            self.predicted_mask = torch.zeros(self.estimated_mask.shape, dtype=torch.float32)
        fused_mask = torch.logical_or(self.estimated_mask.cpu(), self.predicted_mask).float()

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
        
        self.detection_windows_times.append(time.time()-t2)
        return detection_windows


    def reset_predictor(self):
        """
        Resets the predictor by reinitializing it with the same tracker.
        Should be used at the beginning of each sequence.
        """
        self.predictor = Predictor(self.tracker_name) if self.is_sequence else None


    def update_predictor(self, img_det):
        """
        Updates the predictor state with new_detections.
        
        Args:
            img_det (np.ndarray): A numpy array with detected objects.
        """
        if self.predictor:
            self.predictor.update_tracker_state(img_det) 


    def get_masks(self, shape):
        """
        Resizes and returns both estimated and predicted masks for visualization.

        Args:
            shape (tuple): Desired shape for resizing the masks.

        Returns:
            tuple: Estimated mask and predicted mask as resized numpy arrays.
        """
        estimated_mask = cv2.resize((self.estimated_mask.cpu().numpy()*255).astype(np.uint8), (shape[1], shape[0]))
        predicted_mask = cv2.resize((self.predicted_mask.cpu().numpy()*255).astype(np.uint8), (shape[1], shape[0]))
        return estimated_mask, predicted_mask


    def get_config_dict(self):
        """
        Returns the configuration dictionaries of the estimator and predictor.

        Returns:
            dict: A dictionary containing ROI estimator and predictor configurations.
        """
        estimator_config = {k:v for k,v in self.estimator.config.items() if k not in ['transform', 'postprocess']}
        predictor_config = self.predictor.config if self.predictor else None
        return {'ROI_estimator': estimator_config, 'ROI_predictor': predictor_config}


    def get_execution_times(self, num_images):
        """
        Calculates average execution times for ROI coordinates, detection windows, 
        prediction, and estimation over a given number of images.

        Args:
            num_images (int): The number of images processed.

        Returns:
            tuple: A tuple containing average times for ROI coordinates, detection windows generation, 
            predictor, and estimator execution.
        """
        return (sum(self.rois_coorinates_times)/num_images, 
                sum(self.detection_windows_times)/num_images, 
                self.predictor.get_execution_times(num_images), 
                self.estimator.get_execution_times(num_images)
                )

