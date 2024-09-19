import numpy as np

from .estimator import Estimator
from .predictor import Predictor
from .windows_proposals import findBboxes, getDetectionWindows


class ROIModule:

    def __init__(self, tracker_name, estimator_name, is_sequence=True, device='cuda',
        bbox_type='sorted', allow_resize=True):

        self.predictor = Predictor(tracker_name)
        self.estimator = Estimator(estimator_name, device=device)
        self.is_sequence = is_sequence
        self.bbox_type = bbox_type
        self.allow_resize = allow_resize


    def get_fused_roi(self, frame_id, img_tensor, orig_shape, det_shape):

        estimated_mask = self.estimator.get_estimated_roi(img_tensor, orig_shape)
        estimated_shape = estimated_mask.shape[:2]

        if self.is_sequence:
            predicted_mask = self.predictor.get_predicted_roi(frame_id, orig_shape, estimated_shape)
            fused_mask = np.logical_or(estimated_mask, predicted_mask).astype(np.uint8)*255 # ?

        fused_mask = estimated_mask
        fused_bboxes = findBboxes(fused_mask, orig_shape, estimated_shape)
        detection_windows = getDetectionWindows( # TODO split into detection windows coordinates & filtering
            fused_bboxes, 
            orig_shape, 
            det_size=det_shape, 
            bbox_type=self.bbox_type,
            allow_resize=self.allow_resize,
        )

        detection_windows = np.array(detection_windows).astype(np.int32)
                
        if len(detection_windows) > 0:
            indices = np.nonzero(((detection_windows[:,2]-detection_windows[:,0]) > 0) & ((detection_windows[:,3]-detection_windows[:,1]) > 0))
            detection_windows = detection_windows[indices[0], :]
        return detection_windows

