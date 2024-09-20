import os
import torch

from .configs import DETECTION_MODELS
from .aggregation import non_max_suppression, rot90points, scale_coords

    
class Detector:


    def __init__(self, model_name, device='cuda'):
        assert model_name in DETECTION_MODELS.keys(), f'{model_name} not in DETECTION_MODELS.keys()'

        self.config = DETECTION_MODELS[model_name]
        weights = self.config['weights']
        print(f"Loading detector weights: {os.path.basename(weights)}")
        self.net = torch.jit.load(weights)
        self.net.to(device)
        self.net.eval() # ?

        self.input_size = self.config['in_size']

        # NMS 
        self.conf_thresh = self.config['conf_thresh']
        self.iou_thresh = self.config['iou_thresh']
        self.multi_label = self.config['multi_label']
        self.labels = self.config['labels']
        self.merge = self.config['merge']
        self.agnostic = self.config['agnostic']

        self.preprocess = self.config['transform']
        self.postprocess = self.config['postprocess']
        self.device = device


    @torch.no_grad()
    def get_detections(self, img_tensor):

        detections = self.net(img_tensor.to(self.device))
        detections = self.postprocess(detections)
        detections = non_max_suppression(
            detections, 
            conf_thres = self.conf_thresh, 
            iou_thres = self.iou_thresh,
            multi_label = self.multi_label,
            labels = self.labels,
            merge = self.merge,
            agnostic = self.agnostic
        )

        return detections

    def get_config_dict(self):
        return {'detector': {k:v for k,v in self.config.items() if k not in ['transform', 'postprocess']}}


    def postprocess_detections(self, detections, det_metadata):

        img_det_list = []
        img_win_list = []

        for si, detection in enumerate(detections):
            if len(detection) == 0:
                continue
                
            # Collect detection windows using list
            img_win_list.append(det_metadata['bbox'][si].repeat(len(detection), 1))

            # Resize detections if required
            if det_metadata['resize'][si].item():
                detection[:, :4] = scale_coords(
                    det_metadata['unpadded_shape'][si], 
                    detection[:, :4], 
                    det_metadata['crop_shape'][si]
                )

            # Rotate detections if required
            if det_metadata['rotation'][si].item():
                h_window, w_window = det_metadata['roi_shape'][si]
                xmin_, ymax_ = rot90points(detection[:, 0], detection[:, 1], [w_window.item(), h_window.item()])
                xmax_, ymin_ = rot90points(detection[:, 2], detection[:, 3], [w_window.item(), h_window.item()])
                detection[:, 0] = xmin_
                detection[:, 1] = ymin_
                detection[:, 2] = xmax_
                detection[:, 3] = ymax_

            # Translate coordinates back to the original image
            detection[:, :4] += det_metadata['translate'][si].to(self.device)

            img_det_list.append(detection)

        return img_det_list, img_win_list