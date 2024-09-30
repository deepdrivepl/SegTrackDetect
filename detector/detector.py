import os
import torch
import time

from statistics import mean

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


        self.inference_times = []
        self.postprocess_times = []
        self.nms_times = []
        self.postprocess_to_orig_times = []


    @torch.no_grad()
    def get_detections(self, img_tensor):
        t1 = time.time()
        detections = self.net(img_tensor.to(self.device))
        self.inference_times.append(time.time()-t1)

        t2 = time.time()
        detections = self.postprocess(detections)
        self.postprocess_times.append(time.time()-t2)

        t3 = time.time()
        detections = non_max_suppression(
            detections, 
            conf_thres = self.conf_thresh, 
            iou_thres = self.iou_thresh,
            multi_label = self.multi_label,
            labels = self.labels,
            merge = self.merge,
            agnostic = self.agnostic
        )
        self.nms_times.append(time.time()-t3)

        return detections

    def get_config_dict(self):
        return {'detector': {k:v for k,v in self.config.items() if k not in ['transform', 'postprocess']}}


    def postprocess_detections(self, detections, det_metadata):

        t1 = time.time()

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

        self.postprocess_to_orig_times.append(time.time()-t1)

        return torch.cat(img_det_list), torch.cat(img_win_list).to(self.device)


    def get_execution_times(self, num_images):
        return sum(self.inference_times)/num_images, sum(self.postprocess_times)/num_images, sum(self.nms_times)/num_images, sum(self.postprocess_to_orig_times)/num_images