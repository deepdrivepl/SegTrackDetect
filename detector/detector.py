import os
import torch

from .configs import DETECTION_MODELS
from .aggregation import rot90points, scale_coords

    
class Detector:
    """A class for loading a detection model and performing object detection.

    Attributes:
        config (dict): Configuration settings for the model, including weights, input size, preprocessing, and postprocessing methods.
        net (torch.jit.ScriptModule): Loaded detection model.
        device (str): Device being used ('cuda' or 'cpu'). Default is 'cuda'.
        input_size (tuple): Size of the input image expected by the model.
        preprocess (callable): Preprocessing function for input images.
        preprocess_args (dict): Additional arguments for the preprocessing function.
        postprocess (callable): Postprocessing function for model outputs.
        postprocess_args (dict): Additional arguments for the postprocessing function.

    Args:
        model_name (str): Name of the detection model to be loaded. Must be a key in the `DETECTION_MODELS` configuration.
        device (str, optional): Device to run the model on, either 'cuda' or 'cpu'. Default is 'cuda'.
    """

    def __init__(self, model_name, device='cuda'):
        assert model_name in DETECTION_MODELS.keys(), f'{model_name} not in DETECTION_MODELS.keys()'

        self.config = DETECTION_MODELS[model_name]
        weights = self.config['weights']
        print(f"Loading detector weights: {os.path.basename(weights)}")
        self.net = torch.jit.load(weights)
        self.net.to(device)

        self.input_size = self.config['in_size']

        self.preprocess = self.config['preprocess']
        self.preprocess_args = self.config['preprocess_args']
        self.postprocess = self.config['postprocess']
        self.postprocess_args = self.config['postprocess_args']
        self.device = device


    @torch.no_grad()
    def get_detections(self, img_tensor):
        """Perform detection on the input image tensor.

        Args:
            img_tensor (torch.Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            list: List of detections for each image in the batch.
        """
        img_tensor = self.preprocess(img_tensor, **self.preprocess_args)
        detections = self.net(img_tensor.to(self.device))
        detections = self.postprocess(detections, **self.postprocess_args)

        return detections


    def get_config_dict(self):
        """Retrieve the configuration settings of the detector.

        Returns:
            dict: Dictionary containing the configuration settings, excluding preprocess and postprocess
                functions.
        """
        return {'detector': {k:v for k,v in self.config.items() if k not in ['preprocess', 'postprocess']}}


    def postprocess_detections(self, detections, det_metadata):
        """Postprocess the detections to translate them back to the original image dimensions.

        Args:
            detections (list): List of detections for each image.
            det_metadata (dict): Metadata containing information about the original image dimensions.

        Returns:
            tuple:
                - img_det (torch.Tensor): Tensor containing the processed detection results.
                - img_win (torch.Tensor): Tensor containing the bounding box windows for detections.
        """

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

        if img_det_list and img_win_list:
            img_det = torch.cat(img_det_list)
            img_win = torch.cat(img_win_list).to(self.device)

            # Remove NaNs from both img_det and img_win
            valid_mask = ~torch.any(img_det.isnan(), dim=1)
            img_win = img_win[valid_mask]
            img_det = img_det[valid_mask]

        else:
            img_det = torch.empty((0,6), device=self.device)
            img_win = torch.empty((0,4), device=self.device)

        
        return img_det, img_win