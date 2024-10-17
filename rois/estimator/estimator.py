import os
import torch
import time

from statistics import mean
from .configs import ESTIMATOR_MODELS


    
class Estimator:
    """
    A class to handle ROI estimation using pre-trained models loaded via TorchScript.

    Attributes:
        config (dict): Configuration for the selected model containing weights, input size, and processing methods.
        net (torch.jit.ScriptModule): Loaded TorchScript model for ROI estimation.
        device (str): Device to run the model on (default is 'cuda').
        input_size (tuple): Input size expected by the model.
        preprocess (callable): Preprocessing function to apply to input data.
        postprocess (callable): Postprocessing function to apply to the model's output.
        sigmoid_incl (bool): Whether the sigmoid activation is included in the model's output.
        thresh (float): Threshold to apply during postprocessing.
        dilate (bool): Whether to apply dilation to the output mask.
        k_size (int): Kernel size for dilation.
        iter (int): Number of iterations for morphological operations.
        postprocess_times (list): Stores times taken for postprocessing each image.
        infer_times (list): Stores times taken for inference on each image.

    Args:
        model_name (str): The name of the model to load. Must be a key in the `ESTIMATOR_MODELS` configuration.
        device (str, optional): The device to run the model on, either 'cuda' or 'cpu'. Default is 'cuda'.
    """

    def __init__(self, model_name, device='cuda'):
        assert model_name in ESTIMATOR_MODELS.keys(), f'{model_name} not in ESTIMATOR_MODELS.keys()'

        self.config = ESTIMATOR_MODELS[model_name]
        weights = self.config['weights']
        print(f"Loading ROI estimator weights: {os.path.basename(weights)}")
        self.net = torch.jit.load(weights)
        self.net.to(device)
        self.net.eval() # ?

        self.input_size = self.config['in_size']
        self.preprocess = self.config['preprocess']
        self.preprocess_args = self.config['preprocess_args']
        self.postprocess = self.config['postprocess']
        self.postprocess_args = self.config['postprocess_args']
        self.device = device

        self.postprocess_times = []
        self.infer_times = []


    @torch.no_grad()
    def get_estimated_roi(self, img_tensor, orig_shape):
        """
        Estimates the ROI mask from the input image tensor.

        Args:
            img_tensor (torch.Tensor): The input image tensor to process.
            orig_shape (tuple): The original shape of the image, used in postprocessing.

        Returns:
            torch.Tensor: The post-processed estimated ROI mask.
        """
        t1 = time.time()
        estimated_mask = self.net(img_tensor.to(self.device))
        self.infer_times.append(time.time()-t1)

        t2 = time.time()
        estimated_mask = self.postprocess(
            estimated_mask, 
            orig_shape,
            **self.postprocess_args
            
        )
        self.postprocess_times.append(time.time()-t2)
        return estimated_mask


    def get_execution_times(self, num_images):
        """
        Returns the average inference and postprocessing times.

        Args:
            num_images (int): The number of images processed.

        Returns:
            tuple: A tuple containing:
                - float: The average inference time.
                - float: The average postprocessing time.
        """
        return sum(self.infer_times)/num_images, sum(self.postprocess_times)/num_images
