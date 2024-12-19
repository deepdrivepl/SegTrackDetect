import os
import torch

from .configs import ESTIMATOR_MODELS


    
class Estimator:
    """
    A class to handle ROI estimation using pre-trained models loaded via TorchScript.

    Attributes:
        config (dict): Configuration for the selected model, containing weights, input size, preprocessing, and postprocessing methods.
        net (torch.jit.ScriptModule): Loaded TorchScript model for ROI estimation.
        device (str): Device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.
        input_size (tuple): Input size expected by the model.
        preprocess (callable): Preprocessing function to apply to the input data.
        preprocess_args (dict): Additional arguments for the preprocessing function.
        postprocess (callable): Postprocessing function to apply to the model's output.
        postprocess_args (dict): Additional arguments for the postprocessing function.

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
        
        dtypes = {param.dtype for param in self.net.parameters()}
        self.dtype = next(iter(dtypes))

        self.input_size = self.config['in_size']
        self.preprocess = self.config['preprocess']
        self.preprocess_args = self.config['preprocess_args']
        self.postprocess = self.config['postprocess']
        self.postprocess_args = self.config['postprocess_args']
        self.device = device


    @torch.inference_mode()
    def get_estimated_roi(self, img_tensor, orig_shape):
        """
        Estimates the ROI mask from the input image tensor.

        Args:
            img_tensor (torch.Tensor): The input image tensor to process.
            orig_shape (tuple): The original shape of the image, used in postprocessing.

        Returns:
            torch.Tensor: The post-processed estimated ROI mask.
        """
        estimated_mask = self.net(img_tensor.to(self.device).to(self.dtype))
        estimated_mask = self.postprocess(
            estimated_mask, 
            orig_shape,
            **self.postprocess_args
            
        )
        return estimated_mask