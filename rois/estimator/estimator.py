import os
import torch
import time

from statistics import mean
from .configs import ESTIMATOR_MODELS


    
class Estimator:


    def __init__(self, model_name, device='cuda'):
        assert model_name in ESTIMATOR_MODELS.keys(), f'{model_name} not in ESTIMATOR_MODELS.keys()'

        self.config = ESTIMATOR_MODELS[model_name]
        weights = self.config['weights']
        print(f"Loading ROI estimator weights: {os.path.basename(weights)}")
        self.net = torch.jit.load(weights)
        self.net.to(device)
        self.net.eval() # ?

        self.input_size = self.config['in_size']
        self.preprocess = self.config['transform']
        self.postprocess = self.config['postprocess']
        self.device = device

        self.sigmoid_incl = self.config['sigmoid_included']
        self.thresh = self.config['thresh']
        self.dilate = self.config['dilate']
        self.k_size = self.config['k_size']
        self.iter = self.config['iter']

        self.postprocess_times = []
        self.infer_times = []

        print(f'Postprocessing estimated mask with: threshold={self.thresh}, dilate={self.dilate}, k_size={self.k_size}, iter={self.iter}\nEdit ESTIMATOR_MODELS to change these values')

    @torch.no_grad()
    def get_estimated_roi(self, img_tensor, orig_shape):
        t1 = time.time()
        estimated_mask = self.net(img_tensor.to(self.device))
        self.infer_times.append(time.time()-t1)

        t2 = time.time()
        estimated_mask = self.postprocess(
            estimated_mask, 
            orig_shape,
            self.sigmoid_incl,
            self.thresh,
            self.dilate,
            self.k_size,
            self.iter
        )
        self.postprocess_times.append(time.time()-t2)
        return estimated_mask


    def get_execution_times(self, num_images):
        return sum(self.infer_times)/num_images, sum(self.postprocess_times)/num_images
