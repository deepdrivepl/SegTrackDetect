import os
import time


import cv2
import numpy as np
import kornia

import torch
from torchvision import transforms as T
from collections import defaultdict



def resize_keep_ar(img, new_shape=(640, 640)):
    """
    Resize an image while maintaining its aspect ratio.

    Args:
        img (torch.Tensor): Input image tensor of shape [C, H, W].
        new_shape (tuple, optional): Desired output shape as (height, width). 
                                      Defaults to (640, 640).

    Returns:
        tuple: 
            - resized_img (torch.Tensor): Resized image tensor of shape [C, height, width].
            - new_unpad (tuple): Dimensions of the resized image (height, width).
    """

    # shape of input image
    shape = img.shape[-2:]  # H, W
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute shape
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))  # height, width
    resized_img = kornia.geometry.resize(img, new_unpad, interpolation='bilinear')

    # Normalize to fix AP/AR ml
    resized_img = (resized_img * 255).to(torch.uint8)
    resized_img = (resized_img / 255.0).to(torch.float32)

    return resized_img, (new_unpad[0], new_unpad[1])



class ROIDataset(torch.utils.data.Dataset):
    """A dataset class for loading full images and their metadata for ROI Estimation Network.

    Args:
        paths (list): List of file paths to images.
        dataset (object): Dataset object containing metadata retrieval method.
        roi_inf_size (tuple): ROI Estimation Network input size.
        roi_transform (callable): Transformation function for the ROI Estimation Network.

    Attributes:
        paths (list): Filtered list of image file paths.
        roi_transform (callable): Transformation function for ROI.
        dataset (object): Metadata dataset object.
    """
    def __init__(self, paths, dataset, roi_inf_size, roi_transform):
        self.paths = [x for x in paths if os.path.isfile(x)]
        self.roi_transform = roi_transform(roi_inf_size[0], roi_inf_size[1])
        self.dataset = dataset


    def __len__(self):
        """Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.paths)


    def __getitem__(self,idx):
        """Retrieve an image and its metadata by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple:
                - image (torch.Tensor): Image tensor of shape [C, H, W].
                - metadata (dict): Metadata dictionary containing image path and COCO metadata.
        """
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W = image.shape[:2]

        metadata = {
            'image_path': os.path.abspath(self.paths[idx]),
            'coco': self.dataset.get_image_metadata(self.paths[idx])
        }    

        return T.ToTensor()(image), metadata



class WindowDetectionDataset():
    """A dataset class for cropping detection windows returned by the ROI Fusion Module.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape [C, H, W].
        dataset (object): Dataset object containing metadata retrieval method.
        bboxes (list): List of bounding boxes for the image.
        size (tuple): Desired output size for the image windows.

    Attributes:
        image (torch.Tensor): Image tensor.
        bboxes (torch.Tensor): Tensor of detection window bounding boxes.
        size (tuple): Desired output size.
        transform (callable): Transformation function for the image.
        h (int): Height of the image.
        w (int): Width of the image.
    """
    def __init__(self, img_tensor, dataset, bboxes, size):
        self.image = img_tensor[0] 
        self.bboxes = torch.tensor(bboxes, dtype=torch.int)

        self.size = size
        self.transform = T.ToTensor()
        self.h, self.w = self.image.shape[-2:]


    def get_subwindow(self, idx):
        """Retrieve a cropped and processed subwindow from the image based on the bounding box.

        Args:
            idx (int): Index of the bounding box.

        Returns:
            tuple:
                - img_in (torch.Tensor): Cropped and resized image tensor of shape [1, C, H_out, W_out].
                - metadata (dict): Metadata dictionary containing transformation details.
        """
        # Get the bounding box and crop the image
        xmin, ymin, xmax, ymax = self.bboxes[idx].tolist()
        crop_image = self.image[..., ymin:ymax, xmin:xmax]  # Crop: [C, H, W]
        
        
        # Check if rotation is needed
        rotate = (crop_image.shape[-2] > crop_image.shape[-1]) != (self.size[0] > self.size[1])

        # Rotate if necessary
        if rotate:
            crop_image = torch.rot90(crop_image, k=-1, dims=(-2, -1))


        # Resize and handle padding
        crop_h, crop_w = crop_image.shape[-2:]
        if crop_h > self.size[0] or crop_w > self.size[1]:
            crop_image, unpadded = resize_keep_ar(crop_image, self.size)
        else:
            unpadded = (crop_h, crop_w)


        # Pre-allocate output image with padding
        img_in = torch.full((crop_image.shape[0], *self.size), 0.56, dtype=crop_image.dtype, device=crop_image.device)
        img_in[..., :crop_image.shape[-2], :crop_image.shape[-1]] = crop_image


        metadata = {
            'translate': torch.tensor([[xmin, ymin, xmin, ymin]], dtype=torch.float32, device=crop_image.device),
            'rotation': torch.tensor([[rotate]], dtype=torch.bool, device=crop_image.device),
            'resize': torch.tensor([[crop_h > self.size[0] or crop_w > self.size[1]]], dtype=torch.bool, device=crop_image.device),
            'crop_shape': torch.tensor([[crop_h, crop_w]], dtype=torch.float32, device=crop_image.device),
            'unpadded_shape': torch.tensor([unpadded], dtype=torch.int64, device=crop_image.device),
            'roi_shape': torch.tensor([[ymax - ymin, xmax - xmin]], dtype=torch.float32, device=crop_image.device),
        }

        return img_in.unsqueeze(0), metadata


    def get_batch(self):
        """Retrieve a batch of subwindows and their metadata.

        Returns:
            tuple:
                - batch (torch.Tensor): Batch tensor containing all subwindow tensors.
                - metadata (dict): Metadata dictionary containing all transformation details.
        """
        batch = []
        metadata = defaultdict(list)
        for i in range(self.bboxes.shape[0]):
            subwindow, meta = self.get_subwindow(i)
            for k, v in meta.items():
                metadata[k].append(v)
            batch.append(subwindow)

        batch = torch.cat(batch, dim=0)  # Concatenate all subwindow tensors
        metadata = {k: torch.cat(v) for k, v in metadata.items()}
        metadata['bbox'] = self.bboxes
        return batch, metadata


    