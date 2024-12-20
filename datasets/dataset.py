import os
import time
import json

from glob import glob

import cv2
import numpy as np
import pandas as pd
import kornia

import torch
from torchvision import transforms as T
from collections import defaultdict
from PIL import Image
from tqdm import tqdm



class DirectoryDataset:
    """A dataset class for loading images and annotations from a directory.

    This class handles loading images and their corresponding COCO-format annotations 
    from a directory. It supports both predefined splits and custom splits.

    Args:
        data_root (str): Root directory containing images and annotation files.
        split (str, optional): Name of the predefined dataset split. Defaults to 'val'.
        flist (str, optional): Path to the file containing a list of image file paths. Defaults to None.
        name (str, optional): Name of the custom split for which annotations will be generated. Defaults to None.

    Attributes:
        data_root (str): Root directory containing images and annotation files.
        images (list of str): List of file paths to the images.
        annotations (dict): COCO-format annotations including images, annotations, and categories.
        is_sequential (bool): Whether the dataset has sequential image directories.
        seq2images (dict): Mapping of sequence names to lists of image file paths.
        metadata (pandas.DataFrame): Metadata of the images loaded from the annotations.
    """

    def __init__(self, data_root, split='val', flist=None, name=None):

        self.data_root = data_root

        if flist is not None:
            assert name is not None, "Provide a --name for your --flist"
            self.images = [x.rstrip() for x in open(flist)]
            self.images = list(set(self.images))
            self.images = sorted([x for x in self.images if os.path.isfile(x)])

            anno_path = f"{data_root}/{name}.json"
            if not os.path.isfile(anno_path):
                print(f"Creating COCO annotations for custom split {name}.")
                self.annotations = self.load_all_annos()
                with open(anno_path, 'w', encoding='utf-8') as f:
                    json.dump(self.annotations, f, ensure_ascii=False, indent=4)
            else:
                self.annotations = json.load(open(anno_path))

        else:
            anno_path = f"{data_root}/{split}.json"
            if not os.path.isfile(anno_path):
                print(f"{anno_path} does not exist, provide a valid split")
                exit(1)
            self.annotations = json.load(open(anno_path))
            self.images = [x['file_name'] for x in self.annotations['images']]
            self.images = list(set(self.images))
            self.images = sorted([x for x in self.images if os.path.isfile(x)])

        
        print(f"Found {len(self.images)} images")
        self.is_sequential = self.images[0].split(os.sep)[-2] != 'images'
        self.seq2images = self.get_seq2images()
        print({k:f'len(seq): {len(v)}' for k,v in self.seq2images.items()})

        self.metadata = pd.DataFrame(self.annotations['images'])
        self.metadata = self.metadata.drop_duplicates(ignore_index=True)


    def get_seq2images(self):
        """Creates a mapping of sequences to their corresponding images.

        Returns:
            dict: A dictionary where the keys are sequence names and the values are lists of image file paths.
        """
        sequences = list(set([x.split(os.sep)[-2] for x in self.images]))
        sequences = sorted(sequences)
        
        if sequences[0] != 'images':
            seq2images = {seq: sorted([x for x in self.images if x.split(os.sep)[-2]==seq]) for seq in sequences}
        else:
            seq2images = {'images': sorted(self.images)}
        return seq2images


    def load_all_annos(self):
        """Loads all COCO annotations from the root directory.

        Looks for JSON files in the root directory and aggregates their content into a single dictionary.

        Returns:
            dict: A dictionary containing 'images', 'annotations', and 'categories' keys, each mapping to aggregated lists.
        """
        annos_paths = glob(f"{self.data_root}/*.json")
        print(f"Found {len(annos_paths)} annotation files is {self.data_root}")

        if len(annos_paths) == 0:
            print("Annotations not available. Creating empty annotations")
            images = []
            for img_id, img_path in tqdm(enumerate(self.images)):
                W,H = Image.open(img_path).size
                images.append({
                    "id": img_id, "file_name": os.path.abspath(img_path), "height": H, "width": W
                })
            return {"images": images, "annotations": [], "categories": []}


        images, annotations, categories = [], [], []
        for anno_path in annos_paths:
            annos = json.load(open(anno_path))
            images+=annos["images"]
            annotations+=annos["annotations"]
            categories = annos["categories"]

        return {"images": images, "annotations": annotations, "categories": categories}


    def get_image_metadata(self, img_path):
        """Fetches COCO metadata for a given image.

        Args:
            img_path (str): The file path to the image.

        Returns:
            dict: A dictionary containing metadata of the image (e.g., file_name, width, height).

        Raises:
            AssertionError: If no matching metadata is found for the image.
        """
        metadata = self.metadata[self.metadata.file_name == os.path.abspath(img_path)]
        assert len(metadata) == 1
        return metadata.iloc[0].to_dict()




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
    resized_img = (resized_img / 255.0).to(img.dtype)

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
        self.roi_transform = roi_transform
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
        image = np.ascontiguousarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W = image.shape[:2]

        metadata = {
            'image_path': os.path.abspath(self.paths[idx]),
            'coco': self.dataset.get_image_metadata(self.paths[idx])
        }
        
        return T.ToTensor()(image).half(), metadata


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


    