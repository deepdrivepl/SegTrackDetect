import os
import time


import cv2
import numpy as np
import kornia

import torch
from torchvision import transforms as T
from collections import defaultdict



# based on https://github.com/WongKinYiu/yolov7
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=64):
    
    shape = img.shape[:2]  # orig hw
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # if not scaleup:  
    r = min(r, 1.0) # only scale down, do not scale up (for better test mAP)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    img_in = np.full((*new_shape, 3), color).astype(np.uint8)
    img_in[:new_unpad[1],:new_unpad[0],...] = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return img_in, new_unpad[::-1]



class ROIDataset(torch.utils.data.Dataset):
    def __init__(self, paths, dataset, roi_inf_size, roi_transform):
        self.paths = [x for x in paths if os.path.isfile(x)]
        # self.roi_transform = roi_transform(roi_inf_size[0], roi_inf_size[1])
        self.roi_transform = T.Compose([
            # T.ToTensor(),
            T.Resize((roi_inf_size[0], roi_inf_size[1]), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        self.dataset = dataset



    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W = image.shape[:2]

        metadata = {
            'image_path': os.path.abspath(self.paths[idx]),
            'coco': self.dataset.get_image_metadata(self.paths[idx])
        }    

    
        # return self.roi_transform(image), metadata
        return T.ToTensor()(image), metadata



class WindowDetectionDataset():
    def __init__(self, img_tensor, path, dataset, bboxes, size):
        self.path = os.path.abspath(path)

        self.image = img_tensor.cpu().numpy()[0,...]
        self.image = np.transpose(self.image, (1,2,0))
        self.image = (self.image * 255).astype(np.uint8)

        self.bboxes = torch.tensor(bboxes).int()
        self.size = size
        self.transform = T.ToTensor()
        self.h, self.w = self.image.shape[:2]


    def get_subwindow(self, idx):
        # Get the bounding box and crop the image
        xmin, ymin, xmax, ymax = self.bboxes[idx].tolist()
        crop_image = self.image[ymin:ymax, xmin:xmax] # H,W,C

        # Check if rotation is needed using a simpler conditional
        rotate = (crop_image.shape[0] > crop_image.shape[1]) != (self.size[0] > self.size[1])

        # Rotate the image if necessary
        if rotate:
            crop_image = cv2.rotate(crop_image, cv2.ROTATE_90_CLOCKWISE)

        # Resize and handle padding
        crop_h, crop_w = crop_image.shape[:2]
        if crop_h > self.size[0] or crop_w > self.size[1]:
            crop_image, unpadded = letterbox(crop_image, self.size, auto=False)
        else:
            unpadded = (crop_h, crop_w)

        # Pre-allocate output image with padding
        img_in = np.full((*self.size, 3), 114, dtype=np.uint8)
        img_in[:crop_image.shape[0], :crop_image.shape[1]] = crop_image

        # Pre-construct metadata tensor
        metadata = {
            'translate': torch.tensor([[xmin, ymin, xmin, ymin]], dtype=torch.float32),
            'rotation': torch.tensor([[rotate]], dtype=bool),
            'resize': torch.tensor([[crop_h > self.size[0] or crop_w > self.size[1]]], dtype=bool),
            'crop_shape': torch.tensor([[crop_h, crop_w]], dtype=torch.float32),
            'unpadded_shape': torch.tensor([unpadded], dtype=torch.int64),
            'roi_shape': torch.tensor([[ymax - ymin, xmax - xmin]], dtype=torch.float32),
        }

        return self.transform(img_in).unsqueeze(0), metadata



    def get_batch(self):

        batch = []
        metadata = defaultdict(list)
        for i in range(self.bboxes.shape[0]):
            subwindow, meta = self.get_subwindow(i)
            for k,v in meta.items():
                metadata[k].append(v)
            batch.append(subwindow)

        batch = torch.cat(batch)
        metadata = {k: torch.cat(v) for k,v in metadata.items()}
        metadata['bbox'] = self.bboxes
        return batch, metadata





    