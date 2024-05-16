import os

import cv2
import numpy as np

import torch
from torchvision import transforms as T


class SingleDetectionDataset(torch.utils.data.Dataset):
    
    def __init__(self, paths, dataset, det_inf_size, det_transform):
        self.paths = [x for x in paths if os.path.isfile(x)]
        print(f'len(dataset) before: {len(paths)}')
        print(f'len(dataset) after: {len(self.paths)}')
        self.dataset = dataset
        self.size = det_inf_size
        self.det_transform = det_transform(self.size[0], self.size[1])
        
        
    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        h,w = img.shape[:2]
        
        img = self.det_transform(img)
        metadata = {
            "image_path": os.path.abspath(self.paths[idx]), 
            "image_idx": idx,
            "coco": self.dataset.get_image_metadata(self.paths[idx])
        }
        return img, metadata
        
        

class ROIDataset(torch.utils.data.Dataset):
    def __init__(self, paths, dataset, roi_inf_size, roi_transform):
        self.paths = [x for x in paths if os.path.isfile(x)]
        print(f'len(dataset) before: {len(paths)}')
        print(f'len(dataset) after: {len(self.paths)}')
        self.roi_transform = roi_transform(roi_inf_size[0], roi_inf_size[1])
        self.dataset = dataset



    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W = image.shape[:2]
        
        image = self.roi_transform(image)

        metadata = {
            'image_path': os.path.abspath(self.paths[idx]),
            'coco': self.dataset.get_image_metadata(self.paths[idx])
        }            
    
        return image, metadata
    

    
class WindowDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset, bboxes, size):
        self.path = os.path.abspath(path)
        self.image = cv2.imread(path)[:,:,::-1]
        self.image = np.ascontiguousarray(self.image)
        self.bboxes = torch.tensor(bboxes)
        self.size = size
        self.transform = T.ToTensor()
        self.h, self.w = self.image.shape[:2]
        self.dataset = dataset


    def __len__(self):
        return len(self.bboxes)


    def __getitem__(self, idx):
        # img = letterbox(img, self.size, auto=False, scaleup=False)[0]
        xmin, ymin, xmax, ymax = map(int, self.bboxes[idx])
        h_window, w_window = ymax-ymin, xmax-xmin # vertical or horizontal

        if (h_window > w_window and self.size[0] > self.size[1]) or (h_window <= w_window and self.size[0] <= self.size[1]):
            rotate = False
        else:
            rotate = True
            
        color=(114, 114, 114)
        img_in = np.full((*self.size, 3), color).astype(np.uint8)
        if not rotate:
            img_in[:h_window,:w_window,...] = self.image[ymin:ymax,xmin:xmax,:]
        else:
            im = cv2.rotate(self.image[ymin:ymax,xmin:xmax,:], cv2.ROTATE_90_CLOCKWISE)
            img_in[:w_window,:h_window,...] = im
            
        xmin,ymin,xmax,ymax = self.bboxes[idx]
        
        metadata = {
            'bbox': self.bboxes[idx], 
            'translate': torch.tensor([xmin, ymin, xmin, ymin]),  
            'image_path': self.path, 
            'rotation': rotate, 
            'shape': torch.tensor([h_window, w_window]),
            'coco': self.dataset.get_image_metadata(self.path)
        }

        return self.transform(img_in), metadata


