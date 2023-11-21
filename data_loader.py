import os

import cv2
import numpy as np

import torch
from torchvision import transforms as T




class SingleDetectionDataset(torch.utils.data.Dataset):
    
    def __init__(self, paths, det_inf_size):
        self.paths = [x for x in paths if os.path.isfile(x)]
        print(f'len(dataset) before: {len(paths)}')
        print(f'len(dataset) after: {len(self.paths)}')
        self.size = det_inf_size
        self.transform = T.Compose([
            T.ToTensor()
        ])
        
        
    def __len__(self):
        return len(self.paths)
    
    
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        h,w = img.shape[:2]
        # print('self.size', self.size)
        img, (unpadded) = self.letterbox(img, self.size, auto=False) #[0]
        h_,w_ = unpadded
        img = np.ascontiguousarray(img[:, :, ::-1])
        # cv2.imwrite('test.jpg', img[:,:,::-1])
        metadata = {"image_path": os.path.abspath(self.paths[idx]), "image_h": h, "image_w": w, "image_idx": idx,
                   "unpadded_h": h_, "unpadded_w": w_}
        return self.transform(img).float(), metadata



    # based on https://github.com/WongKinYiu/yolov7
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=64):
        
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
        # print(img_in.shape, new_unpad)
        return img_in, new_unpad[::-1]
        
        

class ROIDataset(torch.utils.data.Dataset):
    def __init__(self, paths, roi_inf_size, roi_transform):
        self.paths = [x for x in paths if os.path.isfile(x)]
        print(f'len(dataset) before: {len(paths)}')
        print(f'len(dataset) after: {len(self.paths)}')
        self.roi_transform = roi_transform(roi_inf_size[0], roi_inf_size[1])


    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H,W = image.shape[:2]
        
        image = self.roi_transform(image)

        metadata = {
            'image_path': os.path.abspath(self.paths[idx]),
            'image_h': H,
            'image_w': W
        }            
    
        return image.to(torch.float), metadata
    

    
class WindowDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, path, bboxes, size):
        self.path = os.path.abspath(path)
        self.image = cv2.imread(path)[:,:,::-1]
        self.image = np.ascontiguousarray(self.image)
        self.bboxes = torch.tensor(bboxes)
        self.size = size
        self.transform = T.ToTensor()
        self.h, self.w = self.image.shape[:2]


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
        
        metadata = {'bbox': self.bboxes[idx], 'translate': torch.tensor([xmin, ymin, xmin, ymin]),  'image_path': self.path, 'rotation': rotate, 'shape': torch.tensor([h_window, w_window])}

        return self.transform(img_in), metadata


