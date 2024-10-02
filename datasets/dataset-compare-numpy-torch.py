import os
import time


import cv2
import numpy as np
import kornia

import torch
from torchvision import transforms as T
import torch.nn.functional as F
from collections import defaultdict


class WindowDetectionDataset():
    def __init__(self):
        pass


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


# based on https://github.com/WongKinYiu/yolov7
def letterbox_numpy(img, new_shape=(640, 640), color=(114, 114, 114)):
    
    shape = img.shape[:2]  # orig hw
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) 
    r = min(r, 1.0) # only scale down, do not scale up (for better test mAP)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    img_in = np.full((*new_shape, 3), color).astype(np.uint8)
    # print('before cv2 resize: ', np.unique(img))
    img_in[:new_unpad[1],:new_unpad[0],...] = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # print('after cv2 resize: ', np.unique(img_in))
    return img_in, new_unpad[::-1]


def letterbox_torch(img, new_shape=(640, 640), color=(114, 114, 114)):
    # shape of input image
    shape = img.shape[-2:]  # H, W
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)  # only scale down

    # Compute padding
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))  # height, width
    # print('before kornia resize: ', torch.unique(img))
    original_dtype = img.dtype
    # print(img.shape)
    resized_img = kornia.geometry.resize(img, new_unpad, interpolation='bilinear') #, align_corners=False, antialias=True)
    resized_img = (resized_img * 255).to(torch.uint8)
    resized_img = (resized_img / 255.0).to(torch.float32)
    #resized_img = resized_img.to(original_dtype)
    # print(img.shape)
    # resized_img = F.interpolate(img, size=(new_unpad[0], new_unpad[1]), mode='linear', align_corners=False)
    # resized_img = resized_img.to(original_dtype)
    # print('after kornia resize: ', torch.unique(resized_img))
    # Add padding and fill with specified color
    padded_img = torch.full((img.shape[0], *new_shape), 0.56, dtype=img.dtype, device=img.device)
    padded_img[..., :new_unpad[0], :new_unpad[1]] = resized_img
    # print('after pasting: ', torch.unique(padded_img))
    return padded_img, (new_unpad[0], new_unpad[1])


class WindowDetectionDatasetNumpy():
    def __init__(self, img_tensor, path, dataset, bboxes, size):
        self.path = os.path.abspath(path)
        self.basename = os.path.basename(os.path.splitext(path)[0])

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
            crop_image, unpadded = letterbox_numpy(crop_image, self.size)
        else:
            unpadded = (crop_h, crop_w)

        # Pre-allocate output image with padding
        img_in = np.full((*self.size, 3), 114, dtype=np.uint8)
        img_in[:crop_image.shape[0], :crop_image.shape[1]] = crop_image
        # cv2.imwrite(f'checks-images/{self.basename}-{idx}-numpy.png', img_in)

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
            # if i!=3:
            #     continue
            subwindow, meta = self.get_subwindow(i)
            for k,v in meta.items():
                metadata[k].append(v)
            batch.append(subwindow)

        batch = torch.cat(batch)
        metadata = {k: torch.cat(v) for k,v in metadata.items()}
        metadata['bbox'] = self.bboxes
        return batch, metadata


def save_torch_img(tensor_img, path):
    img = (np.transpose(tensor_img.cpu().numpy(), (1,2,0)) * 255).astype(np.uint8)
    cv2.imwrite(path, img)


class WindowDetectionDatasetTorch():
    def __init__(self, img_tensor, path, dataset, bboxes, size):
        self.path = path
        self.basename = os.path.basename(os.path.splitext(path)[0])
        self.image = img_tensor[0]  # Keep in tensor form, assume input is [N, C, H, W] and we take first image
        self.bboxes = torch.tensor(bboxes, dtype=torch.int)
        self.size = size
        self.transform = T.ToTensor()
        self.h, self.w = self.image.shape[-2:]

    def get_subwindow(self, idx):
        # Get the bounding box and crop the image
        xmin, ymin, xmax, ymax = self.bboxes[idx].tolist()
        crop_image = self.image[..., ymin:ymax, xmin:xmax]  # Crop: [C, H, W]
        # print('after crop: ', torch.unique(crop_image))
        
        # Check if rotation is needed
        rotate = (crop_image.shape[-2] > crop_image.shape[-1]) != (self.size[0] > self.size[1])

        # Rotate if necessary
        if rotate:
            # crop_image = kornia.geometry.rotate(crop_image.unsqueeze(0), torch.tensor([-90.0], device=crop_image.device)).squeeze(0)
            crop_image = torch.rot90(crop_image, k=-1, dims=(-2, -1))

        # print('after rot: ', torch.unique(crop_image))
        # Resize and handle padding
        crop_h, crop_w = crop_image.shape[-2:]
        if crop_h > self.size[0] or crop_w > self.size[1]:
            crop_image, unpadded = letterbox_torch(crop_image, self.size)
        else:
            unpadded = (crop_h, crop_w)

        # print('after letterbox: ', torch.unique(crop_image))

        # Pre-allocate output image with padding
        img_in = torch.full((crop_image.shape[0], *self.size), 0.56, dtype=crop_image.dtype, device=crop_image.device)
        img_in[..., :crop_image.shape[-2], :crop_image.shape[-1]] = crop_image
        # save_torch_img(img_in, f'checks-images/{self.basename}-{idx}-torch.png')

        # Metadata tensor
        metadata = {
            'translate': torch.tensor([[xmin, ymin, xmin, ymin]], dtype=torch.float32, device=crop_image.device),
            'rotation': torch.tensor([[rotate]], dtype=torch.bool, device=crop_image.device),
            'resize': torch.tensor([[crop_h > self.size[0] or crop_w > self.size[1]]], dtype=torch.bool, device=crop_image.device),
            'crop_shape': torch.tensor([[crop_h, crop_w]], dtype=torch.float32, device=crop_image.device),
            'unpadded_shape': torch.tensor([unpadded], dtype=torch.int64, device=crop_image.device),
            'roi_shape': torch.tensor([[ymax - ymin, xmax - xmin]], dtype=torch.float32, device=crop_image.device),
        }
        img_in = img_in.unsqueeze(0)
        # print('after all: ', torch.unique(img_in))


        return img_in, metadata


    def get_batch(self):
        batch = []
        metadata = defaultdict(list)
        for i in range(self.bboxes.shape[0]):
            # if i != 3:
            #     continue
            subwindow, meta = self.get_subwindow(i)
            for k, v in meta.items():
                metadata[k].append(v)
            # print(f'sample {i}: {torch.unique(subwindow)}')
            # print()
            batch.append(subwindow)

        batch = torch.cat(batch, dim=0)  # Concatenate all subwindow tensors
        # print('whole batch: ', torch.unique(batch))
        metadata = {k: torch.cat(v) for k, v in metadata.items()}
        metadata['bbox'] = self.bboxes
        return batch, metadata
    