import os
import json

from glob import glob
from tqdm import tqdm

from PIL import Image

import pandas as pd
import scipy.io


# warning val is a part of test set
class DroneCrowdDataset():
    
    def __init__(self, split='val', root_dir='data/DroneCrowd'): #'/tinyROI/data/DroneCrowd'):
        
        assert split in ['train', 'val', 'test']
        self.split = split
        
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, f'{split}_data', 'images')
        self.imgs = sorted(glob(f'{self.img_dir}/*.jpg'))
        self.seqs = sorted(list(set([os.path.basename(x)[3:6] for x in self.imgs])))
        print(f'Found {len(self.imgs)} images, {len(self.seqs)} sequences in {self.split}')
        print(f'Sequences: {self.seqs}')
        

        self.ann_dir = os.path.join(self.root_dir, 'annotations')
        self.coco_json = os.path.join(self.root_dir, f'{split}.json')
        self.categories = [{"id": 0, "name": "human", "supercategory": "human"}]
        if not os.path.isfile(self.coco_json):
            print('Converting annotations to coco json.')
            self.annotations = self.annotations2coco()
            print(f'Saving {self.coco_json}')
            with open(self.coco_json, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=4)
        else:
            with open(self.coco_json) as f:
                self.annotations = json.load(f)
        
        self.imgs_metadata = pd.DataFrame(self.annotations['images'])
        self.imgs_metadata['sequence'] = self.imgs_metadata.apply(lambda x: x.file_name[3:6], axis=1)
        self.imgs_metadata['frame_id'] = self.imgs_metadata.apply(lambda x: x.file_name[6:].replace('.jpg', ''), axis=1)

        self.seq2imgs = {seq:[] for seq in self.seqs}
        for img_path in self.imgs:
            seq = self.get_image_metadata(img_path)['sequence']
            self.seq2imgs[seq].append(img_path)
        print({k:len(v) for k,v in self.seq2imgs.items()})

        print(self.imgs_metadata)

           
        
        
    def annotations2coco(self):
        images, annos = [], []
        obj_id = 0
        
        for img_id, img_path in tqdm(enumerate(self.imgs)):
            seq_id = os.path.basename(img_path)[3:6]
            frame_id = os.path.basename(img_path)[6:].replace('.jpg', '')
            ann_path = os.path.join(self.ann_dir, f'00{seq_id}.mat')
            
            W,H = Image.open(img_path).size
            images.append({
                "id": img_id, 
                "file_name": os.path.basename(img_path), 
                "height": H, 
                "width": W
            })
            
            ann = scipy.io.loadmat(ann_path)
            df = pd.DataFrame(ann['anno'], columns=['frame_id', 'track_id', 'xmin', 'ymin', 'xmax', 'ymax'])
            df = df[df.frame_id == int(frame_id)]
            
            for i in range(len(df)):
                obj = df.iloc[i]
                w,h = obj.xmax-obj.xmin, obj.ymax-obj.ymin
                annos.append({
                    "id": obj_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [int(obj.xmin), int(obj.ymin), int(w), int(h)],
                    "segmentation": [],
                    "area": int(w*h),
                    "iscrowd": 0
                })
                obj_id+=1
        annotations = {"images": images, "categories": self.categories, "annotations": annos}
        return annotations


    def get_image_metadata(self, img_path):
        metadata = self.imgs_metadata[self.imgs_metadata.file_name == os.path.basename(img_path)]
        assert len(metadata) == 1
        return metadata.iloc[0].to_dict()


    def get_seq2imgs(self):
        return self.seq2imgs


    def get_sequences(self):
        return self.seqs


    def get_images(self):
        return self.imgs


    def get_gt(self):
        return self.annotations


    def get_gt_path(self):
        return self.coco_json