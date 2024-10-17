import os
import json

from glob import glob
from tqdm import tqdm
from shutil import copyfile, rmtree
from PIL import Image

import pandas as pd
import scipy.io


class DroneCrowdDataset():
    
    def __init__(self, rm_old=True):

        self.root_dir = '/SegTrackDetect/data/DroneCrowd'
        self.categories = [{"id": 0, "name": "human", "supercategory": "human"}]
        self.splits = ['train', 'val', 'test']
        self.ann_dir = f"{self.root_dir}/annotations"
        self.out_img_dir = f"{self.root_dir}/images"
        self.rm_old = rm_old


    def convert_dataset(self):

        print(f"Converting annnotations to coco format")
        for split_name in self.splits:

            split_paths = sorted(glob(f"{self.root_dir}/{split_name}_data/images/*.jpg"))
            print(f'Converting {split_name}, number of images: {len(split_paths)}')

            images, annos = [], []
            obj_id = 0

            for img_id, split_path in tqdm(enumerate(split_paths)):
                seq_id = os.path.basename(split_path)[3:6]
                frame_id = os.path.basename(split_path)[6:].replace('.jpg', '')
                ann_path = os.path.join(self.ann_dir, f'00{seq_id}.mat')

                out_image_path = f"{self.out_img_dir}/{seq_id}/{os.path.basename(split_path)}"
                os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
                copyfile(split_path, out_image_path)

                W,H = Image.open(out_image_path).size
                images.append({
                    "id": img_id, 
                    "file_name": os.path.abspath(out_image_path), 
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
            out_path = f'{self.root_dir}/{split_name}.json'
            print(f'Saving {out_path}')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4)


        if self.rm_old:
            rmtree(f'{self.root_dir}/train_data')
            rmtree(f'{self.root_dir}/test_data')
            rmtree(f'{self.root_dir}/val_data')
            rmtree(f'{self.root_dir}/annotations')
           
    
if __name__=='__main__':
    ds = DroneCrowdDataset()
    ds.convert_dataset()