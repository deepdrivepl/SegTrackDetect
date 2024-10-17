import os
import json

from collections import defaultdict
from shutil import copyfile, rmtree
from glob import glob
from tqdm import tqdm

from PIL import Image

import pandas as pd
import scipy.io




class ZebraFishDataset():

    def __init__(self, rm_old=True):

        self.root_dir = '/SegTrackDetect/data/3DZeF20'
        self.categories = [{"id": 0, "name": "fish", "supercategory": "fish"}]
        self.annotations_orig = self.load_orig_annos()
        self.seqs2splits = {
            "ZebraFish-01": "train",
            "ZebraFish-02": "train",
            "ZebraFish-03": "val",
            "ZebraFish-04": "val",
            "ZebraFish-05": "test",
            "ZebraFish-06": "test",
            "ZebraFish-07": "test",
            "ZebraFish-08": "test"
        }
        
        self.out_img_dir = f"{self.root_dir}/images"
        self.images = glob(f"{self.root_dir}/**/*.jpg", recursive=True)
        self.images = sorted([x for x in self.images if self.out_img_dir not in x])
        self.rm_old = rm_old



    def convert_dataset(self):

        self.split_images = defaultdict(list)
        
        print(f"Copying images to {self.out_img_dir}")
        for orig_image_path in tqdm(self.images):

            seq, view, fname = orig_image_path.split(os.sep)[-3:]
            out_image_path = f"{self.out_img_dir}/{seq}-{view}/{fname}"

            os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
            copyfile(orig_image_path, out_image_path)

            self.split_images[self.seqs2splits[seq]].append(
                (orig_image_path, out_image_path)
            )

        print(f"Converting annnotations to coco format")
        for split_name, split_paths in self.split_images.items():
            print(f'Converting {split_name}, number of images: {len(split_paths)}')
            
            images, annos = [], []
            obj_id = 0

            for img_id, (orig_path, new_path) in tqdm(enumerate(split_paths)):

                with Image.open(new_path) as im:
                    W,H = im.size
                
                images.append({
                    "id": img_id, "file_name": os.path.abspath(new_path), "height": H, "width": W
                })


                img_annos = self.annotations_orig[self.annotations_orig.fname == orig_path]
                for i in range(len(img_annos)):
                    obj = img_annos.iloc[i]
                    xmin,ymin,w,h = obj['left'], obj['top'], obj['width'], obj['height']
                    annos.append({
                        "id": obj_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "bbox": [int(xmin), int(ymin), int(w), int(h)],
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
            rmtree(f'{self.root_dir}/train')
            rmtree(f'{self.root_dir}/test')


    def load_orig_annos(self):
        names = ["frame", "id", "3d_x", "3d_y", "3d_z", "camT_x", "camT_y", "camT_left", 
        "camT_top", "camT_width", "camT_height", "camT_occlusion", "camF_x", "camF_y", 
        "camF_left", "camF_top", "camF_width", "camF_height", "camF_occlusion"
        ]

        gt_paths = glob(os.path.join(self.root_dir, "**", "gt", "gt.txt"), recursive=True)

        to_concat = []
        for gt_path in gt_paths:
            img_F_dir = os.path.dirname(gt_path).replace("gt", "imgF")
            img_T_dir = os.path.dirname(gt_path).replace("gt", "imgT")

            gt = pd.read_csv(gt_path, names=names)

            gt_F = gt[["frame", "id"]+[x for x in gt.columns if x.startswith("camF")]]
            gt_F = gt_F.rename(columns={k:k.replace("camF_","") for k in gt_F})
            gt_F["fname"] = gt_F.apply(lambda x: os.path.abspath(os.path.join(img_F_dir, f"{x.frame:06d}.jpg")), axis=1)


            gt_T = gt[["frame", "id"]+[x for x in gt.columns if x.startswith("camT")]]
            gt_T = gt_T.rename(columns={k:k.replace("camT_","") for k in gt_T})
            gt_T["fname"] = gt_T.apply(lambda x: os.path.abspath(os.path.join(img_T_dir, f"{x.frame:06d}.jpg")), axis=1)

            to_concat.append(gt_F)
            to_concat.append(gt_T)

        if len(to_concat) > 0:
            gt = pd.concat(to_concat, ignore_index=True)
            gt = gt.sort_values(by=['fname', 'frame'])
        else:
            gt = pd.DataFrame(columns=names)

        return gt


if __name__=='__main__':
    ds = ZebraFishDataset()
    ds.convert_dataset()