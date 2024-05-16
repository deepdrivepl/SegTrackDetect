import os
import json

from glob import glob
from tqdm import tqdm

from PIL import Image

import pandas as pd
import scipy.io


# warning val is a part of test set
class ZebraFishDataset():
    
    def __init__(self, split='train', root_dir='/tinyROI/data/3DZeF20', flist=None, name=None):
        
        assert split in ['train', 'val', 'test']
        self.split = split
        
        self.root_dir = root_dir
        
        self.coco_json = os.path.join(self.root_dir, f'{split}.json')
        self.categories = [{"id": 0, "name": "fish", "supercategory": "fish"}]
        if not os.path.isfile(self.coco_json):
            self.imgs = sorted(glob(f'{self.root_dir}/{self.split}/**/img*/*.jpg', recursive=True))
            self.seqs = sorted(list(set(['_'.join([os.path.dirname(x).split(os.sep)[-2], os.path.dirname(x).split(os.sep)[-1]]) for x in self.imgs])))
            print(f'Found {len(self.imgs)} images, {len(self.seqs)} sequences in {self.split}')
            print(f'Sequences: {self.seqs}')
            
            self.orig_annos = self.load_orig_annos()
            print('Converting annotations to coco json.')
            self.annotations = self.annotations2coco()
            print(f'Saving {self.coco_json}')
            with open(self.coco_json, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=4)
        else:
            print(f'Loading {self.coco_json}')
            with open(self.coco_json) as f:
                self.annotations = json.load(f)

            self.imgs = sorted([x["file_name"] for x in self.annotations["images"]])
            self.seqs = sorted(list(set(['_'.join([os.path.dirname(x).split(os.sep)[-2], os.path.dirname(x).split(os.sep)[-1]]) for x in self.imgs])))
        
            print(f'Found {len(self.imgs)} images, {len(self.seqs)} sequences in {self.split}')
            print(f'Sequences: {self.seqs}')

        self.imgs_metadata = pd.DataFrame(self.annotations['images'])
        self.imgs_metadata['sequence'] = self.imgs_metadata.apply(lambda x: '_'.join([os.path.dirname(x.file_name).split(os.sep)[-2], os.path.dirname(x.file_name).split(os.sep)[-1]]), axis=1)
        self.imgs_metadata['frame_id'] = self.imgs_metadata.apply(lambda x: int(os.path.basename(x.file_name).replace('.jpg', '')), axis=1)

        self.seq2imgs = {seq:[] for seq in self.seqs}
        for img_path in self.imgs:
            seq = self.get_image_metadata(img_path)['sequence']
            self.seq2imgs[seq].append(img_path)
        print({k:len(v) for k,v in self.seq2imgs.items()})



    def load_orig_annos(self):
        names = ["frame", "id", "3d_x", "3d_y", "3d_z", "camT_x", "camT_y", "camT_left", 
        "camT_top", "camT_width", "camT_height", "camT_occlusion", "camF_x", "camF_y", 
        "camF_left", "camF_top", "camF_width", "camF_height", "camF_occlusion"
        ]

        gt_paths = glob(os.path.join(self.root_dir, self.split, "*", "gt", "gt.txt"))

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


    def annotations2coco(self):
        images, annos = [], []
        obj_id = 0
        
        for img_id, img_path in tqdm(enumerate(self.imgs)):

            with Image.open(img_path) as im:
                W,H = im.size
            images.append({
                "id": img_id, 
                "file_name": os.path.abspath(img_path), 
                "height": H, 
                "width": W
            })

            if self.orig_annos.empty:
                continue
        
            img_annos = self.orig_annos[self.orig_annos.fname==os.path.abspath(img_path)]
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
        return annotations


    def get_image_metadata(self, img_path):
        metadata = self.imgs_metadata[self.imgs_metadata.file_name == os.path.abspath(img_path)]
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