import os
import json

from glob import glob
from tqdm import tqdm

from PIL import Image

import pandas as pd
import scipy.io


class SeaDronesSeeDataset():
    
    def __init__(self, split='val', root_dir='/tinyROI/data/SeaDronesSee', flist=None, name=None):
        
        assert split in ['train', 'val', 'test']
        self.split = split
        
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'images', self.split)
        self.imgs = sorted(glob(f'{self.img_dir}/*.jpg'))

        anno_path = os.path.join(self.root_dir, f"instances_{self.split}_objects_in_water.json")
        with open(anno_path) as f:
            self.orig_annos = json.load(f)

        self.seqs = [os.path.splitext(os.path.basename(x['name:']))[0] for x in self.orig_annos['videos']] 
        if self.split == 'train' and 'DJI_0003_d3' not in self.seqs:
            self.seqs+=['DJI_0003_d3'] # fix
        print(f'Found {len(self.imgs)} images, {len(self.seqs)} sequences in {self.split}')
        print(f'Sequences: {self.seqs}')
        

        self.coco_json = os.path.join(self.root_dir, f'{split}.json')
        self.class_mapping = {1:0, 2:1, 3:2, 6:3}
        self.categories = [{'id': self.class_mapping[x['id']], 'name': x['name'], 'supercategory': x['supercategory']} for x in self.orig_annos['categories']]
        if not os.path.isfile(self.coco_json):
            print('Converting annotations to coco json.')
            self.annotations = self.annotations2coco()
            print(f'Saving {self.coco_json}')
            with open(self.coco_json, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=4)
        else:
            with open(self.coco_json) as f:
                self.annotations = json.load(f)
        
        self.imgs_metadata = self.get_metadata()
        print(self.imgs_metadata)

        self.seq2imgs = {seq:[] for seq in self.seqs}
        for img_path in self.imgs:
            meta = self.get_image_metadata(img_path)
            if meta is None:
                print(os.path.basename(img_path))
                self.imgs.pop(self.imgs.index(img_path))
                continue
            self.seq2imgs[meta['sequence']].append(img_path)
        print({k:len(v) for k,v in self.seq2imgs.items()})

        
    def annotations2coco(self):
        images = [{'id': x['id'], 'file_name': x['file_name'].replace('.png', '.jpg'), 'height': x['height'], 'width':x['width']} 
                   for x in self.orig_annos['images']]

        annos = []
        if self.orig_annos['annotations'] is not None:
            annos = [{'id': x['id'], 'image_id': x['image_id'], 'bbox': x['bbox'], 'area':x['area'], 
                         'category_id': self.class_mapping[x['category_id']], "segmentation": [], "iscrowd": 0
                          } 
                          for x in self.orig_annos['annotations']]
        annotations = {"images": images, "categories": self.categories, "annotations": annos}
        return annotations


    def get_metadata(self):
        id2vid = {x['id']: os.path.splitext(os.path.basename(x['name:']))[0] for x in self.orig_annos['videos']}
        id2vid[20] = 'DJI_0003_d3' # train set fix, DJI_0003_d3 absent in annotations['videos']
        coco_metadata = pd.DataFrame(self.annotations['images'])
        orig_metadata = pd.DataFrame(self.orig_annos['images'])

        merged = pd.merge(coco_metadata, orig_metadata, on="id")
        
        coco_metadata['sequence'] = merged.apply(lambda x: id2vid[x.video_id], axis=1)
        coco_metadata['frame_id'] = merged['frame_index']
        return coco_metadata


    def get_image_metadata(self, img_path):
        metadata = self.imgs_metadata[self.imgs_metadata.file_name == os.path.basename(img_path)]
        if metadata.empty:
            return None
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