import os
import json

from glob import glob
from tqdm import tqdm
from collections import defaultdict

from PIL import Image

import pandas as pd
import scipy.io


class SeaDronesSeeDataset():
    
    def __init__(self, split='val', root_dir='/tinyROI/data/SeaDronesSee', flist=None, name=None):
        
        self.splits = ['train', 'val', 'test']
        self.root_dir = root_dir

        if flist is not None:
            print(f"Inferencing {len(flist)} images provided in a text file, with a name {name}")
            assert name is not None, f"Provide a name for the images listed in a text file"
            assert name not in self.splits, f"Provide a name different than {self.splits}"
            self.imgs = sorted(flist)
            self.split = name

            # TODO load all original annotations, merge, and filer by images
            anno_paths = glob(os.path.join(self.root_dir, f"instances_*_objects_in_water.json"))
            self.orig_annos = defaultdict(list)
            for anno_path in anno_paths:
                with open(anno_path) as f:
                    orig_annos = json.load(f)
                    self.orig_annos['videos']+=orig_annos['videos']
                    self.orig_annos['images']+=orig_annos['images']
                    if orig_annos['annotations'] is not None:
                        self.orig_annos['annotations']+=orig_annos['annotations']
                    self.orig_annos['categories'] = orig_annos['categories']

            fnames = [os.path.basename(x) for x in self.imgs]
            self.orig_annos['images'] = [x for x in self.orig_annos['images'] if x['file_name'].replace('.png', '.jpg') in fnames]
            img_ids = [x['id'] for x in self.orig_annos['images']]
            vid_ids = list(set([x['video_id'] for x in self.orig_annos['images']]))
            self.orig_annos['annotations'] = [x for x in self.orig_annos['annotations'] if x['image_id'] in img_ids]
            self.orig_annos['videos'] = [x for x in self.orig_annos['videos'] if x['id'] in vid_ids]
        else:
            print(f"Inferencing {split} split images")
            assert split in self.splits
            self.split = split
            self.img_dir = os.path.join(self.root_dir, 'images', self.split)
            self.imgs = sorted(glob(f'{self.img_dir}/*.jpg'))

            anno_path = os.path.join(self.root_dir, f"instances_{self.split}_objects_in_water.json")
            with open(anno_path) as f:
                self.orig_annos = json.load(f)

        self.seqs = list(set([x['id'] for x in self.orig_annos['videos']])) 
        if self.split == 'train' and 20 not in self.seqs:
            self.seqs+=[20] # fix
        print(f'Found {len(self.imgs)} images, {len(self.seqs)} sequences in {self.split}')
        print(f'Sequences: {self.seqs}')
        

        self.coco_json = os.path.join(self.root_dir, f'{self.split}.json')
        self.class_mapping = {1:0, 2:1, 3:2, 6:3}
        self.categories = [{'id': self.class_mapping[x['id']], 'name': x['name'], 'supercategory': x['supercategory']} for x in self.orig_annos['categories']]
        if not os.path.isfile(self.coco_json):
            print('Converting annotations to coco json.')
            self.annotations = self.annotations2coco()
            print(f'Saving {self.coco_json}')
            with open(self.coco_json, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, ensure_ascii=False, indent=4)
        else:
            print(f'Loading {self.coco_json}')
            with open(self.coco_json) as f:
                self.annotations = json.load(f)

        self.classes =  ["swimmer", "swimmer with life jacket", "boat", "life jacket"]
        self.colors = [(203, 179, 11), (222, 135, 191), (40, 195, 132), (75, 140, 112)]
        
        self.imgs_metadata = self.get_metadata()
        self.validate_sequences()
        self.seqs = self.imgs_metadata.sequence.unique().tolist()
        print(f'New split: {len(self.imgs)} images, {len(self.seqs)} sequences in {self.split}')

        self.seq2imgs = {seq:[] for seq in self.seqs}
        for img_path in self.imgs:
            meta = self.get_image_metadata(img_path)
            if meta is None: # remove, or better keep meta without annotations?
                print(f'Image not found in annotations" {os.path.basename(img_path)}, skipping')
                self.imgs.pop(self.imgs.index(img_path))
                continue
            self.seq2imgs[meta['sequence']].append(img_path)
        print({k:f'len(seq): {len(v)}' for k,v in self.seq2imgs.items()})

        
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
        coco_metadata = pd.DataFrame(self.annotations['images'])
        orig_metadata = pd.DataFrame(self.orig_annos['images'])
        merged = pd.merge(coco_metadata, orig_metadata, on="id")
        
        coco_metadata['sequence'] = merged.apply(lambda x: x.video_id, axis=1)
        coco_metadata['frame_id'] = merged['frame_index']

        return coco_metadata


    def validate_sequences(self):
        unique_sequences = self.imgs_metadata.sequence.unique()

        dfs = []
        for unique_sequence in unique_sequences:
            seq_df = self.imgs_metadata[self.imgs_metadata.sequence == unique_sequence]
            seq_df = seq_df.sort_values(by='frame_id')
            seq_df = seq_df.reset_index()

            prev_frame, current_frame = None, None
            seq_id = 1
            for i in range(len(seq_df)):
                current_frame = seq_df.iloc[i].frame_id

                if prev_frame is None:
                    prev_frame = current_frame
                    continue

                if current_frame - prev_frame != 1:
                    current_name = f'{unique_sequence}_{seq_id:03d}'
                    seq_df.loc[i:,'sequence'] = current_name
                    seq_id+=1
                prev_frame = current_frame

            dfs.append(seq_df)
        self.imgs_metadata = pd.concat(dfs, ignore_index=True)




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