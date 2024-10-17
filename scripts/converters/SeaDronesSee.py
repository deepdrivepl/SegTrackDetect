import os
import json

from glob import glob
from tqdm import tqdm
from shutil import copyfile, rmtree

from PIL import Image

import pandas as pd


class SeaDronesSeeDataset():
    
    def __init__(self, rm_old=True):
        
        self.root_dir = '/SegTrackDetect/data/SeaDronesSee'
        self.splits = ['train', 'val', 'test']
        self.class_mapping = {1:0, 2:1, 3:2, 6:3}
        self.out_img_dir = f"{self.root_dir}/images"
        self.rm_old = rm_old


    def convert_dataset(self):

        self.orig_annos = []
        print(f"Converting annnotations to coco format")
        for split_name in self.splits:

            anno_path = f"{self.root_dir}/instances_{split_name}_objects_in_water.json"
            self.orig_annos.append(anno_path)
            with open(anno_path) as f:
                orig_annos = json.load(f)
            seqs = list(set([x['id'] for x in orig_annos['videos']])) 
            seqs = seqs+[20] if split_name=='train' else seqs
            categories = [{
                'id': self.class_mapping[x['id']], 'name': x['name'], 'supercategory': x['supercategory']} 
                for x in orig_annos['categories']]

            split_paths = glob(f"{self.root_dir}/images/{split_name}/*.jpg")
            print(f'Converting {split_name}, number of images: {len(split_paths)}')
            coco_annos = self.annotations2coco(orig_annos, categories)
            meta = self.get_metadata(orig_annos, coco_annos)
            

            images = []
            obj_id = 0

            for img_id in tqdm(range(len(meta))):

                img_meta = meta.iloc[img_id]
                old_image_path = f"{self.root_dir}/images/{split_name}/{img_meta['file_name']}"
                out_image_path = f"{self.out_img_dir}/{img_meta['sequence']}/{img_meta['file_name']}"

                os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
                copyfile(old_image_path, out_image_path)

                W,H = Image.open(out_image_path).size
                images.append({
                    "id": int(img_meta['id']), 
                    "file_name": os.path.abspath(out_image_path), 
                    "height": H, 
                    "width": W
                })

            annos = []
            if orig_annos['annotations'] is not None:
                annos = [{'id': x['id'], 'image_id': x['image_id'], 'bbox': x['bbox'], 'area':x['area'], 
                         'category_id': self.class_mapping[x['category_id']], "segmentation": [], "iscrowd": 0
                          } 
                          for x in orig_annos['annotations']]

            annotations = {"images": images, "categories": categories, "annotations": annos}
            out_path = f'{self.root_dir}/{split_name}.json'
            print(f'Saving {out_path}')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4)


        if self.rm_old:
            rmtree(f'{self.root_dir}/images/train')
            rmtree(f'{self.root_dir}/images/test')
            rmtree(f'{self.root_dir}/images/val')
            for orig_anno in self.orig_annos:
                os.remove(orig_anno)
        

    def annotations2coco(self, orig_annos, categories):
        images = [{'id': x['id'], 'file_name': x['file_name'].replace('.png', '.jpg'), 'height': x['height'], 'width':x['width']} 
                   for x in orig_annos['images']]

        annos = []
        if orig_annos['annotations'] is not None:
            annos = [{'id': x['id'], 'image_id': x['image_id'], 'bbox': x['bbox'], 'area':x['area'], 
                         'category_id': self.class_mapping[x['category_id']], "segmentation": [], "iscrowd": 0
                          } 
                          for x in orig_annos['annotations']]
        annotations = {"images": images, "categories": categories, "annotations": annos}
        return annotations


    def get_metadata(self, orig_annos, coco_annos):
        coco_metadata = pd.DataFrame(coco_annos['images'])
        orig_metadata = pd.DataFrame(orig_annos['images'])
        merged = pd.merge(coco_metadata, orig_metadata, on="id")
       
        coco_metadata['sequence'] = merged.apply(lambda x: x.video_id, axis=1)
        coco_metadata['frame_id'] = merged['frame_index']
        coco_metadata['sequence'] = coco_metadata['sequence'].astype(str)

        unique_sequences = coco_metadata.sequence.unique()

        dfs = []
        for unique_sequence in unique_sequences:
            seq_df = coco_metadata[coco_metadata.sequence == unique_sequence]
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
        coco_metadata = pd.concat(dfs, ignore_index=True)

        return coco_metadata


if __name__=='__main__':
    ds = SeaDronesSeeDataset()
    ds.convert_dataset()