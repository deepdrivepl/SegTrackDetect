import os
import itertools

import cv2
import numpy as np
import seaborn as sns

from PIL import Image, ImageFont, ImageDraw



def plot_mask(mask, image, random_color=False, alpha=0.4):
    mask = cv2.merge((mask, mask, mask))
    color = np.full(mask.shape, np.array([255, 144, 30]))
    
    mask = np.where(mask==255, color, image).astype(np.uint8) # mask > 0 ?
    image_new = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0) 
    return image_new


def plot_one_box(x, img, color=None, label=None, line_thickness=3, draw_label=True):
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 3) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label and draw_label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 8, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 8, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def get_colors():
    palette = itertools.cycle(sns.color_palette())

    colors = {}
    for i, (_, clr) in enumerate(zip(cat2label, palette)):
        colors[i] = tuple([int(x*255) for x in clr])
    return colors


def denormalize_numpy(img):
    img=img.copy()
    img[:,:,0] = img[:,:,0]*0.229+0.485
    img[:,:,1] = img[:,:,1]*0.224+0.456
    img[:,:,2] = img[:,:,2]*0.225+0.406
    return (img*255).astype(np.uint8)


def draw_text(frame_numpy, text, x, y, color=(250,0,0), font='JetBrainsMono-ExtraBold.ttf', frac=0.01):
    
    font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts', font)
    font = ImageFont.truetype(font_path, int(frac*np.sum(frame_numpy.shape[:2])))

    frame_PIL = Image.fromarray(frame_numpy)
    draw = ImageDraw.Draw(frame_PIL)
    draw.multiline_text((x, y), text, fill=color, stroke_fill=(0,0,0), stroke_width=3, font=font, spacing=0)

    return np.array(frame_PIL)



def make_vis(d0_fullres, roi_bboxes, trk_bboxes, det_bboxes, metadata, out_dir, frame_id):
    
    seq, view, fname = metadata['image_path'][0].split(os.sep)[-3:]
    out_path = os.path.join(out_dir, seq, view, fname)
    out_path_mask = out_path.replace(view, f'{view}-masks').replace('.jpg','.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_path_mask), exist_ok=True)
    
    frame = cv2.imread(metadata['image_path'][0])
    frame = plot_mask(d0_fullres, frame)
                
    for roi_bbox in roi_bboxes:
        frame = plot_one_box(list(map(int, roi_bbox)), frame, color=(200,0,0), label='ROI', line_thickness=4, draw_label=True)         
    for trk_bbox in trk_bboxes:
        frame = plot_one_box(list(map(int, trk_bbox)), frame, color=(0,200,0), label='KF', line_thickness=4, draw_label=True) 
    for bbox_det in det_bboxes:
        frame = plot_one_box(list(map(int, bbox_det)), frame, color=(0,0,180), label='CONCAT', line_thickness=4, draw_label=True)
        
    stats = ["FRAME  %03d" % (frame_id), 
             "", 
             "ROI  %02d" % (len(roi_bboxes)), 
             "MOT  %02d" % (len(trk_bboxes)), 
             "DET  %02d" % (len(det_bboxes))
            ]
    frame = draw_text(frame, "\n".join(stats), 20, 40, color=(255,255,255))
    cv2.imwrite(out_path, frame)
    cv2.imwrite(out_path_mask, d0_fullres)