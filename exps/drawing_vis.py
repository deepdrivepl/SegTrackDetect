import os
import itertools

import cv2
import numpy as np
import seaborn as sns

from PIL import Image, ImageFont, ImageDraw



def plot_mask(mask, image, alpha=0.4, color=[255, 144, 30]):
    mask = cv2.merge((mask, mask, mask))
    color = np.full(mask.shape, np.array(color))
    
    mask = np.where(mask>0, color, image).astype(np.uint8)
    masked_image = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0) 
    return masked_image


def plot_one_box(bbox, img, color, label=None, lw=2, draw_label=True):
    
    xmin,ymin,xmax,ymax = bbox
    img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, lw)

    if draw_label:
        ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        img = cv2.rectangle(img, (xmin, ymin - int(1.3 * text_height)), (xmin + text_width, ymin), color, -1)
        
        img = cv2.putText(
            img,
            text=label,
            org=(xmin, ymin - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.45,
            color=(0, 0, 0),
            lineType=cv2.LINE_AA,
    )
    return img


def draw_text(frame_numpy, text, x, y, color=(250,0,0), font='JetBrainsMono-ExtraBold.ttf', frac=0.01):
    
    font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts', font)
    font = ImageFont.truetype(font_path, int(frac*np.sum(frame_numpy.shape[:2])))

    frame_PIL = Image.fromarray(frame_numpy)
    draw = ImageDraw.Draw(frame_PIL)
    draw.multiline_text((x, y), text, fill=color, stroke_fill=(0,0,0), stroke_width=3, font=font, spacing=0)

    return np.array(frame_PIL)



def make_vis(frame, seg_mask, mot_mask, seg_bboxes, mot_bboxes, det_bboxes, detections, classes, colors, vis_conf_th=0.1, show_label=True):
    
    # print(frame.shape, mot_mask.shape, seg_mask.shape)
    # exit()
    frame_wins = frame.copy()  
    if seg_mask is not None:
        frame_wins = plot_mask(seg_mask, frame_wins, alpha=0.6)

    if mot_mask is not None:
        frame_wins = plot_mask(mot_mask, frame_wins, color=[0, 128, 255], alpha=0.6)


    for i,det_bbox in enumerate(det_bboxes.tolist()):
        # color = tuple(map(int, list(np.random.choice(range(256), size=3))))
        bbox = list(map(int, det_bbox))
        color = (0,0,0)
        # frame_wins = plot_one_box(bbox, frame_wins, color=color, draw_label=True, label=f'W{i:02d}', lw=2)
        frame_wins = plot_one_box(bbox, frame_wins, color=color, draw_label=False, label=f'W{i:02d}', lw=2)


    
    detections = detections[detections[:, -2] >= vis_conf_th]
    for det in detections.tolist():
        xmin,ymin,xmax,ymax,conf,cls = det
        frame = plot_one_box(
            list(map(int, [xmin,ymin,xmax,ymax])), 
            frame, 
            color=colors[int(cls)],  #(0,0,190)
            label=f'{classes[int(cls)]} {int(conf*100)}%' if show_label else '',
            draw_label=show_label,
            lw=2,
        )

    _xmin = min([max(int(x[0]-20),0) for x in det_bboxes])
    _ymin = min([max(int(x[1]-20),0) for x in det_bboxes])
    _xmax = max([min(int(x[2]+20), frame.shape[1]) for x in det_bboxes])
    _ymax = max([min(int(x[3]+20), frame.shape[0]) for x in det_bboxes])
    # print(_xmin, _ymin, _xmax, _ymax)
    # exit()
    frame = frame[_ymin:_ymax, _xmin:_xmax,:]

    # _xmin = min([max(int(x[0]-20),0) for x in det_bboxes])
    # _ymin = min([max(int(x[1]-20),0) for x in det_bboxes])
    # _xmax = max([min(int(x[2]+20), frame.shape[1]) for x in det_bboxes])
    # _ymax = max([min(int(x[3]+20), frame.shape[0]) for x in det_bboxes])
    # # print(_xmin, _ymin, _xmax, _ymax)
    # # exit()
    # frame_wins = frame_wins[_ymin:_ymax, _xmin:_xmax,:]
    # frame = frame[_ymin:_ymax, _xmin:_xmax,:]

    # stats = [
    #          "SEG  %02d" % (len(seg_bboxes)), 
    #          "MOT  %02d" % (len(mot_bboxes)), 
    #          "WIN  %02d" % (len(det_bboxes))
    #         ]
    # frame_wins = draw_text(frame_wins, "\n".join(stats), 20, 40, color=(255,255,255))




    
      
    # for seg_bbox in seg_bboxes:
    #     frame_wins = plot_one_box(list(map(int, seg_bbox)), frame_wins, color=(200,0,0), label='SEG')       
    # for mot_bbox in mot_bboxes:
    #     frame_wins = plot_one_box(list(map(int, mot_bbox)), frame_wins, color=(0,200,0), label='MOT') 
    

    
    
    
    return frame_wins, frame