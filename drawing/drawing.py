import os
import itertools

import cv2
import numpy as np
import seaborn as sns

from PIL import Image, ImageFont, ImageDraw



def plot_mask(mask, image, color = [255, 144, 30], alpha=0.4):
    mask = cv2.merge((mask, mask, mask))
    color = np.full(mask.shape, np.array(color))
    
    mask = np.where(mask>0, color, image).astype(np.uint8)
    masked_image = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0) 
    return masked_image


def plot_one_box(bbox, img, color, label=None, lw=2, draw_label=True):
    
    xmin,ymin,xmax,ymax = list(map(int, bbox))
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


def make_vis(frame, estim_mask, pred_mask, detection_windows, detections, classes, colors, vis_conf_th=0.1, show_label=True):

    if estim_mask is not None:
        frame = plot_mask(estim_mask, frame, alpha=0.6)

    if pred_mask is not None:
        frame = plot_mask(pred_mask, frame, color=[0, 128, 255], alpha=0.6)

    for detection_window in detection_windows:
        frame = plot_one_box(list(map(int, detection_window)), frame, color=(0,0,0), label='WINDOW', draw_label=show_label) 

    
    detections = detections[detections[:, -2] >= vis_conf_th]
    for det in detections.tolist():
        xmin,ymin,xmax,ymax,conf,cls = det
        frame = plot_one_box(
            list(map(int, [xmin,ymin,xmax,ymax])), 
            frame, 
            color=colors[int(cls)], 
            label=f'{classes[int(cls)]} {int(conf*100)}%' if show_label else ''
        )

    return frame