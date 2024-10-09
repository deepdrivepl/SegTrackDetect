import os
import itertools

import cv2
import numpy as np
import seaborn as sns

from PIL import Image, ImageFont, ImageDraw



def plot_mask(mask, image, color = [255, 144, 30], alpha=0.4):
    """Overlay a mask on an image with a specified color and transparency.

    Args:
        mask (ndarray): A binary mask of shape (H, W) where non-zero values indicate the mask area.
        image (ndarray): The original image to overlay the mask on, with shape (H, W, C).
        color (list): A list of three integers representing the RGB color for the mask overlay.
        alpha (float): The transparency factor for the mask overlay.

    Returns:
        ndarray: The masked image with the overlay applied.
    """
    mask = cv2.merge((mask, mask, mask))
    color = np.full(mask.shape, np.array(color))
    
    mask = np.where(mask>0, color, image).astype(np.uint8)
    masked_image = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0) 
    return masked_image



def plot_one_box(bbox, img, color, label=None, lw=2, draw_label=True):
    """Draw a bounding box on an image.

    Args:
        bbox (list): A list of four integers representing the bounding box in (xmin, ymin, xmax, ymax) format.
        img (ndarray): The image to draw the bounding box on.
        color (tuple): A tuple of three integers representing the RGB color of the bounding box.
        label (str): An optional label to draw above the bounding box.
        lw (int): The line width of the bounding box.
        draw_label (bool): Whether to draw the label above the bounding box.

    Returns:
        ndarray: The image with the bounding box drawn on it.
    """
    
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
    """Draw text on a numpy array frame.

    Args:
        frame_numpy (ndarray): The image as a numpy array.
        text (str): The text to be drawn.
        x (int): The x-coordinate of the text position.
        y (int): The y-coordinate of the text position.
        color (tuple): A tuple of three integers representing the RGB color of the text.
        font (str): The font file name to use for the text.
        frac (float): A fraction of the frame size used to scale the font size.

    Returns:
        ndarray: The image with the text drawn on it.
    """

    font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts', font)
    font = ImageFont.truetype(font_path, int(frac*np.sum(frame_numpy.shape[:2])))

    frame_PIL = Image.fromarray(frame_numpy)
    draw = ImageDraw.Draw(frame_PIL)
    draw.multiline_text((x, y), text, fill=color, stroke_fill=(0,0,0), stroke_width=3, font=font, spacing=0)

    return np.array(frame_PIL)



def make_vis(frame, estim_mask, pred_mask, detection_windows, detections, classes, colors, vis_conf_th=0.1, show_label=True):
    """Create a visualization by overlaying masks and bounding boxes on the frame.

    Args:
        frame (ndarray): The image frame to visualize.
        estim_mask (ndarray): The estimated mask to overlay on the frame.
        pred_mask (ndarray): The predicted mask to overlay on the frame.
        detection_windows (list): A list of bounding boxes representing detection windows.
        detections (ndarray): An array of detections with shape (N, 6) where each row contains
                              (xmin, ymin, xmax, ymax, confidence, class_id).
        classes (list): A list of class names corresponding to the class IDs.
        colors (list): A list of colors for each class.
        vis_conf_th (float): The confidence threshold for displaying detections.
        show_label (bool): Whether to display labels on the bounding boxes.

    Returns:
        ndarray: The visualized image with masks and bounding boxes.
    """
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