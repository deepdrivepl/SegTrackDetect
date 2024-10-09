"""
Configuration dictionary for the SORT (Simple Online and Realtime Tracking) tracker.

This dictionary contains the necessary parameters for initializing the SORT tracker, 
including settings for tracking behavior and performance.

Keys:
- module_name: The name of the module where the SORT tracker class is defined.
- class_name: The name of the SORT tracker class to be instantiated.
- args: A dictionary of arguments passed to the SORT tracker constructor, which includes:
  - max_age: The maximum age (in frames) before a track is considered lost.
  - min_hits: The minimum number of hits required to establish a track.
  - iou_threshold: The Intersection over Union (IoU) threshold for determining 
    whether detections match existing tracks.
  - min_confidence: The minimum confidence score required to consider a detection valid.
- frame_delay: The number of frames to delay before using predictions to select detection windows.
"""


sort = dict(
    module_name = 'rois.predictor.SORT', 
    class_name = 'Sort',
    args = dict(
        max_age = 10,
        min_hits = 1,
        iou_threshold = 0.3,
        min_confidence = 0.3
    ),
    frame_delay = 3,
)
