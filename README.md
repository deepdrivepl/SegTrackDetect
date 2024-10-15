# SegTrackDetect

SegTrackDetect is a modular framework designed for accurate small object detection using a combination of segmentation and tracking techniques. It performs detection within selected Regions of Interest (ROIs), providing a highly efficient solution for scenarios where detecting tiny objects with precision is critical. The framework's modularity empowers users to easily customize key components, including the ROI Estimation Module, ROI Prediction Module, and Object Detector. It also features our Overlapping Box Suppression Algorithm that efficiently combines detected objects from multiple sub-windows, filtering them to overcome the limitations of window-based detection methods. See the following sections for more details on the framework, its components, and customization options:
- [SegTrackDetect Architecural Design](#architecture)
- [ROI Fusion Module](#roi-fusion-module)
- [Object Detection Module](#object-detection)
- [Detection Aggregation and Filtering](#detection-aggregation-and-filtering).

To get started with the framework right away, head to the [Getting Started](#getting-started) section.



# Architecture
## ROI Fusion Module
### ROI Prediction with Object Trackers
### ROI Estimation with Segmentation
## Object Detection
## Detection Aggregation and Filtering



# Getting Started

## Depencencies
We provide a Dockerfile that manages all the dependencies for you. To download all the trained models described in the model zoo and build a Docker image, simply run:
```console
./build_and_run.sh
```

## Examples
All available models can be found in [Model ZOO](#model-zoo). Currently, we provide trained models for 4 detection tasks. 

## Customization
### Existing Models
### New Models
### New Datasets

## Metrics
We convert all datasets to coco format, and we provide a script for metrics computation.



# Model ZOO
## Region of Interest Prediction

|  Model | Objects of Interest |    Dataset   | Input size | Weights                                                                                    |
|:------:|:-------------------:|:------------:|:----------:|:------------------------------------------------------------------------------------------:|
| u2netp |    traffic signs    |     MTSD     |   576x576  | [here](https://github.com/koseq/tinyROI-track/releases/download/v0.1/u2netp_MTSD.pt)       |
|  unet  |         fish        |   ZebraFish  |   160x256  | [here](https://github.com/koseq/tinyROI-track/releases/download/v0.1/unetR18-ZebraFish.pt) |
|  unet  |        people       |  DroneCrowd  |   192x320  | [here]() |
|  unet  |    people, boats    | SeaDronesSee |   224x384  | [here]() |

## Object Detectors

|  Model        | Objects of Interest |    Dataset   | Input size | Weights                                                                                    |
|:------------: |:-------------------:|:------------:|:----------:|:------------------------------------------------------------------------------------------:|
| yolov4        |    traffic signs    |     MTSD     |   960x960  | [here](https://github.com/koseq/tinyROI-track/releases/download/v0.1/yolov4_MTSD.pt)       |
| yolov7 tiny   |         fish        |   ZebraFish  |   160x256  | [here](https://github.com/koseq/tinyROI-track/releases/download/v0.1/yolov7t-ZebraFish.pt) |
| yolov7 tiny   |        people       |  DroneCrowd  |   320x512  | [here]() |
| yolov7 tiny   |    people, boats    | SeaDronesSee |   320x512  | [here]() |



# Datasets
## Mapillary Traffic Sign Dataset
## ZebraFish
## DroneCrowd
## SeaDronesSee



# Licence



# Acknowledgements
