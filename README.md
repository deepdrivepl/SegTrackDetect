# SegTrackDetect

SegTrackDetect is a modular framework designed for accurate small object detection using a combination of segmentation and tracking techniques. It performs detection at full resolution within selected Regions of Interest (ROIs), providing a highly efficient solution for scenarios where detecting tiny objects with precision is critical. The framework's modularity empowers users to easily customize key components, including the ROI Estimation Module, ROI Prediction Module, and Object Detector. See the following sections for more details on how the framework works and its key components:
- [SegTrackDetect Architecural Design](#architecture)
- [ROI Fusion Module](#place-2)
- [Object Detection Module]

# Architecture

## ROI Fusion Module
### ROI Prediction with Object Trackers
### ROI Estimation with Segmentation
## Object Deetection
## Detection Aggregation and Filtering




# Depencencies
We provide a Dockerfile that manages all the dependencies for you. To download all the trained models described in the model zoo and build a Docker image, simply run:
```console
./build_and_run.sh
```

# Models

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

## Metrics

# Datasets
## Mapillary Traffic Sign Dataset
## ZebraFish
## DroneCrowd
## SeaDronesSee


# Examples

# Licence

# Acknowledgements
### Header Name
