# SegTrackDetect

SegTrackDetect is a modular framework designed for accurate small object detection using a combination of segmentation and tracking techniques. It performs detection within selected Regions of Interest (ROIs), providing a highly efficient solution for scenarios where detecting tiny objects with precision is critical. The framework's modularity empowers users to easily customize key components, including the ROI Estimation Module, ROI Prediction Module, and Object Detector. It also features our Overlapping Box Suppression Algorithm that efficiently combines detected objects from multiple sub-windows, filtering them to overcome the limitations of window-based detection methods. 

![example](images/MTSD-example.png)

See the following sections for more details on the framework, its components, and customization options:
- [SegTrackDetect Architecural Design](#architecture)
- [ROI Fusion Module](#roi-fusion-module)
- [Object Detection Module](#object-detection)
- [Detection Aggregation and Filtering](#detection-aggregation-and-filtering).

To get started with the framework right away, head to the [Getting Started](#getting-started) section.



# Getting Started

## Depencencies

We provide a Dockerfile that handles all the dependencies for you. 
Simply install the [Docker Engine](https://docs.docker.com/engine/install/) and, if you plan to run detection on a GPU, the [NVIDIA Container Toolkit](https://docs.docker.com/engine/install/).

To download all the trained models described in [Model ZOO](#model-zoo) and build a Docker image, simply run:
```bash
./build_and_run.sh
```
We currently support four [datasets](#datasets), and we provide scripts that downloads the datasets and converts them into supported format.
To download and convert all of them, run:
```bash
./download_and_convert.sh
```
You can also download selected datasets by running corresponding scripts in the [`scripts`](scripts/) directory.

## Examples

SegTrackDetect framework supports tiny object detection on consecutive frames (video detection), as well as detection on independent windows.

To run detection on video data using one of the supported datasets, e.g. `SeaDronesSee`:
```bash
python inference_vid.py \
--roi_model 'SDS_large' --det_model 'SDS' --tracker 'sort' \
--ds 'SeaDronesSee' --split 'val' \
--bbox_type 'sorted' --allow_resize --obs_iou_th 0.1 \
--out_dir 'results/SDS/val' --debug
```
To run the detection on independent windows, e.g. `MTSD`, use:
```bash
python inference_img.py \
--roi_model 'MTSD' --det_model 'MTSD' \
--ds 'MTSD' --split 'val' \
--bbox_type 'sorted' --allow_resize --obs_iou_th 0.7 \
--out_dir 'results/MTSD/val' --debug
```
| Argument          | Type      | Description                                                                                                                                 |
|:-------------------:|-----------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `--roi_model`     | `str`     | Specifies the ROI model to use (e.g., `SDS_large`). All available ROI models are defined [here](rois/estimator/configs/__init__.py)         |
| `--det_model`     | `str`     | Specifies the detection model to use (e.g., `SDS`). All available detectors are defined [here](detector/configs/__init__.py)                |
| `--tracker`       | `str`     | Specifies the tracker to use (e.g., `sort`). All available trackers are defoned [here](rois/predictor/configs/__init__.py)                  |
| `--ds`            | `str`     | Dataset to use for inference (e.g., `SeaDronesSee`). Available [datasets](datasets/__init__.py)                                             |
| `--split`         | `str`     | Data split to use (e.g., `val` for validation). If present, the script will save the detections using the coco image ids used in `val.json` |
| `--flist`         | `str`     | An alternative version of providing an image list, path to the file with absolute paths to images.                                          |
| `--name`          | `str`     | A name for provided `flist`, coco annotations `name.json` will be generated and saved in the dataset root directory                         |
| `--bbox_type`     | `str`     | Type of the detection window filtering algorithm (`all` - no filtering, `naive`, `sorted`).                                                 |
| `--allow_resize`  | `flag`    | Enables resizing of cropped detection windows. Siling window within large ROIs will be used otherwise.                                      |
| `--obs_iou_th`    | `float`   | Sets the IoU threshold for Overlapping Box Suppresion (default is 0.7).                                                                     |
| `--cpu`           | `flag`    | Use `cpu` for computations, if not set use `cuda`                                                                                           |
| `--out_dir`       | `str`     | Directory to save output results (e.g., `results/SDS/val`).                                                                                 |
| `--debug`         | `flag`    | Enables saving visualisation in `out_dir`                                                                                                   |
| `--vis_conf_th`   | `float`   | Confidence threshold for the detections in visualisation, default 0.3.                                                                      |


All available models can be found in [Model ZOO](#model-zoo). Currently, we provide trained models for 4 detection tasks. 

## Customization
### Existing Models
### New Models
### New Datasets

## Metrics
We convert all datasets to coco format, and we provide a script for metrics computation.



# Architecture
![architecture](images/architecture.png)
## ROI Fusion Module
### ROI Prediction with Object Trackers
### ROI Estimation with Segmentation
## Object Detection
## Detection Aggregation and Filtering



# Model ZOO

All models we use, are in TorchScrpt format. 

## Region of Interest Estimation

|  Model | Objects of Interest |    Dataset   | Model name | Input size | Weights                                                                                    |
|:------:|:-------------------:|:------------:|:----------:|:----------:|:------------------------------------------------------------------------------------------:|
| u2netp |    traffic signs    |     MTSD     |  MTSD      |   576x576  | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/u2netp_MTSD.pt)       |
|  unet  |         fish        |   ZebraFish  |  ZeF20     | 160x256    | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/unetR18-ZebraFish.pt) |
|  unet  |        people       |  DroneCrowd  |  DC_tiny   |  96x160    | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/DroneCrowd-001-R18-96x160-best-loss.pt) |
|  unet  |        people       |  DroneCrowd  |  DC_small  |  192x320   | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/DroneCrowd-001-R18-192x320-best-loss.pt) |
|  unet  |        people       |  DroneCrowd  |  DC_medium |  384x640   | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/DroneCrowd-001-R18-384x640-best-loss.pt) |
|  unet  |    people, boats    | SeaDronesSee |  SDS_tiny  |   64x96    | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/SeaDronesSee-000-R18-64x96-best-loss.pt) |
|  unet  |    people, boats    | SeaDronesSee | SDS_small  |   128x192  | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/SeaDronesSee-000-R18-128x192-best-loss.pt) |
|  unet  |    people, boats    | SeaDronesSee | SDS_medium |   224x384  | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/SeaDronesSee-000-R18-224x384-best-loss.pt) |
|  unet  |    people, boats    | SeaDronesSee | SDS_large  |   448x768  | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/SeaDronesSee-000-R18-448x768-best-loss.pt) |

## Object Detectors

|  Model        | Objects of Interest |    Dataset   | Model name | Input size | Weights                                                                                    |
|:------------: |:-------------------:|:------------:|:----------:|:----------:|:------------------------------------------------------------------------------------------:|
| yolov4        |    traffic signs    |     MTSD     | MTSD       |   960x960  | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/yolov4_MTSD.pt)      |
| yolov7 tiny   |         fish        |   ZebraFish  |  ZeF20     |   160x256  | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/yolov7t-ZebraFish.pt) |
| yolov7 tiny   |        people       |  DroneCrowd  |  SDS       |   320x512  | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/006-SeaDronesSee-yolov7-tiny-512x512-crops-only-multiple-scales-300ep-best.torchscript.pt) |
| yolov7 tiny   |    people, boats    | SeaDronesSee |  DC        |   320x512  | [here](https://github.com/deepdrivepl/SegTrackDetect/releases/download/v0.1/004-DroneCrowd-yolov7-tiny-512x512-crops-only-multiple-scales-50ep-best.torchscript.pt) |



# Datasets
## Mapillary Traffic Sign Dataset
## ZebraFish
## DroneCrowd
## SeaDronesSee



# Licence



# Acknowledgements
