./scripts/download_models.sh
git clone git@github.com:deepdrivepl/SORT.git rois/predictor/SORT

docker build -t segtrackdetect .
docker run --ipc=host --shm-size=20g --gpus all -it -v $(pwd):/SegTrackDetect segtrackdetect 
