./scripts/download_models.sh
./scripts/download_3DZeF20.sh
./scripts/download_DroneCrowd.sh
./scripts/download_SeaDronesSee.sh
./scripts/download_MTSD.sh
git clone git@github.com:deepdrivepl/SORT.git rois/predictor/SORT

docker build -t tinyroi .
docker run --ipc=host --shm-size=20g --gpus all -it -v $(pwd):/tinyROI tinyroi 
