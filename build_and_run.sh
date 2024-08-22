./scripts/download_models.sh
docker build -t tinyroi .
docker run --ipc=host --shm-size=20g --gpus all -it -v $(pwd):/tinyROI tinyroi 