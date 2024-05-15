docker build -t tinyroi .
docker run --ipc=host --gpus all -it -v $(pwd):/tinyROI tinyroi 