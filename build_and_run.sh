docker build -t tinyroi .
docker run --gpus all -it -v $(pwd):/tinyROI tinyroi 