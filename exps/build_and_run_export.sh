docker build -t trt -f Dockerfile-export .
docker run --ipc=host --shm-size=20g --gpus all -it -v $(pwd):/tinyROI -v $(pwd)/onnx:/onnx trt 