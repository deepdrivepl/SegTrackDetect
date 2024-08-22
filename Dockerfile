FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt update && apt install -y git gcc zip htop screen libgl1-mesa-glx wget libglib2.0-0 g++
RUN pip install seaborn cython thop tensorboard opencv-python gdown Pillow scikit-image filterpy lap
RUN pip install git+"https://github.com/Cufix/tinycocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

WORKDIR /tinyROI