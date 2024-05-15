FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx wget libglib2.0-0
RUN pip install seaborn thop tensorboard opencv-python gdown Pillow scikit-image filterpy

WORKDIR /tinyROI