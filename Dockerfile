FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt update && apt install -y git gcc zip htop screen libgl1-mesa-glx wget libglib2.0-0 g++
RUN pip install seaborn==0.13.2 cython==3.0.11 thop==0.1.1.post2209072238 \
	opencv-python==4.10.0.84 gdown==5.2.0 Pillow==10.2.0 scikit-image==0.24.0 \
	filterpy==1.4.5 lap==0.4.0 kornia==0.7.3
RUN pip install git+"https://github.com/Cufix/tinycocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

WORKDIR /SegTrackDetect