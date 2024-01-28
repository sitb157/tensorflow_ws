FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

ARG USER_NAME
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd ${USER_NAME} --gid ${USER_ID}\
    && useradd -l -m ${USER_NAME} -u ${USER_ID} -g ${USER_ID} -s /bin/bash

USER root

# Run update
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update 

# Install pip
RUN apt-get update && apt-get install -y \
    python3-pip 

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install pykitti
RUN pip3 install pykitti \
    opencv-python 

# Install libgl for opencv-python
RUN apt-get install -y \
    libgl1-mesa-glx \
    eog
