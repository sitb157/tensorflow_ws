FROM tensorflow/tensorflow:2.15.0.post1-gpu

ARG USER_NAME
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd ${USER_NAME} --gid ${USER_ID}\
    && useradd -l -m ${USER_NAME} -u ${USER_ID} -g ${USER_ID} -s /bin/bash

USER root

RUN apt-get update 

# Install pip
RUN apt-get update && apt-get install -y \
    python3-pip 

# Install pykitti
RUN pip3 install pykitti \
    opencv-python 

# Install libgl for opencv-python
RUN apt-get install -y \
    libgl1-mesa-glx \
    eog


