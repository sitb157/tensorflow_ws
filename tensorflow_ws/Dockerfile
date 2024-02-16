#FROM tensorflow/tensorflow:nightly
FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3

# Setup User
ARG USER_NAME
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd ${USER_NAME} --gid ${USER_ID}\
    && useradd -l -m ${USER_NAME} -u ${USER_ID} -g ${USER_ID} -s /bin/bash

USER root

# Run update
RUN apt-get update 

# Install pip
RUN apt-get update && apt-get install -y \
    python3-pip 

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install OpenVINO
WORKDIR /root
RUN apt-get install -y git
RUN git clone https://github.com/openvinotoolkit/openvino.git
WORKDIR /root/openvino
RUN git submodule update --init --recursive 
RUN ./install_build_dependencies.sh
RUN mkdir build
WORKDIR /root/openvino/build
RUN cmake -DCMAKE_BUILD_TYPE=Release \
-DENABLE_PYTHON=ON \
-DPYTHON_EXECUTABLE=`which python3.8` \
-DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
-DPYTHON_INCLUDE_DIR=/usr/include/python3.8 ..
RUN make --jobs=$(nproc --all)
RUN echo "export PYTHONPATH=$PYTHONPATH:~/openvino/bin/intel64/Release/python/" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/openvino/bin/intel64/Release/" >> ~/.bashrc

# Install openvino-dev
RUN pip3 install openvino-dev

# Install python lib
RUN pip3 install numpy \
    matplotlib
