# FROM nvidia/cuda:10.0-cudnn7-runtime
# FROM tensorflow/tensorflow
FROM ubuntu

RUN apt-get update
RUN apt-get install -y python3-dev python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==2.2 tensorflow-probability==0.10 tf-agents==0.5.0 gin-config==0.1.3
RUN pip3 install cloudml-hypertune cloudpickle==1.4.1 gym opencv-python imageio

RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip wget libboost-all-dev

RUN pip3 install vizdoom

WORKDIR /root
COPY ./ /root/

ENTRYPOINT ["python3", "train.py"]

# tensorflow 2.3.0 requires numpy<1.19.0,>=1.16.0, but you'll have numpy 1.19.2 which is incompatible.
# tf-agents 0.5.0 requires gin-config==0.1.3, but you'll have gin-config 0.3.0 which is incompatible.