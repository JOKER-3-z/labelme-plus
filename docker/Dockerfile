FROM ubuntu:bionic
LABEL maintainer "Kentaro Wada <www.kentaro.wada@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

ENV LANG C.UTF-8
RUN \
  apt-get update -qq && \
  apt-get install -qq -y \
    git \
    python3 \
    python3-pip \
    python3-matplotlib \
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

#RUN python3 -m pip install -v git+https://code.raintai.com:8090/ATA/labelPose.git@pose
COPY . /tmp/
RUN cd /tmp/ && pip install .




WORKDIR /root/workdir
ENTRYPOINT [ "labelme" ]
