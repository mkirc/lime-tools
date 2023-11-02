FROM ubuntu:16.04

RUN apt-get update \
        && apt-get install -y \
            build-essential \
            libgsl-dev \
            libcfitsio-dev \
            libatlas-base-dev \
            libncurses-dev \
            libqhull-dev \
            libqhull7 \
            gcc-6-base \
            libhdf5-dev \
            python-dev \
        && mkdir /lime /lime-tools /model

COPY configure /lime-tools
COPY lime-run /lime-tools

ENV PATH=/lime-tools:/lime:${PATH}
