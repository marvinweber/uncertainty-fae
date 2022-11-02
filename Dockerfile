FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN mkdir /install

# Bash as default Shell
RUN ln -sf /bin/bash /bin/sh

# Container User
RUN mkdir /workspace && \
    useradd -u 1111 -d /workspace -M -s /bin/bash -p cds cds && \
    chown cds:cds /workspace && \
    chmod -R a+rwx /workspace

# Package Installation
RUN apt-get update && \
    apt-get install -y apt-utils curl wget git rsync software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Python Setup
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10-dev python3.10-distutils python3-pip libffi-dev build-essential && \
    apt-get install -y python3.10 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN python -m pip install \
        torch==1.12.* \
        torchvision \ 
        torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu113 && \
    python -m pip cache purge

ADD requirements.txt /install/requirements_fae_uncertainty.txt
RUN python -m pip install -r /install/requirements_fae_uncertainty.txt && \
    python -m pip cache purge

USER cds
ENV HOME /workspace

WORKDIR /app

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
