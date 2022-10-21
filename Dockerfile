FROM gitlab.lrz.de:5005/clinicaldatascience/radler:pytorch_v2.5

# Bash as default Shell
RUN ln -sf /bin/bash /bin/sh

RUN apt update && \
    apt install -y git && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /install/requirements_fae_uncertainty.txt
RUN pip install -r /install/requirements_fae_uncertainty.txt && \
    rm -r /workspace/.cache || true

RUN chmod go+rw /workspace
