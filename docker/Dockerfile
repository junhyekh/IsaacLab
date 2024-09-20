ARG BASE_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04

FROM ${BASE_IMAGE} as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

ARG USE_CUDA=on
ARG USERNAME=user
ARG UID=1000
ARG GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    sudo \
    gnupg2 \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    g++ \
    gdb \
    python3-dev \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists*

# install vulkan
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.290-jammy.list https://packages.lunarg.com/vulkan/1.3.290/lunarg-vulkan-1.3.290-jammy.list
RUN apt update
RUN apt install -y vulkan-sdk
RUN apt install -y libxt-dev
# REQUIRED?
RUN apt install -y libxt6

COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo "useradd -m -s /bin/bash ${USERNAME}"
RUN [ $(getent group ${GID}) ] || groupadd -g ${GID} ${USERNAME} && \
    useradd -m -s /bin/bash ${USERNAME} -u ${UID} -g ${GID} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} && \
    chmod 440 /etc/sudoers.d/${USERNAME} && \
    usermod -u ${UID} ${USERNAME} && \
    usermod -g ${GID} ${USERNAME}

RUN usermod -aG video ${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Add ~/.local/bin to path.
ENV HOME="/home/${USERNAME}"
ENV PATH="${PATH}:${HOME}/.local/bin"

# Update python deps.
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade wheel setuptools Cython

RUN echo 'export PATH="$HOME/.local/bin:${PATH}"' >> "${HOME}/.bashrc"
# export `CUDA_HOME` env variable to .bashrc.
RUN printf 'export CUDA_HOME=/usr/local/cuda\nexport LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"\nexport PATH="${PATH}:${CUDA_HOME}/bin"' >> "${HOME}/.bashrc"

RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

CMD ["/bin/bash"]
