#!/bin/bash
set -e -u -o pipefail -o noglob
set -x

CUDA_VERSION=${CUDA_VERSION:-11.5}
CUDNN_VERSION=${CUDNN_VERSION:-8}

UBUNTU_RELEASE=$(lsb_release -rs) # 18.04
DISTRO=ubuntu${UBUNTU_RELEASE//\./} # ubuntu1804

sudo apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/7fa2af80.pub"
# For CUDA
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list
# For cuDNN and TensorRT
echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/${DISTRO}/x86_64 /" | sudo tee -a /etc/apt/sources.list.d/cuda.list
# Need the latest drivers, but the ones installed by cuda-drivers cause issues for 32-bit applications, such as Wine and Steam
sudo add-apt-repository --no-update -y ppa:graphics-drivers/ppa
sudo apt-get update

# Optionally, uncomment to remove any existing packages first
 sudo apt-get purge 'cuda-*' 'libcudnn*' 'libnvidia-compute-*' 'nvidia-driver-*' 'libcublas*' 'libnvinfer*' 'libnvparsers*' 'libnvonnx*' 'python-libnvinfer*' 'python3-libnvinfer*'

sudo tee /etc/apt/preferences.d/cuda-nvidia-driver-400 <<EOF
# Pin drivers from the CUDA repo to a lower priority than the default 500
Package: nvidia-driver-*
Pin: release l=NVIDIA CUDA
Pin-Priority: 400
# Disable installation of cuda-drivers
Package: cuda-drivers
Pin: release *
Pin-Priority: -1
EOF

sudo tee /etc/apt/preferences.d/cuda-version-pin <<EOF
# Pin packages using on CUDA to depend on CUDA ${CUDA_VERSION}
Package: libcudnn* libnccl* graphsurgeon-tf
Pin: version *+cuda${CUDA_VERSION}
Pin-Priority: 991
Package: cuda-* libcublas*
Pin: version ${CUDA_VERSION}*
Pin-Priority: 991
Package: libnvinfer* libnvparsers* libnvonnx* python-libnvinfer* python3-libnvinfer* uff-converter-tf
Pin: version ${TENSORRT_VERSION}.*+cuda${CUDA_VERSION}
Pin-Priority: 991
EOF

# Install the latest driver
sudo ubuntu-drivers install

# Installing the cuda-toolkit-x-x rather than the cuda-x-x meta-package,
# since the toolkit one does not add the unwanted cuda-drivers dependency.
sudo apt-get install -y cuda-toolkit-${CUDA_VERSION//\./-}
sudo apt-get install -y libcudnn${CUDNN_VERSION}-dev
