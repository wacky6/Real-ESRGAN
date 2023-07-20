#!/bin/bash

# Assuming a nvidia/cuda cudnn runtime image.
export DEBIAN_FRONTEND=noninteractive
apt update

apt install -y python3 python3-pip build-essential curl git libgl1-mesa-glx libglib2.0-0 ffmpeg

# Assuming a Ampere GPU.
pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip3 install tqdm opencv-python
pip3 install 'gfpgan>=0.2.1' 'facexlib>=0.2.0.3' 'basicsr>=1.3.3.11'
python3 ./setup.py build

# For torch 2, confirmed working with inference_video_esrgan_legacy, about 50% speedup
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118