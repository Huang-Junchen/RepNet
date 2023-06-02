# CALL: docker build -t jasonhuang1999/repnet .
FROM jasonhuang1999/runtime:cuda10.2-cudnn7-python3.8-opencv-ffmpeg-ubuntu18.04
# FROM jasonhuang1999/runtime:cuda11.2-cudnn8-python3.8-opencv-ffmpeg-ubuntu18.04

LABEL Author="Junchen Huang"

WORKDIR /repnet
COPY . .
RUN pip install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com 

# CUDA 11.2
# RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
