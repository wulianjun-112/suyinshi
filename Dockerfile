FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04
FROM python:3.8
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build bash tmux ssh
# RUN ln -sv /usr/bin/python3 /usr/bin/python
RUN ln -sv /home/lianjunwu/.conda/envs/eva/bin/python /usr/bin/python
# create a non-root user
# ARG USER_ID=1000
# RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER appuser
# WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/get-pip.py && \
	python3 get-pip.py --user -i https://pypi.tuna.tsinghua.edu.cn/simple && \
	rm get-pip.py
# RUN pip install --upgrade pip

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard cmake onnx 
RUN pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.13.1 torchvision==0.14.1 -f https://download.pytorch.org/whl/cu117/torch_stable.html

RUN pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple 'git+https://github.com/facebookresearch/fvcore'
COPY /EVA/ /EVA
# RUN chmod 777 -R /EVA 
WORKDIR /EVA/EVA-02/det
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --user -e .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  mmcv==2.0.0rc4 
# install detectron2
# RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
# WORKDIR /home/appuser/detectron2_repo

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

CMD [ "bash" ]