FROM nvcr.io/nvidia/pytorch:21.02-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof

# Create working directory
RUN mkdir -p /product-pricing
WORKDIR /product-pricing
ENV HOME=/product-pricing

# yolov5
RUN git clone https://github.com/ultralytics/yolov5 && \
    cd yolov5 && \
    git checkout 06372b1465f5a58463bf8c32bdf65fc679c17ebf && \
    pip install -r requirements.txt && \
    cd ..

# Install PP
RUN python -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple && \
    git clone https://github.com/PaddlePaddle/PaddleOCR && \
    cd PaddleOCR && \
    pip install -r requirements.txt && \
    cd ..

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Install segmentation requirements
RUN git clone git@github.com:tyomj/prc-seg.git && \
    cd prc-seg && \
    pip install . && \
    cd ..

# Copy files to working directory
COPY ./prod_pricing /product-pricing/prod_pricing
COPY ./requirements.txt /product-pricing/requirements.txt

# Install ResNest
RUN python -m pip install -r requirements.txt
