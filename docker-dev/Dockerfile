FROM bvlc/caffe:gpu

ARG uid=1001
ARG group=
ARG http=proxy=

ENV HTTP_PROXY=$http_proxy \
    PATH="/opt/ide/bin:$PATH"

# Install extra packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python-tk \
        python3-tk \
        nano \
        python3-setuptools \
        python3-pip \
        libhdf5-10 \
        libhdf5-serial-dev \
        libhdf5-dev \
        libhdf5-cpp-11 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip3 install --upgrade pip \
    && pip3 install jupyter \
    && pip2 install opencv-python \
    && python2 -m pip install ipykernel \
    && python2 -m ipykernel install --user \
    && python3 -m pip install ipykernel \
    && python3 -m ipykernel install --user \
    && pip2 install -U --pre pip setuptools wheel \
    && pip2 install -U --pre numpy scipy matplotlib scikit-learn scikit-image Pillow future tqdm Augmentor\
    && pip3 install -U --pre pip setuptools wheel \
    && pip3 install -U --pre numpy scipy matplotlib scikit-learn scikit-image Pillow future tqdm Augmentor\
    && pip2 install -U --pre tensorflow-gpu \
    && pip3 install -U --pre tensorflow-gpu \
    && pip2 install -U --pre keras \
    && pip3 install -U --pre keras
#    && cd ~ \
#    && git clone https://github.com/zeakey/hed.git \
#    && cd hed \
#    && rm -r caffe \
#    && git clone https://github.com/BVLC/caffe.git \
#    && chmod +x build.sh
#    && ./build.sh
#    && cd ~ \
#    && mkdir hed_origin \
#    && git clone https://github.com/s9xie/hed.git \
#    && adduser --uid $uid $group --disabled-password --gecos "" dev

#EXPOSE 8888
#USER dev
WORKDIR /home/dev/project
