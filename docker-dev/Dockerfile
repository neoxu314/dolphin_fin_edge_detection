FROM bvlc/caffe:cpu

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
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip3 install --upgrade pip \
    && pip3 install jupyter \
    && python2 -m pip install ipykernel \
    && python2 -m ipykernel install --user \
    && python3 -m pip install ipykernel \
    && python3 -m ipykernel install --user \
    && pip install -U --pre pip setuptools wheel \
    && pip install -U --pre numpy scipy matplotlib scikit-learn scikit-image Pillow \
    && pip3 install -U --pre pip setuptools wheel \
    && pip3 install -U --pre numpy scipy matplotlib scikit-learn scikit-image Pillow \
    && adduser --uid $uid $group --disabled-password --gecos "" dev

#EXPOSE 8888
USER dev
WORKDIR /home/dev/project
