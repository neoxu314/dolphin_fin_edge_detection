#!/bin/bash

# This script's absolute dir
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Assume this script is in the scripts directory.
projectDir=$( cd $SCRIPT_DIR/.. && pwd )
docker run -ti --name hed-dev \
    -h hed-dev \
    -e http_proxy=http://alb-cache.massey.ac.nz:8080 \
    -p 8888:8888 \
    -v "$projectDir":/home/dev/project \
    --restart unless-stopped \
    --privileged \
    hed-dev \
    /bin/bash

















#    jupyter notebook --no-browser --ip 0.0.0.0