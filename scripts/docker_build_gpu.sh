#!/bin/bash

# This script's absolute dir
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Assume this script is in the scripts directory.
projectDir=$( cd $SCRIPT_DIR/.. && pwd )

cd "$projectDir/docker-dev"

docker build -t hed-dev-gpu --network=host --build-arg=http_proxy=$http_proxy --build-arg=https_proxy=$https_proxy --build-arg=uid=$UID .