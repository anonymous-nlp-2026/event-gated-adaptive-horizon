#!/bin/sh

# Install Java 8 before running this script by either of the following methods.

# 1. Use docker
# $ apt-get update
# $ apt-get install -y openjdk-8-jdk
# 2. Use conda
# $ conda install -c conda-forge openjdk=8

pip3 install <MINERL_WHEEL_URL>
pip3 install cloudpickle==3.0.0
