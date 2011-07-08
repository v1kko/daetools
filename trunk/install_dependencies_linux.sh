#!/bin/sh

set -e

# Check the dependencies and install missing packages
# There are three group of packages:
# 1. DAE Tools related
# 2. Development related (compiler, tools, libraries, etc)
# 3. Utilities (wget, subversion, etc)
sudo apt-get update
sudo apt-get install libboost-all-dev python-qt4 python-numpy python-scipy python-matplotlib \
                     liblapack3gf libblas3gf libsuitesparse-dev libqt4-dev qtcreator \
                     autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake \
                     wget subversion
