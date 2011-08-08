#!/bin/sh

set -e

HOST_ARCH=`uname -m`
PLATFORM=`uname -s | tr "[:upper:]" "[:lower:]"`
DISTRIBUTOR_ID=`echo $(lsb_release -si) | tr "[:upper:]" "[:lower:]"`
CODENAME=`echo $(lsb_release -sc) | tr "[:upper:]" "[:lower:]"`

# Check the dependencies and install missing packages
# There are three group of packages:
# 1. DAE Tools related
# 2. Development related (compiler, tools, libraries, etc)
# 3. Utilities (wget, subversion, etc)

if [ ${DISTRIBUTOR_ID} = "debian" ]; then
  sudo apt-get update
  sudo apt-get install libboost-all-dev python-qt4 python-numpy python-scipy python-matplotlib mayavi2 \
                       liblapack3gf libblas3gf libsuitesparse-dev libqt4-dev qtcreator libamd2.2.0 libumfpack5.4.0 \
                       autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake \
                       wget subversion fakeroot

elif [ ${DISTRIBUTOR_ID} = "ubuntu" ]; then
  sudo apt-get update
  sudo apt-get install libboost-all-dev python-qt4 python-numpy python-scipy python-matplotlib mayavi2 \
                       liblapack3gf libblas3gf libsuitesparse-dev libqt4-dev qtcreator libamd2.2.0 libumfpack5.4.0 \
                       autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake \
                       wget subversion fakeroot

elif [ ${DISTRIBUTOR_ID} = "fedora" ]; then
  sudo yum check-update
  sudo yum install boost-devel PyQt4 numpy scipy python-matplotlib \
                   blas lapack suitesparse-devel qt-creator qt-devel \
                   automake make autoconf gcc gcc-c++ gcc-gfortran binutils cmake \
                   wget subversion fakeroot

else
  echo "ERROR: unsupported GNU/Linux distribution; please edit the script to add support for: ${DISTRIBUTOR_ID}/${CODENAME}"
  exit -1
fi

