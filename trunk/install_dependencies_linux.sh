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
  #sudo apt-get update
  sudo apt-get install python-qt4 python-numpy python-scipy python-matplotlib python-tk mayavi2 \
                       liblapack3gf libblas3gf libsuitesparse-dev libqt4-dev qtcreator libamd2.2.0 libumfpack5.4.0 \
                       autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake \
                       wget subversion fakeroot libfreetype6-dev swig python-dev libpng12-dev libxext-dev libbz2-dev

elif [ ${DISTRIBUTOR_ID} = "ubuntu" ]; then
  #sudo apt-get update
  sudo apt-get install python-qt4 python-numpy python-scipy python-matplotlib python-tk mayavi2 \
                       liblapack3gf libblas3gf libsuitesparse-dev libqt4-dev qtcreator libamd2.2.0 libumfpack5.4.0 \
                       autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake \
                       wget subversion fakeroot libfreetype6-dev swig python-dev libpng12-dev libxext-dev libbz2-dev

elif [ ${DISTRIBUTOR_ID} = "linuxmint" ]; then
  #sudo apt-get update
  sudo apt-get install python-qt4 python-numpy python-scipy python-matplotlib python-tk mayavi2 \
                       liblapack3gf libblas3gf libsuitesparse-dev libqt4-dev qtcreator libamd2.2.0 libumfpack5.4.0 \
                       autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake \
                       wget subversion fakeroot libfreetype6-dev swig python-dev libpng12-dev libxext-dev libbz2-dev

elif [ ${DISTRIBUTOR_ID} = "fedora" ]; then
  #sudo yum check-update
  sudo yum install PyQt4 numpy scipy python-matplotlib python-tk \
                   blas blas-devel lapack lapack-devel suitesparse-devel qt-creator qt-devel \
                   automake make autoconf gcc gcc-c++ gcc-gfortran binutils cmake \
                   wget subversion fakeroot rpm-build libbz2-devel

elif [ ${DISTRIBUTOR_ID} = "centos" ]; then
  #sudo yum check-update
  # Missing: scipy, suitesparse-devel, qt-creator 
  # Should be manually installed, ie. from http://pkgs.org
  sudo yum install PyQt4 numpy python-matplotlib python-tk python-devel \
                   blas blas-devel lapack lapack-devel qt-devel \
                   automake make autoconf gcc gcc-c++ gcc-gfortran binutils cmake \
                   wget subversion fakeroot rpm-build libbz2-devel

elif [ ${DISTRIBUTOR_ID} = "suse linux" ]; then
  # Missing: scipy, suitesparse-devel, mayavi
  # Should be manually installed, ie. from http://pkgs.org
  sudo zypper in python-qt4 python-numpy python-matplotlib python-tk python-devel \
                 blas lapack libqt4 libqt4-devel qt-creator \
                 automake make autoconf gcc gcc-c++ gcc-fortran binutils cmake \
                 wget subversion devel_rpm_build libbz2-devel
                 
elif [ ${DISTRIBUTOR_ID} = "archlinux" ]; then
  sudo pacman -S python2-pyqt4 python2-numpy python2-scipy python2-matplotlib mayavi \
                 lapack blas libsuitesparse qt4 qtcreator \
                 automake make pkg-config autoconf gcc gcc-fortran binutils cmake \
                 wget subversion fakeroot swig libpng libxext bzip2

else
  echo "ERROR: unsupported GNU/Linux distribution; please edit the script to add support for: ${DISTRIBUTOR_ID}/${CODENAME}"
  exit -1
fi

