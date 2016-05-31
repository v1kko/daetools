#!/bin/bash

set -e

usage()
{
cat << EOF
usage: $0 [options]

This script detects the system and installs packages required to compile 
daetools libraries and python extension modules. Currently supported:
    - Debian GNU/Linux
    - Ubuntu
    - Fedora
    - CentOS
    - Linux Mint
    - Suse Linux
    - Arch Linux

NOTE: If the script fails check whether 'lsb_release' package is installed
      and whether your GNU/Linux distribution is recognized.

Typical use:
    sh install_dependencies_linux.sh

OPTIONS:
   -h | --help  Show this message.
EOF
}

args=`getopt -a -o "h" -l "help" -n "install_dependencies_linux" -- $*`

# Process options
for i; do
  case "$i" in
    -h|--help)  usage
                exit 1
                ;;
                  
  esac
done

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
                       python-xlwt python-lxml python-h5py python-pandas pyqt4-dev-tools \
                       libblas3gf liblapack3gf libqt4-dev qtcreator qt4-designer \
                       autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake patch \
                       wget subversion fakeroot libfreetype6-dev swig python-dev libpng12-dev libxext-dev libbz2-dev

elif [ ${DISTRIBUTOR_ID} = "ubuntu" ]; then
  #sudo apt-get update
  sudo apt-get install python-qt4 python-numpy python-scipy python-matplotlib python-tk mayavi2 \
                       python-xlwt python-lxml python-h5py python-pandas pyqt4-dev-tools \
                       libblas3gf liblapack3gf libqt4-dev qtcreator qt4-designer \
                       autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake patch \
                       wget subversion fakeroot libfreetype6-dev swig python-dev libpng12-dev libxext-dev libbz2-dev

elif [ ${DISTRIBUTOR_ID} = "linuxmint" ]; then
  #sudo apt-get update
  sudo apt-get install python-qt4 python-numpy python-scipy python-matplotlib python-tk mayavi2 \
                       python-xlwt python-lxml python-h5py python-pandas pyqt4-dev-tools \
                       libblas3gf liblapack3gf libqt4-dev qtcreator qt4-designer \
                       autotools-dev automake make pkg-config autoconf gcc g++ gfortran binutils cmake patch \
                       wget subversion fakeroot libfreetype6-dev swig python-dev libpng12-dev libxext-dev libbz2-dev

elif [ ${DISTRIBUTOR_ID} = "fedora" ]; then
  #sudo yum check-update
  sudo yum install PyQt4 numpy scipy python-matplotlib python-tk \
                   python-xlwt python-lxml h5py python-pandas PyQt4-devel \
                   blas blas-devel lapack lapack-devel qt-creator qt-devel \
                   automake make autoconf gcc gcc-c++ gcc-gfortran binutils cmake patch \
                   wget subversion fakeroot rpm-build libbz2-devel

elif [ ${DISTRIBUTOR_ID} = "centos" ]; then
  #sudo yum check-update
  # Missing: scipy, suitesparse-devel, qt-creator 
  # Should be manually installed, ie. from http://pkgs.org
  sudo yum install PyQt4 numpy python-matplotlib python-tk python-devel PyQt4-devel \
                   python-xlwt python-lxml h5py python-pandas \
                   blas blas-devel lapack lapack-devel qt-devel \
                   automake make autoconf gcc gcc-c++ gcc-gfortran binutils cmake patch \
                   wget subversion fakeroot rpm-build libbz2-devel

elif [ ${DISTRIBUTOR_ID} = "suse linux" ]; then
  # Missing: scipy, suitesparse-devel, mayavi
  # Should be manually installed, ie. from http://pkgs.org
  sudo zypper in python-qt4 python-numpy python-matplotlib python-tk python-devel \
                 python-xlwt python-lxml h5py python-pandas \
                 blas lapack libqt4 libqt4-devel qt-creator \
                 automake make autoconf gcc gcc-c++ gcc-fortran binutils cmake patch \
                 wget subversion devel_rpm_build libbz2-devel
                 
elif [ ${DISTRIBUTOR_ID} = "arch" ]; then
  sudo pacman -S python2-pyqt4 python2-numpy python2-scipy python2-matplotlib mayavi \
                 python2-xlwt python-lxml python-h5py python-pandas \
                 blas lapack qt4 qtcreator \
                 base-devel automake make pkg-config autoconf gcc gcc-fortran binutils cmake patch \
                 wget subversion fakeroot swig libpng libxext bzip2

else
  echo "ERROR: unsupported GNU/Linux distribution; please edit the script to add support for: ${DISTRIBUTOR_ID}/${CODENAME}"
  exit -1
fi

