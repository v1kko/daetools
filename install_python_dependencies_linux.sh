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
    sh install_python_dependencies.sh

OPTIONS:
   -h | --help  Show this message.
EOF
}

args=`getopt -a -o "h" -l "help,with-python-version:" -n "install_python_dependencies" -- $*`

PYTHON="python"

# Process options
for i; do
  case "$i" in
    --with-python-version)  PYTHON="python$2"
                            echo ${PYTHON}
                            shift; shift
                            ;;

    -h|--help)  usage
                exit 1
                ;;
                  
  esac
done

echo ${PYTHON}

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
  sudo apt-get install ${PYTHON}-pyqt5 ${PYTHON}-pyqt5.qtwebkit ${PYTHON}-numpy ${PYTHON}-scipy ${PYTHON}-matplotlib ${PYTHON}-tk mayavi2 \
                       ${PYTHON}-xlwt ${PYTHON}-lxml ${PYTHON}-h5py ${PYTHON}-pandas pyqt5-dev-tools

elif [ ${DISTRIBUTOR_ID} = "ubuntu" ]; then
  #sudo apt-get update
  sudo apt-get install ${PYTHON}-pyqt5 ${PYTHON}-pyqt5.qtwebkit ${PYTHON}-numpy ${PYTHON}-scipy ${PYTHON}-matplotlib ${PYTHON}-tk mayavi2 \
                       ${PYTHON}-xlwt ${PYTHON}-lxml ${PYTHON}-h5py ${PYTHON}-pandas pyqt5-dev-tools

elif [ ${DISTRIBUTOR_ID} = "linuxmint" ]; then
  #sudo apt-get update
  sudo apt-get install ${PYTHON}-pyqt5 ${PYTHON}-pyqt5.qtwebkit ${PYTHON}-numpy ${PYTHON}-scipy ${PYTHON}-matplotlib ${PYTHON}-tk mayavi2 \
                       ${PYTHON}-lxml ${PYTHON}-h5py ${PYTHON}-pandas pyqt5-dev-tools

elif [ ${DISTRIBUTOR_ID} = "fedora" ]; then
  #sudo yum check-update
  sudo yum install PyQt5 numpy scipy ${PYTHON}-matplotlib ${PYTHON}-tk \
                   ${PYTHON}-xlwt ${PYTHON}-lxml h5py ${PYTHON}-pandas PyQt5-devel

elif [ ${DISTRIBUTOR_ID} = "centos" ]; then
  #sudo yum check-update
  # Missing: scipy, suitesparse-devel, qt-creator 
  # Should be manually installed, ie. from http://pkgs.org
  sudo yum install PyQt5 numpy ${PYTHON}-matplotlib ${PYTHON}-tk ${PYTHON}-devel PyQt5-devel \
                   ${PYTHON}-xlwt ${PYTHON}-lxml h5py ${PYTHON}-pandas

elif [ ${DISTRIBUTOR_ID} = "suse linux" ]; then
  # Missing: scipy, suitesparse-devel, mayavi
  # Should be manually installed, ie. from http://pkgs.org
  sudo zypper in ${PYTHON}-qt5 ${PYTHON}-numpy ${PYTHON}-matplotlib ${PYTHON}-tk ${PYTHON}-devel \
                 ${PYTHON}-xlwt ${PYTHON}-lxml h5py ${PYTHON}-pandas
                 
elif [ ${DISTRIBUTOR_ID} = "arch" ]; then
  sudo pacman -S ${PYTHON}-pyqt5 ${PYTHON}-numpy ${PYTHON}-scipy ${PYTHON}-matplotlib mayavi ${PYTHON}-xlwt \
                 ${PYTHON}-lxml ${PYTHON}-h5py ${PYTHON}-pandas

else
  echo "ERROR: unsupported GNU/Linux distribution; please edit the script to add support for: ${DISTRIBUTOR_ID}/${CODENAME}"
  exit -1
fi

