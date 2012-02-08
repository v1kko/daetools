#!/bin/sh

set -e

TRUNK=`pwd`
Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
Ncpu=$(($Ncpu+1))
HOST_ARCH=`uname -m`

# Set SSE flags for x86
SSE_FLAGS=
#if [ ${HOST_ARCH} != "x86_64" ]; then
#  SSE_FLAGS="-mfpmath=sse"
#  SSE_TAGS=`grep -m 1 flags /proc/cpuinfo | grep -o 'sse\|sse2\|sse3\|ssse3\|sse4a\|sse4.1\|sse4.2\|sse5'`
#  for SSE_TAG in ${SSE_TAGS}
#  do
#    SSE_FLAGS="${SSE_FLAGS} -m${SSE_TAG}"
#  done
#fi

vBONMIN=1.5.1
vSUPERLU=4.1
vSUPERLU_MT=2.0
vNLOPT=2.2.4
vIDAS=1.0.0
vTRILINOS=10.8.0

DAETOOLS_HTTP=http://sourceforge.net/projects/daetools/files/gnu-linux-libs
IDAS_HTTP=${DAETOOLS_HTTP}
BONMIN_HTTP=http://www.coin-or.org/download/source/Bonmin
SUPERLU_HTTP=http://crd.lbl.gov/~xiaoye/SuperLU
TRILINOS_HTTP=http://trilinos.sandia.gov/download/files
NLOPT_HTTP=http://ab-initio.mit.edu/nlopt

# Get the archives DAE Tools website
if [ ! -e Bonmin-${vBONMIN}.zip ]; then
    wget ${BONMIN_HTTP}/Bonmin-${vBONMIN}.zip
fi
if [ ! -e superlu_${vSUPERLU}.tar.gz ]; then
    wget ${SUPERLU_HTTP}/superlu_${vSUPERLU}.tar.gz
fi
if [ ! -e superlu_makefiles.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/superlu_makefiles.tar.gz
fi
if [ ! -e superlu_mt_${vSUPERLU_MT}.tar.gz ]; then
    wget ${SUPERLU_HTTP}/superlu_mt_${vSUPERLU_MT}.tar.gz
fi
if [ ! -e superlu_mt_makefiles.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/superlu_mt_makefiles.tar.gz
fi
if [ ! -e nlopt-${vNLOPT}.tar.gz ]; then
    wget ${NLOPT_HTTP}/nlopt-${vNLOPT}.tar.gz
fi
if [ ! -e idas-${vIDAS}.tar.gz ]; then
    wget ${IDAS_HTTP}/idas-${vIDAS}.tar.gz
fi
if [ ! -e trilinos-${vTRILINOS}-Source.tar.gz ]; then
    wget ${TRILINOS_HTTP}/trilinos-${vTRILINOS}-Source.tar.gz
fi

# Unpack and compile libraries
unzip Bonmin-${vBONMIN}.zip
mv Bonmin-${vBONMIN} bonmin
cd bonmin/ThirdParty/Mumps
sh get.Mumps
cd ../..
mkdir build
cd build
../configure --enable-shared=no --enable-static=yes CFLAGS="-fPIC ${SSE_FLAGS}" CXXFLAGS="-fPIC ${SSE_FLAGS}" FFLAGS="-fPIC ${SSE_FLAGS}"
make -j${Ncpu}
make test
make install
make clean
cd ${TRUNK}

tar -xzf superlu_${vSUPERLU}.tar.gz
mv SuperLU_${vSUPERLU} superlu
cd superlu
tar -xzf ../superlu_makefiles.tar.gz
make superlulib
cd ${TRUNK}

tar -xzf superlu_mt_${vSUPERLU_MT}.tar.gz
mv SuperLU_MT_${vSUPERLU_MT} superlu_mt
cd superlu_mt
tar -xzf ../superlu_mt_makefiles.tar.gz
make lib
cd ${TRUNK}

tar -xzf nlopt-${vNLOPT}.tar.gz
mv nlopt-${vNLOPT} nlopt
cd nlopt
mkdir build
cd build
../configure -prefix=${TRUNK}/nlopt/build CFLAGS="-fPIC ${SSE_FLAGS}" CXXFLAGS="-fPIC ${SSE_FLAGS}"
make
make install
make clean
cd ${TRUNK}

tar -xzf idas-${vIDAS}.tar.gz
mv idas-${vIDAS} idas
cd idas
mkdir build
./configure --prefix=${TRUNK}/idas/build --with-pic --disable-mpi --enable-examples --enable-static=yes --enable-shared=no CFLAGS="-O3 ${SSE_FLAGS}"
make
make install
make clean
cd ${TRUNK}

tar -xzf trilinos-${vTRILINOS}-Source.tar.gz
mv trilinos-${vTRILINOS}-Source trilinos
cd trilinos
mkdir build
cp ../do-configure-trilinos.sh build
cd build
sh do-configure-trilinos.sh
make -j${Ncpu}
make install
make clean
cd ${TRUNK}


