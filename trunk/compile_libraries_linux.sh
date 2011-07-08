#!/bin/sh

set -e

DAETOOLS_HTTP=http://daetools.sourceforge.net/compile-linux
TRUNK=`pwd`
Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
Ncpu=$(($Ncpu+1))

vBONMIN=1.4.1
vSUPERLU=4.1
vSUPERLU_MT=2.0
vNLOPT=2.2.1
vIDAS=1.0.0
vTRILINOS=10.6.2

# Get the archives DAE Tools website
wget ${DAETOOLS_HTTP}/Bonmin-${vBONMIN}.zip
wget ${DAETOOLS_HTTP}/superlu_${vSUPERLU}.tar.gz
wget ${DAETOOLS_HTTP}/superlu_makefiles.tar.gz
wget ${DAETOOLS_HTTP}/superlu_mt_${vSUPERLU_MT}.tar.gz
wget ${DAETOOLS_HTTP}/superlu_mt_makefiles.tar.gz
wget ${DAETOOLS_HTTP}/nlopt-${vNLOPT}.tar.gz
wget ${DAETOOLS_HTTP}/idas-${vIDAS}.tar.gz
wget ${DAETOOLS_HTTP}/trilinos-${vTRILINOS}-Source.tar.gz
wget ${DAETOOLS_HTTP}/do-configure-trilinos.sh

# Unpack and compile libraries
unzip Bonmin-${vBONMIN}.zip
mv Bonmin-${vBONMIN} bonmin
cd bonmin/ThirdParty/Mumps
sh get.Mumps
cd ../..
mkdir build
cd build
../configure --enable-shared=no --enable-static=yes CFLAGS=-fPIC CXXFLAGS=-fPIC FFLAGS=-fPIC
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
../configure -prefix=${TRUNK}/nlopt/build CFLAGS=-fPIC CXXFLAGS=-fPIC
make
make install
make clean
cd ${TRUNK}

tar -xzf idas-${vIDAS}.tar.gz
mv idas-${vIDAS} idas
cd idas
mkdir build
./configure --prefix=${TRUNK}/idas/build --with-pic --disable-mpi --enable-examples --enable-static=yes --enable-shared=no CFLAGS=-O3
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

