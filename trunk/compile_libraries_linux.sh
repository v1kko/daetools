#!/bin/sh
# -*- coding: utf-8 -*-

set -e

TRUNK="$( cd "$( dirname "$0" )" && pwd )"
Ncpu=1
HOST_ARCH=`uname -m`
PLATFORM=`uname -s`

# daetools specific compiler flags
DAE_COMPILER_FLAGS="-fPIC"

if [ ${PLATFORM} = "Darwin" ]; then
  DAE_COMPILER_FLAGS="${DAE_COMPILER_FLAGS} -arch i386 -arch ppc -arch x86_64"
  
  if type "wget" > /dev/null ; then
    echo "wget found"
  else
    echo "cURL have problems to get files from Source Forge: geting wget instead..."
    curl -O ftp://ftp.gnu.org/gnu/wget/wget-1.13.tar.gz
    tar -xvzf wget-1.13.tar.gz
    cd wget-1.13
    ./configure --with-ssl=openssl
    make
    sudo make install
  fi

else
  Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
  
  if [ ${HOST_ARCH} != "x86_64" ]; then
    DAE_COMPILER_FLAGS="${DAE_COMPILER_FLAGS} -mfpmath=sse"
    SSE_TAGS=`grep -m 1 flags /proc/cpuinfo | grep -o 'sse\|sse2\|sse3\|ssse3\|sse4a\|sse4.1\|sse4.2\|sse5'`
    for SSE_TAG in ${SSE_TAGS}
    do
      DAE_COMPILER_FLAGS="${DAE_COMPILER_FLAGS} -m${SSE_TAG}"
    done
  fi
fi

if [ ${Ncpu} -gt 1 ]; then
  Ncpu=$(($Ncpu+1))
fi

export DAE_COMPILER_FLAGS
echo $DAE_COMPILER_FLAGS

vBONMIN=1.5.1
vSUPERLU=4.1
vSUPERLU_MT=2.0
vNLOPT=2.2.4
vIDAS=1.1.0
vTRILINOS=10.8.0

DAETOOLS_HTTP=http://sourceforge.net/projects/daetools/files/gnu-linux-libs
IDAS_HTTP=${DAETOOLS_HTTP}
BONMIN_HTTP=http://www.coin-or.org/download/source/Bonmin
SUPERLU_HTTP=http://crd.lbl.gov/~xiaoye/SuperLU
TRILINOS_HTTP=http://trilinos.sandia.gov/download/files
NLOPT_HTTP=http://ab-initio.mit.edu/nlopt

# ACHTUNG! cd to TRUNK (in case the script is called from some other folder)
cd ${TRUNK}

#######################################################
#                       IDAS                          #
#######################################################
if [ ! -e idas ]; then
  echo "Setting-up idas..."
  if [ ! -e idas-${vIDAS}.tar.gz ]; then
    wget ${IDAS_HTTP}/idas-${vIDAS}.tar.gz
  fi
  tar -xzf idas-${vIDAS}.tar.gz
  mv idas-${vIDAS} idas
  cd idas
  mkdir build
  echo "./configure --prefix=${TRUNK}/idas/build --with-pic --disable-mpi --enable-examples --enable-static=yes --enable-shared=no --enable-lapack F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}""
  ./configure --prefix=${TRUNK}/idas/build --with-pic --disable-mpi --enable-examples --enable-static=yes --enable-shared=no --enable-lapack F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}"
  cd ${TRUNK}
fi
cd idas
if [ ! -e build/lib/libsundials_idas.a ]; then
  echo "Building idas..."
  make
  make install
  make clean
else
  echo "   idas library already built"
fi
cd ${TRUNK}

#######################################################
#                     SUPERLU                         #
#######################################################
if [ ! -e superlu ]; then
  echo "Setting-up superlu..."
  if [ ! -e superlu_${vSUPERLU}.tar.gz ]; then
    wget ${SUPERLU_HTTP}/superlu_${vSUPERLU}.tar.gz
  fi
  if [ ! -e superlu_makefiles.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/superlu_makefiles.tar.gz
  fi
  tar -xzf superlu_${vSUPERLU}.tar.gz
  mv SuperLU_${vSUPERLU} superlu
  cd superlu
  tar -xzf ../superlu_makefiles.tar.gz
  cd ${TRUNK}
fi
cd superlu
if [ ! -e lib/libsuperlu_${vSUPERLU}.a ]; then
  echo "Building superlu..."
  make superlulib
else
  echo "   superlu library already built"
fi
cd ${TRUNK}

#######################################################
#                    SUPERLU_MT                       #
#######################################################
if [ ! -e superlu_mt ]; then
  echo "Setting-up superlu_mt..."
  if [ ! -e superlu_mt_${vSUPERLU_MT}.tar.gz ]; then
    wget ${SUPERLU_HTTP}/superlu_mt_${vSUPERLU_MT}.tar.gz
  fi
  if [ ! -e superlu_mt_makefiles.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/superlu_mt_makefiles.tar.gz
  fi
  tar -xzf superlu_mt_${vSUPERLU_MT}.tar.gz
  mv SuperLU_MT_${vSUPERLU_MT} superlu_mt
  cd superlu_mt
  tar -xzf ../superlu_mt_makefiles.tar.gz
  cd ${TRUNK}
fi
cd superlu_mt
if [ ! -e lib/libsuperlu_mt_${vSUPERLU_MT}.a ]; then
  echo "Building superlu_mt..."
  make lib
else
  echo "   superlu_mt library already built"
fi
cd ${TRUNK}

#######################################################
#                      BONMIN                         #
#######################################################
if [ ! -e bonmin ]; then
  echo "Setting-up bonmin..."
  if [ ! -e Bonmin-${vBONMIN}.zip ]; then
    wget ${BONMIN_HTTP}/Bonmin-${vBONMIN}.zip
  fi
  unzip Bonmin-${vBONMIN}.zip
  rm -rf bonmin/Bonmin-${vBONMIN}
  mv Bonmin-${vBONMIN} bonmin
  cd bonmin/ThirdParty/Mumps
  sh get.Mumps
  cd ../..
  mkdir -p build
  cd build
  ../configure --disable-dependency-tracking --enable-shared=no --enable-static=yes ARCHFLAGS="${DAE_COMPILER_FLAGS}" CFLAGS="${DAE_COMPILER_FLAGS}" CXXFLAGS="${DAE_COMPILER_FLAGS}" FFLAGS="${DAE_COMPILER_FLAGS}" LDFLAGS="${DAE_COMPILER_FLAGS}"
  cd ${TRUNK}
fi
cd bonmin/build
if [ ! -e lib/libbonmin.a ]; then
  echo "Building bonmin..."
  make -j${Ncpu}
  make test
  make install
  make clean
else
  echo "   bonmin library already built"
fi
cd ${TRUNK}

#######################################################
#                      NLOPT                          #
#######################################################
if [ ! -e nlopt ]; then
  echo "Setting-up nlopt..."
  if [ ! -e nlopt-${vNLOPT}.tar.gz ]; then
    wget ${NLOPT_HTTP}/nlopt-${vNLOPT}.tar.gz
  fi
  tar -xzf nlopt-${vNLOPT}.tar.gz
  mv nlopt-${vNLOPT} nlopt
  cd nlopt
  mkdir build
  cd build
  ../configure --disable-dependency-tracking -prefix=${TRUNK}/nlopt/build CFLAGS="${DAE_COMPILER_FLAGS}" CXXFLAGS="${DAE_COMPILER_FLAGS}" FFLAGS="${DAE_COMPILER_FLAGS}"
  cd ${TRUNK}
fi
cd nlopt/build
if [ ! -e lib/libnlopt.a ]; then
  echo "Building nlopt..."
  make
  make install
  make clean
else
  echo "   nlopt library already built"
fi
cd ${TRUNK}

#######################################################
#                   TRILINOS                          #
#######################################################
if [ ! -e trilinos ]; then
  echo "Setting-up trilinos..."
  if [ ! -e trilinos-${vTRILINOS}-Source.tar.gz ]; then
    wget ${TRILINOS_HTTP}/trilinos-${vTRILINOS}-Source.tar.gz
  fi
  tar -xzf trilinos-${vTRILINOS}-Source.tar.gz
  mv trilinos-${vTRILINOS}-Source trilinos
  cd trilinos
  mkdir build
  cd build

  export TRILINOS_HOME="${TRUNK}/trilinos"
  EXTRA_ARGS=

  echo $TRILINOS_HOME

  if [ ${PLATFORM} = "Darwin" ]; then
    SUITE_SPARSE_DIR="/opt/local/include/ufsparse"
  else
    SUITE_SPARSE_DIR="/usr/include/suitesparse"
  fi
  
  cmake \
    -DCMAKE_BUILD_TYPE:STRING=RELEASE \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DTrilinos_ENABLE_Amesos:BOOL=ON \
    -DTrilinos_ENABLE_Epetra:BOOL=ON \
    -DTrilinos_ENABLE_AztecOO:BOOL=ON \
    -DTrilinos_ENABLE_ML:BOOL=ON \
    -DTrilinos_ENABLE_Ifpack:BOOL=ON \
    -DTrilinos_ENABLE_Teuchos:BOOL=ON \
    -DAmesos_ENABLE_SuperLU:BOOL=ON \
    -DIfpack_ENABLE_SuperLU:BOOL=ON \
    -DTPL_SuperLU_INCLUDE_DIRS:FILEPATH=${TRUNK}/superlu/SRC \
    -DTPL_SuperLU_LIBRARIES:STRING=superlu_4.1 \
    -DTPL_ENABLE_UMFPACK:BOOL=ON \
    -DTPL_UMFPACK_INCLUDE_DIRS:FILEPATH=${SUITE_SPARSE_DIR} \
    -DTPL_ENABLE_MPI:BOOL=OFF \
    -DDART_TESTING_TIMEOUT:STRING=600 \
    -DCMAKE_INSTALL_PREFIX:PATH=. \
    -DCMAKE_CXX_FLAGS:STRING="-DNDEBUG ${DAE_COMPILER_FLAGS}" \
    -DCMAKE_C_FLAGS:STRING="-DNDEBUG ${DAE_COMPILER_FLAGS}" \
    -DCMAKE_Fortran_FLAGS:STRING="-DNDEBUG ${DAE_COMPILER_FLAGS}" \
    $EXTRA_ARGS \
    ${TRILINOS_HOME}
  
  cd ${TRUNK}
fi

cd trilinos/build
if [ ! -e lib/libamesos.a ]; then
  echo "Building trilinos..."
  make -j${Ncpu}
  make install
  make clean
else
  echo "   trilinos library already built"
fi
cd ${TRUNK}


