#!/bin/sh
# -*- coding: utf-8 -*-

set -e

TRUNK="$( cd "$( dirname "$0" )" && pwd )"
HOST_ARCH=`uname -m`
PLATFORM=`uname -s`

if [ ${PLATFORM} = "Darwin" ]; then
  Ncpu=$(/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | awk '/Total Number Of Cores/ {print $5};')
else
  Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
fi

PYTHON_MAJOR=`python -c "import sys; print(sys.version_info[0])"`
PYTHON_MINOR=`python -c "import sys; print(sys.version_info[1])"`
PYTHON_VERSION=${PYTHON_MAJOR}.${PYTHON_MINOR}
PYTHON_INCLUDE_DIR=`python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())"`
PYTHON_SITE_PACKAGES_DIR=`python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib())"`
PYTHON_LIB_DIR=`python -c "import sys; print(sys.prefix)"`/lib
echo $PYTHON_INCLUDE_DIR
echo $PYTHON_SITE_PACKAGES_DIR
echo $PYTHON_LIB_DIR

# daetools specific compiler flags
DAE_COMPILER_FLAGS="-fPIC"
BOOST_MACOSX_FLAGS=

if [ ${PLATFORM} = "Darwin" ]; then
  DAE_COMPILER_FLAGS="${DAE_COMPILER_FLAGS} -arch i386 -arch x86_64"
  BOOST_MACOSX_FLAGS="macosx-version-min=10.5 architecture=x86 address-model=32_64"
  
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

vBOOST=1.49.0
vBOOST_=1_49_0
vBONMIN=1.7.0
vLAPACK=3.4.1
vSUPERLU=4.1
vSUPERLU_MT=2.0
vNLOPT=2.2.4
vIDAS=1.1.0
vTRILINOS=10.8.0
vUMFPACK=5.6.2
vAMD=2.3.1
vMETIS=5.1.0
vCHOLMOD=2.1.2
vCAMD=2.3.1
vCOLAMD=2.8.0
vCCOLAMD=2.8.0
vSUITESPARSE_CONFIG=4.2.1
vOPENBLAS=0.2.8

BOOST_BUILD_ID=daetools-py${PYTHON_MAJOR}${PYTHON_MINOR}
BOOST_PYTHON_BUILD_ID=

BOOST_HTTP=http://sourceforge.net/projects/boost/files/boost
LAPACK_HTTP=http://www.netlib.org/lapack
DAETOOLS_HTTP=http://sourceforge.net/projects/daetools/files/gnu-linux-libs
IDAS_HTTP=${DAETOOLS_HTTP}
BONMIN_HTTP=http://www.coin-or.org/download/source/Bonmin
SUPERLU_HTTP=http://crd.lbl.gov/~xiaoye/SuperLU
TRILINOS_HTTP=http://trilinos.sandia.gov/download/files
NLOPT_HTTP=http://ab-initio.mit.edu/nlopt
UMFPACK_HTTP=http://www.cise.ufl.edu/research/sparse/umfpack
AMD_HTTP=http://www.cise.ufl.edu/research/sparse/amd
METIS_HTTP=http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis
CHOLMOD_HTTP=http://www.cise.ufl.edu/research/sparse/cholmod
CAMD_HTTP=http://www.cise.ufl.edu/research/sparse/camd
COLAMD_HTTP=http://www.cise.ufl.edu/research/sparse/colamd
CCOLAMD_HTTP=http://www.cise.ufl.edu/research/sparse/ccolamd
SUITESPARSE_CONFIG_HTTP=http://www.cise.ufl.edu/research/sparse/UFconfig

# ACHTUNG! cd to TRUNK (in case the script is called from some other folder)
cd ${TRUNK}

#######################################################
#                       BOOST                         #
#######################################################
if [ ! -e boost ]; then
  echo "Setting-up BOOST..."
  if [ ! -e boost_${vBOOST_}.tar.gz ]; then
    wget ${BOOST_HTTP}/${vBOOST}/boost_${vBOOST_}.tar.gz
  fi
  tar -xzf boost_${vBOOST_}.tar.gz
  mv boost_${vBOOST_} boost
  cd boost
  sh bootstrap.sh
  cd ${TRUNK}
fi
cd boost
if [ ! -e stage/lib/libboost_python-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}.so ]; then
  echo "Building BOOST..."
  
  if [ -e build ]; then
    rm -r build
  fi
  
  echo "using python : ${PYTHON_MAJOR}.${PYTHON_MINOR} : python${PYTHON_MAJOR}.${PYTHON_MINOR} ;" > user-config.jam

  ./bjam --build-dir=./build --debug-building --layout=system --buildid=${BOOST_BUILD_ID} \
         --with-date_time --with-system --with-regex --with-serialization --with-thread --with-python python=${PYTHON_VERSION} \
         variant=release link=shared threading=multi runtime-link=shared ${BOOST_MACOSX_FLAGS}

  cp -a stage/lib/libboost_python-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}* ../daetools-package/solibs
  cp -a stage/lib/libboost_system-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}* ../daetools-package/solibs
  cp -a stage/lib/libboost_thread-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}* ../daetools-package/solibs
fi
cd ${TRUNK}

#######################################################
#                   OpenBLAS                          #
#######################################################
# if [ ! -e openblas ]; then
#   echo "Setting-up openblas..."
#   if [ ! -e openblas-${vOPENBLAS}.tar.gz ]; then
#     wget ${DAETOOLS_HTTP}/openblas-${vOPENBLAS}.tar.gz
#   fi
#   if [ ! -e Makefile-openblas.rule ]; then
#     wget ${DAETOOLS_HTTP}/Makefile-openblas.rule
#   fi
#   tar -xzf openblas-${vOPENBLAS}.tar.gz
#   cp Makefile-openblas.rule openblas/Makefile.rule
#   cd openblas
#   mkdir build
#   cd ${TRUNK}
# fi
# cd openblas
# if [ ! -e libopenblas.so ]; then
#   echo "Building openblas..."
#   make -j${Ncpu} libs
#   make 
#   make prefix=build install
#   cp -a libopenblas_daetools* ../daetools-package/solibs
#   make clean
# else
#   echo "   openblas library already built"
# fi
# cd ${TRUNK}

#######################################################
#                   LAPACK + BLAS                     #
#######################################################
if [ ! -e lapack ]; then
  echo "Setting-up lapack..."
  if [ ! -e lapack-${vLAPACK}.tgz ]; then
    wget ${LAPACK_HTTP}/lapack-${vLAPACK}.tgz
  fi
  if [ ! -e daetools_lapack_make.inc ]; then
    wget ${DAETOOLS_HTTP}/daetools_lapack_make.inc
  fi
  tar -xzf lapack-${vLAPACK}.tgz
  mv lapack-${vLAPACK} lapack
  cp daetools_lapack_make.inc lapack/make.inc
  cd ${TRUNK}
fi
cd lapack
if [ ! -e liblapack.a ]; then
  echo "Building lapack..."
  make -j${Ncpu} lapacklib
  make -j${Ncpu} blaslib
  make clean
else
  echo "   lapack library already built"
fi
cd ${TRUNK}

#######################################################
#                      UMFPACK                        #
#######################################################
if [ ! -e umfpack ]; then
  echo "Setting-up umfpack and friends..."
  #if [ ! -e metis-${vMETIS}.tar.gz ]; then
  #  wget ${METIS_HTTP}/metis-${vMETIS}.tar.gz
  #fi
  if [ ! -e SuiteSparse_config-${vSUITESPARSE_CONFIG}.tar.gz ]; then
    wget ${SUITESPARSE_CONFIG_HTTP}/SuiteSparse_config-${vSUITESPARSE_CONFIG}.tar.gz
  fi 
  if [ ! -e CHOLMOD-${vCHOLMOD}.tar.gz ]; then
    wget ${CHOLMOD_HTTP}/CHOLMOD-${vCHOLMOD}.tar.gz
  fi
  if [ ! -e AMD-${vAMD}.tar.gz ]; then
    wget ${AMD_HTTP}/AMD-${vAMD}.tar.gz
  fi
  if [ ! -e CAMD-${vCAMD}.tar.gz ]; then
    wget ${CAMD_HTTP}/CAMD-${vCAMD}.tar.gz
  fi
  if [ ! -e COLAMD-${vCOLAMD}.tar.gz ]; then
    wget ${COLAMD_HTTP}/COLAMD-${vCOLAMD}.tar.gz
  fi
  if [ ! -e CCOLAMD-${vCCOLAMD}.tar.gz ]; then
    wget ${CCOLAMD_HTTP}/CCOLAMD-${vCCOLAMD}.tar.gz
  fi
  if [ ! -e UMFPACK-${vUMFPACK}.tar.gz ]; then
    wget ${UMFPACK_HTTP}/UMFPACK-${vUMFPACK}.tar.gz
  fi
  if [ ! -e SuiteSparse_config.mk ]; then
    wget ${DAETOOLS_HTTP}/SuiteSparse_config.mk
  fi
  #if [ ! -e metis.h ]; then
  #  wget ${DAETOOLS_HTTP}/metis.h
  #fi
  #if [ ! -e Makefile-CHOLMOD.patch ]; then
  #  wget ${DAETOOLS_HTTP}/Makefile-CHOLMOD.patch
  #fi
  
  mkdir umfpack
  cd umfpack
  #tar -xzf ../metis-${vMETIS}.tar.gz
  #cp ../metis.h metis-${vMETIS}/include
  tar -xzf ../SuiteSparse_config-${vSUITESPARSE_CONFIG}.tar.gz
  tar -xzf ../CHOLMOD-${vCHOLMOD}.tar.gz
  tar -xzf ../AMD-${vAMD}.tar.gz
  tar -xzf ../CAMD-${vCAMD}.tar.gz
  tar -xzf ../COLAMD-${vCOLAMD}.tar.gz
  tar -xzf ../CCOLAMD-${vCCOLAMD}.tar.gz
  tar -xzf ../UMFPACK-${vUMFPACK}.tar.gz
  cp ../SuiteSparse_config.mk SuiteSparse_config

  # Apply Metis 5.1.0 patch for CHOLMOD
  #cd CHOLMOD/Lib
  #patch < ../../../Makefile-CHOLMOD.patch
  
  mkdir build
  mkdir build/lib
  mkdir build/include
  cd ${TRUNK}
fi

DAE_UMFPACK_INSTALL_DIR="${TRUNK}/umfpack/build"
export DAE_UMFPACK_INSTALL_DIR

# cd ${TRUNK}
# cd umfpack/metis-${vMETIS}
# if [ ! -e ${DAE_UMFPACK_INSTALL_DIR}/lib/libmetis.a ]; then
#   echo "Building metis..."
#   echo "make config prefix=${DAE_UMFPACK_INSTALL_DIR} shared=0 -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}""
#   make config prefix=${DAE_UMFPACK_INSTALL_DIR} -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}"
#   make install
#   make clean
# else
#   echo "   metis library already built"
# fi

cd ${TRUNK}
cd umfpack/SuiteSparse_config
if [ ! -e ${DAE_UMFPACK_INSTALL_DIR}/lib/libsuitesparseconfig.a ]; then
  echo "Building suitesparseconfig..."
  echo "make cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  make clean
else
  echo "   suitesparseconfig library already built"
fi

cd ${TRUNK}
cd umfpack/AMD
if [ ! -e ${DAE_UMFPACK_INSTALL_DIR}/lib/libamd.a ]; then
  echo "Building amd..."
  echo "make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  make clean
else
  echo "   amd library already built"
fi

cd ${TRUNK}
cd umfpack/CAMD
if [ ! -e ${DAE_UMFPACK_INSTALL_DIR}/lib/libcamd.a ]; then
  echo "Building camd..."
  echo "make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  make clean
else
  echo "   camd library already built"
fi

cd ${TRUNK}
cd umfpack/COLAMD
if [ ! -e ${DAE_UMFPACK_INSTALL_DIR}/lib/libcolamd.a ]; then
  echo "Building colamd..."
  echo "make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  make clean
else
  echo "   colamd library already built"
fi

cd ${TRUNK}
cd umfpack/CCOLAMD
if [ ! -e ${DAE_UMFPACK_INSTALL_DIR}/lib/libccolamd.a ]; then
  echo "Building ccolamd..."
  echo "make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  make clean
else
  echo "   ccolamd library already built"
fi

cd ${TRUNK}
cd umfpack/CHOLMOD
if [ ! -e ${DAE_UMFPACK_INSTALL_DIR}/lib/libcholmod.a ]; then
  echo "Building cholmod..."
  echo "make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  make clean
else
  echo "   cholmod library already built"
fi

cd ${TRUNK}
cd umfpack/UMFPACK
if [ ! -e ${DAE_UMFPACK_INSTALL_DIR}/lib/libumfpack.${vUMFPACK}.a ]; then
  echo "Building umfpack..."
  echo "make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} cc=gcc F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  make clean
else
  echo "   umfpack library already built"
fi

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
    UMFPACK_INCLUDE_DIR="/opt/local/include/ufsparse"
  else
    UMFPACK_INCLUDE_DIR="${DAE_UMFPACK_INSTALL_DIR}/include"
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
    -DTPL_UMFPACK_INCLUDE_DIRS:FILEPATH=${UMFPACK_INCLUDE_DIR} \
    -DTPL_UMFPACK_LIBRARIES:STRING=umfpack \
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


