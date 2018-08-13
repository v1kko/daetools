#!/bin/bash
# -*- coding: utf-8 -*-

set -e
#set -x

usage()
{
cat << EOF
usage: $0 [OPTIONS] LIBRARY1 [LIBRARY2 LIBRARY3 ...]

This script compiles third party libraries/solvers necessary to build daetools.

Typical usage (configure and then build all libraries/solvers):
    sh $0 all

Compiling only specified libraries:
    sh $0 boost idas trilinos opencs

Achtung, Achtung!!
On MACOS gcc should be used (the XCode does not provide OpenMP).
getopt command might be missing - that line should be commented out.

OPTIONS:
   -h | --help                  Show this message.

   Control options (if not set default is: --clean and --build):
    --configure                 Configure the specified library(ies)/solver(s).
    --build                     Build  the specified library(ies)/solver(s).
    --clean                     Clean  the specified library(ies)/solver(s).

LIBRARY:
    all             All libraries and solvers: boost cblas_clapack idas trilinos opencs

    Individual libraries/solvers:
    boost           Boost (static) libraries (no building required, only headers)
    cblas_clapack   CBLAS and CLapack libraries
    cvodes          CVodes solver with MPI interface enabled
    idas            IDAS solver with MPI interface enabled
    trilinos        Trilinos Amesos and AztecOO solvers
    metis           METIS graph partitioning library
    opencs          OpenCS libraries
EOF
}


ROOT_DIR="$( cd "$( dirname "$0" )" && pwd )"
HOST_ARCH=`uname -m`
PLATFORM=`uname -s`
if [[ "${PLATFORM}" == *"MSYS_"* ]]; then
  PLATFORM="Windows"
  # Platform should be set by i.e. vcbuildtools.bat
  VC_PLAT=`cmd "/C echo %Platform%"`
  echo $VC_PLAT
  if [[ "${VC_PLAT}" == *"X86"* ]]; then
    HOST_ARCH="win32"
  elif [[ "${VC_PLAT}" == *"x86"* ]]; then
    HOST_ARCH="win32"
  elif [[ "${VC_PLAT}" == *"x64"* ]]; then
    HOST_ARCH="win64"
  elif [[ "${VC_PLAT}" == *"X64"* ]]; then
    HOST_ARCH="win64"
  else
    echo unknown HOST_ARCH: $HOST_ARCH
    exit 1
  fi
fi

if [ ${PLATFORM} = "Darwin" ]; then
  args=
else
  args=`getopt -a -o "h" -l "help,configure,build,clean:,host:" -n "compile" -- $*`
fi
# daetools specific compiler flags
OPENCS_COMPILER_FLAGS="-fPIC"
BOOST_MACOSX_FLAGS=

DO_CONFIGURE="no"
DO_BUILD="no"
DO_CLEAN="no"

# Process options
for i; do
  case "$i" in
    -h|--help)  usage
                exit 1
                ;;

    --configure) DO_CONFIGURE="yes"
                    shift
                    ;;

    --build) DO_BUILD="yes"
                shift
                ;;

    --clean) DO_CLEAN="yes"
                shift
                ;;

    --) shift; break
       ;;
  esac
done

MINGW_MAKE=
if [[ "${PLATFORM}" == "Windows" ]]; then
  MINGW_MAKE="mingw32-make"
fi

if [ ${PLATFORM} = "Darwin" ]; then
  Ncpu=$(/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | awk '/Total Number Of Cores/ {print $5};')
  echo $Ncpu
  # If there are problems with memory and speed of compilation set:
  Ncpu=1
elif [ ${PLATFORM} = "Linux" ]; then
  Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
else
  Ncpu=4
fi

MAKE="make"
MAKE_Ncpu="make -j${Ncpu}"
CMAKE_GENERATOR="Unix Makefiles"
WGET="wget"
if [ ${PLATFORM} = "Windows" ]; then
  OPENCS_COMPILER_FLAGS=""
  MAKE="nmake"
  MAKE_Ncpu="nmake"
  CMAKE_GENERATOR="NMake Makefiles"
  WGET="wget --no-check-certificate"
fi

if [ ${PLATFORM} = "Darwin" ]; then
  OPENCS_COMPILER_FLAGS="${OPENCS_COMPILER_FLAGS} -arch x86_64"
  BOOST_MACOSX_FLAGS="architecture=x86"
  export CC=/usr/local/bin/gcc
  export CXX=/usr/local/bin/g++
  export FC=/usr/local/bin/gfortran
  export F77=/usr/local/bin/gfortran

elif [ ${PLATFORM} = "Linux" ]; then
  if [ ${HOST_ARCH} != "x86_64" ]; then
    OPENCS_COMPILER_FLAGS="${OPENCS_COMPILER_FLAGS} -march=pentium4 -mfpmath=sse -msse -msse2"
  fi
fi

if [ ${Ncpu} -gt 1 ]; then
 Ncpu=$(($Ncpu+1))
fi

export OPENCS_COMPILER_FLAGS

vBOOST=1.65.1
vBOOST_=1_65_1
vCLAPACK=3.2.1
vCVODES=2.8.2
vIDAS=1.3.0
vTRILINOS=12.10.1
vMETIS=5.1.0
DAETOOLS_HTTP=http://sourceforge.net/projects/daetools/files/gnu-linux-libs
DAETOOLS_WIN_HTTP=http://sourceforge.net/projects/daetools/files/windows-libs
CLAPACK_HTTP=${DAETOOLS_HTTP}
IDAS_HTTP=${DAETOOLS_HTTP}
CVODES_HTTP=${DAETOOLS_HTTP}
TRILINOS_HTTP=http://sourceforge.net/projects/daetools/files/gnu-linux-libs
METIS_HTTP=http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis

# If no option is set use defaults
if [ "${DO_CONFIGURE}" = "no" -a "${DO_BUILD}" = "no" -a "${DO_CLEAN}" = "no" ]; then
    DO_CONFIGURE="yes"
    DO_BUILD="yes"
    DO_CLEAN="no"
fi

if [ -z "$@" ]; then
    # If no project is specified compile all
    usage
    exit
else
    # Check if requested solver exist
    for solver in "$@"
    do
    case "$solver" in
        all)              ;;
        boost)            ;;
        cblas_clapack)    ;;
        cvodes)           ;;
        idas)             ;;
        opencs)           ;;
        metis)            ;;
        trilinos)         ;;
        *) echo Unrecognized solver: "$solver"
        exit
        ;;
    esac
    done
fi
echo ""
echo "###############################################################################"
echo "Proceed with the following options:"
echo "  - Platform:                     $PLATFORM"
echo "  - Architecture:                 $HOST_ARCH"
echo "  - Additional compiler flags:    ${OPENCS_COMPILER_FLAGS}"
echo "  - Number of threads:            ${Ncpu}"
echo "  - Projects to compile:          $@"
echo "     + Configure:  [$DO_CONFIGURE]"
echo "     + Build:      [$DO_BUILD]"
echo "     + Clean:      [$DO_CLEAN]"
echo "###############################################################################"
echo ""

# ACHTUNG! cd to ROOT_DIR (in case the script is called from some other folder)
cd "${ROOT_DIR}"

#######################################################
#                   BOOST STATIC                      #
#######################################################
configure_boost_static()
{
  if [ -e boost ]; then
    rm -r boost
  fi
  
  echo ""
  echo "[*] Setting-up boost..."
  echo ""
  
  # Building Boost libs is currently not required.
  
  #BOOST_USER_CONFIG=~/user-config.jam
  #echo "using mpi ;" > ${BOOST_USER_CONFIG}
  
  if [ ${PLATFORM} = "Windows" ]; then
    if [ ! -e boost_${vBOOST_}.zip ]; then
      $WGET ${BOOST_HTTP}/${vBOOST}/boost_${vBOOST_}.zip
    fi
    unzip boost_${vBOOST_}.zip
    mv boost_${vBOOST_} boost
    cd boost
    
    # There is a problem when there are multiple vc++ version installed.
    # Therefore, specify the version in the jam file.
    #echo "using msvc : 14.0 : : ;" >>  ${BOOST_USER_CONFIG}
    
    # There is a problem with executing bootstrap.bat with arguments from bash.
    # Solution: create a proxy batch file which runs 'bootstrap vc14'. 
    #echo "call bootstrap vc14" > dae_bootstrap.bat
    #cmd "/C dae_bootstrap" 
    
  else
    if [ ! -e boost_${vBOOST_}.tar.gz ]; then
      $WGET ${BOOST_HTTP}/${vBOOST}/boost_${vBOOST_}.tar.gz
    fi
    tar -xzf boost_${vBOOST_}.tar.gz
    mv boost_${vBOOST_} boost
    cd boost
    #sh bootstrap.sh
  fi

  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_boost_static()
{
  # Building Boost libs is currently not required.
  echo "[*] Building boost skipped (not currently required)"
  return
  
#   cd boost
#   echo ""
#   echo "[*] Building boost..."
#   echo ""
# 
#   BOOST_USER_CONFIG=~/user-config.jam
#   if [ ${PLATFORM} = "Windows" ]; then
#     if [[ $HOST_ARCH == "win64" ]]; then
# 	    ADDRESS_MODEL="address-model=64"
#     else
# 	    ADDRESS_MODEL=""
# 	fi
# 	
#     ./bjam --build-dir=./build --debug-building --layout=system ${ADDRESS_MODEL} \
#            --with-system --with-filesystem --with-thread --with-regex --with-mpi --with-serialization \
#            variant=release link=static threading=multi runtime-link=shared cxxflags="\MD"
# 
#   else
#     ./bjam --build-dir=./build --debug-building --layout=system \
#            --with-system --with-filesystem --with-thread --with-regex --with-mpi --with-serialization \
#            variant=release link=static threading=multi runtime-link=shared cxxflags="-fPIC"
#   fi
# 
#   echo ""
#   echo "[*] Done!"
#   echo ""
#   cd "${ROOT_DIR}"
}

clean_boost_static()
{
  # Building Boost libs is currently not required.
  echo "[*] Skipping cleaning boost..."
  return

#   echo ""
#   echo "[*] Cleaning boost..."
#   echo ""
#   cd boost
#   ./bjam --clean
#   cd "${ROOT_DIR}"
#   echo ""
#   echo "[*] Done!"
#   echo ""
}

#######################################################
#                 CBLAS and CLAPACK                   #
#######################################################
configure_cblas_clapack()
{
  if [ -e clapack ]; then
    rm -r clapack
  fi

  echo ""
  echo "[*] Setting-up cblas & clapack..."
  echo ""
  if [ ! -e clapack-${vCLAPACK}-CMAKE.tgz ]; then
    $WGET ${CLAPACK_HTTP}/clapack-${vCLAPACK}-CMAKE.tgz
  fi
  tar -xzf clapack-${vCLAPACK}-CMAKE.tgz
  mv clapack-${vCLAPACK}-CMAKE clapack

  cd clapack
  mkdir -p build

  # Might need configure from the cmake-gui
  cmake \
    -G"${CMAKE_GENERATOR}" \
    -DBUILD_TESTING:BOOL=OFF \
    -DUSE_BLAS_WRAP:BOOL=OFF \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_INSTALL_PREFIX:PATH="${ROOT_DIR}/clapack/build" \
    -DBUILD_STATIC_LIBS:BOOL=ON

  cmake-gui .
  cd "${ROOT_DIR}"

  echo ""
  echo "[*] Done!"
  echo ""
}

compile_cblas_clapack()
{
  cd clapack
  echo ""
  echo "[*] Building cblas & clapack..."
  echo ""
  ${MAKE_Ncpu} install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${ROOT_DIR}"
}

clean_cblas_clapack()
{
  echo ""
  echo "[*] Cleaning cblas & clapack..."
  echo ""
  cd clapack
  cd build
  ${MAKE} clean
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                      CVODES                         #
#######################################################
configure_cvodes()
{
  CVODES_MPI_ENABLE=$1
  if [ "${CVODES_MPI_ENABLE}" = "" ]; then
    CVODES_MPI_ENABLE=OFF
  fi
  #echo "CVODES_MPI_ENABLE = ${CVODES_MPI_ENABLE}"

  if [ -e cvodes ]; then
    rm -r cvodes
  fi
  echo ""
  echo "[*] Setting-up cvodes..."
  echo ""
  if [ ! -e cvodes-${vCVODES}.tar.gz ]; then
    $WGET ${CVODES_HTTP}/cvodes-${vCVODES}.tar.gz
  fi
  tar -xzf cvodes-${vCVODES}.tar.gz
  mv cvodes-${vCVODES} cvodes
  cd cvodes

  # IDAS fails to find MS-MPI using their SundialsMPIC function: use FIND_PACKAGE(MPI) instead.
  if [ ${PLATFORM} = "Windows" ]; then
    sed -i 's/INCLUDE(SundialsMPIC)/FIND_PACKAGE(MPI REQUIRED)\n  SET(MPIC_FOUND ON)/gi' CMakeLists.txt
  fi
  
  mkdir -p build
  cd build

  export CVODES_HOME="${ROOT_DIR}/cvodes"
  EXTRA_ARGS=

  cmake \
    -G"${CMAKE_GENERATOR}" \
    -DMPI_ENABLE:BOOL=${CVODES_MPI_ENABLE} \
    -DCMAKE_BUILD_TYPE:STRING=RELEASE \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=. \
    -DEXAMPLES_ENABLE:BOOL=OFF \
    -DEXAMPLES_INSTALL:BOOL=OFF \
    -DEXAMPLES_INSTALL_PATH:PATH=. \
    -DMPI_INCLUDE_PATH:PATH="${MSMPI_INC}" \
    -DMPI_C_LIBRARIES:PATH="${MSMPI_LIB64}msmpi.lib" \
    -DMPI_LIBRARY:STRING="${MSMPI_LIB64}msmpi.lib" \
    -DCMAKE_CXX_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    -DCMAKE_C_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    -DCMAKE_Fortran_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    $EXTRA_ARGS \
    ${CVODES_HOME}

  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_cvodes()
{
  cd cvodes
  cd build
  echo ""
  echo "[*] Building cvodes..."
  echo ""
  ${MAKE_Ncpu} install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${ROOT_DIR}"
}

clean_cvodes()
{
  echo ""
  echo "[*] Cleaning cvodes..."
  echo ""
  cd cvodes
  ${MAKE} clean
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                       IDAS                          #
#######################################################
configure_idas()
{
  IDAS_MPI_ENABLE=$1
  if [ "${IDAS_MPI_ENABLE}" = "" ]; then
    IDAS_MPI_ENABLE=OFF
  fi
  #echo "IDAS_MPI_ENABLE = ${IDAS_MPI_ENABLE}"

  if [ -e idas ]; then
    rm -r idas
  fi
  echo ""
  echo "[*] Setting-up idas..."
  echo ""
  if [ ! -e idas-${vIDAS}.tar.gz ]; then
    $WGET ${IDAS_HTTP}/idas-${vIDAS}.tar.gz
  fi
  tar -xzf idas-${vIDAS}.tar.gz
  mv idas-${vIDAS} idas
  cd idas

  # IDAS fails to find MS-MPI using their SundialsMPIC function: use FIND_PACKAGE(MPI) instead.
  if [ ${PLATFORM} = "Windows" ]; then
    sed -i 's/INCLUDE(SundialsMPIC)/FIND_PACKAGE(MPI REQUIRED)\n  SET(MPIC_FOUND ON)/gi' CMakeLists.txt
  fi
  
  mkdir -p build
  cd build

  export IDAS_HOME="${ROOT_DIR}/idas"
  EXTRA_ARGS=

  cmake \
    -G"${CMAKE_GENERATOR}" \
    -DMPI_ENABLE:BOOL=${IDAS_MPI_ENABLE} \
    -DCMAKE_BUILD_TYPE:STRING=RELEASE \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=. \
    -DEXAMPLES_ENABLE:BOOL=OFF \
    -DEXAMPLES_INSTALL:BOOL=OFF \
    -DEXAMPLES_INSTALL_PATH:PATH=. \
    -DMPI_INCLUDE_PATH:PATH="${MSMPI_INC}" \
    -DMPI_C_LIBRARIES:PATH="${MSMPI_LIB64}msmpi.lib" \
    -DMPI_LIBRARY:STRING="${MSMPI_LIB64}msmpi.lib" \
    -DCMAKE_CXX_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    -DCMAKE_C_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    -DCMAKE_Fortran_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    $EXTRA_ARGS \
    ${IDAS_HOME}

  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_idas()
{
  cd idas
  cd build
  echo ""
  echo "[*] Building idas..."
  echo ""
  ${MAKE_Ncpu} install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${ROOT_DIR}"
}

clean_idas()
{
  echo ""
  echo "[*] Cleaning idas..."
  echo ""
  cd idas
  ${MAKE} clean
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                   TRILINOS                          #
#######################################################
configure_trilinos()
{
  if [ -e trilinos ]; then
    rm -r trilinos
  fi

  echo "[*] Setting-up trilinos..."
  if [ ! -e trilinos-${vTRILINOS}-Source.tar.gz ]; then
    $WGET ${TRILINOS_HTTP}/trilinos-${vTRILINOS}-Source.tar.gz
  fi

  tar -xzf trilinos-${vTRILINOS}-Source.tar.gz
  mv trilinos-${vTRILINOS}-Source trilinos
  cd trilinos
  mkdir build
  cd build

  export TRILINOS_HOME="${ROOT_DIR}/trilinos"
  EXTRA_ARGS=

  UMFPACK_INCLUDE_DIR="${DAE_UMFPACK_INSTALL_DIR}/include"

  UMFPACK_ENABLED=OFF
  BLAS_LIBRARIES="${ROOT_DIR}/lapack/lib/libblas.a -lgfortran"
  LAPACK_LIBRARIES="${ROOT_DIR}/lapack/lib/liblapack.a -lgfortran"
  if [ ${PLATFORM} = "Windows" ]; then
    UMFPACK_ENABLED=OFF
    BLAS_LIBRARIES="${ROOT_DIR}/clapack/build/lib/blas.lib ${ROOT_DIR}/clapack/build/lib/libf2c.lib"
    LAPACK_LIBRARIES="${ROOT_DIR}/clapack/build/lib/lapack.lib ${ROOT_DIR}/clapack/build/lib/libf2c.lib"
  fi

  cmake \
    -G"${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE:STRING=RELEASE \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DTrilinos_ENABLE_CXX11=OFF \
    -DTrilinos_ENABLE_Amesos:BOOL=ON \
    -DTrilinos_ENABLE_Epetra:BOOL=ON \
    -DTrilinos_ENABLE_AztecOO:BOOL=ON \
    -DTrilinos_ENABLE_ML:BOOL=ON \
    -DTrilinos_ENABLE_Ifpack:BOOL=ON \
    -DTrilinos_ENABLE_Teuchos:BOOL=ON \
    -DTrilinos_ENABLE_Zoltan:BOOL=OFF \
    -DAmesos_ENABLE_LAPACK:BOOL=ON \
    -DAmesos_ENABLE_SuperLU:BOOL=OFF \
    -DIfpack_ENABLE_SuperLU:BOOL=OFF \
    -DTeuchos_ENABLE_COMPLEX:BOOL=OFF \
    -DTeuchos_ENABLE_LAPACK:BOOL=ON \
    -DEpetra_ENABLE_BLAS:BOOL=ON \
    -DTPL_ENABLE_UMFPACK:BOOL=${UMFPACK_ENABLED} \
    -DTPL_UMFPACK_INCLUDE_DIRS:FILEPATH=${UMFPACK_INCLUDE_DIR} \
    -DTPL_UMFPACK_LIBRARIES:STRING=umfpack \
    -DTPL_BLAS_LIBRARIES:STRING="${BLAS_LIBRARIES}" \
    -DTPL_LAPACK_LIBRARIES:STRING="${LAPACK_LIBRARIES}" \
    -DTPL_ENABLE_MPI:BOOL=OFF \
    -DDART_TESTING_TIMEOUT:STRING=600 \
    -DCMAKE_INSTALL_PREFIX:PATH=. \
    -DCMAKE_CXX_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    -DCMAKE_C_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    -DCMAKE_Fortran_FLAGS:STRING="-DNDEBUG ${OPENCS_COMPILER_FLAGS}" \
    $EXTRA_ARGS \
    ${TRILINOS_HOME}

  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_trilinos()
{
  cd trilinos/build
  echo "[*] Building trilinos..."
  ${MAKE_Ncpu}
  ${MAKE} install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${ROOT_DIR}"
}

clean_trilinos()
{
  echo ""
  echo "[*] Cleaning trilinos..."
  echo ""
  cd trilinos/build
  ${MAKE} clean
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}


#######################################################
#                      Metis                          #
#######################################################
configure_metis()
{
  if [ -e metis ]; then
    rm -r metis
  fi

  echo "[*] Setting-up Metis..."
  if [ ! -e metis-${vMETIS}.tar.gz ]; then
    $WGET ${METIS_HTTP}/metis-${vMETIS}.tar.gz
  fi

  tar -xzf metis-${vMETIS}.tar.gz
  mv metis-${vMETIS} metis
  cd metis
  mkdir -p build

  if [ ${PLATFORM} = "Windows" ]; then
    METIS_HOME="${ROOT_DIR}/metis"
    METIS_INSTALL_DIR="${ROOT_DIR}/metis/build"
    cmake \
      -G"${CMAKE_GENERATOR}" \
      -DCMAKE_INSTALL_PREFIX:STRING="${METIS_INSTALL_DIR}" \
      -DCMAKE_BUILD_TYPE:STRING=Release \
      -DBUILD_SHARED_LIBS:BOOL=OFF \
      ${METIS_HOME}
      
    # This is required for metis 5.1.0 and MSVC v140 due to 
    # the compiler error in ucrt/corecrt_math.h: error C2059: syntax error : '('
    sed -i 's/#define rint(x) ((int)((x)+0.5))/\/*#define rint(x) ((int)((x)+0.5))*\//gi' GKlib/gk_arch.h
  else
    ${MAKE} config prefix="${ROOT_DIR}/metis/build"
  fi
  
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_metis()
{
  echo ""
  echo "[*] Building Metis..."
  echo ""
  cd metis
  
  if [ ${PLATFORM} = "Windows" ]; then
    ${MAKE_Ncpu}
    #if [[ "${HOST_ARCH}" == "win32" ]]; then
    #  MS_BUILD_PLATFORM="x86"
    #elif [[ "${HOST_ARCH}" == "win64" ]]; then
    #  MS_BUILD_PLATFORM="x64"
    #else
    #  echo unknown HOST_ARCH: $HOST_ARCH
    #  exit 1
    #fi
    #
    #echo "msbuild.exe METIS.sln /target:rebuild /p:Platform=${MS_BUILD_PLATFORM} /p:Configuration=Release /p:PlatformToolset=v140 /p:UseEnv=true /maxcpucount:3"
    #msbuild.exe METIS.sln -target:rebuild -p:Platform="${MS_BUILD_PLATFORM}" -p:Configuration="Release" -p:PlatformToolset="v140" -p:UseEnv=true -maxcpucount:3
    mkdir -p build/include
    mkdir -p build/lib
    cp include/metis.h    build/include
    cp libmetis/metis.lib build/lib
  else
    ${MAKE_Ncpu} install
  fi
  
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

clean_metis()
{
  echo ""
  echo "[*] Cleaning Metis..."
  echo ""
  cd metis
  ${MAKE} clean
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                     OpenCS                          #
#######################################################
configure_opencs()
{
  if [ -e build ]; then
    rm -r build
  fi
  
  echo ""
  echo "[*] Setting-up OpenCS..."
  echo ""

  mkdir -p build
  cd build
  
  OPEN_CS_HOME="${ROOT_DIR}"
  OPEN_CS_INSTALL_DIR="${ROOT_DIR}/build"
  
  cmake \
     -G"${CMAKE_GENERATOR}" \
     -DCMAKE_INSTALL_PREFIX:STRING="${OPEN_CS_INSTALL_DIR}" \
     -DCMAKE_BUILD_TYPE:STRING=Release \
     -DBUILD_SHARED_LIBS:BOOL=ON \
     -DOPENCL_INCLUDE_DIR:PATH="/opt/intel/opencl/include" \
     -DOPENCL_LIB_DIR:PATH="/opt/intel/opencl" \
     ${OPEN_CS_HOME}

  cd "${ROOT_DIR}"
}

compile_opencs()
{
  echo ""
  echo "[*] Building OpenCS..."
  echo ""
  cd build
  ${MAKE_Ncpu} install
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

clean_opencs()
{
  echo ""
  echo "[*] Cleaning OpenCS..."
  echo ""
  cd build
  ${MAKE} clean
  cd "${ROOT_DIR}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                Actual work                          #
#######################################################
for solver in "$@"
do
  case "$solver" in
    all)              if [ "${DO_CONFIGURE}" = "yes" ]; then
                          configure_boost_static
                          configure_cblas_clapack
                          configure_idas  "ON"
                          configure_trilinos
                          configure_metis
                          configure_opencs
                      fi

                      if [ "${DO_BUILD}" = "yes" ]; then
                          compile_boost_static
                          compile_cblas_clapack
                          compile_idas
                          compile_trilinos
                          compile_metis
                          compile_opencs
                      fi

                      if [ "${DO_CLEAN}" = "yes" ]; then
                          clean_boost_static
                          clean_cblas_clapack
                          clean_idas
                          clean_trilinos
                          clean_metis
                          clean_opencs
                      fi
                      ;;

    boost)            if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_boost_static
                      fi

                      if [ "${DO_BUILD}" = "yes" ]; then
                        compile_boost_static
                      fi

                      if [ "${DO_CLEAN}" = "yes" ]; then
                        clean_boost_static
                      fi
                      ;;
                      
    cblas_clapack)  if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_cblas_clapack
                      fi

                      if [ "${DO_BUILD}" = "yes" ]; then
                        compile_cblas_clapack
                      fi

                      if [ "${DO_CLEAN}" = "yes" ]; then
                        clean_cblas_clapack
                      fi
                      ;;

    cvodes)           if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_cvodes "ON"
                      fi

                      if [ "${DO_BUILD}" = "yes" ]; then
                        compile_cvodes
                      fi

                      if [ "${DO_CLEAN}" = "yes" ]; then
                        clean_cvodes
                      fi
                      ;;

    idas)             if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_idas "ON"
                      fi

                      if [ "${DO_BUILD}" = "yes" ]; then
                        compile_idas
                      fi

                      if [ "${DO_CLEAN}" = "yes" ]; then
                        clean_idas
                      fi
                      ;;

    trilinos)         if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_trilinos
                      fi

                      if [ "${DO_BUILD}" = "yes" ]; then
                        compile_trilinos
                      fi

                      if [ "${DO_CLEAN}" = "yes" ]; then
                        clean_trilinos
                      fi
                      ;;

    metis)            if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_metis
                      fi

                      if [ "${DO_BUILD}" = "yes" ]; then
                        compile_metis
                      fi

                      if [ "${DO_CLEAN}" = "yes" ]; then
                        clean_metis
                      fi
                      ;;

    opencs)           if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_opencs
                      fi

                      if [ "${DO_BUILD}" = "yes" ]; then
                        compile_opencs
                      fi

                      if [ "${DO_CLEAN}" = "yes" ]; then
                        clean_opencs
                      fi
                      ;;

    *) echo Unrecognized solver: "$solver"
       exit
       ;;
  esac
done

cd "${ROOT_DIR}"
