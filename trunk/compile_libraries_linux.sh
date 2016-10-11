#!/bin/bash
# -*- coding: utf-8 -*-

set -e
usage()
{
cat << EOF
usage: $0 [OPTIONS] LIBRARY1 [LIBRARY2 LIBRARY3 ...]

This script compiles third party libraries/solvers necessary to build daetools.

Typical usage (configure and then build all libraries/solvers):
    sh $0 all

Compiling only specified libraries:
    sh $0 superlu bonmin trilinos

Achtung, Achtung!!
On MACOS gcc does not work well. llvm-gcc and llvm-g++ should be used.
Add the "llvm-gcc" compiler to the PATH variable if necessary. For instance:
    export PATH=/Developer/usr/bin:$PATH
Make sure there are: QMAKE_CC = llvm-gcc and QMAKE_CXX = llvm-g++ defined in dae.pri
getopt command might be missing - that line should be commented out.

OPTIONS:
   -h | --help                  Show this message.
   
   Control options (if not set default is: --clean and --build):
    --configure                 Configure the specified library(ies)/solver(s).
    --build                     Build  the specified library(ies)/solver(s).
    --clean                     Clean  the specified library(ies)/solver(s).
   
   Python options (if not set use system's default python). One of the following:
    --with-python-binary          Path to python binary to use.
    --with-python-version         Version of the system's python in the format: major.minor (i.e 2.7).
   
   Cross compiling options:
    --host                        Example: --host i686-w64-mingw32 (defines --host option for cross-compiling with the GNU gcc toolchain).    
    --cross-compile-python-root   An absolute path to the python root folder (for the native python cannot be run under the build OS).
                                  Has to be: ~/daetools/trunk/Python[Major][Minor]-[arch] (i.e. ~/daetools-win32-cross/trunk/Python27-win32)
                   
LIBRARY:
    all    All libraries and solvers.
           Equivalent to: boost ref_blas_lapack umfpack idas superlu superlu_mt bonmin nlopt trilinos deal.ii
    
    Individual libraries/solvers:
    boost            Boost libraries (system, filesystem, thread, python)
    ref_blas_lapack  reference BLAS and Lapack libraries
    umfpack          Umfpack solver
    idas             IDAS solver
    superlu          SuperLU solver
    superlu_mt       SuperLU solver
    bonmin           Bonmin solver
    nlopt            NLopt solver 
    trilinos         Trilinos Amesos and AztecOO solvers
    deal.ii          deal.II finite elements library

CROSS-COMPILATION (GNU/Linux -> Windows):
Prerequisities:
  1. Install the mingw-w64 package from the main Debian repository.
 
  2. Install Python using the binary from the python.org website 
     and copy it to trunk/PythonXY-arch (i.e. Python35-win32). 
 
  3. cmake cross-compilation requires the toolchain file: set it up using -DCMAKE_TOOLCHAIN_FILE=[path_to_toolchain_file].cmake
     Cross-compile .cmake files are provided by daetools and located in the trunk folder.
       cross-compile-i686-w64-mingw32.cmake   file targets a toolchain located in /usr/mingw32-i686 directory.
       cross-compile-x86_64-w64-mingw32.cmake file targets a toolchain located in /usr/mingw32-x86_64 directory.
      
  4. deal.II specific options: 
     The native "expand_instantiations_exe" is required but cannot be run under the build architecture.
     and must be used from the native build.
     Therefore, set up a native deal.II build directory first and run the following command in it:
         make expand_instantiations_exe
     Typically, it is located in the deal.II/common/scripts directory.
     That directory will be added to the PATH environment variable by this script.
     If necessary, modify the line 'export PATH=...:${PATH}' to match the actual location.
      
  5. Boost specific options: 
     boost-python linking will fail. Append the value of:
        ${DAE_CROSS_COMPILE_PYTHON_ROOT}/libs/libpython${PYTHON_MAJOR}${PYTHON_MINOR}.a
     at the end of the failed linking command, re-run it, and manually copy the stage/lib/*.dll(s) to the "daetools/solibs/${PLATFORM}_${HOST_ARCH}" directory.
     Win64 (x86_64-w64-mingw32):
      - Python 2.7 won't compile (probably issues with the MS Universal CRT voodoo mojo)
      - dl and util libraries are missing when compiling with x86_64-w64-mingw32.
        solution: just remove -ldl and -lutil from the linking line.
     
  6. Trilinos specific options
     i686-w64-mingw32 specific:
       1. In the file: 
            - trilinos/packages/teuchos/src/Teuchos_BLAS.cpp  
          "template BLAS<...>" (lines 96-104)
             #ifdef _WIN32
             #ifdef HAVE_TEUCHOS_COMPLEX
                 template class BLAS<long int, std::complex<float> >;
                 template class BLAS<long int, std::complex<double> >;
             #endif
                 template class BLAS<long int, float>;
                 template class BLAS<long int, double>;
             #endif
          should be replaced by "template class BLAS<...>"
       2. In the files:
            - trilinos/packages/ml/src/Utils/ml_epetra_utils.cpp, 
            - trilinos/packages/ml/src/Utils/ml_utils.c
            - trilinos/packages/ml/src/MLAPI/MLAPI_Workspace.cpp:
           the functions "gethostname" and "sleep" do not exist
             a) Add include file:
                   #include <winsock2.h> 
                and if that does not work (getting unresolved _gethostname function in pyTrilinos),
                then comment-out all "gethostname" occurences (they are not important - just for printing some info)
             b) Rename sleep() to Sleep() (if needed, wasn't needed for 10.12.2)
     
     x86_64-w64-mingw32 specific:
       All the same as above. Additionally:
       1. trilinos/packages/teuchos/src/Teuchos_SerializationTraits.hpp
          Comment lines: UndefinedSerializationTraits<T>::notDefined();
       2. trilinos/packages/epetra/src/Epetra_C_wrappers.cpp
          Add lines at the beggining of the file:
            #pragma GCC diagnostic push
            #pragma GCC diagnostic warning "-fpermissive"          
     
Cross compiling notes:
  1. Requirements for Boost:
       --with-python-version 3.5
       --cross-compile-python-root .../trunk/Python35-win32
       --host i686-w64-mingw32
    
  2. The other libraries:
       --host i686-w64-mingw32 (the only necessary)
                
Example cross-compile call:
    sh compile_libraries_linux.sh --with-python-version 3.5 --cross-compile-python-root ~/daetools-win32-cross/trunk/Python35-win32 --host i686-w64-mingw32 boost
    sh compile_libraries_linux.sh --host i686-w64-mingw32 ref_blas_lapack umfpack idas superlu superlu_mt trilinos bonmin nlopt deal.ii
EOF
}

# Default python binary:
PYTHON=python
PYTHON_MAJOR=
PYTHON_MINOR=
PYTHON_VERSION=
PYTHON_INCLUDE_DIR=
PYTHON_LIB_DIR=

TRUNK="$( cd "$( dirname "$0" )" && pwd )"
HOST_ARCH=`uname -m`
PLATFORM=`uname -s`

if [ ${PLATFORM} = "Darwin" ]; then
  args=
else
  args=`getopt -a -o "h" -l "help,with-python-binary:,with-python-version:,cross-compile-python-root:,configure,build,clean:,host:" -n "compile_libraries_linux" -- $*`
fi
# daetools specific compiler flags
DAE_COMPILER_FLAGS="-fPIC"
BOOST_MACOSX_FLAGS=

DO_CONFIGURE="no"
DO_BUILD="no"
DO_CLEAN="no"

DAE_IF_CROSS_COMPILING=0
DAE_CROSS_COMPILE_FLAGS=
DAE_CROSS_COMPILER_PREFIX=
DAE_CROSS_COMPILE_TOOLCHAIN_FILE=
DAE_CROSS_COMPILE_PYTHON_ROOT=

# Process options
for i; do
  case "$i" in
    -h|--help)  usage
                exit 1
                ;;
                  
    --with-python-binary)  PYTHON=$2
                           shift ; shift 
                           ;;
                            
    --with-python-version )  PYTHON=python$2
                             PYTHON_MAJOR=${2%.*}
                             PYTHON_MINOR=${2##*.}
                             shift ; shift 
                             ;;
                                    
    --cross-compile-python-root)  DAE_CROSS_COMPILE_PYTHON_ROOT=$2
                                  shift ; shift 
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

    --host) DAE_IF_CROSS_COMPILING=1
            DAE_CROSS_COMPILER=$2
            DAE_CROSS_COMPILER_PREFIX="$2-"
            DAE_CROSS_COMPILE_FLAGS="--host=$2"
            DAE_CROSS_COMPILE_TOOLCHAIN_FILE="-DCMAKE_TOOLCHAIN_FILE=${TRUNK}/cross-compile-$2.cmake"
            DAE_COMPILER_FLAGS=
            HOST_ARCH="win32"
            PLATFORM="Windows"

            shift ; shift
            ;;

    --) shift; break 
       ;;
  esac
done

if [ ${PLATFORM} = "Darwin" ]; then
  Ncpu=$(/usr/sbin/system_profiler -detailLevel full SPHardwareDataType | awk '/Total Number Of Cores/ {print $5};')
  # If there are problems with memory and speed of compilation set:
  # Ncpu=1
elif [ ${PLATFORM} = "Linux" ]; then
  Ncpu=`cat /proc/cpuinfo | grep processor | wc -l`
else
  Ncpu=4
fi

if [ "${DAE_IF_CROSS_COMPILING}" = "0" ]; then
    PYTHON_MAJOR=`${PYTHON} -c "import sys; print(sys.version_info[0])"`
    PYTHON_MINOR=`${PYTHON} -c "import sys; print(sys.version_info[1])"`
    PYTHON_INCLUDE_DIR=`${PYTHON} -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())"`
    PYTHON_LIB_DIR=`${PYTHON} -c "import sys; print(sys.prefix)"`/lib
fi
PYTHON_VERSION=$PYTHON_MAJOR.$PYTHON_MINOR

SOLIBS_DIR="${TRUNK}/daetools-package/daetools/solibs/${PLATFORM}_${HOST_ARCH}"
if [ ! -e ${SOLIBS_DIR} ]; then
    mkdir ${SOLIBS_DIR}
fi

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

elif [ ${PLATFORM} = "Linux" ]; then
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

export DAE_CROSS_COMPILER_PREFIX

DAE_UMFPACK_INSTALL_DIR="${TRUNK}/umfpack/build"
export DAE_UMFPACK_INSTALL_DIR

vBOOST=1.52.0
vBOOST_=1_52_0
vBONMIN=1.7.4
vLAPACK=3.4.1
vSUPERLU=4.1
vSUPERLU_MT=2.0
vNLOPT=2.4.1
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
vDEALII=8.4.1

BOOST_BUILD_ID=daetools-py${PYTHON_MAJOR}${PYTHON_MINOR}
BOOST_PYTHON_BUILD_ID=

BOOST_HTTP=http://sourceforge.net/projects/boost/files/boost
LAPACK_HTTP=http://www.netlib.org/lapack
DAETOOLS_HTTP=http://sourceforge.net/projects/daetools/files/gnu-linux-libs
IDAS_HTTP=${DAETOOLS_HTTP}
BONMIN_HTTP=http://www.coin-or.org/download/source/Bonmin
SUPERLU_HTTP=http://crd.lbl.gov/~xiaoye/SuperLU
#TRILINOS_HTTP=http://trilinos.csbsju.edu/download/files
TRILINOS_HTTP=http://sourceforge.net/projects/daetools/files/gnu-linux-libs
NLOPT_HTTP=http://ab-initio.mit.edu/nlopt
METIS_HTTP=http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis
UMFPACK_HTTP=http://www.cise.ufl.edu/research/sparse/umfpack
AMD_HTTP=http://www.cise.ufl.edu/research/sparse/amd
CHOLMOD_HTTP=http://www.cise.ufl.edu/research/sparse/cholmod
CAMD_HTTP=http://www.cise.ufl.edu/research/sparse/camd
COLAMD_HTTP=http://www.cise.ufl.edu/research/sparse/colamd
CCOLAMD_HTTP=http://www.cise.ufl.edu/research/sparse/ccolamd
SUITESPARSE_CONFIG_HTTP=http://www.cise.ufl.edu/research/sparse/UFconfig
LIBMESH_HTTP=http://sourceforge.net/projects/libmesh/files/libmesh
DEALII_HTTP=https://github.com/dealii/dealii/releases/download

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
        ref_blas_lapack)  ;;
        openblas)         ;;
        umfpack)          ;;
        idas)             ;;
        trilinos)         ;;
        superlu)          ;;
        superlu_mt)       ;;
        bonmin)           ;;
        nlopt)            ;; 
        libmesh)          ;;
        deal.ii)          ;;
        *) echo Unrecognized solver: "$solver"
        exit
        ;;
    esac
    done
fi
echo ""
echo "###############################################################################"
echo "Proceed with the following options:"
echo "  - Python:                       ${PYTHON}"
echo "  - Python version:               ${PYTHON_MAJOR}${PYTHON_MINOR}"
echo "  - Python include dir:           ${PYTHON_INCLUDE_DIR}"
echo "  - Python lib dir:               ${PYTHON_LIB_DIR}"
echo "  - Platform:                     $PLATFORM"
echo "  - Architecture:                 $HOST_ARCH"
echo "  - Additional compiler flags:    ${DAE_COMPILER_FLAGS}"
echo "  - Cross-compile flags:          ${DAE_CROSS_COMPILE_FLAGS}"
echo "  - Number of threads:            ${Ncpu}"
echo "  - Projects to compile:          $@"
echo "     + Configure:  [$DO_CONFIGURE]"
echo "     + Build:      [$DO_BUILD]"
echo "     + Clean:      [$DO_CLEAN]"
echo "###############################################################################"
echo ""

# ACHTUNG! cd to TRUNK (in case the script is called from some other folder)
cd "${TRUNK}"

#######################################################
#                       BOOST                         #
#######################################################
configure_boost() 
{
  if [ -e boost${PYTHON_VERSION} ]; then
    rm -r boost${PYTHON_VERSION}
  fi
  echo ""
  echo "[*] Setting-up boost"
  echo ""
  if [ ! -e boost_${vBOOST_}.tar.gz ]; then
    wget ${BOOST_HTTP}/${vBOOST}/boost_${vBOOST_}.tar.gz
  fi
  tar -xzf boost_${vBOOST_}.tar.gz
  mv boost_${vBOOST_} boost${PYTHON_VERSION}
  cd boost${PYTHON_VERSION}
  sh bootstrap.sh --with-python=${PYTHON}
  
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_boost() 
{
  if [ "${DAE_IF_CROSS_COMPILING}" = "1" ]; then
    cd boost${PYTHON_VERSION}
    echo ""
    echo "[*] Building boost"
    echo ""
    
    BOOST_USER_CONFIG=~/user-config.jam
    
    #GCC_CROSS="${DAE_CROSS_COMPILER}-g++ -Wl,${DAE_CROSS_COMPILE_PYTHON_ROOT}/libs/libpython${PYTHON_MAJOR}${PYTHON_MINOR}.a"
    echo "using gcc : : ${DAE_CROSS_COMPILER}-g++ ;"                                              > ${BOOST_USER_CONFIG}
    echo "using python"                                                                          >> ${BOOST_USER_CONFIG}
    echo "    : ${PYTHON_MAJOR}.${PYTHON_MINOR}"                                                 >> ${BOOST_USER_CONFIG}
    echo "    : "                                                                                >> ${BOOST_USER_CONFIG}
    echo "    : ${DAE_CROSS_COMPILE_PYTHON_ROOT}/include "                                       >> ${BOOST_USER_CONFIG}
    echo "    : ${DAE_CROSS_COMPILE_PYTHON_ROOT}/libs/libpython${PYTHON_MAJOR}${PYTHON_MINOR}.a" >> ${BOOST_USER_CONFIG}
    echo "    : <toolset>gcc"                                                                    >> ${BOOST_USER_CONFIG}
    echo "    ;"                                                                                 >> ${BOOST_USER_CONFIG}
       
    # https://stackoverflow.com/questions/3778370/python-extensions-for-win64-via-gcc
    # There is a mechanism in Python to prevent linking a module against the wrong version of the library. 
    # The Py_InitModule4 function is renamed to Py_InitModule4_64 (via a macro) when the library / module is compiled 
    # for a 64-bit architecture (see modsupport.h) :
    #    #if SIZEOF_SIZE_T != SIZEOF_INT
    #    /* On a 64-bit system, rename the Py_InitModule4 so that 2.4
    #       modules cannot get loaded into a 2.5 interpreter */
    #    #define Py_InitModule4 Py_InitModule4_64
    #    #endif
    if [ "${DAE_CROSS_COMPILER}" = "x86_64-w64-mingw32" ]; then
      export CXX=MS_WIN64
      export CPPFLAGS=MS_WIN64
    fi
    export CPLUS_INCLUDE_PATH=${DAE_CROSS_COMPILE_PYTHON_ROOT}/include
    
    ./bjam --build-dir=./build --debug-building --layout=system --buildid=${BOOST_BUILD_ID} \
           --with-date_time --with-system --with-filesystem --with-regex --with-serialization --with-thread \
           toolset=gcc target-os=windows threadapi=win32 \
           variant=release link=shared threading=multi runtime-link=shared ${BOOST_MACOSX_FLAGS}

    cp -a stage/lib/libboost_system-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}*       ${SOLIBS_DIR}
    cp -a stage/lib/libboost_thread_win32-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}* ${SOLIBS_DIR}
    cp -a stage/lib/libboost_filesystem-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}*   ${SOLIBS_DIR}

    # Achtung, Achtung!
    # The following will fail at the linking phase!
    # Redo the link with the value of: 
    #     ${DAE_CROSS_COMPILE_PYTHON_ROOT}/libs/libpython${PYTHON_MAJOR}${PYTHON_MINOR}.a
    # appended at the end of the linking line, and manually copy the .dll to the "daetools/solibs/${PLATFORM}_${HOST_ARCH}" directory.
    ./bjam --build-dir=./build --debug-building --layout=system --buildid=${BOOST_BUILD_ID} \
           --with-python \
           toolset=gcc target-os=windows threadapi=win32 \
           variant=release link=shared threading=multi runtime-link=shared ${BOOST_MACOSX_FLAGS}
    
    # Restore the variables
    export CXX=
    export CPPFLAGS=
    export CPLUS_INCLUDE_PATH=
    
    LIBBOOST_PYTHON_SUF="${PYTHON_MAJOR}"
    if [ "${PYTHON_MAJOR}" = "2" ]; then
      LIBBOOST_PYTHON_SUF=""
    fi
    cp -a stage/lib/libboost_python${LIBBOOST_PYTHON_SUF}-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}*  ${SOLIBS_DIR}
  
  else # regular compiler (not a cross-compiler)
    
    cd boost${PYTHON_VERSION}
    echo ""
    echo "[*] Building boost"
    echo ""
    
    BOOST_USER_CONFIG=~/user-config.jam
    echo "using python"                           >  ${BOOST_USER_CONFIG}
    echo "    : ${PYTHON_MAJOR}.${PYTHON_MINOR}"  >> ${BOOST_USER_CONFIG}
    echo "    : ${PYTHON}"                        >> ${BOOST_USER_CONFIG}
    echo "    : ${PYTHON_INCLUDE_DIR}"            >> ${BOOST_USER_CONFIG}
    echo "    : ${PYTHON_LIB_DIR}"                >> ${BOOST_USER_CONFIG}
    echo "    : <toolset>gcc"                     >> ${BOOST_USER_CONFIG}
    echo "    ;"                                  >> ${BOOST_USER_CONFIG}
    
    ./bjam --build-dir=./build --debug-building --layout=system --buildid=${BOOST_BUILD_ID} \
           --with-date_time --with-system --with-filesystem --with-regex --with-serialization --with-thread --with-python \
           variant=release link=shared threading=multi runtime-link=shared ${BOOST_MACOSX_FLAGS}

    LIBBOOST_PYTHON_SUF="${PYTHON_MAJOR}"
    if [ "${PYTHON_MAJOR}" = "2" ]; then
      LIBBOOST_PYTHON_SUF=""
    fi

    cp -a stage/lib/libboost_python${LIBBOOST_PYTHON_SUF}-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}*  ${SOLIBS_DIR}
    cp -a stage/lib/libboost_system-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}*                        ${SOLIBS_DIR}
    cp -a stage/lib/libboost_thread-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}*                        ${SOLIBS_DIR}
    cp -a stage/lib/libboost_filesystem-${BOOST_BUILD_ID}${BOOST_PYTHON_BUILD_ID}*                    ${SOLIBS_DIR}

  fi
  
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_boost()
{
  echo ""
  echo "[*] Cleaning boost..."
  echo ""
  cd boost${PYTHON_VERSION}
  ./bjam --clean
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                   OpenBLAS                          #
#######################################################
configure_openblas() 
{
  if [ "${DAE_IF_CROSS_COMPILING}" = 1 ]; then
    echo "OpenBLAS not configured for cross-compiling at the moment"
    exit
  fi
  
  if [ -e openblas ]; then
    rm -r openblas
  fi
  
  echo ""
  echo "Setting-up openblas..."
  echo ""
  if [ ! -e openblas-${vOPENBLAS}.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/openblas-${vOPENBLAS}.tar.gz
  fi
  if [ ! -e Makefile-openblas.rule ]; then
    wget ${DAETOOLS_HTTP}/Makefile-openblas.rule
  fi
  tar -xzf openblas-${vOPENBLAS}.tar.gz
  cp Makefile-openblas.rule openblas/Makefile.rule
  cd openblas
  mkdir build
  cd "${TRUNK}"
  
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_openblas() 
{
  cd openblas
  echo ""
  echo "[*] Building openblas..."
  echo ""
  make -j${Ncpu} libs
  make 
  make prefix=build install
  cp -a libopenblas_daetools* ${SOLIBS_DIR}
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_openblas()
{
  echo ""
  echo "[*] Cleaning openblas..."
  echo ""
  cd openblas
  make clean
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#            Reference BLAS and LAPACK                #
#######################################################
configure_ref_blas_lapack() 
{
  if [ -e lapack ]; then
    rm -r lapack
  fi
  
  echo ""
  echo "[*] Setting-up reference blas & lapack..."
  echo ""
  if [ ! -e lapack-${vLAPACK}.tgz ]; then
    wget ${LAPACK_HTTP}/lapack-${vLAPACK}.tgz
  fi
  if [ ! -e daetools_lapack_make.inc ]; then
    wget ${DAETOOLS_HTTP}/daetools_lapack_make.inc
  fi
  tar -xzf lapack-${vLAPACK}.tgz
  mv lapack-${vLAPACK} lapack
  
  cd lapack
  
  cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_INSTALL_PREFIX:STRING="${TRUNK}/lapack" \
    -DBUILD_DOUBLE:BOOL=ON \
    -DBUILD_STATIC_LIBS:BOOL=ON \
    -DCMAKE_Fortran_FLAGS:STRING="${DAE_COMPILER_FLAGS}" \
    ${DAE_CROSS_COMPILE_TOOLCHAIN_FILE}
  
  cd "${TRUNK}"
  
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_ref_blas_lapack() 
{
  cd lapack
  echo ""
  echo "[*] Building reference blas & lapack..."
  echo ""
  make -j${Ncpu} lapack
  make -j${Ncpu} blas
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_ref_blas_lapack()
{
  echo ""
  echo "[*] Cleaning reference blas & lapack..."
  echo ""
  cd lapack
  make clean
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                      UMFPACK                        #
#######################################################
configure_umfpack() 
{
  if [ -e umfpack ]; then
    rm -rf umfpack
  fi
  echo ""
  echo "[*] Setting-up umfpack and friends..."
  echo ""
  if [ ! -e SuiteSparse_config-${vSUITESPARSE_CONFIG}.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/SuiteSparse_config-${vSUITESPARSE_CONFIG}.tar.gz
  fi 
  if [ ! -e CHOLMOD-${vCHOLMOD}.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/CHOLMOD-${vCHOLMOD}.tar.gz
  fi
  if [ ! -e AMD-${vAMD}.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/AMD-${vAMD}.tar.gz
  fi
  if [ ! -e CAMD-${vCAMD}.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/CAMD-${vCAMD}.tar.gz
  fi
  if [ ! -e COLAMD-${vCOLAMD}.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/COLAMD-${vCOLAMD}.tar.gz
  fi
  if [ ! -e CCOLAMD-${vCCOLAMD}.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/CCOLAMD-${vCCOLAMD}.tar.gz
  fi
  if [ ! -e UMFPACK-${vUMFPACK}.tar.gz ]; then
    wget ${DAETOOLS_HTTP}/UMFPACK-${vUMFPACK}.tar.gz
  fi
  if [ ! -e SuiteSparse_config.mk ]; then
    wget ${DAETOOLS_HTTP}/SuiteSparse_config.mk
  fi
  #if [ ! -e metis-${vMETIS}.tar.gz ]; then
  #  wget ${METIS_HTTP}/metis-${vMETIS}.tar.gz
  #fi
  #if [ ! -e metis.h ]; then
  #  wget ${DAETOOLS_HTTP}/metis.h
  #fi
  #if [ ! -e Makefile-CHOLMOD.patch ]; then
  #  wget ${DAETOOLS_HTTP}/Makefile-CHOLMOD.patch
  #fi
  
  mkdir umfpack
  cd umfpack
  tar -xzf ../SuiteSparse_config-${vSUITESPARSE_CONFIG}.tar.gz
  tar -xzf ../CHOLMOD-${vCHOLMOD}.tar.gz
  tar -xzf ../AMD-${vAMD}.tar.gz
  tar -xzf ../CAMD-${vCAMD}.tar.gz
  tar -xzf ../COLAMD-${vCOLAMD}.tar.gz
  tar -xzf ../CCOLAMD-${vCCOLAMD}.tar.gz
  tar -xzf ../UMFPACK-${vUMFPACK}.tar.gz
  cp ../SuiteSparse_config.mk SuiteSparse_config

  #tar -xzf ../metis-${vMETIS}.tar.gz
  #cp ../metis.h metis-${vMETIS}/include
  # Apply Metis 5.1.0 patch for CHOLMOD
  #cd CHOLMOD/Lib
  #patch < ../../../Makefile-CHOLMOD.patch
  
  mkdir build
  mkdir build/lib
  mkdir build/include
  cd "${TRUNK}"
  
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_umfpack() 
{
#  cd umfpack/metis-${vMETIS}
#  echo "[*] Building metis..."
#  echo "make config prefix=${DAE_UMFPACK_INSTALL_DIR} shared=0 -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}""
#  make config prefix=${DAE_UMFPACK_INSTALL_DIR} -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}"
#  make install
#  echo ""
#  echo "[*] Done!"
#  echo ""
#  cd "${TRUNK}"

  cd umfpack/SuiteSparse_config
  echo ""
  echo "[*] Building suitesparseconfig..."
  echo ""
  echo "make CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar RANLIB=${DAE_CROSS_COMPILER_PREFIX}ranlib CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  make clean
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"

  cd umfpack/AMD
  echo ""
  echo "[*] Building amd..."
  echo ""
  #echo "make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar RANLIB=${DAE_CROSS_COMPILER_PREFIX}ranlib CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"

  cd umfpack/CAMD
  echo ""
  echo "[*] Building camd..."
  echo ""
  #echo "make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar RANLIB=${DAE_CROSS_COMPILER_PREFIX}ranlib CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  echo "[*] Done!"
  cd "${TRUNK}"

  cd umfpack/COLAMD
  echo ""
  echo "[*] Building colamd..."
  echo ""
  #echo "make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar RANLIB=${DAE_CROSS_COMPILER_PREFIX}ranlib CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"

  cd umfpack/CCOLAMD
  echo ""
  echo "[*] Building ccolamd..."
  echo ""
  #echo "make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar RANLIB=${DAE_CROSS_COMPILER_PREFIX}ranlib CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"

  cd umfpack/CHOLMOD
  echo ""
  echo "[*] Building cholmod..."
  echo ""
  #echo "make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar RANLIB=${DAE_CROSS_COMPILER_PREFIX}ranlib CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"

  cd umfpack/UMFPACK
  echo ""
  echo "[*] Building umfpack..."
  echo ""
  #echo "make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library"
  make -j${Ncpu} CC=${DAE_CROSS_COMPILER_PREFIX}gcc CXX=${DAE_CROSS_COMPILER_PREFIX}g++ AR=${DAE_CROSS_COMPILER_PREFIX}ar RANLIB=${DAE_CROSS_COMPILER_PREFIX}ranlib CFLAGS="${DAE_COMPILER_FLAGS} -O3" CPPFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}" library
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_umfpack()
{
  echo ""
  echo "[*] Cleaning umfpack..."
  echo ""
  #cd umfpack/metis-${vMETIS}
  #make clean
  #cd "${TRUNK}"
  
  cd umfpack/SuiteSparse_config
  make clean
  cd "${TRUNK}"
  
  cd umfpack/AMD
  make clean
  cd "${TRUNK}"
  
  cd umfpack/CAMD
  make clean
  cd "${TRUNK}"
  
  cd umfpack/COLAMD
  make clean
  cd "${TRUNK}"
  
  cd umfpack/CCOLAMD
  make clean
  cd "${TRUNK}"
  
  cd umfpack/CHOLMOD
  make clean
  cd "${TRUNK}"
  
  cd umfpack/UMFPACK
  make clean
  cd "${TRUNK}"
  
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                       IDAS                          #
#######################################################
configure_idas() 
{
  if [ "${DAE_IF_CROSS_COMPILING}" = "1" ]; then
    echo ""
    echo "Sundials IDAS:"
    echo "  Cross-compilation must be performed using the cmake-gui and the ${DAE_CROSS_COMPILE_TOOLCHAIN_FILE} cross compile toolchain"
    exit 1
  fi
    
  if [ -e idas ]; then
    rm -r idas
  fi
  echo ""
  echo "[*] Setting-up idas..."
  echo ""
  if [ ! -e idas-${vIDAS}.tar.gz ]; then
    wget ${IDAS_HTTP}/idas-${vIDAS}.tar.gz
  fi
  if [ ! -e idasMakefile.in.patch ]; then
    wget ${DAETOOLS_HTTP}/idasMakefile.in.patch
  fi
  tar -xzf idas-${vIDAS}.tar.gz
  mv idas-${vIDAS} idas
  cd idas
  patch < ../idasMakefile.in.patch
  
  ./configure ${DAE_CROSS_COMPILE_FLAGS} --prefix=${TRUNK}/idas/build --with-pic --enable-mpi --enable-static=yes --enable-shared=no --enable-examples --enable-lapack F77=gfortran CFLAGS="${DAE_COMPILER_FLAGS} -O3" FFLAGS="${DAE_COMPILER_FLAGS}"
  
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_idas() 
{
  cd idas
  echo ""
  echo "[*] Building idas..."
  echo ""
  make all
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_idas()
{
  echo ""
  echo "[*] Cleaning idas..."
  echo ""
  cd idas
  make clean
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                     SUPERLU                         #
#######################################################
configure_superlu() 
{
  if [ -e superlu ]; then
    rm -r superlu
  fi
  echo ""
  echo "[*] Setting-up superlu..."
  echo ""
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
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_superlu() 
{
  cd superlu
  echo ""
  echo "[*] Building superlu..."
  echo ""
  make superlulib DAE_CROSS_COMPILER_PREFIX=${DAE_CROSS_COMPILER_PREFIX} DAE_COMPILER_FLAGS="${DAE_COMPILER_FLAGS}"
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_superlu()
{
  echo ""
  echo "[*] Cleaning superlu..."
  echo ""
  cd superlu
  make cleanlib
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                    SUPERLU_MT                       #
#######################################################
configure_superlu_mt() 
{
  if [ -e superlu_mt ]; then
    rm -r superlu_mt
  fi
  echo ""
  echo "[*] Setting-up superlu_mt..."
  echo ""
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
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_superlu_mt() 
{
  cd superlu_mt
  echo ""
  echo "[*] Building superlu_mt..."
  echo ""
  
  make lib DAE_CROSS_COMPILER_PREFIX=${DAE_CROSS_COMPILER_PREFIX} DAE_COMPILER_FLAGS="${DAE_COMPILER_FLAGS}"
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_superlu_mt()
{
  echo ""
  echo "[*] Cleaning superlu_mt..."
  echo ""
  cd superlu_mt
  make cleanlib
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                      BONMIN                         #
#######################################################
configure_bonmin() 
{
  if [ -e bonmin ]; then
    rm -r bonmin
  fi
  echo ""
  echo "[*] Setting-up bonmin..."
  echo ""
  if [ ! -e Bonmin-${vBONMIN}.zip ]; then
    wget ${BONMIN_HTTP}/Bonmin-${vBONMIN}.zip
  fi
  unzip Bonmin-${vBONMIN}.zip
  rm -rf bonmin/Bonmin-${vBONMIN}
  mv Bonmin-${vBONMIN} bonmin
  
  cd bonmin
  
  cd ThirdParty/Mumps
  sh get.Mumps
  cd ../..
  
  cd ThirdParty/Blas
  sh get.Blas
  cd ../..
  
  cd ThirdParty/Lapack
  sh get.Lapack
  cd ../..
  
  mkdir -p build
  cd build
  ../configure ${DAE_CROSS_COMPILE_FLAGS} --disable-dependency-tracking --enable-shared=no --enable-static=yes ARCHFLAGS="${DAE_COMPILER_FLAGS}" CFLAGS="${DAE_COMPILER_FLAGS}" CXXFLAGS="${DAE_COMPILER_FLAGS}" FFLAGS="${DAE_COMPILER_FLAGS}" LDFLAGS="${DAE_COMPILER_FLAGS}"
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_bonmin() 
{
  cd bonmin/build
  echo "[*] Building bonmin..."
  make -j${Ncpu}
  #make test
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_bonmin()
{
  echo ""
  echo "[*] Cleaning bonmin..."
  echo ""
  cd bonmin/build
  make clean
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                      NLOPT                          #
#######################################################
configure_nlopt() 
{
  if [ -e nlopt ]; then
    rm -r nlopt
  fi
  echo ""
  echo "[*] Setting-up nlopt..."
  echo ""
  if [ ! -e nlopt-${vNLOPT}.tar.gz ]; then
    wget ${NLOPT_HTTP}/nlopt-${vNLOPT}.tar.gz
  fi
  tar -xzf nlopt-${vNLOPT}.tar.gz
  mv nlopt-${vNLOPT} nlopt
  cd nlopt
  mkdir build
  cd build
  ../configure ${DAE_CROSS_COMPILE_FLAGS} --disable-dependency-tracking -prefix=${TRUNK}/nlopt/build CFLAGS="${DAE_COMPILER_FLAGS}" CXXFLAGS="${DAE_COMPILER_FLAGS}" FFLAGS="${DAE_COMPILER_FLAGS}"
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_nlopt() 
{
  cd nlopt/build
  echo "[*] Building nlopt..."
  make
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_nlopt()
{
  echo ""
  echo "[*] Cleaning nlopt..."
  echo ""
  cd nlopt/build
  make clean
  cd "${TRUNK}"
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

  UMFPACK_INCLUDE_DIR="${DAE_UMFPACK_INSTALL_DIR}/include"
  
  if [ "${DAE_IF_CROSS_COMPILING}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS ${DAE_CROSS_COMPILE_TOOLCHAIN_FILE} -DHAVE_GCC_ABI_DEMANGLE_EXITCODE=0 "
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
    -DTrilinos_ENABLE_Zoltan:BOOL=OFF \
    -DAmesos_ENABLE_SuperLU:BOOL=ON \
    -DIfpack_ENABLE_SuperLU:BOOL=ON \
    -DTeuchos_ENABLE_COMPLEX:BOOL=OFF \
    -DTPL_SuperLU_INCLUDE_DIRS:FILEPATH=${TRUNK}/superlu/SRC \
    -DTPL_SuperLU_LIBRARIES:STRING=superlu_4.1 \
    -DTPL_ENABLE_UMFPACK:BOOL=ON \
    -DTPL_UMFPACK_INCLUDE_DIRS:FILEPATH=${UMFPACK_INCLUDE_DIR} \
    -DTPL_UMFPACK_LIBRARIES:STRING=umfpack \
    -DTPL_BLAS_LIBRARIES:STRING="${TRUNK}/lapack/lib/libblas.a -lgfortran" \
    -DTPL_LAPACK_LIBRARIES:STRING="${TRUNK}/lapack/lib/liblapack.a ${TRUNK}/lapack/lib/libblas.a -lgfortran" \
    -DTPL_ENABLE_MPI:BOOL=OFF \
    -DDART_TESTING_TIMEOUT:STRING=600 \
    -DCMAKE_INSTALL_PREFIX:PATH=. \
    -DCMAKE_CXX_FLAGS:STRING="-DNDEBUG ${DAE_COMPILER_FLAGS}" \
    -DCMAKE_C_FLAGS:STRING="-DNDEBUG ${DAE_COMPILER_FLAGS}" \
    -DCMAKE_Fortran_FLAGS:STRING="-DNDEBUG ${DAE_COMPILER_FLAGS}" \
    $EXTRA_ARGS \
    ${TRILINOS_HOME}
    
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_trilinos() 
{
# Note: there may be problems while cross-compiling with some of the files:
#  i686-w64-mingw32:
#  1. trilinos/packages/teuchos/src/Teuchos_BLAS.cpp
#     template BLAS<...> should be replaced by "template class BLAS<...>" (lines 96-104):
#      #ifdef _WIN32
#      #  ifdef HAVE_TEUCHOS_COMPLEX
#          template class BLAS<long int, std::complex<float> >;
#          template class BLAS<long int, std::complex<double> >;
#      #  endif
#          template class BLAS<long int, float>;
#          template class BLAS<long int, double>;
#      #endif      
#  2. trilinos/packages/ml/src/Utils/ml_epetra_utils.cpp, 
#     trilinos/packages/ml/src/Utils/ml_utils.c
#     trilinos/packages/ml/src/MLAPI/MLAPI_Workspace.cpp:
#     Functions gethostname and sleep do not exist
#      - Add include #include <winsock2.h> or if that does not work (unresolved _gethostname function)
#        then comment-out all gethostname occurences (not important - just for printing some info)
#      - Rename sleep() to Sleep() (if needed, wasn't needed for 10.12.2)
#
#  x86_64-w64-mingw32:
#

  cd trilinos/build
  echo "[*] Building trilinos..."
  make -j${Ncpu}
  make install
  echo ""
  echo "[*] Done!"
  echo ""
  cd "${TRUNK}"
}

clean_trilinos()
{
  echo ""
  echo "[*] Cleaning trilinos..."
  echo ""
  cd trilinos/build
  make clean
  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

#######################################################
#                     deal.II                         #
#######################################################
configure_dealii() 
{
  if [ "${DAE_IF_CROSS_COMPILING}" = "1" ]; then
    # Cross-compilation requires the toolchain file: set it up using: -DCMAKE_TOOLCHAIN_FILE=path_to_toolchain_file.cmake
    # 
    # The problem with cross-compilation is that expand_instantiations_exe cannot be run under the build architecture.
    # deal.II/bin/expand_instantiations.exe is required but not existing and must be used from the native build.
    # Therefore, set up a native deal.II build directory first and run the following command in it:
    #    make expand_instantiations_exe
    # Locate the expand_instantiations executable (it usually resides under ${CMAKE_BINARY_DIR}/common/scripts
    # and export its location using the PATH environment variable.
    export PATH=${TRUNK}/../../daetools/trunk/deal.II/bin:${PATH}
        
    DEALII_CROSS_COMPILE_OPTIONS="${DAE_CROSS_COMPILE_TOOLCHAIN_FILE}"
  fi
  
  if [ -e deal.II ]; then
    rm -r deal.II
  fi
  echo ""
  echo "[*] Setting-up deal.II..."
  echo ""
  if [ ! -e dealii-${vDEALII}.tar.gz ]; then
    wget ${DEALII_HTTP}/v${vDEALII}/dealii-${vDEALII}.tar.gz
  fi
  
  tar -xzf dealii-${vDEALII}.tar.gz

  # The line below should be enabled for newer versions of deal.ii
  mv dealii-${vDEALII} deal.II

  cd deal.II
  mkdir build
  cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DDEAL_II_PACKAGE_NAME:STRING=deal.II-daetools \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DDEAL_II_ALLOW_AUTODETECTION=OFF \
    -DDEAL_II_PREFER_STATIC_LIBS:BOOL=ON \
    -DDEAL_II_COMPONENT_EXAMPLES:BOOL=ON \
    -DDEAL_II_WITH_LAPACK:BOOL=OFF \
    -DDEAL_II_WITH_THREADS:BOOL=OFF \
    -DDEAL_II_WITH_MPI:BOOL=OFF \
    -DDEAL_II_WITH_P4EST:BOOL=OFF \
    -DDEAL_II_WITH_SLEPC:BOOL=OFF \
    -DDEAL_II_WITH_BZIP2:BOOL=OFF \
    -DDEAL_II_WITH_HDF5:BOOL=OFF \
    -DDEAL_II_WITH_OPENCASCADE:BOOL=OFF \
    -DDEAL_II_WITH_NETCDF:BOOL=OFF \
    -DDEAL_II_WITH_PETSC:BOOL=OFF \
    -DDEAL_II_WITH_ARPACK:BOOL=OFF \
    -DDEAL_II_WITH_UMFPACK:BOOL=OFF \
    -DDEAL_II_WITH_TRILINOS:BOOL=OFF \
    -DDEAL_II_WITH_METIS:BOOL=OFF \
    -DDEAL_II_WITH_ZLIB:BOOL=OFF \
    -DDEAL_II_WITH_MUMPS:BOOL=OFF \
    -DDEAL_II_WITH_MUPARSER:BOOL=OFF \
    -DDEAL_II_COMPONENT_PARAMETER_GUI:BOOL=OFF \
    -DDEAL_II_COMPONENT_MESH_CONVERTER:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:STRING="${TRUNK}/deal.II/build" \
    -DDEAL_II_CMAKE_CXX_FLAGS:STRING="${DAE_COMPILER_FLAGS} " \
    -DDEAL_II_CMAKE_C_FLAGS:STRING="${DAE_COMPILER_FLAGS} " \
    ${DEALII_CROSS_COMPILE_OPTIONS}

  cd "${TRUNK}"
  echo ""
  echo "[*] Done!"
  echo ""
}

compile_dealii() 
{
  cd deal.II
  echo ""
  echo "[*] Building deal.II..."
  echo ""
  
  make expand_instantiations_exe
  make -j${Ncpu} install
  
  # Nota bene:
  #   No need to copy anything since we are producing a static lib (.a)
  # Not anymore!
  if [ ${PLATFORM} = "Darwin" ]; then
    cp -a build/lib/libdeal_II-daetools.dylib.${vDEALII} ${SOLIBS_DIR}
  elif [ ${PLATFORM} = "Linux" ]; then
    cp -a build/lib/libdeal_II-daetools.so.${vDEALII} ${SOLIBS_DIR}/libdeal_II-daetools.so.${vDEALII}
  else
    echo "..."
  fi
  cd "${TRUNK}"
}

clean_dealii()
{
  echo ""
  echo "[*] Cleaning deal.II..."
  echo ""
  cd deal.II
  make clean
  cd "${TRUNK}"
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
                        configure_boost
                        configure_ref_blas_lapack
                        configure_umfpack
                        configure_idas
                        configure_superlu
                        configure_superlu_mt
                        configure_trilinos
                        configure_bonmin
                        configure_nlopt
                        configure_dealii
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_boost
                        compile_ref_blas_lapack
                        compile_umfpack
                        compile_idas
                        compile_superlu
                        compile_superlu_mt
                        compile_trilinos
                        compile_bonmin
                        compile_nlopt
                        compile_dealii
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_boost
                        clean_ref_blas_lapack
                        clean_umfpack
                        clean_idas
                        clean_superlu
                        clean_superlu_mt
                        clean_trilinos
                        clean_bonmin
                        clean_nlopt
                        clean_dealii
                      fi
                      ;;
    
    boost)            if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_boost
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_boost
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_boost
                      fi
                      ;;
                      
    ref_blas_lapack)  if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_ref_blas_lapack
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_ref_blas_lapack
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_ref_blas_lapack
                      fi
                      ;;
                      
    openblas)         if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_openblas
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_openblas
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_openblas
                      fi
                      ;;
                      
    umfpack)          if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_umfpack
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_umfpack
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_umfpack
                      fi
                      ;;
                      
    idas)             if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_idas
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
                      
    superlu)          if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_superlu
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_superlu
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_superlu
                      fi
                      ;;
                      
    superlu_mt)       if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_superlu_mt
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_superlu_mt
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_superlu_mt
                      fi
                      ;;
                      
    bonmin)           if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_bonmin
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_bonmin
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_bonmin
                      fi
                      ;;
                      
    nlopt)            if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_nlopt
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_nlopt
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_nlopt
                      fi
                      ;; 
                      
    libmesh)          if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_libmesh
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_libmesh
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_libmesh
                      fi
                      ;; 
                      
    deal.ii)          if [ "${DO_CONFIGURE}" = "yes" ]; then
                        configure_dealii
                      fi
                      
                      if [ "${DO_BUILD}" = "yes" ]; then 
                        compile_dealii
                      fi
                      
                      if [ "${DO_CLEAN}" = "yes" ]; then 
                        clean_dealii
                      fi
                      ;; 
                      
    *) echo Unrecognized solver: "$solver"
       exit
       ;;
  esac
done

cd "${TRUNK}"
