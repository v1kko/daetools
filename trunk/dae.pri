#************************************************************************************
#                 DAE Tools Project: www.daetools.com
#                 Copyright (C) Dragan Nikolic, 2010
#************************************************************************************
# DAE Tools is free software; you can redistribute it and/or modify it under the 
# terms of the GNU General Public License version 3 as published by the Free Software
# Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with the
# DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
#************************************************************************************
DAE_TOOLS_MAJOR = 1
DAE_TOOLS_MINOR = 3
DAE_TOOLS_BUILD = 0

# DAE Tools version (major, minor, build)
VERSION = $${DAE_TOOLS_MAJOR}.$${DAE_TOOLS_MINOR}.$${DAE_TOOLS_BUILD}

QMAKE_CXXFLAGS += -DDAE_MAJOR=$${DAE_TOOLS_MAJOR}
QMAKE_CXXFLAGS += -DDAE_MINOR=$${DAE_TOOLS_MINOR}
QMAKE_CXXFLAGS += -DDAE_BUILD=$${DAE_TOOLS_BUILD}

# daetools always use the current system python version and custom compiled boost libs
# located in ../boost with the libraries in ../boost/stage/lib

# Build universal binaries for MAC OS-X
# There is a problem with ppc64 under OSX 10.6 so it is excluded
# Otherwise ppc64 should be added as well
macx-g++::CONFIG += x86 x86_64

win32::SHARED_LIB_EXT     = dll
linux-g++::SHARED_LIB_EXT = so
macx-g++::SHARED_LIB_EXT  = dylib

win32::SHARED_LIB_PREFIX     =
linux-g++::SHARED_LIB_PREFIX = lib
macx-g++::SHARED_LIB_PREFIX  = lib

win32::SHARED_LIB_APPEND     = pyd
linux-g++::SHARED_LIB_APPEND = so.$${VERSION}
macx-g++::SHARED_LIB_APPEND  = $${VERSION}.dylib

# Set CONFIG += enable_mpi to use MPI libraries

CONFIG(debug, debug|release):message(debug){
    DAE_DEST_DIR = ../debug
    OBJECTS_DIR = debug
}

CONFIG(release, debug|release):message(release){
	DAE_DEST_DIR = ../release
    OBJECTS_DIR = release
}

DESTDIR = $${DAE_DEST_DIR}

win32{
COPY_FILES = copy /y
}
unix{
COPY_FILES = cp -fa
}

#####################################################################################
#                           System + Machine + Python info
#####################################################################################
PYTHON       = python
PYTHON_MAJOR = $$system($${PYTHON} -c \"import sys; print(sys.version_info[0])\")
PYTHON_MINOR = $$system($${PYTHON} -c \"import sys; print(sys.version_info[1])\")

message(python: v$${PYTHON_MAJOR}.$${PYTHON_MINOR} - $${PYTHON})

# Numpy version
NUMPY_VERSION = $$system($${PYTHON} -c \"import numpy; print(\'\'.join(numpy.__version__.split(\'.\')[0:2]))\")

# System := {'Linux', 'Windows', 'Darwin'}
DAE_SYSTEM   = $$system($${PYTHON} -c \"import platform; print(platform.system())\")

# Machine := {'i386', ..., 'i686', 'AMD64'}
MACHINE_COMMAND = "import platform; p={'Linux':platform.machine(), 'Darwin':'universal', 'Windows':'win32'}; machine = p[platform.system()]; print(machine)"
DAE_MACHINE = $$system($${PYTHON} -c \"$${MACHINE_COMMAND}\")

####################################################################################
# Remove all symbol table and relocation information from the executable.
# Necessary to pass lintian test in debian  
####################################################################################
#CONFIG(release, debug|release){
#    unix:QMAKE_LFLAGS += -s
#}

####################################################################################
#                       Compiler flags
####################################################################################
win32::QMAKE_CXXFLAGS += -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX
win32::QMAKE_CXXFLAGS += -DBOOST_ALL_NO_LIB=1
win32::QMAKE_CXXFLAGS += /bigobj

QMAKE_CXXFLAGS_DEBUG += -fno-default-inline

QMAKE_CXXFLAGS_DEBUG += -DDAE_DEBUG
QMAKE_CFLAGS_DEBUG   += -DDAE_DEBUG

#unix::QMAKE_CXXFLAGS += -ansi -pedantic

# Older GCCs complain about -Wno-unused-but-set-variable
linux-g++::QMAKE_CXXFLAGS_WARN_ON += -Wextra -Wno-sign-compare \
                                     -Wno-unused-parameter \
                                     -Wno-unused-variable \
                                     -Wno-unused-but-set-variable

macx-g++::QMAKE_CXXFLAGS_WARN_ON += -Wextra -Wno-sign-compare \
                                    -Wno-unused-parameter \
                                    -Wno-unused-variable
               
unix::QMAKE_CFLAGS_RELEASE   -= -O2
unix::QMAKE_CXXFLAGS_RELEASE -= -O2

unix::QMAKE_CFLAGS_RELEASE   += -O3
unix::QMAKE_CXXFLAGS_RELEASE += -O3

# On some low-RAM machines certain boost.python modules cannot compile
# The workaround is to set the following flags:
#unix::QMAKE_CXXFLAGS += --param ggc-min-expand=30 --param ggc-min-heapsize=8192

# Use SSE for x86 32 bit machines (not used by default)
# When building for Mac-OS we build for all architectures and SSE flags should go away
#QMAKE_CXXFLAGS_RELEASE += -mfpmath=sse -msse -msse2 -msse3

####################################################################################
# Creating .vcproj under windows:
# cd trunk
# qmake -tp vc -r dae.pro
####################################################################################

####################################################################################
#                       single/double precision control
####################################################################################
# SUNDIALS_DOUBLE/SINGLE_PRECISION must be also defined in sundials_config.h
####################################################################################
#QMAKE_CXXFLAGS += -DDAE_SINGLE_PRECISION

####################################################################################
#				                     ARM port
####################################################################################
# Undefined reference to `__sync_fetch_and_add_4' issue):
####################################################################################
# QMAKE_CXXFLAGS += -DBOOST_SP_USE_PTHREADS


#####################################################################################
#                                PYTHON + NUMPY
#####################################################################################
PYTHON_INCLUDE_DIR       = $$system($${PYTHON} -c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\")
PYTHON_SITE_PACKAGES_DIR = $$system($${PYTHON} -c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_lib())\")
unix::PYTHON_LIB_DIR     = $$system($${PYTHON} -c \"import sys; print(sys.prefix)\")/lib
win32::PYTHON_LIB_DIR    = $$system($${PYTHON} -c \"import sys; print(sys.prefix)\")/libs

message(Python dirs: v$${PYTHON_MAJOR}.$${PYTHON_MINOR} - $${PYTHON_INCLUDE_DIR} - $${PYTHON_SITE_PACKAGES_DIR})

win32::NUMPY_INCLUDE_DIR     = $${PYTHON_SITE_PACKAGES_DIR}/numpy/core/include/numpy \
                               $${PYTHON_INCLUDE_DIR}/numpy/core/include/numpy
linux-g++::NUMPY_INCLUDE_DIR = $${PYTHON_SITE_PACKAGES_DIR}/numpy/core/include/numpy \
                               $${PYTHON_INCLUDE_DIR}/numpy/core/include/numpy \
                               /usr/include/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}/numpy \
                               /usr/share/pyshared/numpy/core/include/numpy
macx-g++::NUMPY_INCLUDE_DIR  = $${PYTHON_SITE_PACKAGES_DIR}/numpy/core/include/numpy \
                               $${PYTHON_INCLUDE_DIR}/numpy/core/include/numpy


#####################################################################################
#                                  RT/GFORTRAN
#####################################################################################
# librt does not exist in Windows/MacOS
# gfortran does not exist in MacOS XCode and must be installed separately
#####################################################################################
win32::RT     =
linux-g++::RT = -lrt
macx-g++::RT  =

win32::GFORTRAN     =
linux-g++::GFORTRAN = -lgfortran
macx-g++::GFORTRAN  = -lgfortran


#####################################################################################
#                                    BOOST
#####################################################################################
# Boost version installed must be 1.42+ (asio, system, python, thread, regex)
# Starting with the version 1.2.1 daetools use manually compiled boost libraries.
# The compilation is done in the shell script compile_libraries_linux.sh
#####################################################################################
win32::BOOSTDIR              = ../boost
win32::BOOSTLIBPATH          = $${BOOSTDIR}/stage/lib
win32::BOOST_PYTHON_LIB_NAME = boost_python-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32::BOOST_SYSTEM_LIB_NAME = boost_system-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32::BOOST_THREAD_LIB_NAME = boost_thread-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32::BOOST_PYTHON_LIB      = $${BOOST_PYTHON_LIB_NAME}.lib python$${PYTHON_MAJOR}$${PYTHON_MINOR}.lib
win32::BOOST_LIBS            = $${BOOST_SYSTEM_LIB_NAME}.lib $${BOOST_THREAD_LIB_NAME}.lib

unix::BOOSTDIR              = ../boost
unix::BOOSTLIBPATH          = $${BOOSTDIR}/stage/lib
unix::BOOST_PYTHON_LIB_NAME = boost_python-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::BOOST_SYSTEM_LIB_NAME = boost_system-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::BOOST_THREAD_LIB_NAME = boost_thread-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::BOOST_PYTHON_LIB      = -L$${BOOSTLIBPATH} -l$${BOOST_PYTHON_LIB_NAME} \
                              -L$${PYTHON_LIB_DIR} -lpython$${PYTHON_MAJOR}.$${PYTHON_MINOR} $${RT}
unix::BOOST_LIBS            = -L$${BOOSTLIBPATH} -l$${BOOST_SYSTEM_LIB_NAME} -l$${BOOST_THREAD_LIB_NAME} $${RT}


#####################################################################################
#                                 BLAS/LAPACK
#####################################################################################
win32::BLAS_LAPACK_LIBDIR = ../clapack/LIB/Win32
unix::BLAS_LAPACK_LIBDIR  =

win32::BLAS_LAPACK_LIBS = $${BLAS_LAPACK_LIBDIR}/BLAS_nowrap.lib \
                          $${BLAS_LAPACK_LIBDIR}/clapack_nowrap.lib \
                          $${BLAS_LAPACK_LIBDIR}/libf2c.lib

#unix::BLAS_LAPACK_LIBS = -llapack -lblas -lm
unix::BLAS_LAPACK_LIBS = ../lapack/liblapack.a ../lapack/librefblas.a -lgfortran -lm


#####################################################################################
#                                 SUNDIALS IDAS
#####################################################################################
# ./configure --prefix=${HOME}/Data/daetools/trunk/idas/build --disable-mpi
#             --enable-examples --enable-static=yes --enable-shared=no --with-pic
#             --enable-lapack CFLAGS=-O3
#####################################################################################
SUNDIALS = ../idas/build
SUNDIALS_INCLUDE = $${SUNDIALS}/include
SUNDIALS_LIBDIR = $${SUNDIALS}/lib

win32::SUNDIALS_LIBS = sundials_idas.lib \
                       sundials_nvecserial.lib
unix::SUNDIALS_LIBS = -lsundials_idas \
                      -lsundials_nvecserial


#####################################################################################
#                                  MUMPS
#####################################################################################
MUMPS_DIR  = ../mumps

win32::G95_LIBDIR = c:/g95/lib/gcc-lib/i686-pc-mingw32/4.1.2

MUMPS_LIBDIR   = $${MUMPS_DIR}/lib \
                 $${MUMPS_DIR}/libseq \
                 $${MUMPS_DIR}/blas \
                 $${G95_LIBDIR}

win32::MUMPS_LIBS = blas.lib \
					libmpiseq.lib \
					libdmumps.lib \
					libmumps_common.lib \
					libpord.lib \
					libf95.a \
					libgcc.a
unix::MUMPS_LIBS = -lcoinmumps -lpthread $${RT}


#####################################################################################
#                                  IPOPT
#####################################################################################
IPOPT_DIR = ../bonmin/build

IPOPT_INCLUDE = $${IPOPT_DIR}/include/coin
IPOPT_LIBDIR  = $${IPOPT_DIR}/lib

win32::IPOPT_LIBS = libCoinBlas.lib libCoinLapack.lib libf2c.lib libIpopt.lib 
unix::IPOPT_LIBS  = -lipopt -ldl


#####################################################################################
#                                  BONMIN
#####################################################################################
BONMIN_DIR = ../bonmin/build

BONMIN_INCLUDE = $${BONMIN_DIR}/include/coin
BONMIN_LIBDIR  = $${BONMIN_DIR}/lib

win32::BONMIN_LIBS = libCoinBlas.lib libCoinLapack.lib libf2c.lib \
                     libBonmin.lib libIpopt.lib libCbc.lib \
                     libCgl.lib libClp.lib libCoinUtils.lib \
                     libOsiCbc.lib libOsiClp.lib libOsi.lib
linux-g++::BONMIN_LIBS = -lbonmin -lCbc -lCbcSolver -lCgl \
                         -lClp -lCoinUtils -lipopt -lOsiCbc \
                         -lOsiClp -lOsi \
                         -ldl -lz -lbz2
macx-g++::BONMIN_LIBS  = -lbonmin -lCbc -lCbcSolver -lCgl \
                         -lClp -lCoinUtils -lipopt -lOsiCbc \
                         -lOsiClp -lOsi \
                         -ldl -lz -lbz2


#####################################################################################
#                                 NLOPT
#####################################################################################
NLOPT_DIR     = ../nlopt/build
NLOPT_INCLUDE = $${NLOPT_DIR}/include
NLOPT_LIBDIR  = $${NLOPT_DIR}/lib

win32::NLOPT_LIBS = nlopt.lib
unix::NLOPT_LIBS  = -lnlopt -lm


######################################################################################
#                                   SuperLU
######################################################################################
SUPERLU_PATH    = ../superlu
SUPERLU_LIBPATH = $${SUPERLU_PATH}/lib
SUPERLU_INCLUDE = $${SUPERLU_PATH}/SRC

win32::SUPERLU_LIBS  = -L$${SUPERLU_LIBPATH} superlu.lib $${BLAS_LAPACK_LIBS}
unix::SUPERLU_LIBS   = -L$${SUPERLU_LIBPATH} -lsuperlu_4.1 $${RT} -lpthread


######################################################################################
#                                SuperLU_MT
######################################################################################
SUPERLU_MT_PATH    = ../superlu_mt
SUPERLU_MT_LIBPATH = $${SUPERLU_MT_PATH}/lib
SUPERLU_MT_INCLUDE = $${SUPERLU_MT_PATH}/SRC

win32::SUPERLU_MT_LIBS  = 
unix::SUPERLU_MT_LIBS   = -L$${SUPERLU_MT_LIBPATH} -lsuperlu_mt_2.0 $${RT} -lpthread


######################################################################################
#                                SuperLU_CUDA
######################################################################################
win32::CUDA_PATH     =
linux-g++::CUDA_PATH = /usr/local/cuda

SUPERLU_CUDA_PATH    = ../superlu_mt-GPU
SUPERLU_CUDA_LIBPATH = $${SUPERLU_CUDA_PATH}/lib
SUPERLU_CUDA_INCLUDE = $${SUPERLU_CUDA_PATH} \
                       $${CUDA_PATH}/include

win32::CUDA_LIBS     =
linux-g++::CUDA_LIBS = -L$${CUDA_PATH}/lib   -lcuda -lcudart


#####################################################################################
#                                  TRILINOS 
#####################################################################################
TRILINOS_DIR  = ../trilinos/build

win32::TRILINOS_INCLUDE = $${TRILINOS_DIR}/include \
                          $${TRILINOS_DIR}/../commonTools/WinInterface/include
unix::TRILINOS_INCLUDE  = $${TRILINOS_DIR}/include

win32::TRILINOS_LIBS = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                        $${SUPERLU_LIBS} \
                        aztecoo.lib ml.lib ifpack.lib amesos.lib epetra.lib epetraext.lib teuchos.lib

linux-g++::TRILINOS_LIBS  = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                            -laztecoo -lml -lifpack -lamesos -lepetra -lepetraext -lteuchos -lumfpack -lamd \
                            $${SUPERLU_LIBS}

macx-g++::TRILINOS_LIBS  = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib -L/opt/local/lib \
                           -laztecoo -lml -lifpack -lamesos -lepetra -lepetraext -lteuchos -lumfpack -lamd \
                           $${SUPERLU_LIBS}

#####################################################################################
#                                 MPI SUPPORT
#####################################################################################
enable_mpi {
QMAKE_CXXFLAGS += -DDAE_MPI

win32::MPI = 
unix::MPI  = 

win32::MPI_INCLUDE = 
unix::MPI_INCLUDE  = /usr/include/mpi

win32::MPI_LIBDIR = 
unix::MPI_LIBDIR  = 

win32::MPI_LIBS = 
unix::MPI_LIBS  = -lboost_mpi-mt -lboost_serialization -lmpi_cxx -lmpi
}


#####################################################################################
#                                  DAE Tools
#####################################################################################
win32::DAE_CORE_LIB                = cdaeCore.lib
win32::DAE_DATAREPORTING_LIB       = cdaeDataReporting.lib
win32::DAE_ACTIVITY_LIB            = cdaeActivity.lib
win32::DAE_IDAS_SOLVER_LIB         = cdaeIDAS_DAESolver.lib
win32::DAE_UNITS_LIB               = cdaeUnits.lib
win32::DAE_SUPERLU_SOLVER_LIB      = cdaeSuperLU_LASolver.lib
win32::DAE_SUPERLU_MT_SOLVER_LIB   = cdaeSuperLU_MT_LASolver.lib
win32::DAE_SUPERLU_CUDA_SOLVER_LIB = cdaeSuperLU_CUDA_LASolver.lib
win32::DAE_BONMIN_SOLVER_LIB       = cdaeBONMIN_MINLPSolver.lib
win32::DAE_IPOPT_SOLVER_LIB        = cdaeIPOPT_NLPSolver.lib
win32::DAE_NLOPT_SOLVER_LIB        = cdaeNLOPT_NLPSolver.lib
win32::DAE_TRILINOS_SOLVER_LIB     = cdaeTrilinos_LASolver.lib

unix::DAE_CORE_LIB                = -lcdaeCore
unix::DAE_DATAREPORTING_LIB       = -lcdaeDataReporting
unix::DAE_ACTIVITY_LIB            = -lcdaeActivity
unix::DAE_IDAS_SOLVER_LIB         = -lcdaeIDAS_DAESolver
unix::DAE_UNITS_LIB               = -lcdaeUnits
unix::DAE_SUPERLU_SOLVER_LIB      = -lcdaeSuperLU_LASolver
unix::DAE_SUPERLU_MT_SOLVER_LIB   = -lcdaeSuperLU_MT_LASolver
unix::DAE_SUPERLU_CUDA_SOLVER_LIB = -lcdaeSuperLU_CUDA_LASolver
unix::DAE_BONMIN_SOLVER_LIB       = -lcdaeBONMIN_MINLPSolver
unix::DAE_IPOPT_SOLVER_LIB        = -lcdaeIPOPT_NLPSolver
unix::DAE_NLOPT_SOLVER_LIB        = -lcdaeNLOPT_NLPSolver
unix::DAE_TRILINOS_SOLVER_LIB     = -lcdaeTrilinos_LASolver

QMAKE_LIBDIR += $${DAE_DEST_DIR} $${BOOSTLIBPATH} $${PYTHON_LIB_DIR}

#######################################################
#            Settings for installing files
#######################################################
SOLVERS_DIR  = ../daetools-package/daetools/solvers/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR}_numpy$${NUMPY_VERSION}
PYDAE_DIR    = ../daetools-package/daetools/pyDAE/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR}_numpy$${NUMPY_VERSION}

win32::DUMMY = $$system(mkdir daetools-package\daetools\solvers\\$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR}_numpy$${NUMPY_VERSION})
win32::DUMMY = $$system(mkdir daetools-package\daetools\pyDAE\\$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR}_numpy$${NUMPY_VERSION})

unix::DUMMY = $$system(mkdir -p daetools-package/daetools/solvers/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR}_numpy$${NUMPY_VERSION})
unix::DUMMY = $$system(mkdir -p daetools-package/daetools/pyDAE/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR}_numpy$${NUMPY_VERSION})

STATIC_LIBS_DIR = ../daetools-package/daetools/usr/local/lib
HEADERS_DIR     = ../daetools-package/daetools/usr/local/include

#####################################################################################
#         Write compiler settings (needed to build installations packages)
#####################################################################################
# Python settings
COMPILER_SETTINGS_FOLDER = .compiler_settings
win32::COMPILER = $$system(mkdir $${COMPILER_SETTINGS_FOLDER})
unix::COMPILER  = $$system(mkdir -p $${COMPILER_SETTINGS_FOLDER})

COMPILER = $$system(echo $${DAE_TOOLS_MAJOR} > $${COMPILER_SETTINGS_FOLDER}/dae_major)
COMPILER = $$system(echo $${DAE_TOOLS_MINOR} > $${COMPILER_SETTINGS_FOLDER}/dae_minor)
COMPILER = $$system(echo $${DAE_TOOLS_BUILD} > $${COMPILER_SETTINGS_FOLDER}/dae_build)
