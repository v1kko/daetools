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
DAE_TOOLS_MINOR = 2
DAE_TOOLS_BUILD = 0

# Set CONFIG += use_system_python to use the default system's python
# 1. On GNU/LINUX:
#    Set CONFIG += use_custom_python and for instance PYTHON_MAJOR=2 and PYTHON_MINOR=7 
#    to use Python located in /usr/lib/python2.7
# 2. On Windows:
#    Set CONFIG += use_custom_python and for instance PYTHON_MAJOR=2 and PYTHON_MINOR=7 
#    to use Python located in c:\Python27
CONFIG += use_system_python
PYTHON_MAJOR = 2
PYTHON_MINOR = 6

# 1. On GNU/LINUX:
#    a) Set CONFIG += use_system_boost to use the system's default version
#    b) Set CONFIG += use_custom_boost and for instance BOOST_MAJOR = 1, BOOST_MINOR = 42 
#       and BOOST_BUILD = 0 to use the boost build in ../boost_1_42_0
# 2. On Windows: 
#    BOOST_MAJOR, BOOST_MINOR and BOOST_BUILD must always be set!!
#    and Boost build must be located in ../boost_1_42_0 (for instance)
CONFIG += use_system_boost
BOOST_MAJOR = 1
BOOST_MINOR = 42
BOOST_BUILD = 0

# Set CONFIG += enable_mpi to use MPI libraries
#CONFIG += enable_mpi

# DAE Tools version (major, minor, build)
VERSION = $${DAE_TOOLS_MAJOR}.$${DAE_TOOLS_MINOR}.$${DAE_TOOLS_BUILD}

QMAKE_CXXFLAGS += -DDAE_MAJOR=$${DAE_TOOLS_MAJOR}
QMAKE_CXXFLAGS += -DDAE_MINOR=$${DAE_TOOLS_MINOR}
QMAKE_CXXFLAGS += -DDAE_BUILD=$${DAE_TOOLS_BUILD}

CONFIG(debug, debug|release):message(debug){
	DAE_DEST_DIR = ../debug
    OBJECTS_DIR = debug
}

CONFIG(release, debug|release):message(release){
	DAE_DEST_DIR = ../release
    OBJECTS_DIR = release
}

DESTDIR = $${DAE_DEST_DIR}

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

QMAKE_CXXFLAGS_DEBUG += -fno-default-inline

QMAKE_CXXFLAGS_DEBUG += -DDAE_DEBUG
QMAKE_CFLAGS_DEBUG   += -DDAE_DEBUG

#unix::QMAKE_CXXFLAGS += -ansi -pedantic
unix::QMAKE_CXXFLAGS_WARN_ON         += -Wextra -Wno-sign-compare \
                                        -Wno-unused-parameter \
                                        -Wno-unused-variable 
linux-g++-32::QMAKE_CXXFLAGS_WARN_ON += -Wno-unused-but-set-variable
linux-g++-64::QMAKE_CXXFLAGS_WARN_ON += -Wno-unused-but-set-variable

unix::QMAKE_CFLAGS_RELEASE   -= -O2
unix::QMAKE_CXXFLAGS_RELEASE -= -O2

unix::QMAKE_CFLAGS_RELEASE   += -O3
unix::QMAKE_CXXFLAGS_RELEASE += -O3

# On some low-RAM machines pyCore cannot compile
# The workaround is to set the following flags:
#unix::QMAKE_CXXFLAGS += --param ggc-min-expand=30 --param ggc-min-heapsize=8192

# Use SSE for x86 (32 bit machines)
linux-g++-32::QMAKE_CXXFLAGS_RELEASE += -mfpmath=sse -msse -msse2 -msse3
macx-g++::QMAKE_CXXFLAGS_RELEASE     += -mfpmath=sse -msse -msse2 -msse3

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
#                                   PYTHON
#####################################################################################
# Numpy and Scipy must be installed
# Debian Squeeze: sometimes there are problems with _numpyconfig.h
# Add: /usr/include/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}/numpy 
# to:  PYTHON_INCLUDE_DIR 
#####################################################################################
use_system_python {
PYTHON_MAJOR = $$system(python -c \"import sys; print sys.version_info[0]\")
PYTHON_MINOR = $$system(python -c \"import sys; print sys.version_info[1]\")
message(use_system_python: $${PYTHON_MAJOR}.$${PYTHON_MINOR})
}

use_custom_python { 
message(use_custom_python: $${PYTHON_MAJOR}.$${PYTHON_MINOR})
}

win32::PYTHONDIR                = C:\Python$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32::PYTHON_INCLUDE_DIR       = $${PYTHONDIR}\include
win32::PYTHON_SITE_PACKAGES_DIR = $${PYTHONDIR}\Lib\site-packages
win32::PYTHON_LIB_DIR           = $${PYTHONDIR}\libs

linux-g++-32::PYTHONDIR         = /usr/lib/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}
linux-g++-64::PYTHONDIR         = /usr/lib64/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}
macx-g++::PYTHONDIR             = /usr/lib/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}

unix::PYTHON_INCLUDE_DIR        = /usr/include/python$${PYTHON_MAJOR}.$${PYTHON_MINOR} \
							      /usr/include/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}/numpy \
                                  /usr/share/pyshared
unix::PYTHON_SITE_PACKAGES_DIR  = $${PYTHONDIR}/dist-packages \
                                  $${PYTHONDIR}/site-packages
unix::PYTHON_LIB_DIR            =


#####################################################################################
#                                  RT/GFORTRAN
#####################################################################################
win32::RT        =
linux-g++-32::RT = -lrt
linux-g++-64::RT = -lrt
macx-g++::RT     =

win32::GFORTRAN        =
linux-g++-32::GFORTRAN = -lgfortran
linux-g++-64::GFORTRAN = -lgfortran
macx-g++::GFORTRAN     = -lgfortran


#####################################################################################
#                                    BOOST
#####################################################################################
# Boost version installed must be 1.41+ (asio, system, python, thread, regex)
# 1. Install bjam and Boost.Build
#    a) On windows:
#       - Go to the directory tools\build\v2\.
#       - Run bootstrap.bat
#       - Run bjam install --prefix=PREFIX where PREFIX is the directory where you 
#         want Boost.Build to be installed
#       - Add PREFIX\bin to your PATH environment variable.
# 2) Build boost libraries (toolset=msvc or gcc; both static|shared)
#      bjam --build-dir=./build  
#      --with-date_time --with-python --with-system --with-regex --with-serialization --with-thread 
#      variant=release link=static,shared threading=multi runtime-link=shared
#    Python version could be set:
#    a) --with-python-version=2.7
#    b) in user-config.jam located in $BOOST_BUILD or $HOME directory
#       Add line: using python : 2.7 ;
#####################################################################################
win32::BOOSTDIR         = ../boost_$${BOOST_MAJOR}_$${BOOST_MINOR}_$${BOOST_BUILD}
win32::BOOSTLIBPATH     = $${BOOSTDIR}/lib
win32::BOOST_PYTHON_LIB =
win32::BOOST_LIBS       =

use_system_boost {
unix::BOOSTDIR         = /usr/include/boost
unix::BOOSTLIBPATH     = 
unix::BOOST_PYTHON_LIB = -lboost_python
unix::BOOST_LIBS       = -lboost_system -lboost_thread $${RT}

macx-g++::BOOSTDIR         = ../boost_$${BOOST_MAJOR}_$${BOOST_MINOR}_$${BOOST_BUILD}
macx-g++::BOOSTLIBPATH     = $${BOOSTDIR}/stage/lib
macx-g++::BOOST_PYTHON_LIB = -lboost_python
macx-g++::BOOST_LIBS       = -lboost_system -lboost_thread-mt
}

use_custom_boost { 
unix::BOOSTDIR         = ../boost_$${BOOST_MAJOR}_$${BOOST_MINOR}_$${BOOST_BUILD}
unix::BOOSTLIBPATH     = $${BOOSTDIR}/stage/lib
unix::BOOST_PYTHON_LIB = -lboost_python
unix::BOOST_LIBS       = -lboost_system -lboost_thread $${RT}
}


#####################################################################################
#                                 BLAS/LAPACK
#####################################################################################
win32::BLAS_LAPACK_LIBDIR = ../clapack/LIB/Win32
unix::BLAS_LAPACK_LIBDIR  =

win32::BLAS_LAPACK_LIBS = $${BLAS_LAPACK_LIBDIR}/BLAS_nowrap.lib \
                          $${BLAS_LAPACK_LIBDIR}/clapack_nowrap.lib \
                          $${BLAS_LAPACK_LIBDIR}/libf2c.lib

unix::BLAS_LAPACK_LIBS = -L$${BLAS_LAPACK_LIBDIR} -lblas -llapack -lm
#unix::BLAS_LAPACK_LIBS = ../GotoBLAS2/libgoto2.a -lm


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
                       sundials_nvecserial.lib \
                       $${BLAS_LAPACK_LIBS}
unix::SUNDIALS_LIBS = -lsundials_idas \
                      -lsundials_nvecserial \
                       $${BLAS_LAPACK_LIBS}


#####################################################################################
#                                  MUMPS
#####################################################################################
MUMPS_DIR  = ../mumps

win32::G95_LIBDIR = c:\g95\lib\gcc-lib\i686-pc-mingw32\4.1.2

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
unix::MUMPS_LIBS = -lcoinmumps -lpthread $${BLAS_LAPACK_LIBS} $${RT} $${GFORTRAN}


#####################################################################################
#                                  IPOPT
#####################################################################################
IPOPT_DIR = ../bonmin/build

IPOPT_INCLUDE = $${IPOPT_DIR}/include/coin
IPOPT_LIBDIR  = $${IPOPT_DIR}/lib

win32::IPOPT_LIBS = libCoinBlas.lib libCoinLapack.lib libf2c.lib libIpopt.lib 
unix::IPOPT_LIBS  = -lipopt -ldl $${BLAS_LAPACK_LIBS}


#####################################################################################
#                                  BONMIN
#####################################################################################
BONMIN_DIR = ../bonmin/build

BONMIN_INCLUDE = $${BONMIN_DIR}/include/coin
BONMIN_LIBDIR  = $${BONMIN_DIR}/lib

win32::BONMIN_LIBS =  libCoinBlas.lib libCoinLapack.lib libf2c.lib \
					  libBonmin.lib \
					  libIpopt.lib \
					  libCbc.lib \
					  libCgl.lib \
					  libClp.lib \
					  libCoinUtils.lib \
					  libOsiCbc.lib \
					  libOsiClp.lib \
					  libOsi.lib
unix::BONMIN_LIBS  =    -ldl $${BLAS_LAPACK_LIBS} -lz \
						-lbonmin \
						-lCbc \
						-lCbcSolver \
						-lCgl \
						-lClp \
						-lCoinUtils \
						-lipopt \
						-lOsiCbc \
						-lOsiClp \
						-lOsi


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

win32::SUPERLU_LIBS          = -L$${SUPERLU_LIBPATH} superlu.lib $${BLAS_LAPACK_LIBS}
linux-g++-32::SUPERLU_LIBS   = -L$${SUPERLU_LIBPATH} -lsuperlu_4.1 $${RT} -lpthread $${BLAS_LAPACK_LIBS}
linux-g++-64::SUPERLU_LIBS   = -L$${SUPERLU_LIBPATH} -lsuperlu_4.1 $${RT} -lpthread $${BLAS_LAPACK_LIBS}


######################################################################################
#                                SuperLU_MT
######################################################################################
SUPERLU_MT_PATH    = ../superlu_mt
SUPERLU_MT_LIBPATH = $${SUPERLU_MT_PATH}/lib
SUPERLU_MT_INCLUDE = $${SUPERLU_MT_PATH}/SRC

win32::SUPERLU_MT_LIBS          = 
linux-g++-32::SUPERLU_MT_LIBS   = -L$${SUPERLU_MT_LIBPATH} -lsuperlu_mt_2.0 $${RT} -lpthread $${BLAS_LAPACK_LIBS}
linux-g++-64::SUPERLU_MT_LIBS   = -L$${SUPERLU_MT_LIBPATH} -lsuperlu_mt_2.0 $${RT} -lpthread $${BLAS_LAPACK_LIBS}


######################################################################################
#                                SuperLU_CUDA
######################################################################################
win32::CUDA_PATH          = 
linux-g++-32::CUDA_PATH   = /usr/local/cuda
linux-g++-64::CUDA_PATH   = /usr/local/cuda

SUPERLU_CUDA_PATH    = ../superlu_mt-GPU
SUPERLU_CUDA_LIBPATH = $${SUPERLU_CUDA_PATH}/lib
SUPERLU_CUDA_INCLUDE = $${SUPERLU_CUDA_PATH} \
	                   $${CUDA_PATH}/include

win32::CUDA_LIBS = 
linux-g++-32::CUDA_LIBS   = -L$${CUDA_PATH}/lib   -lcuda -lcudart
linux-g++-64::CUDA_LIBS   = -L$${CUDA_PATH}/lib64 -lcuda -lcudart


#####################################################################################
#                                  TRILINOS 
#####################################################################################
TRILINOS_DIR  = ../trilinos/build

win32::TRILINOS_INCLUDE = $${TRILINOS_DIR}/include \
                          $${TRILINOS_DIR}/../commonTools/WinInterface/include
unix::TRILINOS_INCLUDE  = $${TRILINOS_DIR}/include

win32::TRILINOS_LIBS = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                        $${BLAS_LAPACK_LIBS} \
                        $${SUPERLU_LIBS} \
                        aztecoo.lib ml.lib ifpack.lib amesos.lib epetra.lib epetraext.lib teuchos.lib

linux-g++-32::TRILINOS_LIBS  = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
							   -laztecoo -lml -lifpack -lamesos -lepetra -lepetraext -lteuchos \
							   -lumfpack -lamd \
							    $${SUPERLU_LIBS} \
							    $${BLAS_LAPACK_LIBS}

linux-g++-64::TRILINOS_LIBS = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
							  -laztecoo -lml -lifpack -lamesos -lepetra -lepetraext -lteuchos \
							  -lumfpack -lamd \
							   $${SUPERLU_LIBS} \
							   $${BLAS_LAPACK_LIBS}


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
win32::DAE_SUPERLU_SOLVER_LIB      = cdaeSuperLU_LASolver.lib
win32::DAE_SUPERLU_MT_SOLVER_LIB   = cdaeSuperLU_MT_LASolver.lib
win32::DAE_SUPERLU_CUDA_SOLVER_LIB = cdaeSuperLU_CUDA_LASolver.lib
win32::DAE_BONMIN_SOLVER_LIB       = cdaeBONMIN_MINLPSolver.lib
win32::DAE_IPOPT_SOLVER_LIB        = cdaeIPOPT_NLPSolver.lib
win32::DAE_NLOPT_SOLVER_LIB        = cdaeNLOPT_NLPSolver.lib
win32::DAE_UNITS_LIB               = cdaeUnits.lib

unix::DAE_CORE_LIB                = -lcdaeCore
unix::DAE_DATAREPORTING_LIB       = -lcdaeDataReporting
unix::DAE_ACTIVITY_LIB            = -lcdaeActivity
unix::DAE_IDAS_SOLVER_LIB         = -lcdaeIDAS_DAESolver
unix::DAE_SUPERLU_SOLVER_LIB      = -lcdaeSuperLU_LASolver
unix::DAE_SUPERLU_MT_SOLVER_LIB   = -lcdaeSuperLU_MT_LASolver
unix::DAE_SUPERLU_CUDA_SOLVER_LIB = -lcdaeSuperLU_CUDA_LASolver
unix::DAE_BONMIN_SOLVER_LIB       = -lcdaeBONMIN_MINLPSolver
unix::DAE_IPOPT_SOLVER_LIB        = -lcdaeIPOPT_NLPSolver
unix::DAE_NLOPT_SOLVER_LIB        = -lcdaeNLOPT_NLPSolver
unix::DAE_UNITS_LIB               = -lcdaeUnits

QMAKE_LIBDIR += $${DAE_DEST_DIR} $${BOOSTLIBPATH}

HEADERS += \
    ../config.h \
    ../dae_develop.h \
    ../dae.h
