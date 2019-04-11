#************************************************************************************
#                 DAE Tools Project: www.daetools.com
#                 Copyright (C) Dragan Nikolic
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
DAE_TOOLS_MINOR = 9
DAE_TOOLS_BUILD = 0

# DAE Tools version (major, minor, build)
VERSION = $${DAE_TOOLS_MAJOR}.$${DAE_TOOLS_MINOR}.$${DAE_TOOLS_BUILD}

QMAKE_CXXFLAGS += -DDAE_MAJOR=$${DAE_TOOLS_MAJOR}
QMAKE_CXXFLAGS += -DDAE_MINOR=$${DAE_TOOLS_MINOR}
QMAKE_CXXFLAGS += -DDAE_BUILD=$${DAE_TOOLS_BUILD}

# daetools always use the current system python version and custom compiled boost libs
# located in ../boost with the libraries in ../boost/stage/lib

# Build only for x86_64 for MAC OS-X
macx-g++::CONFIG    += x86_64
macx-g++::QMAKE_CC   = /usr/local/bin/gcc
macx-g++::QMAKE_CXX  = /usr/local/bin/g++

win32-msvc2015::SHARED_LIB_EXT  = dll
win32-g++-*::SHARED_LIB_EXT     = dll
win64-g++-*::SHARED_LIB_EXT     = dll
linux-g++::SHARED_LIB_EXT       = so
macx-g++::SHARED_LIB_EXT        = dylib

win32-msvc2015::SHARED_LIB_PREFIX   =
win32-g++-*::SHARED_LIB_PREFIX      =
win64-g++-*::SHARED_LIB_PREFIX      =
linux-g++::SHARED_LIB_PREFIX        = lib
macx-g++::SHARED_LIB_PREFIX         = lib

win32-msvc2015::SHARED_LIB_POSTFIX  = $${DAE_TOOLS_MAJOR}
win32-g++-*::SHARED_LIB_POSTFIX     = $${DAE_TOOLS_MAJOR}
win64-g++-*::SHARED_LIB_POSTFIX     = $${DAE_TOOLS_MAJOR}
linux-g++::SHARED_LIB_POSTFIX       =
macx-g++::SHARED_LIB_POSTFIX        =

win32-msvc2015::SHARED_LIB_APPEND   = dll
win32-g++-*::SHARED_LIB_APPEND      = dll
win64-g++-*::SHARED_LIB_APPEND      = dll
linux-g++::SHARED_LIB_APPEND        = so.$${VERSION}
macx-g++::SHARED_LIB_APPEND         = $${VERSION}.dylib

win32-msvc2015::PYTHON_EXTENSION_MODULE_EXT = pyd
win32-g++-*::PYTHON_EXTENSION_MODULE_EXT    = pyd
win64-g++-*::PYTHON_EXTENSION_MODULE_EXT    = pyd
linux-g++::PYTHON_EXTENSION_MODULE_EXT      = so
macx-g++::PYTHON_EXTENSION_MODULE_EXT       = so

# Old: copy /y
win32-msvc2015::COPY_FILE = cp -fa
win32-g++-*::COPY_FILE    = cp -fa
win64-g++-*::COPY_FILE    = cp -fa
unix::COPY_FILE           = cp -fa

win32-msvc2015::MOVE_FILE = mv -f
win32-g++-*::MOVE_FILE    = mv -f
win64-g++-*::MOVE_FILE    = mv -f
unix::MOVE_FILE           = mv -f

# Set CONFIG += enable_mpi to use MPI libraries

CONFIG(debug, debug|release) {
    DAE_DEST_DIR = ../debug
    OBJECTS_DIR = debug
}

CONFIG(release, debug|release) {
    DAE_DEST_DIR = ../release
    OBJECTS_DIR = release
}

DESTDIR = $${DAE_DEST_DIR}

#####################################################################################
#                           System + Machine + Python info
#####################################################################################
crossCompile{
    # Not used
    PYTHON = ""

    PYTHON_MAJOR = 3
    PYTHON_MINOR = 4
    PYTHON_ABI   =

    # System := {'Windows'}
    DAE_SYSTEM   = Windows

    # Machine := {'i386', ..., 'i686', 'AMD64'}
    DAE_MACHINE = win32

    RT =

    WIN_PYTHON_DIR     = Python$${PYTHON_MAJOR}$${PYTHON_MINOR}-$${DAE_MACHINE}
    PYTHON_INCLUDE_DIR = ../$${WIN_PYTHON_DIR}/include
    PYTHON_LIB_DIR     = ../$${WIN_PYTHON_DIR}/libs
    PYTHON_LIB         = -lpython$${PYTHON_MAJOR}$${PYTHON_MINOR}$${PYTHON_ABI} $${RT}
}

!crossCompile{
    # If compiling from the shell compile_linux.sh script specify the python binary
    shellCompile {
        PYTHON = $$customPython
    }

    # If compiling from the GUI use system's default python version
    !shellCompile {
        PYTHON = python
    }

    PYTHON_MAJOR = $$system($${PYTHON} -c \"import sys; print(sys.version_info[0])\")
    PYTHON_MINOR = $$system($${PYTHON} -c \"import sys; print(sys.version_info[1])\")
    PYTHON_ABI   = $$system($${PYTHON} -c \"import sysconfig; flags = sysconfig.get_config_vars(); abi = flags[\'ABIFLAGS\'] if (\'ABIFLAGS\' in flags) else \'\'; print(abi)\")

    # Numpy version
    #NUMPY_VERSION = $$system($${PYTHON} -c \"import numpy; print(\'\'.join(numpy.__version__.split(\'.\')[0:2]))\")

    # System := {'Linux', 'Windows', 'Darwin'}
    DAE_SYSTEM   = $$system($${PYTHON} -c \"import platform; print(platform.system())\")

    # DAE Machine := {'i686', 'x86_64', 'win32', 'win64'}
    win32-msvc2015 {
      MACHINE_COMMAND = "import platform; arch = platform.architecture()[0]; p={'32bit':'win32', '64bit':'win64'}; machine = p[arch]; print(machine)"
    } else {
      MACHINE_COMMAND  = "import platform; p = {'Linux':platform.machine(), 'Darwin':platform.machine(), 'Windows':('win32' if platform.architecture()[0] == '32bit' else 'win64')}; machine = p[platform.system()]; print(machine)"
    }
    DAE_MACHINE = $$system($${PYTHON} -c \"$${MACHINE_COMMAND}\")

    message(Compiling on $${DAE_SYSTEM}_$${DAE_MACHINE})

    win32-msvc2015::RT =
    linux-g++::RT      = -lrt
    macx-g++::RT       =

    win32::PYTHON_INCLUDE_DIR  = $$system($${PYTHON} -c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\")
    unix::PYTHON_INCLUDE_DIR   = $$system($${PYTHON} -c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\")

    win32::PYTHON_LIB_DIR      = $$system($${PYTHON} -c \"import sys; print(sys.prefix)\")/libs
    unix::PYTHON_LIB_DIR       = $$system($${PYTHON} -c \"import sys; print(sys.prefix)\")/lib

    win32::PYTHON_LIB          = python$${PYTHON_MAJOR}$${PYTHON_MINOR}.lib
    unix::PYTHON_LIB           = -lpython$${PYTHON_MAJOR}.$${PYTHON_MINOR}$${PYTHON_ABI} $${RT}

    !shellCompile {
        message(Using python [$${PYTHON}] v$${PYTHON_MAJOR}.$${PYTHON_MINOR}$${PYTHON_ABI})
    }

    #PYTHON_SITE_PACKAGES_DIR  = $$system($${PYTHON} -c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_lib())\")
    #win32-msvc2015::NUMPY_INCLUDE_DIR     = $${PYTHON_SITE_PACKAGES_DIR}/numpy/core/include/numpy \
    #                               $${PYTHON_INCLUDE_DIR}/numpy/core/include/numpy
    #linux-g++::NUMPY_INCLUDE_DIR = $${PYTHON_SITE_PACKAGES_DIR}/numpy/core/include/numpy \
    #                               $${PYTHON_INCLUDE_DIR}/numpy/core/include/numpy \
    #                               /usr/include/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}/numpy \
    #                               /usr/share/pyshared/numpy/core/include/numpy
    #macx-g++::NUMPY_INCLUDE_DIR  = $${PYTHON_SITE_PACKAGES_DIR}/numpy/core/include/numpy \
    #                               $${PYTHON_INCLUDE_DIR}/numpy/core/include/numpy

}

# RPATH for python extension modules
# win32-msvc2015::SOLIBS_RPATH =
# win32-g++-*::SOLIBS_RPATH    = -Wl,-rpath,\'\$$ORIGIN/../../solibs/$${DAE_SYSTEM}_$${DAE_MACHINE}\'
# win64-g++-*::SOLIBS_RPATH    = -Wl,-rpath,\'\$$ORIGIN/../../solibs/$${DAE_SYSTEM}_$${DAE_MACHINE}\'
# linux-g++::SOLIBS_RPATH      = -Wl,-rpath,\'\$$ORIGIN/../../solibs/$${DAE_SYSTEM}_$${DAE_MACHINE}\',-z,origin
# macx-g++::SOLIBS_RPATH       = -Wl,-rpath,\'@loader_path/../../solibs/$${DAE_SYSTEM}_$${DAE_MACHINE}\'
win32-msvc2015::SOLIBS_RPATH =
win32-g++-*::SOLIBS_RPATH    = -Wl,-rpath,\'\$$ORIGIN/../lib\'
win64-g++-*::SOLIBS_RPATH    = -Wl,-rpath,\'\$$ORIGIN/../lib\'
linux-g++::SOLIBS_RPATH      = -Wl,-rpath,\'\$$ORIGIN/../lib\',-z,origin
macx-g++::SOLIBS_RPATH       = -Wl,-rpath,\'@loader_path/../lib\'

# RPATH for the simulation_loader
win32-msvc2015::SOLIBS_RPATH_SL =
win32-g++-*::SOLIBS_RPATH_SL    = -Wl,-rpath,\'\$$ORIGIN\'
win64-g++-*::SOLIBS_RPATH_SL    = -Wl,-rpath,\'\$$ORIGIN\',-z,origin
linux-g++::SOLIBS_RPATH_SL      = -Wl,-rpath,\'\$$ORIGIN\',-z,origin
macx-g++::SOLIBS_RPATH_SL       = -Wl,-rpath,\'@loader_path\'

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
CONFIG += rtti

# For profiling with Valgrind
#QMAKE_CXXFLAGS += -g

# c++11
unix::QMAKE_CXXFLAGS  += -std=c++11
unix::QMAKE_LFLAGS    += -std=c++11
win32::QMAKE_CXXFLAGS += /std:c++14
win32::QMAKE_LFLAGS   +=

# OpenMP
unix::QMAKE_CXXFLAGS           += -fopenmp
win32-msvc2015::QMAKE_CXXFLAGS += /openmp

unix::QMAKE_LFLAGS += -fopenmp

win32-msvc2015::QMAKE_CXXFLAGS += -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX
win32-msvc2015::QMAKE_CXXFLAGS += -DBOOST_ALL_NO_LIB=1
win32-msvc2015::QMAKE_CXXFLAGS += /bigobj

win32-g++-*::QMAKE_LFLAGS += -mwindows
win64-g++-*::QMAKE_LFLAGS += -mwindows

QMAKE_CXXFLAGS_DEBUG += -fno-default-inline

QMAKE_CXXFLAGS_DEBUG += -DDAE_DEBUG
QMAKE_CFLAGS_DEBUG   += -DDAE_DEBUG

#unix::QMAKE_CXXFLAGS += -ansi -pedantic

# Unresolved _gethostname problem in MinGW
#win32-g++-*::QMAKE_LFLAGS += -Wl,--enable-stdcall-fixup

macx-g++::QMAKE_LFLAGS += -mmacosx-version-min=10.6

QMAKE_CXXFLAGS += -DDAE_PYTHON_MAJOR=$${PYTHON_MAJOR} -DDAE_PYTHON_MINOR=$${PYTHON_MINOR}

# If compiling from the compile_linux.sh shell script supress all warnings
shellCompile {
    unix::QMAKE_CXXFLAGS_WARN_ON = -w
    unix::QMAKE_CFLAGS_WARN_ON   = -w

    win32-g++-*::QMAKE_CXXFLAGS_WARN_ON = -w
    win32-g++-*::QMAKE_CFLAGS_WARN_ON   = -w

    win64-g++-*::QMAKE_CXXFLAGS_WARN_ON = -w
    win64-g++-*::QMAKE_CFLAGS_WARN_ON   = -w
}
# If compiling from the GUI enable warnings
!shellCompile:message(Compiling from the GUI) {
win32-g++-*::QMAKE_CXXFLAGS_WARN_ON = -Wall -Wextra \
                                      -Wno-sign-compare \
                                      -Wno-unused-parameter \
                                      -Wno-unused-variable \
                                      -Wno-unused-but-set-variable

win64-g++-*::QMAKE_CXXFLAGS_WARN_ON = -Wall -Wextra \
                                      -Wno-sign-compare \
                                      -Wno-unused-parameter \
                                      -Wno-unused-variable \
                                      -Wno-unused-but-set-variable

linux-g++::QMAKE_CXXFLAGS_WARN_ON = -Wall -Wextra \
                                    -Wno-sign-compare \
                                    -Wno-unused-parameter \
                                    -Wno-unused-variable \
                                    -Wno-unused-but-set-variable

macx-g++::QMAKE_CXXFLAGS_WARN_ON = -Wall -Wextra \
                                   -Wno-sign-compare \
                                   -Wno-unused-parameter \
                                   -Wno-unused-variable
}

unix::QMAKE_CFLAGS_RELEASE   -= -O2
unix::QMAKE_CXXFLAGS_RELEASE -= -O2

unix::QMAKE_CFLAGS_RELEASE   += -O3
unix::QMAKE_CXXFLAGS_RELEASE += -O3

# On some low-RAM machines certain boost.python modules cannot compile
# The workaround is to set the following flags:
#unix::QMAKE_CXXFLAGS += --param ggc-min-expand=0 --param ggc-min-heapsize=8192 -fno-var-tracking-assignments
#win32-g++-*::QMAKE_CXXFLAGS += -fno-var-tracking-assignments

# Use SSE for x86 32 bit machines (not used by default)
#linux-g++::QMAKE_CXXFLAGS_RELEASE += -march=pentium4 -mfpmath=sse -msse -msse2

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
#                                  FORTRAN
#####################################################################################
# gfortran does not exist in MacOS XCode and must be installed separately
#####################################################################################
win32-msvc2015::GFORTRAN    =
win32-g++-*::GFORTRAN       = -lgfortran
win64-g++-*::GFORTRAN       = -lgfortran
linux-g++::GFORTRAN         = -lgfortran
macx-g++::GFORTRAN          = -lgfortran


#####################################################################################
#                                  pthreads
#####################################################################################
win32-msvc2015::PTHREADS_LIB =
unix::PTHREADS_LIB           = -lpthread
win32-g++-*::PTHREADS_LIB    = -lwinpthread
win64-g++-*::PTHREADS_LIB    = -lwinpthread


#####################################################################################
#                                    BOOST
#####################################################################################
# Boost version installed must be 1.42+ (asio, system, python, thread, regex)
# Starting with the version 1.2.1 daetools use manually compiled boost libraries.
# The compilation is done in the shell script compile_libraries_linux.sh
#####################################################################################
BOOST_PYTHON_SUFFIX =
equals(PYTHON_MAJOR, "3") {
    message("If python major is 3; otherwise the suffix is empty.")
    BOOST_PYTHON_SUFFIX = $${PYTHON_MAJOR}
}

win32-msvc2015::BOOSTDIR                  = ../boost$${PYTHON_MAJOR}.$${PYTHON_MINOR}
win32-msvc2015::BOOSTLIBPATH              = $${BOOSTDIR}/stage/lib
win32-msvc2015::BOOST_PYTHON_LIB_NAME     = boost_python$${BOOST_PYTHON_SUFFIX}-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-msvc2015::BOOST_SYSTEM_LIB_NAME     = boost_system-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-msvc2015::BOOST_THREAD_LIB_NAME     = boost_thread-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-msvc2015::BOOST_FILESYSTEM_LIB_NAME = boost_filesystem-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-msvc2015::BOOST_PYTHON_LIB          = $${BOOST_PYTHON_LIB_NAME}.lib $${PYTHON_LIB}
win32-msvc2015::BOOST_LIBS                = $${BOOST_SYSTEM_LIB_NAME}.lib \
                                            $${BOOST_THREAD_LIB_NAME}.lib \
                                            $${BOOST_FILESYSTEM_LIB_NAME}.lib

win32-g++-*::BOOSTDIR                   = ../boost$${PYTHON_MAJOR}.$${PYTHON_MINOR}
win32-g++-*::BOOSTLIBPATH               = $${BOOSTDIR}/stage/lib
win32-g++-*::BOOST_PYTHON_LIB_NAME      = boost_python$${BOOST_PYTHON_SUFFIX}-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-g++-*::BOOST_SYSTEM_LIB_NAME      = boost_system-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-g++-*::BOOST_THREAD_LIB_NAME      = boost_thread_win32-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-g++-*::BOOST_FILESYSTEM_LIB_NAME  = boost_filesystem-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-g++-*::BOOST_PYTHON_LIB           = -L$${BOOSTLIBPATH} -l$${BOOST_PYTHON_LIB_NAME} \
                                          -L$${PYTHON_LIB_DIR} $${PYTHON_LIB}
win32-g++-*::BOOST_LIBS                 = -L$${BOOSTLIBPATH} -l$${BOOST_SYSTEM_LIB_NAME} \
                                                             -l$${BOOST_THREAD_LIB_NAME} \
                                                             -l$${BOOST_FILESYSTEM_LIB_NAME} \
                                                             -lws2_32 -lmswsock \
                                                             $${PTHREADS_LIB} \
                                                             $${RT}

win64-g++-*::BOOSTDIR                   = ../boost$${PYTHON_MAJOR}.$${PYTHON_MINOR}
win64-g++-*::BOOSTLIBPATH               = $${BOOSTDIR}/stage/lib
win64-g++-*::BOOST_PYTHON_LIB_NAME      = boost_python$${BOOST_PYTHON_SUFFIX}-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win64-g++-*::BOOST_SYSTEM_LIB_NAME      = boost_system-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win64-g++-*::BOOST_THREAD_LIB_NAME      = boost_thread_win32-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win64-g++-*::BOOST_FILESYSTEM_LIB_NAME  = boost_filesystem-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win64-g++-*::BOOST_PYTHON_LIB           = -L$${BOOSTLIBPATH} -l$${BOOST_PYTHON_LIB_NAME} \
                                          -L$${PYTHON_LIB_DIR} $${PYTHON_LIB}
win64-g++-*::BOOST_LIBS                 = -L$${BOOSTLIBPATH} -l$${BOOST_SYSTEM_LIB_NAME} \
                                                             -l$${BOOST_THREAD_LIB_NAME} \
                                                             -l$${BOOST_FILESYSTEM_LIB_NAME} \
                                                             -lws2_32 -lmswsock \
                                                             $${PTHREADS_LIB} \
                                                             $${RT}

unix::BOOSTDIR                   = ../boost$${PYTHON_MAJOR}.$${PYTHON_MINOR}
unix::BOOSTLIBPATH               = $${BOOSTDIR}/stage/lib
unix::BOOST_PYTHON_LIB_NAME      = boost_python$${BOOST_PYTHON_SUFFIX}-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::BOOST_SYSTEM_LIB_NAME      = boost_system-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::BOOST_THREAD_LIB_NAME      = boost_thread-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::BOOST_FILESYSTEM_LIB_NAME  = boost_filesystem-daetools-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::BOOST_PYTHON_LIB           = -L$${BOOSTLIBPATH} -l$${BOOST_PYTHON_LIB_NAME} \
                                   -L$${PYTHON_LIB_DIR} $${PYTHON_LIB}
unix::BOOST_LIBS                 = -L$${BOOSTLIBPATH} -l$${BOOST_SYSTEM_LIB_NAME} \
                                                      -l$${BOOST_THREAD_LIB_NAME} \
                                                      -l$${BOOST_FILESYSTEM_LIB_NAME} \
                                                       $${RT}


#####################################################################################
#                                 BLAS/LAPACK
#####################################################################################
# No fortran compiler in windows - use cblas and clapack reference implementation
win32-msvc2015::BLAS_LAPACK_LIBDIR = ../clapack/build/lib
win32-g++-*::BLAS_LAPACK_LIBDIR    = ../lapack/lib
win64-g++-*::BLAS_LAPACK_LIBDIR    = ../lapack/lib
linux-g++::BLAS_LAPACK_LIBDIR      = ../lapack/lib
macx-g++::BLAS_LAPACK_LIBDIR       = ../lapack/lib

# Define DAE_USE_OPEN_BLAS if using OpenBLAS
win32-msvc2015::QMAKE_CXXFLAGS  +=
win32-g++-*::QMAKE_CXXFLAGS     +=
win64-g++-*::QMAKE_CXXFLAGS     +=
linux-g++::QMAKE_CXXFLAGS       += #-DDAE_USE_OPEN_BLAS
macx-g++::QMAKE_CXXFLAGS        +=

win32-msvc2015::BLAS_LAPACK_LIBS = $${BLAS_LAPACK_LIBDIR}/blas.lib \
                                   $${BLAS_LAPACK_LIBDIR}/lapack.lib \
                                   $${BLAS_LAPACK_LIBDIR}/libf2c.lib
win32-g++-*::BLAS_LAPACK_LIBS    =  $${BLAS_LAPACK_LIBDIR}/liblapack.a $${BLAS_LAPACK_LIBDIR}/libblas.a -lgfortran -lm
win64-g++-*::BLAS_LAPACK_LIBS    =  $${BLAS_LAPACK_LIBDIR}/liblapack.a $${BLAS_LAPACK_LIBDIR}/libblas.a -lgfortran -lm

# 1. OpenBLAS dynamically linked:
#linux-g++::BLAS_LAPACK_LIBS = -L$${BLAS_LAPACK_LIBDIR} -lopenblas_daetools -lm
# 2. daetools compiled reference BLAS and Lapack statically linked:
linux-g++::BLAS_LAPACK_LIBS = $${BLAS_LAPACK_LIBDIR}/liblapack.a $${BLAS_LAPACK_LIBDIR}/libblas.a -l:libgfortran.so.3 -lm
macx-g++::BLAS_LAPACK_LIBS  = $${BLAS_LAPACK_LIBDIR}/liblapack.a $${BLAS_LAPACK_LIBDIR}/libblas.a -lgfortran -lm


#####################################################################################
#                                 SUNDIALS IDAS
#####################################################################################
# ./configure --prefix=${HOME}/Data/daetools/trunk/idas/build --disable-mpi
#             --enable-examples --enable-static=yes --enable-shared=no --with-pic
#             --enable-lapack CFLAGS=-O3
#####################################################################################
SUNDIALS         = ../idas/build
SUNDIALS_INCLUDE = $${SUNDIALS}/include
SUNDIALS_LIBDIR  = $${SUNDIALS}/lib

win32-msvc2015::SUNDIALS_LIBS = $${SUNDIALS_LIBDIR}/sundials_idas.lib \
                                $${SUNDIALS_LIBDIR}/sundials_nvecserial.lib
win32-g++-*::SUNDIALS_LIBS = $${SUNDIALS_LIBDIR}/libsundials_idas.a \
                             $${SUNDIALS_LIBDIR}/libsundials_nvecserial.a
win64-g++-*::SUNDIALS_LIBS = $${SUNDIALS_LIBDIR}/libsundials_idas.a \
                             $${SUNDIALS_LIBDIR}/libsundials_nvecserial.a
unix::SUNDIALS_LIBS  = $${SUNDIALS_LIBDIR}/libsundials_idas.a \
                       $${SUNDIALS_LIBDIR}/libsundials_nvecserial.a


#####################################################################################
#                                  MUMPS
#####################################################################################
MUMPS_DIR  = ../mumps

contains(DAE_MACHINE, win32) {
    win32-msvc2015::GFORTRAN_LIBDIR = C:/mingw-w64/win32/mingw32/lib/gcc/i686-w64-mingw32/7.1.0
    message(GFORTRAN_LIBDIR = $${GFORTRAN_LIBDIR})
}

contains(DAE_MACHINE, win64) {
    win32-msvc2015::GFORTRAN_LIBDIR = C:/mingw-w64/win64/mingw64/lib/gcc/x86_64-w64-mingw32/7.1.0
    message(GFORTRAN_LIBDIR = $${GFORTRAN_LIBDIR})
}

MUMPS_LIBDIR   = $${MUMPS_DIR}/lib \
                 $${MUMPS_DIR}/libseq \
                 $${MUMPS_DIR}/blas \
                 $${GFORTRAN_LIBDIR}

win32-msvc2015::MUMPS_LIBS = blas.lib \
                             libmpiseq.lib \
                             libdmumps.lib \
                             libmumps_common.lib \
                             libpord.lib \
                             libgfortran.dll.a \
                             libquadmath.dll.a \
                             libgcc.a
win32-g++-*::MUMPS_LIBS = -lcoinmumps $${PTHREADS_LIB} $${RT} -lmsvcrt -lkernel32 -luser32 -lshell32 -luuid -lole32 -ladvapi32 -lws2_32
win64-g++-*::MUMPS_LIBS = -lcoinmumps $${PTHREADS_LIB} $${RT} -lmsvcrt -lkernel32 -luser32 -lshell32 -luuid -lole32 -ladvapi32 -lws2_32
unix::MUMPS_LIBS        = -lcoinmumps $${PTHREADS_LIB} $${RT}


#####################################################################################
#                                  IPOPT
#####################################################################################
IPOPT_DIR = ../bonmin/build

IPOPT_INCLUDE = $${IPOPT_DIR}/include/coin
IPOPT_LIBDIR  = $${IPOPT_DIR}/lib

win32-msvc2015::IPOPT_LIBS = libCoinBlas.lib libCoinLapack.lib libIpopt.lib
win32-g++-*::IPOPT_LIBS    = -lipopt -lcoinlapack -lcoinblas $${GFORTRAN}
win64-g++-*::IPOPT_LIBS    = -lipopt -lcoinlapack -lcoinblas $${GFORTRAN}
unix::IPOPT_LIBS           = -lipopt -ldl


#####################################################################################
#                                  BONMIN
#####################################################################################
BONMIN_DIR = ../bonmin/build

BONMIN_INCLUDE = $${BONMIN_DIR}/include/coin
BONMIN_LIBDIR  = $${BONMIN_DIR}/lib

win32-msvc2015::BONMIN_LIBS = libCoinBlas.lib libCoinLapack.lib \
                              libBonmin.lib libIpopt.lib libCbc.lib \
                              libCgl.lib libClp.lib libCoinUtils.lib \
                              libOsiCbc.lib libOsiClp.lib libOsi.lib
win32-g++-*::BONMIN_LIBS = -lbonmin -lCbc -lCbcSolver -lCgl \
                           -lClp -lCoinUtils -lipopt -lOsiCbc \
                           -lOsiClp -lOsi \
                           -lcoinlapack -lcoinblas
win64-g++-*::BONMIN_LIBS = -lbonmin -lCbc -lCbcSolver -lCgl \
                           -lClp -lCoinUtils -lipopt -lOsiCbc \
                           -lOsiClp -lOsi \
                           -lcoinlapack -lcoinblas
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

win32-msvc2015::NLOPT_LIBS = $${NLOPT_LIBDIR}/nlopt.lib
win32-g++-*::NLOPT_LIBS    = -lnlopt -lm
win64-g++-*::NLOPT_LIBS    = -lnlopt -lm
unix::NLOPT_LIBS           = -lnlopt -lm


######################################################################################
#                                   SuperLU
######################################################################################
SUPERLU_PATH    = ../superlu/build
SUPERLU_LIBPATH = $${SUPERLU_PATH}/lib
SUPERLU_INCLUDE = $${SUPERLU_PATH}/include

# win32 uses internal cblas lib
win32-msvc2015::SUPERLU_LIBS = -L$${SUPERLU_LIBPATH} superlu.lib \
                               -L$${SUPERLU_PATH}\CBLAS blas.lib
win32-g++-*::SUPERLU_LIBS    = -L$${SUPERLU_LIBPATH} -lsuperlu $${RT} $${PTHREADS_LIB}
win64-g++-*::SUPERLU_LIBS    = -L$${SUPERLU_LIBPATH} -lsuperlu $${RT} $${PTHREADS_LIB}
unix::SUPERLU_LIBS           = -L$${SUPERLU_LIBPATH} -lsuperlu $${RT} $${PTHREADS_LIB}


######################################################################################
#                                SuperLU_MT
######################################################################################
SUPERLU_MT_PATH    = ../superlu_mt
SUPERLU_MT_LIBPATH = $${SUPERLU_MT_PATH}/lib
SUPERLU_MT_INCLUDE = $${SUPERLU_MT_PATH}/SRC

win32-msvc2015::SUPERLU_MT_LIBS =
win32-g++-*::SUPERLU_MT_LIBS    = -L$${SUPERLU_MT_LIBPATH} -lsuperlu_mt -lgomp $${RT}
win64-g++-*::SUPERLU_MT_LIBS    = -L$${SUPERLU_MT_LIBPATH} -lsuperlu_mt -lgomp $${RT}
unix::SUPERLU_MT_LIBS           = -L$${SUPERLU_MT_LIBPATH} -lsuperlu_mt -lgomp $${RT}

######################################################################################
#                           Umfpack + AMD + CHOLMOD
######################################################################################
UMFPACK_LIBPATH = ../umfpack/build/lib

win32-msvc2015::UMFPACK_LIBS     =
linux-g++::UMFPACK_LIBS = $${UMFPACK_LIBPATH}/libumfpack.a \
                          $${UMFPACK_LIBPATH}/libcholmod.a \
                          $${UMFPACK_LIBPATH}/libamd.a \
                          $${UMFPACK_LIBPATH}/libcamd.a \
                          $${UMFPACK_LIBPATH}/libcolamd.a \
                          $${UMFPACK_LIBPATH}/libccolamd.a \
                          $${UMFPACK_LIBPATH}/libsuitesparseconfig.a
win32-g++-*::UMFPACK_LIBS = $${UMFPACK_LIBPATH}/libumfpack.a \
                            $${UMFPACK_LIBPATH}/libcholmod.a \
                            $${UMFPACK_LIBPATH}/libamd.a \
                            $${UMFPACK_LIBPATH}/libcamd.a \
                            $${UMFPACK_LIBPATH}/libcolamd.a \
                            $${UMFPACK_LIBPATH}/libccolamd.a \
                            $${UMFPACK_LIBPATH}/libsuitesparseconfig.a
win64-g++-*::UMFPACK_LIBS = $${UMFPACK_LIBPATH}/libumfpack.a \
                            $${UMFPACK_LIBPATH}/libcholmod.a \
                            $${UMFPACK_LIBPATH}/libamd.a \
                            $${UMFPACK_LIBPATH}/libcamd.a \
                            $${UMFPACK_LIBPATH}/libcolamd.a \
                            $${UMFPACK_LIBPATH}/libccolamd.a \
                            $${UMFPACK_LIBPATH}/libsuitesparseconfig.a
macx-g++::UMFPACK_LIBS = $${UMFPACK_LIBPATH}/libumfpack.a \
                         $${UMFPACK_LIBPATH}/libcholmod.a \
                         $${UMFPACK_LIBPATH}/libamd.a \
                         $${UMFPACK_LIBPATH}/libcamd.a \
                         $${UMFPACK_LIBPATH}/libcolamd.a \
                         $${UMFPACK_LIBPATH}/libccolamd.a \
                         $${UMFPACK_LIBPATH}/libsuitesparseconfig.a

#####################################################################################
#                                  TRILINOS
#####################################################################################
TRILINOS_DIR  = ../trilinos/build

win32-msvc2015::TRILINOS_INCLUDE = $${TRILINOS_DIR}/include \
                                   $${TRILINOS_DIR}/../commonTools/WinInterface/include
win32-g++-*::TRILINOS_INCLUDE    = $${TRILINOS_DIR}/include
win64-g++-*::TRILINOS_INCLUDE    = $${TRILINOS_DIR}/include
unix::TRILINOS_INCLUDE           = $${TRILINOS_DIR}/include

win32-msvc2015::TRILINOS_LIBS = -L$${TRILINOS_DIR}/lib \
                                aztecoo.lib ml.lib ifpack.lib \
                                amesos.lib epetraext.lib triutils.lib epetra.lib \
                                teuchosremainder.lib teuchosnumerics.lib teuchoscomm.lib \
                                teuchosparameterlist.lib teuchoscore.lib \
                                $${BLAS_LAPACK_LIBS}

win32-g++-*::TRILINOS_LIBS  = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                              -laztecoo -lml -lifpack \
                              -lamesos -lepetraext -ltriutils -lepetra \
                              -lteuchosremainder -lteuchosnumerics -lteuchoscomm \
                              -lteuchosparameterlist -lteuchoscore \
                               $${UMFPACK_LIBS} \
                               $${SUPERLU_LIBS}

win64-g++-*::TRILINOS_LIBS  = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                              -laztecoo -lml -lifpack \
                              -lamesos -lepetraext -ltriutils -lepetra \
                              -lteuchosremainder -lteuchosnumerics -lteuchoscomm \
                              -lteuchosparameterlist -lteuchoscore \
                               $${UMFPACK_LIBS} \
                               $${SUPERLU_LIBS}

linux-g++::TRILINOS_LIBS = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                           -laztecoo -lml -lifpack \
                           -lamesos -lepetraext -ltriutils -lepetra \
                           -lteuchosremainder -lteuchosnumerics -lteuchoscomm \
                           -lteuchosparameterlist -lteuchoscore \
                            $${UMFPACK_LIBS} \
                            $${SUPERLU_LIBS}

macx-g++::TRILINOS_LIBS  = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                           -laztecoo -lml -lifpack \
                           -lamesos -lepetraext -ltriutils -lepetra \
                           -lteuchosremainder -lteuchosnumerics -lteuchoscomm \
                           -lteuchosparameterlist -lteuchoscore \
                            $${UMFPACK_LIBS} \
                            $${SUPERLU_LIBS}

######################################################################################
#                                INTEL Pardiso
######################################################################################
# Version: 11.1
# LD_LIBRARY_PATH should be set
# MKL_NUM_THREADS=Ncpu should be set
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
#####################################################################################
win32-msvc2015::MKLPATH = C:/intel/compilers_and_libraries_2017.4.210/windows
linux-g++::MKLPATH      = /opt/intel
macx-g++::MKLPATH       = /opt/intel

INTEL_MKL_INCLUDE = $${MKLPATH}/mkl/include

win32-g++-*::MKL_LIBS       =
win32-g++-*::INTEL_MKL_LIBS =

win64-g++-*::MKL_LIBS       =
win64-g++-*::INTEL_MKL_LIBS =

win32-msvc2015::ARCH = $$QMAKE_TARGET.arch
unix::ARCH           = $$QMAKE_HOST.arch
#message(Building IntelPardiso on $${ARCH} architecture)

contains(ARCH, x86) {
    win32-msvc2015::MKL_LIBS       = -L$${MKLPATH}/mkl/lib/ia32 -L$${MKLPATH}/compiler/lib/ia32
    win32-msvc2015::INTEL_MKL_LIBS = $${MKL_LIBS} mkl_intel_c_dll.lib mkl_intel_thread_dll.lib mkl_core_dll.lib libiomp5md.lib

    linux-g++::INTEL_MKL_LIBS = -L$${MKLPATH}/lib/ia32 -L$${MKLPATH}/mkl/lib/ia32 \
                                -lmkl_rt \
                                -ldl -lpthread -lm
    linux-g++::QMAKE_LFLAGS   += -m32
    linux-g++::QMAKE_CXXFLAGS += -m32

    macx-g++::INTEL_MKL_LIBS = -L$${MKLPATH}/lib/ia32 -L$${MKLPATH}/mkl/lib/ia32 \
                               -lmkl_rt \
                               -ldl -lpthread -lm
    macx-g++::QMAKE_LFLAGS   += -m32
    macx-g++::QMAKE_CXXFLAGS += -m32
}

contains(ARCH, x86_64) {
    win32-msvc2015::MKL_LIBS       = -L$${MKLPATH}/mkl/lib/intel64 -L$${MKLPATH}/compiler/lib/intel64
    win32-msvc2015::INTEL_MKL_LIBS = $${MKL_LIBS} mkl_intel_lp64_dll.lib mkl_intel_thread_dll.lib mkl_core_dll.lib libiomp5md.lib

    linux-g++::INTEL_MKL_LIBS = -L$${MKLPATH}/mkl/lib/intel64 \
                                -lmkl_rt \
                                -ldl -lpthread -lm
    linux-g++::QMAKE_LFLAGS   += -m64
    linux-g++::QMAKE_CXXFLAGS += -m64

    macx-g++::INTEL_MKL_LIBS = -L$${MKLPATH}/lib/intel64 -L$${MKLPATH}/mkl/lib/intel64 \
                               -lmkl_rt \
                               -ldl -lpthread -lm
    macx-g++::QMAKE_LFLAGS   += -m64
    macx-g++::QMAKE_CXXFLAGS += -m64
}


######################################################################################
#                                Pardiso
######################################################################################
PARDISO_DIR = ../pardiso

win32-msvc2015::PARDISO_LIBS = -L$${PARDISO_DIR}/lib libpardiso500-WIN-X86-64.lib
linux-g++::PARDISO_LIBS      = -L$${PARDISO_DIR}/lib \
                               -lpardiso500-GNU472-X86-64 \
                               $${BLAS_LAPACK_LIBS} $${GFORTRAN} \
                               -lgomp
macx-g++::PARDISO_LIBS       =


#####################################################################################
#                                 DEAL.II
#####################################################################################
DEALII_DIR               = ../deal.II/build

DEALII_INCLUDE           = $${DEALII_DIR}/include
DEALII_LIB_DIR           = $${DEALII_DIR}/lib

win32-msvc2015::DEALII_LIBS = $${DEALII_LIB_DIR}/deal_II-daetools.lib
unix::DEALII_LIBS           = -ldeal_II-daetools -lz
win32-g++-*::DEALII_LIBS    = -ldeal_II-daetools
win64-g++-*::DEALII_LIBS    = -ldeal_II-daetools

#####################################################################################
#                        CoolProp thermo package
#####################################################################################
COOLPROP_DIR      = ../coolprop

COOLPROP_INCLUDE = $${COOLPROP_DIR}/include \
                   $${COOLPROP_DIR}/externals/cppformat
COOLPROP_LIB_DIR = $${COOLPROP_DIR}/build/static_library

win32-msvc2015::COOLPROP_LIBS = $${COOLPROP_LIB_DIR}/Windows/CoolProp.lib
unix::COOLPROP_LIBS           = -L$${COOLPROP_LIB_DIR}/Linux   -lCoolProp
win32-g++-*::COOLPROP_LIBS    = -L$${COOLPROP_LIB_DIR}/Windows -lCoolProp
win64-g++-*::COOLPROP_LIBS    = -L$${COOLPROP_LIB_DIR}/Windows -lCoolProp
macx-g++::COOLPROP_LIBS       = -L$${COOLPROP_LIB_DIR}/Darwin  -lCoolProp


#####################################################################################
#                                 OpenMP
#####################################################################################
# OpenMP libraries are typically linked automatically
unix::OPENMP_LIB           = -lgomp
win32-msvc2015::OPENMP_LIB =


#####################################################################################
#                              NVidia OpenCL
#####################################################################################
unix::NVIDIA_OPENCL_DIR               = /usr/local/cuda
win32-msvc2015::NVIDIA_OPENCL_DIR     = "$$(CUDA_PATH)"

unix::NVIDIA_OPENCL_INCLUDE           = $${NVIDIA_OPENCL_DIR}/include
win32-msvc2015::NVIDIA_OPENCL_INCLUDE =

contains(ARCH, x86) {
win32-msvc2015::NVIDIA_OPENCL_LIBDIR    = $${NVIDIA_OPENCL_DIR}/lib/Win32
}
contains(ARCH, x86_64) {
win32-msvc2015::NVIDIA_OPENCL_LIBDIR    = $${NVIDIA_OPENCL_DIR}/lib/x64
}
#message($${ARCH} architecture: $${NVIDIA_OPENCL_LIBDIR})

unix::NVIDIA_OPENCL_LIBS              = -L$${NVIDIA_OPENCL_DIR}/lib64 -lOpenCL
win32-msvc2015::NVIDIA_OPENCL_LIBS    = "$${NVIDIA_OPENCL_LIBDIR}\OpenCL.lib"

#####################################################################################
#                              Intel OpenCL
#####################################################################################
unix::INTEL_OPENCL_DIR               = /opt/intel/opencl
win32-msvc2015::INTEL_OPENCL_DIR     =

unix::INTEL_OPENCL_INCLUDE           = $${INTEL_OPENCL_DIR}/include
win32-msvc2015::INTEL_OPENCL_INCLUDE =

unix::INTEL_OPENCL_LIBS              = -L$${INTEL_OPENCL_DIR} -lOpenCL
macx-g++::INTEL_OPENCL_LIBS          = -framework OpenCL
win32-msvc2015::INTEL_OPENCL_LIBS    =

#####################################################################################
#                                   OpenCS
#####################################################################################
OPEN_CS_DIR = ../OpenCS

OPEN_CS_INCLUDE = $${OPEN_CS_DIR}/build/include
OPEN_CS_LIB_DIR = $${OPEN_CS_DIR}/build/lib

unix::OPEN_CS_LIBS           = -L$${OPEN_CS_LIB_DIR} -lOpenCS_Evaluators
win32-msvc2015::OPEN_CS_LIBS = -L$${OPEN_CS_LIB_DIR} OpenCS_Evaluators.lib

#####################################################################################
#                                  DAE Tools
#####################################################################################
win32-msvc2015::DAE_CONFIG_LIB                  = cdaeConfig-py$${PYTHON_MAJOR}$${PYTHON_MINOR}$${SHARED_LIB_POSTFIX}.lib
win32-msvc2015::DAE_CORE_LIB                    = cdaeCore.lib $${OPENMP_LIB}
win32-msvc2015::DAE_DATAREPORTING_LIB           = cdaeDataReporting.lib
win32-msvc2015::DAE_ACTIVITY_LIB                = cdaeActivity.lib
win32-msvc2015::DAE_IDAS_SOLVER_LIB             = cdaeIDAS_DAESolver.lib
win32-msvc2015::DAE_UNITS_LIB                   = cdaeUnits.lib
win32-msvc2015::DAE_SUPERLU_SOLVER_LIB          = cdaeSuperLU_LASolver.lib
win32-msvc2015::DAE_SUPERLU_MT_SOLVER_LIB       = cdaeSuperLU_MT_LASolver.lib
win32-msvc2015::DAE_SUPERLU_CUDA_SOLVER_LIB     = cdaeSuperLU_CUDA_LASolver.lib
win32-msvc2015::DAE_BONMIN_SOLVER_LIB           = cdaeBONMIN_MINLPSolver.lib
win32-msvc2015::DAE_IPOPT_SOLVER_LIB            = cdaeIPOPT_NLPSolver.lib
win32-msvc2015::DAE_NLOPT_SOLVER_LIB            = cdaeNLOPT_NLPSolver.lib
win32-msvc2015::DAE_TRILINOS_SOLVER_LIB         = cdaeTrilinos_LASolver.lib
win32-msvc2015::DAE_INTEL_PARDISO_SOLVER_LIB    = cdaeIntelPardiso_LASolver.lib
win32-msvc2015::DAE_PARDISO_SOLVER_LIB          = cdaePardiso_LASolver.lib
win32-msvc2015::DAE_DEALII_SOLVER_LIB           = cdaeDealII_FESolver.lib
win32-msvc2015::DAE_SIMULATION_LOADER_LIB       = cdaeSimulationLoader-py$${PYTHON_MAJOR}$${PYTHON_MINOR}$${SHARED_LIB_POSTFIX}.lib
win32-msvc2015::DAE_DAETOOLS_FMI_CS_LIB         = cdaeFMU_CS-py$${PYTHON_MAJOR}$${PYTHON_MINOR}$${SHARED_LIB_POSTFIX}.lib
win32-msvc2015::DAE_DAETOOLS_FMI_CS_WS_LIB      = cdaeFMU_CS_WS$${SHARED_LIB_POSTFIX}.lib
win32-msvc2015::DAE_CAPE_THERMO_PACKAGE_LIB     = cdaeCapeOpenThermoPackage.lib
win32-msvc2015::DAE_COOLPROP_THERMO_PACKAGE_LIB = cdaeCoolPropThermoPackage.lib
win32-msvc2015::DAE_EVALUATOR_OPENCL_LIB        = cdaeEvaluator_OpenCL.lib

win32-g++-*::DAE_CONFIG_LIB                  = -lcdaeConfig-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-g++-*::DAE_CORE_LIB                    = -lcdaeCore $${OPENMP_LIB}
win32-g++-*::DAE_DATAREPORTING_LIB           = -lcdaeDataReporting
win32-g++-*::DAE_ACTIVITY_LIB                = -lcdaeActivity
win32-g++-*::DAE_IDAS_SOLVER_LIB             = -lcdaeIDAS_DAESolver
win32-g++-*::DAE_UNITS_LIB                   = -lcdaeUnits
win32-g++-*::DAE_SUPERLU_SOLVER_LIB          = -lcdaeSuperLU_LASolver
win32-g++-*::DAE_SUPERLU_MT_SOLVER_LIB       = -lcdaeSuperLU_MT_LASolver
win32-g++-*::DAE_SUPERLU_CUDA_SOLVER_LIB     = -lcdaeSuperLU_CUDA_LASolver
win32-g++-*::DAE_BONMIN_SOLVER_LIB           = -lcdaeBONMIN_MINLPSolver
win32-g++-*::DAE_IPOPT_SOLVER_LIB            = -lcdaeIPOPT_NLPSolver
win32-g++-*::DAE_NLOPT_SOLVER_LIB            = -lcdaeNLOPT_NLPSolver
win32-g++-*::DAE_TRILINOS_SOLVER_LIB         = -lcdaeTrilinos_LASolver
win32-g++-*::DAE_INTEL_PARDISO_SOLVER_LIB    = -lcdaeIntelPardiso_LASolver
win32-g++-*::DAE_PARDISO_SOLVER_LIB          = -lcdaePardiso_LASolver
win32-g++-*::DAE_DEALII_SOLVER_LIB           = -lcdaeDealII_FESolver
win32-g++-*::DAE_SIMULATION_LOADER_LIB       = -lcdaeSimulationLoader-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-g++-*::DAE_DAETOOLS_FMI_CS_LIB         = -lcdaeFMU_CS-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-g++-*::DAE_DAETOOLS_FMI_CS_WS_LIB      = -lcdaeFMU_CS_WS
win32-g++-*::DAE_CAPE_THERMO_PACKAGE_LIB     =
win32-g++-*::DAE_COOLPROP_THERMO_PACKAGE_LIB = -lcdaeCoolPropThermoPackage
win32-g++-*::DAE_EVALUATOR_OPENCL_LIB        = -lcdaeEvaluator_OpenCL

win64-g++-*::DAE_CONFIG_LIB                  = -lcdaeConfig-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win64-g++-*::DAE_CORE_LIB                    = -lcdaeCore $${OPENMP_LIB}
win64-g++-*::DAE_DATAREPORTING_LIB           = -lcdaeDataReporting
win64-g++-*::DAE_ACTIVITY_LIB                = -lcdaeActivity
win64-g++-*::DAE_IDAS_SOLVER_LIB             = -lcdaeIDAS_DAESolver
win64-g++-*::DAE_UNITS_LIB                   = -lcdaeUnits
win64-g++-*::DAE_SUPERLU_SOLVER_LIB          = -lcdaeSuperLU_LASolver
win64-g++-*::DAE_SUPERLU_MT_SOLVER_LIB       = -lcdaeSuperLU_MT_LASolver
win64-g++-*::DAE_SUPERLU_CUDA_SOLVER_LIB     = -lcdaeSuperLU_CUDA_LASolver
win64-g++-*::DAE_BONMIN_SOLVER_LIB           = -lcdaeBONMIN_MINLPSolver
win64-g++-*::DAE_IPOPT_SOLVER_LIB            = -lcdaeIPOPT_NLPSolver
win64-g++-*::DAE_NLOPT_SOLVER_LIB            = -lcdaeNLOPT_NLPSolver
win64-g++-*::DAE_TRILINOS_SOLVER_LIB         = -lcdaeTrilinos_LASolver
win64-g++-*::DAE_INTEL_PARDISO_SOLVER_LIB    = -lcdaeIntelPardiso_LASolver
win64-g++-*::DAE_PARDISO_SOLVER_LIB          = -lcdaePardiso_LASolver
win64-g++-*::DAE_DEALII_SOLVER_LIB           = -lcdaeDealII_FESolver
win64-g++-*::DAE_SIMULATION_LOADER_LIB       = -lcdaeSimulationLoader-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win64-g++-*::DAE_DAETOOLS_FMI_CS_LIB         = -lcdaeFMU_CS-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
win64-g++-*::DAE_DAETOOLS_FMI_CS_WS_LIB      = -lcdaeFMU_CS_WS
win64-g++-*::DAE_CAPE_THERMO_PACKAGE_LIB     =
win64-g++-*::DAE_COOLPROP_THERMO_PACKAGE_LIB = -lcdaeCoolPropThermoPackage
win64-g++-*::DAE_EVALUATOR_OPENCL_LIB        = -lcdaeEvaluator_OpenCL

unix::DAE_CONFIG_LIB                    = -lcdaeConfig-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::DAE_CORE_LIB                      = -lcdaeCore $${OPENMP_LIB}
unix::DAE_DATAREPORTING_LIB             = -lcdaeDataReporting
unix::DAE_ACTIVITY_LIB                  = -lcdaeActivity
unix::DAE_IDAS_SOLVER_LIB               = -lcdaeIDAS_DAESolver
unix::DAE_UNITS_LIB                     = -lcdaeUnits
unix::DAE_SUPERLU_SOLVER_LIB            = -lcdaeSuperLU_LASolver
unix::DAE_SUPERLU_MT_SOLVER_LIB         = -lcdaeSuperLU_MT_LASolver
unix::DAE_SUPERLU_CUDA_SOLVER_LIB       = -lcdaeSuperLU_CUDA_LASolver
unix::DAE_BONMIN_SOLVER_LIB             = -lcdaeBONMIN_MINLPSolver
unix::DAE_IPOPT_SOLVER_LIB              = -lcdaeIPOPT_NLPSolver
unix::DAE_NLOPT_SOLVER_LIB              = -lcdaeNLOPT_NLPSolver
unix::DAE_TRILINOS_SOLVER_LIB           = -lcdaeTrilinos_LASolver
unix::DAE_INTEL_PARDISO_SOLVER_LIB      = -lcdaeIntelPardiso_LASolver
unix::DAE_PARDISO_SOLVER_LIB            = -lcdaePardiso_LASolver
unix::DAE_DEALII_SOLVER_LIB             = -lcdaeDealII_FESolver
unix::DAE_SIMULATION_LOADER_LIB         = -lcdaeSimulationLoader-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::DAE_DAETOOLS_FMI_CS_LIB           = -lcdaeFMU_CS-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
unix::DAE_DAETOOLS_FMI_CS_WS_LIB        = -lcdaeFMU_CS_WS
unix::DAE_CAPE_THERMO_PACKAGE_LIB       =
unix::DAE_COOLPROP_THERMO_PACKAGE_LIB   = -lcdaeCoolPropThermoPackage
unix::DAE_EVALUATOR_OPENCL_LIB          = -lcdaeEvaluator_OpenCL

QMAKE_LIBDIR += $${DAE_DEST_DIR} $${BOOSTLIBPATH}

#######################################################
#            Settings for installing files
#######################################################
DAE_INSTALL_DIR             = ../daetools-dev/$${DAE_SYSTEM}_$${DAE_MACHINE}
DAE_INSTALL_HEADERS_DIR     = $${DAE_INSTALL_DIR}/include
DAE_INSTALL_LIBS_DIR        = $${DAE_INSTALL_DIR}/lib
DAE_INSTALL_PY_MODULES_DIR  = $${DAE_INSTALL_DIR}/py$${PYTHON_MAJOR}$${PYTHON_MINOR}

#SOLIBS_DIR   = ../daetools-package/daetools/solibs/$${DAE_SYSTEM}_$${DAE_MACHINE}
#SOLVERS_DIR  = ../daetools-package/daetools/solvers/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR}
#PYDAE_DIR    = ../daetools-package/daetools/pyDAE/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR}
#FMI_DIR      = ../daetools-package/daetools/solibs/$${DAE_SYSTEM}_$${DAE_MACHINE}
SOLIBS_DIR   = ../daetools-package/daetools/$${DAE_SYSTEM}_$${DAE_MACHINE}/lib
SOLVERS_DIR  = ../daetools-package/daetools/$${DAE_SYSTEM}_$${DAE_MACHINE}/py$${PYTHON_MAJOR}$${PYTHON_MINOR}
PYDAE_DIR    = ../daetools-package/daetools/$${DAE_SYSTEM}_$${DAE_MACHINE}/py$${PYTHON_MAJOR}$${PYTHON_MINOR}
FMI_DIR      = ../daetools-package/daetools/$${DAE_SYSTEM}_$${DAE_MACHINE}/lib
message(SOLIBS_DIR: $${SOLIBS_DIR})

# win32-msvc2015::DUMMY = $$system(mkdir daetools-package\daetools\solvers\\$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# win32-msvc2015::DUMMY = $$system(mkdir daetools-package\daetools\pyDAE\\$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# win32-msvc2015::DUMMY = $$system(mkdir daetools-package\daetools\code_generators\fmi\\$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# 
# win32-g++-*::DUMMY = $$system(mkdir -p daetools-package/daetools/solvers/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# win32-g++-*::DUMMY = $$system(mkdir -p daetools-package/daetools/pyDAE/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# win32-g++-*::DUMMY = $$system(mkdir -p daetools-package/daetools/code_generators/fmi/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# 
# win64-g++-*::DUMMY = $$system(mkdir -p daetools-package/daetools/solvers/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# win64-g++-*::DUMMY = $$system(mkdir -p daetools-package/daetools/pyDAE/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# win64-g++-*::DUMMY = $$system(mkdir -p daetools-package/daetools/code_generators/fmi/$${DAE_SYSTEM}_$${DAE_MACHINE}_py$${PYTHON_MAJOR}$${PYTHON_MINOR})
# 
# unix::DUMMY = $$system(mkdir -p $${SOLIBS_DIR})
# unix::DUMMY = $$system(mkdir -p $${SOLVERS_DIR})
# unix::DUMMY = $$system(mkdir -p $${PYDAE_DIR})
# unix::DUMMY = $$system(mkdir -p $${FMI_DIR})
# 
dae_create_dirs1.commands = $(MKDIR) $${SOLIBS_DIR}
dae_create_dirs2.commands = $(MKDIR) $${SOLVERS_DIR}
dae_create_dirs3.commands = $(MKDIR) $${PYDAE_DIR}
dae_create_dirs4.commands = $(MKDIR) $${FMI_DIR}

QMAKE_EXTRA_TARGETS += dae_create_dirs1 dae_create_dirs2 dae_create_dirs3 dae_create_dirs4
PRE_TARGETDEPS      += dae_create_dirs1 dae_create_dirs2 dae_create_dirs3 dae_create_dirs4

#STATIC_LIBS_DIR = ../daetools-package/daetools/usr/local/lib
#HEADERS_DIR     = ../daetools-package/daetools/usr/local/include

#####################################################################################
#         Write compiler settings (needed to build installations packages)
#####################################################################################
# Python settings
#COMPILER_SETTINGS_FOLDER = .compiler_settings
#win32-msvc2015::COMPILER = $$system(mkdir $${COMPILER_SETTINGS_FOLDER})
#unix::COMPILER  = $$system(mkdir -p $${COMPILER_SETTINGS_FOLDER})

#COMPILER = $$system(echo $${DAE_TOOLS_MAJOR} > $${COMPILER_SETTINGS_FOLDER}/dae_major)
#COMPILER = $$system(echo $${DAE_TOOLS_MINOR} > $${COMPILER_SETTINGS_FOLDER}/dae_minor)
#COMPILER = $$system(echo $${DAE_TOOLS_BUILD} > $${COMPILER_SETTINGS_FOLDER}/dae_build)
