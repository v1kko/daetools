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
DAE_TOOLS_MINOR = 1
DAE_TOOLS_BUILD = 1

# 1. On GNU/LINUX:
#    Set PYTHON_MAJOR=2 and PYTHON_MINOR=7 to use Python located in /usr/lib/python2.7
# 2. On Windows:set PYTHON_MAJOR=2 and PYTHON_MINOR=7 to use Python in c:\Python27
PYTHON_MAJOR = 2
PYTHON_MINOR = 6

# 1. On GNU/LINUX:
#    a) Set CONFIG += use_custom_boost and set BOOST_MAJOR, BOOST_MINOR and BOOST_BUILD
#       Boost must be located in ../boost_1_42_0 (for instance)
#    b) Set CONFIG += use_system_boost to use the system's default version
# 2. On Windows:set BOOST_MAJOR, BOOST_MINOR and BOOST_BUILD
#    Boost must be located in ../boost_1_42_0 (for instance)
CONFIG += use_system_boost
BOOST_MAJOR = 1
BOOST_MINOR = 42
BOOST_BUILD = 0

VERSION = $${DAE_TOOLS_MAJOR}.$${DAE_TOOLS_MINOR}.$${DAE_TOOLS_BUILD}

QMAKE_CXXFLAGS += -DDAE_MAJOR=$${DAE_TOOLS_MAJOR}
QMAKE_CXXFLAGS += -DDAE_MINOR=$${DAE_TOOLS_MINOR}
QMAKE_CXXFLAGS += -DDAE_BUILD=$${DAE_TOOLS_BUILD}

CONFIG(debug, debug|release){
	DAE_DEST_DIR = ../debug
    OBJECTS_DIR = debug
}

CONFIG(release, debug|release){
	DAE_DEST_DIR = ../release
    OBJECTS_DIR = release
}

DESTDIR = $${DAE_DEST_DIR}


# Cross compiling:
# configure --prefix=/usr/i586-mingw32msvc --host=i586-mingw32msvc --build=x86_64-linux

####################################################################################
# Remove all symbol table and relocation information from the executable.
# Necessary to pass lintian test in debian  
####################################################################################
#CONFIG(release, debug|release){
#    unix:QMAKE_LFLAGS += -s
#}

####################################################################################
#                       Suppress some warnings
####################################################################################
#unix::QMAKE_CXXFLAGS += -ansi -pedantic
unix::QMAKE_CXXFLAGS += -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable
unix::QMAKE_CFLAGS   += -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable

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
# Numpy must be installed
# OS-specific stuff (python version and a location of site specific packages):
#     Debian:  Lenny          Squeeze
#              2.5            2.6
#              site-packages  dist-packages
#     Ubuntu:  10.4           10.10
#              2.6            2.6
#              site-packages  dist-packages
#     Fedora:  13             14
#              2.6            2.7
#              site-packages  site-packages
#     Windows: WinXP
#              2.6            
#              site-packages  
#
# Debian Squeeze: sometimes there are problems with _numpyconfig.h
# Add: /usr/include/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}/numpy 
# to:  PYTHON_INCLUDE_DIR 
#####################################################################################
win32-msvc2008::PYTHONDIR                = C:\Python$${PYTHON_MAJOR}$${PYTHON_MINOR}
win32-msvc2008::PYTHON_INCLUDE_DIR       = $${PYTHONDIR}\include
win32-msvc2008::PYTHON_SITE_PACKAGES_DIR = $${PYTHONDIR}\Lib\site-packages
win32-msvc2008::PYTHON_LIB_DIR           = $${PYTHONDIR}\libs

linux-g++::PYTHONDIR           = /usr/lib/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}
linux-g++-64::PYTHONDIR        = /usr/lib64/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}

unix::PYTHON_INCLUDE_DIR       = /usr/include/python$${PYTHON_MAJOR}.$${PYTHON_MINOR} \
							     /usr/include/python$${PYTHON_MAJOR}.$${PYTHON_MINOR}/numpy \
                                 /usr/share/pyshared
unix::PYTHON_SITE_PACKAGES_DIR = $${PYTHONDIR}/dist-packages
unix::PYTHON_LIB_DIR           =


#####################################################################################
#                                    BOOST
#####################################################################################
# Boost version installed must be 1.35+ (asio, system, python, thread, regex)
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
win32-msvc2008::BOOSTDIR         = ../boost_$${BOOST_MAJOR}_$${BOOST_MINOR}_$${BOOST_BUILD}
win32-msvc2008::BOOSTLIBPATH     = $${BOOSTDIR}/stage/lib
win32-msvc2008::BOOST_PYTHON_LIB =
win32-msvc2008::BOOST_LIBS       =

use_system_boost {
unix::BOOSTDIR         = /usr/include/boost
unix::BOOSTLIBPATH     = 
unix::BOOST_PYTHON_LIB = -lboost_python
unix::BOOST_LIBS       = -lboost_system \
                         -lboost_thread
}

use_custom_boost { 
unix::BOOSTDIR         = ../boost_$${BOOST_MAJOR}_$${BOOST_MINOR}_$${BOOST_BUILD}
unix::BOOSTLIBPATH     = $${BOOSTDIR}/stage/lib
unix::BOOST_PYTHON_LIB = -lboost_python
unix::BOOST_LIBS       = -lboost_system \
                         -lboost_thread
}

#####################################################################################
#                                 SUNDIALS IDAS
#####################################################################################
# ./configure --prefix=${HOME}/Data/daetools/trunk/idas/build --disable-mpi
#             --enable-examples --enable-static=yes --enable-shared=no --with-pic
#             CFLAGS=-O3
#####################################################################################
SUNDIALS = ../idas/build
SUNDIALS_INCLUDE = $${SUNDIALS}/include
SUNDIALS_LIBDIR = $${SUNDIALS}/lib

win32-msvc2008::SUNDIALS_LIBS = sundials_idas.lib \
                                sundials_nvecserial.lib
unix::SUNDIALS_LIBS = -lsundials_idas \
                      -lsundials_nvecserial


#####################################################################################
#                                  MUMPS
#####################################################################################
MUMPS_DIR  = ../mumps
win32-msvc2008::G95_LIBDIR = c:\MinGW\lib\gcc-lib\i686-pc-mingw32\4.1.2

MUMPS_LIBDIR   = $${MUMPS_DIR}/lib \
                 $${MUMPS_DIR}/libseq \
                 $${MUMPS_DIR}/blas \
                 $${G95_LIBDIR}

win32-msvc2008::MUMPS_LIBS = blas.lib \
                             libmpiseq.lib \
                             libdmumps.lib \
                             libmumps_common.lib \
                             libpord.lib \
                             libf95.a \
                             libgcc.a
unix::MUMPS_LIBS =


#####################################################################################
#                                  IPOPT
#####################################################################################
IPOPT_DIR = ../bonmin/build

IPOPT_INCLUDE = $${IPOPT_DIR}/include/coin
IPOPT_LIBDIR  = $${IPOPT_DIR}/lib

win32-msvc2008::IPOPT_LIBS = libCoinBlas.lib libCoinLapack.lib libf2c.lib libIpopt.lib 
unix::IPOPT_LIBS           = -ldl -lblas -llapack -lipopt


#####################################################################################
#                                  BONMIN
#####################################################################################
BONMIN_DIR = ../bonmin/build

BONMIN_INCLUDE = $${BONMIN_DIR}/include/coin
BONMIN_LIBDIR  = $${BONMIN_DIR}/lib

win32-msvc2008::BONMIN_LIBS = libCoinBlas.lib libCoinLapack.lib libf2c.lib \
                              libBonmin.lib \
                              libIpopt.lib \
                              libCbc.lib \
                              libCgl.lib \
                              libClp.lib \
                              libCoinUtils.lib \
                              libOsiCbc.lib \
                              libOsiClp.lib \
                              libOsi.lib
unix::BONMIN_LIBS  =    -ldl -lblas -llapack \
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
NLOPT_DIR = ../nlopt/build
NLOPT_INCLUDE = $${NLOPT_DIR}/include
NLOPT_LIBDIR  = $${NLOPT_DIR}/lib

win32-msvc2008::NLOPT_LIBS = nlopt.lib
unix::NLOPT_LIBS           = -lnlopt -lm


#####################################################################################
#                                  DAE Tools
#####################################################################################
win32-msvc2008::DAE_CORE_LIB          = cdaeCore.lib
win32-msvc2008::DAE_DATAREPORTERS_LIB = cdaeDataReporting.lib
win32-msvc2008::DAE_SIMULATION_LIB    = cdaeActivity.lib
win32-msvc2008::DAE_SOLVER_LIB        = cdaeIDAS_DAESolver.lib
win32-msvc2008::DAE_BONMINSOLVER_LIB  = cdaeBONMIN_MINLPSolver.lib
win32-msvc2008::DAE_IPOPTSOLVER_LIB   = cdaeIPOPT_NLPSolver.lib
win32-msvc2008::DAE_NLOPTSOLVER_LIB   = cdaeNLOPT_NLPSolver.lib

unix::DAE_CORE_LIB          = -lcdaeCore
unix::DAE_DATAREPORTERS_LIB = -lcdaeDataReporting
unix::DAE_SIMULATION_LIB    = -lcdaeActivity
unix::DAE_SOLVER_LIB        = -lcdaeIDAS_DAESolver
unix::DAE_BONMINSOLVER_LIB  = -lcdaeBONMIN_MINLPSolver
unix::DAE_IPOPTSOLVER_LIB   = -lcdaeIPOPT_NLPSolver
unix::DAE_NLOPTSOLVER_LIB   = -lcdaeNLOPT_NLPSolver

QMAKE_LIBDIR += $${DAE_DEST_DIR} $${BOOSTLIBPATH}

HEADERS += \
    ../config.h \
    ../dae_develop.h \
    ../dae.h
