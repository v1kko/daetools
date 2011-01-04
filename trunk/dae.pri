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

VERSION = 1.1.0

QMAKE_CXXFLAGS += -DDAE_MAJOR=1
QMAKE_CXXFLAGS += -DDAE_MINOR=1
QMAKE_CXXFLAGS += -DDAE_BUILD=0

CONFIG(debug, debug|release){
	DAE_DEST_DIR = ../debug
}

CONFIG(release, debug|release){
	DAE_DEST_DIR = ../release
}

DESTDIR = $${DAE_DEST_DIR}

CONFIG(debug, debug|release){
    OBJECTS_DIR = debug
}

CONFIG(release, debug|release){
    OBJECTS_DIR = release
}

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
# Add /usr/include/python2.6/numpy to PYTHON_INCLUDE_DIR
#####################################################################################
win32-msvc2008::PYTHONDIR                = C:\Python26
win32-msvc2008::PYTHON_INCLUDE_DIR       = $${PYTHONDIR}\include
win32-msvc2008::PYTHON_SITE_PACKAGES_DIR = $${PYTHONDIR}\Lib\site-packages
win32-msvc2008::PYTHON_LIB_DIR           = $${PYTHONDIR}\libs

linux-g++::PYTHONDIR           = /usr/lib/python2.6
linux-g++-64::PYTHONDIR        = /usr/lib64/python2.6

unix::PYTHON_INCLUDE_DIR       = /usr/include/python2.6 \
							   # /usr/include/python2.6/numpy \
                                 /usr/share/pyshared
unix::PYTHON_SITE_PACKAGES_DIR = $${PYTHONDIR}/dist-packages
unix::PYTHON_LIB_DIR           =


#####################################################################################
#                                    BOOST
#####################################################################################
# Boost version installed must be 1.35+ (asio, system, python, thread, regex)
#####################################################################################
win32-msvc2008::BOOSTDIR         = ../boost
win32-msvc2008::BOOSTLIBPATH     = ../boost/stage/lib
win32-msvc2008::BOOST_PYTHON_LIB =
win32-msvc2008::BOOST_LIBS       =

unix::BOOSTDIR         = /usr/include/boost
unix::BOOSTLIBPATH     = 
unix::BOOST_PYTHON_LIB = -lboost_python
unix::BOOST_LIBS       = -lboost_system \
                         -lboost_thread


#####################################################################################
#                                 SUNDIALS IDAS
#####################################################################################
# ./configure --prefix=/home/ciroki/Data/daetools/trunk/idas-1.0.0/build --disable-mpi
#             --enable-examples --enable-static=yes --enable-shared=no --with-pic
#
#####################################################################################
SUNDIALS = ../idas/build
SUNDIALS_INCLUDE = $${SUNDIALS}/include
SUNDIALS_LIBDIR = $${SUNDIALS}/lib

win32-msvc2008::SUNDIALS_LIBS = sundials_idas.lib \
                                sundials_nvecserial.lib
unix::SUNDIALS_LIBS = -lsundials_idas \
                      -lsundials_nvecserial


#####################################################################################
#                                  BONMIN / IPOPT
#####################################################################################
#    Compiling BONMIN on GNU/Linux
# 0) Unpack BONMIN to daetools/trunk/bonmin
# 1) cd bonmin/ThirdParty/Mumps
#    sh get.Mumps
# 2) cd bonmin/ThirdParty/Metis
#    sh get.Metis
# 3) mkdir build
# 4) cd build
# 5) ../configure
# 6) make -jN
# 7) make test
# 8) make install
#####################################################################################
BONMIN_DIR = ../bonmin/build
MUMPS_DIR  = ../mumps

BONMIN_INCLUDE = $${BONMIN_DIR}/include/coin
BONMIN_LIBDIR  = $${BONMIN_DIR}/lib
MUMPS_LIBDIR   = $${MUMPS_DIR}/lib \
                 $${MUMPS_DIR}/libseq

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

win32-msvc2008::BONMIN_LIBS = libCoinBlas.lib libCoinLapack.lib libf2c.lib \
                              libCoinHSL.lib \
                              libBonmin.lib \
                              libIpopt.lib \
                              libCbc.lib \
                              libCgl.lib \
                              libClp.lib \
                              libCoinUtils.lib \
                              libOsiCbc.lib \
                              libOsiClp.lib \
                              libOsi.lib

unix::MUMPS_LIBS =
win32-msvc2008::MUMPS_LIBS = libmpiseq.lib \
                             libdmumps.lib \
                             libmumps_common.lib \
                             libpord.lib


#####################################################################################
#                                  DAE-TOOLS
#####################################################################################
win32-msvc2008::DAE_CORE_LIB          = cdaeCore.lib
win32-msvc2008::DAE_DATAREPORTERS_LIB = cdaeDataReporting.lib
win32-msvc2008::DAE_SIMULATION_LIB    = cdaeActivity.lib
win32-msvc2008::DAE_SOLVER_LIB        = cdaeIDAS_DAESolver.lib
win32-msvc2008::DAE_NLPSOLVER_LIB     = cdaeBONMIN_MINLPSolver.lib

unix::DAE_CORE_LIB          = -lcdaeCore
unix::DAE_DATAREPORTERS_LIB = -lcdaeDataReporting
unix::DAE_SIMULATION_LIB    = -lcdaeActivity
unix::DAE_SOLVER_LIB        = -lcdaeIDAS_DAESolver
unix::DAE_NLPSOLVER_LIB     = -lcdaeBONMIN_MINLPSolver

QMAKE_LIBDIR += $${DAE_DEST_DIR} $${BOOSTLIBPATH}

HEADERS += \
    ../config.h \
    ../dae_develop.h \
    ../dae.h
