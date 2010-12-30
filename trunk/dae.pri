VERSION = 1.1.1

QMAKE_CXXFLAGS += -DDAE_MAJOR=1
QMAKE_CXXFLAGS += -DDAE_MINOR=1
QMAKE_CXXFLAGS += -DDAE_BUILD=1

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
CONFIG(release, debug|release){
    unix:QMAKE_LFLAGS += -s
}

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
# OS-specific:
#     Debian:  python 2.5-6, site-packages, /usr/lib or /usr/lib64
#     Ubuntu:  python 2.6,   dist-packages, /usr/lib or /usr/lib64
#     Fedora:  python 2.6,   site-packages, /usr/lib or /usr/lib64
#     Windows: python 2.6,   site-packages, C:\PythonXY
#
# Under Debian Squeeze sometimes there are problems with _numpyconfig.h
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
#                                   SUNDIALS
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
#                                   BONMIN / IPOPT
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
unix::NLPSOLVER           = ../bonmin/build
win32-msvc2008::NLPSOLVER = ../ipopt/build

NLPSOLVER_INCLUDE = $${NLPSOLVER}/include/coin
NLPSOLVER_LIBDIR  = $${NLPSOLVER}/lib

unix::NLPSOLVER_LIBS =  -ldl -lblas -llapack \
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

# Currently only IPOPT 3.9.1 compiles on windows 
# Pre-buiilt windows binaries: http://www.coin-or.org/Binaries/Bonmin/Bonmin-1.4.0-win32-msvc9.7z
win32-msvc2008::NLPSOLVER_LIBS = Ipopt.lib
#win32-msvc2008::NLPSOLVER_LIBS = libCoinBlas.lib libCoinLapack.lib libf2c.lib \
#                                 libCoinHSL.lib \
#                                 libipopt.lib

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
