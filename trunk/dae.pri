VERSION=1.0.5

QMAKE_CXXFLAGS += -DDAE_MAJOR=1
QMAKE_CXXFLAGS += -DDAE_MINOR=0
QMAKE_CXXFLAGS += -DDAE_BUILD=5

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
unix::BOOST_LIBS       = -lboost_system -lboost_thread


#####################################################################################
#                                   SUNDIALS
#####################################################################################
SUNDIALS = ../ida-2.6.0
SUNDIALS_INCLUDE = $${SUNDIALS}/include $${SUNDIALS}/src


QMAKE_LIBDIR += $${DAE_DEST_DIR} $${BOOSTLIBPATH}

HEADERS += \
    ../config.h \
    ../dae_develop.h \
    ../dae.h
