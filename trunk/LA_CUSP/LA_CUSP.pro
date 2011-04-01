# ************************************************************************************
# DAE Tools Project: www.daetools.com
# Copyright (C) Dragan Nikolic, 2010
# ************************************************************************************
# DAE Tools is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free Software
# Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with the
# DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
# ************************************************************************************
include(../dae.pri)
QT -= core \
    gui
TARGET = CUSP
TEMPLATE = lib

# #####################################################################################
# CUSP
# #####################################################################################
win32-msvc2008::CUDA_PATH = 
linux-g++::CUDA_PATH = /usr/local/cuda
linux-g++-64::CUDA_PATH = /usr/local/cuda
CUSP_LIBPATH = 
CUSP_INCLUDE = $${CUDA_PATH}/include
win32-msvc2008::CUSP_LIBS = 
linux-g++::CUSP_LIBS = 
linux-g++-64::CUSP_LIBS = -L$${CUDA_PATH}/lib64 \
    -lcuda \
    -lcublas \
    -lcudart
INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR} \
    $${SUNDIALS_INCLUDE} \
    $${CUSP_INCLUDE}
QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS += $${BOOST_PYTHON_LIB} \
    $${CUSP_LIBS}
SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    cusp_la_solver.cpp \
	../IDAS_DAESolver/mmio.c
HEADERS += stdafx.h \
    cusp_la_solver.h \
    cusp_solver.h
win32-msvc2008::QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/CUSP1.dll \
    $${DAE_DEST_DIR}/pyCUSP.pyd
unix::QMAKE_POST_LINK = cp \
    -f \
    $${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
    $${DAE_DEST_DIR}/pyCUSP.so
OTHER_FILES += cusp_solver.cu \
    Makefile \
    cusp_solver.o
