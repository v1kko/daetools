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

include(../dae.pri)
QT -= core \
    gui
TARGET = IntelPardiso
TEMPLATE = lib

######################################################################################
# INTEL MKL solvers
######################################################################################
# LD_LIBRARY_PATH should be set
# MKL_NUM_THREADS=Ncpu should be set
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
#####################################################################################
win32-msvc2008::MKLPATH = c:\Intel\MKL\10.2.5.035
linux-g++::MKLPATH = /opt/intel/mkl/10.2.5.035
linux-g++-64::MKLPATH = /opt/intel/mkl/10.2.5.035

win32-msvc2008::MKL_LIBS = $${MKLPATH}\ia32\lib
linux-g++::MKL_LIBS = $${MKLPATH}/lib/32
linux-g++-64::MKL_LIBS = $${MKLPATH}/lib/em64t

INTEL_MKL_INCLUDE = $${MKLPATH}/include

# Sequential
#win32-msvc2008::INTEL_MKL_LIBS =  -L$${MKL_LIBS} mkl_solver_sequential.lib mkl_intel_c.lib mkl_sequential.lib mkl_core.lib
# OpenMP
win32-msvc2008::INTEL_MKL_LIBS =  -L$${MKL_LIBS} mkl_solver.lib mkl_intel_c.lib mkl_intel_thread.lib mkl_core.lib libiomp5mt.lib -Qopenmp

linux-g++::INTEL_MKL_LIBS = -L$${MKL_LIBS} \
                          $${MKL_LIBS}/libmkl_solver.a \
                          -Wl,--start-group \
                              $${MKL_LIBS}/libmkl_intel.a \
                              $${MKL_LIBS}/libmkl_intel_thread.a \
                              $${MKL_LIBS}/libmkl_core.a \
                          -Wl,--end-group \
                          -liomp5 -lpthread

linux-g++-64::INTEL_MKL_LIBS = -L$${MKL_LIBS} \
								$${MKL_LIBS}/libmkl_solver_lp64.a \
								-Wl,--start-group \
								    $${MKL_LIBS}/libmkl_intel_lp64.a \
								    $${MKL_LIBS}/libmkl_intel_thread.a \
								    $${MKL_LIBS}/libmkl_core.a \
								-Wl,--end-group \
								-liomp5 -lpthread

####################################################################################
#                       Suppress some warnings
####################################################################################
#unix::QMAKE_CXXFLAGS += -ansi -pedantic
unix::QMAKE_CXXFLAGS += -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable
unix::QMAKE_CFLAGS   += -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable

INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR} \
    $${SUNDIALS_INCLUDE} \
    $${INTEL_MKL_INCLUDE}
QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS += $${BOOST_PYTHON_LIB} \
    $${INTEL_MKL_LIBS}
SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    mkl_pardiso_sparse_la_solver.cpp
HEADERS += stdafx.h \
    mkl_pardiso_sparse_la_solver.h

win32-msvc2008::QMAKE_POST_LINK = move /y \
    $${DAE_DEST_DIR}/IntelPardiso1.dll \
    $${DAE_DEST_DIR}/pyIntelPardiso.pyd
unix::QMAKE_POST_LINK = cp \
    -f \
    $${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
    $${DAE_DEST_DIR}/pyIntelPardiso.so
