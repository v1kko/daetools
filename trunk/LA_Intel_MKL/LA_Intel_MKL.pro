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
TARGET = pyIntelPardiso
TEMPLATE = lib

######################################################################################
# INTEL MKL solvers
######################################################################################
# Version: 11.1
# LD_LIBRARY_PATH should be set
# MKL_NUM_THREADS=Ncpu should be set
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
#####################################################################################
win32-msvc2008::MKLPATH =
linux-g++::MKLPATH      = /opt/Intel/mkl
macx-g++::MKLPATH       =

INTEL_MKL_INCLUDE = $${MKLPATH}/include
ARCH = $$QMAKE_HOST.arch

win32-msvc2008::INTEL_MKL_LIBS = -L$${MKL_LIBS} mkl_intel_c.lib mkl_core.lib mkl_intel_thread.lib libiomp5md.lib -Qopenmp
win32-msvc2008::MKL_LIBS = $${MKLPATH}\ia32\lib

contains($$ARCH, x86) {
    message(Using 32 bit MKL)
    linux-g++::MKL_LIBS       = $${MKLPATH}/lib/ia32
    linux-g++::INTEL_MKL_LIBS = -L$${MKL_LIBS} \
                                -Wl,--start-group \
                                    $${MKL_LIBS}/libmkl_intel.a \
                                    $${MKL_LIBS}/libmkl_core.a \
                                    $${MKL_LIBS}/libmkl_gnu_thread.a \
                                -Wl,--end-group \
                                -ldl -lpthread -lm
    linux-g++::QMAKE_LFLAGS   += -fopenmp -m32
    linux-g++::QMAKE_CXXFLAGS += -fopenmp -m32

    macx-g++::MKL_LIBS       = $${MKLPATH}/lib/ia32
    macx-g++::INTEL_MKL_LIBS = -L$${MKL_LIBS} \
                               -Wl,--start-group \
                                   $${MKL_LIBS}/libmkl_intel.a \
                                   $${MKL_LIBS}/libmkl_core.a \
                                   $${MKL_LIBS}/libmkl_gnu_thread.a \
                               -Wl,--end-group \
                               -ldl -lpthread -lm
    macx-g++::QMAKE_LFLAGS   += -fopenmp -m32
    macx-g++::QMAKE_CXXFLAGS += -fopenmp -m32
}
contains(ARCH, x86_64) {
    message(Using 64 bit MKL)
    linux-g++::MKL_LIBS   = $${MKLPATH}/lib/intel64
    linux-g++::INTEL_MKL_LIBS = -L$${MKL_LIBS} \
                                -Wl,--start-group \
                                    $${MKL_LIBS}/libmkl_intel_lp64.a \
                                    $${MKL_LIBS}/libmkl_core.a \
                                    $${MKL_LIBS}/libmkl_gnu_thread.a \
                                -Wl,--end-group \
                                -ldl -lpthread -lm
    linux-g++::QMAKE_LFLAGS   += -fopenmp -m64
    linux-g++::QMAKE_CXXFLAGS += -fopenmp -m64

    macx-g++::MKL_LIBS   = $${MKLPATH}/lib/intel64
    macx-g++::INTEL_MKL_LIBS = -L$${MKL_LIBS} \
                               -Wl,--start-group \
                                   $${MKL_LIBS}/libmkl_intel_lp64.a \
                                   $${MKL_LIBS}/libmkl_core.a \
                                   $${MKL_LIBS}/libmkl_gnu_thread.a \
                               -Wl,--end-group \
                               -ldl -lpthread -lm
    macx-g++::QMAKE_LFLAGS   += -fopenmp -m64
    macx-g++::QMAKE_CXXFLAGS += -fopenmp -m64
}

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
    mkl_pardiso_sparse_la_solver.cpp \
    ../mmio.c

HEADERS += stdafx.h \
    mkl_pardiso_sparse_la_solver.h \
    ../mmio.h

#win32-msvc2008::QMAKE_POST_LINK = move /y \
#    $${DAE_DEST_DIR}/IntelPardiso1.dll \
#    $${DAE_DEST_DIR}/pyIntelPardiso.pyd
#unix::QMAKE_POST_LINK = cp \
#    -f \
#    $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
#    $${DAE_DEST_DIR}/py$${TARGET}.$${SHARED_LIB_EXT}

win32{
QMAKE_POST_LINK = move /y \
    $${DAE_DEST_DIR}/pyIntelPardiso11.dll \
    $${SOLVERS_DIR}/pyIntelPardiso.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
    $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
    $${SOLVERS_DIR}/$${TARGET}.$${SHARED_LIB_EXT}
}
