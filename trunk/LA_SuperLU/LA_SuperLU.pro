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
TARGET = SuperLU
TEMPLATE = lib

CONFIG += SuperLU_MT

######################################################################################
#                                   SuperLU
######################################################################################
SuperLU{
QMAKE_CXXFLAGS += -DdaeSuperLU

win32-msvc2008::SUPERLU_PATH = ..\superlu
linux-g++::SUPERLU_PATH      = ../superlu
linux-g++-64::SUPERLU_PATH   = ../superlu

win32-msvc2008::SUPERLU_LIBS = $${SUPERLU_PATH}\lib
linux-g++::SUPERLU_LIBS      = $${SUPERLU_PATH}/lib
linux-g++-64::SUPERLU_LIBS   = $${SUPERLU_PATH}/lib

SUPERLU_INCLUDE = $${SUPERLU_PATH}/SRC

win32-msvc2008::SUPERLU_LIBS = -L$${SUPERLU_LIBS} superlu.lib
linux-g++::SUPERLU_LIBS      = -L$${SUPERLU_LIBS} -lcdaesuperlu
linux-g++-64::SUPERLU_LIBS   = -L$${SUPERLU_LIBS} -lcdaesuperlu

pyObject = pySuperLU
}

######################################################################################
#                                   SuperLU_MT
######################################################################################
SuperLU_MT{
QMAKE_CXXFLAGS += -DdaeSuperLU_MT

win32-msvc2008::SUPERLU_PATH = ..\superlu_mt
linux-g++::SUPERLU_PATH      = ../superlu_mt
linux-g++-64::SUPERLU_PATH   = ../superlu_mt

win32-msvc2008::SUPERLU_LIBS = $${SUPERLU_PATH}\lib
linux-g++::SUPERLU_LIBS      = $${SUPERLU_PATH}/lib
linux-g++-64::SUPERLU_LIBS   = $${SUPERLU_PATH}/lib

SUPERLU_INCLUDE = $${SUPERLU_PATH}/SRC

win32-msvc2008::SUPERLU_LIBS = -L$${SUPERLU_LIBS} superlu_mt.lib
linux-g++::SUPERLU_LIBS      = -L$${SUPERLU_LIBS} -lcdaesuperlu_mt
linux-g++-64::SUPERLU_LIBS   = -L$${SUPERLU_LIBS} -lcdaesuperlu_mt

pyObject = pySuperLU_MT
}


INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR} \
    $${SUNDIALS_INCLUDE} \
    $${SUPERLU_INCLUDE}
QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS += $${BOOST_PYTHON_LIB} \
    $${SUPERLU_LIBS}
SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    superlu_la_solver.cpp
HEADERS += stdafx.h \
    superlu_la_solver.h

SuperLU{
win32-msvc2008::QMAKE_POST_LINK = move /y \
    $${DAE_DEST_DIR}/SuperLU1.dll \
    $${DAE_DEST_DIR}/SuperLU.pyd
}

SuperLU_MT{
win32-msvc2008::QMAKE_POST_LINK = move /y \
    $${DAE_DEST_DIR}/SuperLU_MT1.dll \
    $${DAE_DEST_DIR}/SuperLU_MT.pyd
}

unix::QMAKE_POST_LINK = cp \
            -f \
            $${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
            $${DAE_DEST_DIR}/$${pyObject}.so

