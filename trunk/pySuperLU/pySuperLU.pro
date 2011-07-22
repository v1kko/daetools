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
QT -= core gui
TARGET = pySuperLU
TEMPLATE = lib

#################################################
# Could be: SuperLU, SuperLU_MT, SuperLU_CUDA
#################################################
CONFIG += SuperLU

#####################################################################
# Small hack used when compiling from compile_linux.sh shell script
#####################################################################
shellCompile:message(shellCompile) {
shellSuperLU {
  CONFIG += SuperLU
}
shellSuperLU_MT {
  CONFIG += SuperLU_MT
}
shellSuperLU_CUDA {
  CONFIG += SuperLU_CUDA
}
}

INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR} \
    $${SUNDIALS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${BOOST_PYTHON_LIB}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp

HEADERS += stdafx.h

######################################################################################
#                                   SuperLU
######################################################################################
CONFIG(SuperLU, SuperLU|SuperLU_MT|SuperLU_CUDA):message(SuperLU) { 

QMAKE_CXXFLAGS += -DdaeSuperLU
LIBS += $${SUPERLU_LIBS} \
        $${DAE_SUPERLU_SOLVER_LIB}
INCLUDEPATH += $${SUPERLU_INCLUDE}
pyObject = pySuperLU
}

######################################################################################
#                                SuperLU_MT
######################################################################################
CONFIG(SuperLU_MT, SuperLU|SuperLU_MT|SuperLU_CUDA):message(SuperLU_MT) { 

QMAKE_CXXFLAGS += -DdaeSuperLU_MT
LIBS += $${SUPERLU_MT_LIBS} \
        $${DAE_SUPERLU_MT_SOLVER_LIB}
INCLUDEPATH += $${SUPERLU_MT_INCLUDE}
pyObject = pySuperLU_MT
}

######################################################################################
#                                SuperLU_CUDA
# compile it with: make --file=gpuMakefile
######################################################################################
CONFIG(SuperLU_CUDA, SuperLU|SuperLU_MT|SuperLU_CUDA):message(SuperLU_CUDA) { 

QMAKE_CXXFLAGS += -DdaeSuperLU_CUDA
LIBS += $${SUPERLU_CUDA_LIBS} \
        $${CUDA_LIBS} \
        $${DAE_SUPERLU_CUDA_SOLVER_LIB}
INCLUDEPATH += $${SUPERLU_CUDA_INCLUDE}
pyObject = pySuperLU_CUDA
}

OTHER_FILES += superlu_mt_gpu.cu gpuMakefile

SuperLU {
win32-msvc2008::QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/pySuperLU1.dll \
    $${DAE_DEST_DIR}/pySuperLU.pyd

}

SuperLU_MT {
win32-msvc2008::QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/pySuperLU1.dll \
    $${DAE_DEST_DIR}/pySuperLU_MT.pyd
}

SuperLU_CUDA {
win32-msvc2008::QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/pySuperLU1.dll \
    $${DAE_DEST_DIR}/pySuperLU_CUDA.pyd
}

unix::QMAKE_POST_LINK = cp \
    -f \
    $${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
    $${DAE_DEST_DIR}/$${pyObject}.so
