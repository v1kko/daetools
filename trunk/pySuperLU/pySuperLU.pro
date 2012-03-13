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
LIBS += $${DAE_SUPERLU_SOLVER_LIB} \
        $${SUPERLU_LIBS}
INCLUDEPATH += $${SUPERLU_INCLUDE}
pyObject = pySuperLU
message(SUPERLU_LIBS: $${SUPERLU_LIBS})
}

######################################################################################
#                                SuperLU_MT
######################################################################################
CONFIG(SuperLU_MT, SuperLU|SuperLU_MT|SuperLU_CUDA):message(SuperLU_MT) { 

QMAKE_CXXFLAGS += -DdaeSuperLU_MT
LIBS += $${DAE_SUPERLU_MT_SOLVER_LIB} \
        $${SUPERLU_MT_LIBS}
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

win32{
SuperLU {
QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/pySuperLU1.dll \
    $${DAE_DEST_DIR}/pySuperLU.pyd

}

SuperLU_MT {
QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/pySuperLU1.dll \
    $${DAE_DEST_DIR}/pySuperLU_MT.pyd
}

SuperLU_CUDA {
QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/pySuperLU1.dll \
    $${DAE_DEST_DIR}/pySuperLU_CUDA.pyd
}
}

unix{
QMAKE_POST_LINK = cp \
    -f \
    $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
    $${DAE_DEST_DIR}/$${pyObject}.$${SHARED_LIB_EXT}
}
