include(../dae.pri)
QT -= core gui
TEMPLATE = lib
CONFIG += shared plugin

#################################################
# Could be: SuperLU or SuperLU_MT
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
}

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${SUNDIALS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp

HEADERS += stdafx.h \
    docstrings.h 

######################################################################################
#                                   SuperLU
######################################################################################
CONFIG(SuperLU, SuperLU|SuperLU_MT|SuperLU_CUDA):message(SuperLU) { 

TARGET = pySuperLU
QMAKE_CXXFLAGS += -DdaeSuperLU
LIBS += $${SOLIBS_RPATH}
LIBS += $${DAE_SUPERLU_SOLVER_LIB} \
        $${DAE_CONFIG_LIB} \
        $${SUPERLU_LIBS} \
        $${BLAS_LAPACK_LIBS} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}
INCLUDEPATH += $${SUPERLU_INCLUDE}
message(SUPERLU_LIBS: $${SUPERLU_LIBS})
}

######################################################################################
#                                SuperLU_MT
######################################################################################
CONFIG(SuperLU_MT, SuperLU|SuperLU_MT|SuperLU_CUDA):message(SuperLU_MT) { 

TARGET = pySuperLU_MT
QMAKE_CXXFLAGS += -DdaeSuperLU_MT
LIBS += $${SOLIBS_RPATH}
LIBS += $${DAE_SUPERLU_MT_SOLVER_LIB} \
        $${DAE_CONFIG_LIB} \
        $${SUPERLU_MT_LIBS} \
        $${BLAS_LAPACK_LIBS} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}
INCLUDEPATH += $${SUPERLU_MT_INCLUDE}
}

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${SOLVERS_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

include(../dae_install_py_module.pri)
