include(../dae.pri)
QT -= core
QT -= gui

TARGET = cdaeEvaluator_OpenCL
TEMPLATE = lib
CONFIG += staticlib

unix::QMAKE_CXXFLAGS  += -std=c++11
unix::QMAKE_LFLAGS    += -std=c++11
win32::QMAKE_CXXFLAGS += /std:c++11
win32::QMAKE_LFLAGS   += /std:c++11

INCLUDEPATH += $${INTEL_OPENCL_INCLUDE} \
               $${OPEN_CS_INCLUDE}

LIBS +=	$${INTEL_OPENCL_LIBS}

SOURCES += stdafx.cpp \
           dllmain.cpp \
           $${OPEN_CS_DIR}/OpenCS/evaluators/cs_evaluator_opencl.cpp \
           $${OPEN_CS_DIR}/OpenCS/evaluators/cs_evaluator_opencl_multidevice.cpp \
           $${OPEN_CS_DIR}/OpenCS/evaluators/cs_evaluator_opencl_factory.cpp

HEADERS += stdafx.h \
           $${OPEN_CS_DIR}/OpenCS/evaluators/cs_evaluator_opencl.h \
           $${OPEN_CS_DIR}/OpenCS/evaluators/cs_evaluator_opencl_multidevice.h \
           $${OPEN_CS_DIR}/OpenCS/evaluators/cs_evaluator_opencl_factory.h

OTHER_FILES += $${OPEN_CS_DIR}/OpenCS/evaluators/cs_machine_kernels.cl
