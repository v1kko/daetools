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
               $${OPEN_CS_DIR}

LIBS +=	$${INTEL_OPENCL_LIBS}

SOURCES += stdafx.cpp \
           dllmain.cpp \
           ../opencs/opencl/cs_evaluator_opencl.cpp \
           ../opencs/opencl/cs_evaluator_opencl_multidevice.cpp \
           ../opencs/opencl/cs_evaluator_opencl_factory.cpp

HEADERS += stdafx.h \
           ../opencs/opencl/cs_evaluator_opencl.h \
           ../opencs/opencl/cs_evaluator_opencl_multidevice.h \
           ../opencs/opencl/cs_evaluator_opencl_factory.h

OTHER_FILES += ../opencs/opencl/cs_machine_kernels.cl
