include(../dae.pri)
QT -= core
QT -= gui

TARGET = cdaeEvaluator_OpenCL
TEMPLATE = lib
CONFIG += staticlib

QMAKE_CFLAGS += -g -O0
QMAKE_CXXFLAGS += -g -O0

INCLUDEPATH += $${INTEL_OPENCL_INCLUDE}

LIBS +=	$${INTEL_OPENCL_LIBS}

SOURCES += stdafx.cpp \
           dllmain.cpp \
           compute_stack_opencl.cpp \
           compute_stack_opencl_multi.cpp \
           cs_opencl.cpp

HEADERS += stdafx.h \
           cs_opencl.h \
           compute_stack_opencl.h \
           compute_stack_opencl_multi.h

OTHER_FILES += compute_stack_opencl_kernel_source.cl

