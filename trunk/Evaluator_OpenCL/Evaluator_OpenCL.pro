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

