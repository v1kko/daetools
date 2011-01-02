include(../dae.pri)

QT -= core gui
TARGET = cdaeBONMIN_MINLPSolver
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
               $${BONMIN_INCLUDE}

LIBS += $${DAE_CORE_LIB} \
        $${BONMIN_LIBS} \
        $${MUMPS_LIBS}

HEADERS += stdafx.h \ 
    nlpsolver_class_factory.h \
    nlpsolver.h
SOURCES += stdafx.cpp \
    dllmain.cpp \
    nlpsolver.cpp
