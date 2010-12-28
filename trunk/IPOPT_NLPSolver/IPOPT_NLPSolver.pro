include(../dae.pri)

QT -= core gui
TARGET = cdaeIPOPT_NLPSolver
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
               $${NLPSOLVER_INCLUDE}

LIBS += $${DAE_CORE_LIB} \
        $${NLPSOLVER_LIBS}

HEADERS += stdafx.h \ 
    nlpsolver_class_factory.h \
    nlpsolver.h
SOURCES += stdafx.cpp \
    dllmain.cpp \
    nlpsolver.cpp
