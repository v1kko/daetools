include(../dae.pri)

QT -= core gui
TARGET = cdaeNLOPT_NLPSolver
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
               $${NLOPT_INCLUDE}

QMAKE_LIBDIR += $${NLOPT_LIBDIR}

LIBS += $${DAE_SIMULATION_LIB} \
        $${DAE_DATAREPORTERS_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_SOLVER_LIB} \
        $${BOOST_LIBS} \
        $${NLOPT_LIBS}

HEADERS += stdafx.h \ 
    nlpsolver_class_factory.h \
    nlpsolver.h
SOURCES += stdafx.cpp \
    dllmain.cpp \
    nlpsolver.cpp
