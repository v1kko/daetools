include(../dae.pri)

QT -= core gui
TARGET = cdaeBONMIN_MINLPSolver
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
               $${BONMIN_INCLUDE}

QMAKE_LIBDIR += $${BONMIN_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${DAE_SIMULATION_LIB} \
        $${DAE_DATAREPORTERS_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_SOLVER_LIB} \
        $${BOOST_LIBS} \
        $${BONMIN_LIBS} \
        $${MUMPS_LIBS}

HEADERS += stdafx.h \ 
    nlpsolver_class_factory.h \
    nlpsolver.h
SOURCES += stdafx.cpp \
    dllmain.cpp \
    nlpsolver.cpp
