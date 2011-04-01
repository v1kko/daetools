include(../dae.pri)

QT -= core gui
TEMPLATE = lib
CONFIG += staticlib

CONFIG += BONMIN

######################################################################################
#                                BONMIN
######################################################################################
BONMIN { 
QMAKE_CXXFLAGS += -DdaeBONMIN
TARGET = cdaeBONMIN_MINLPSolver

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
}

######################################################################################
#                                IPOPT
######################################################################################
IPOPT { 
QMAKE_CXXFLAGS += -DdaeIPOPT
TARGET = cdaeIPOPT_NLPSolver

INCLUDEPATH += $${BOOSTDIR} \
               $${IPOPT_INCLUDE}

QMAKE_LIBDIR += $${IPOPT_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${DAE_SIMULATION_LIB} \
        $${DAE_DATAREPORTERS_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_SOLVER_LIB} \
        $${BOOST_LIBS} \
        $${IPOPT_LIBS} \
        $${MUMPS_LIBS}
}


HEADERS += stdafx.h \ 
    nlpsolver_class_factory.h \
    nlpsolver.h
SOURCES += stdafx.cpp \
    dllmain.cpp \
    nlpsolver.cpp
