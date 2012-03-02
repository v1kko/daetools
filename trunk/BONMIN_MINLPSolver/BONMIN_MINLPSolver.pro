include(../dae.pri)

QT -= core gui
TEMPLATE = lib
CONFIG += staticlib

###############################
# Could be: BONMIN, IPOPT
###############################
CONFIG += IPOPT

#####################################################################
# Small hack used when compiling from compile_linux.sh shell script
#####################################################################
shellCompile:message(shellCompile) {
shellIPOPT {
  CONFIG += IPOPT
}
shellBONMIN {
  CONFIG += BONMIN
}
}

######################################################################################
#                                BONMIN
######################################################################################
CONFIG(BONMIN, BONMIN|IPOPT):message(BONMIN) { 

QMAKE_CXXFLAGS += -DdaeBONMIN
TARGET = cdaeBONMIN_MINLPSolver

INCLUDEPATH += $${BOOSTDIR} \
               $${BONMIN_INCLUDE}

QMAKE_LIBDIR += $${BONMIN_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${BOOST_LIBS} \
        $${BONMIN_LIBS} \
        $${MUMPS_LIBS}
}

######################################################################################
#                                IPOPT
######################################################################################
CONFIG(IPOPT, BONMIN|IPOPT):message(IPOPT) { 

QMAKE_CXXFLAGS += -DdaeIPOPT
TARGET = cdaeIPOPT_NLPSolver

INCLUDEPATH += $${BOOSTDIR} \
               $${IPOPT_INCLUDE}

QMAKE_LIBDIR += $${IPOPT_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${BOOST_LIBS} \
        $${IPOPT_LIBS} \
        $${MUMPS_LIBS}
}


HEADERS += stdafx.h \ 
    nlpsolver_class_factory.h \
    nlpsolver.h \ 
    base_solvers.h \
	../nlp_common.h 
SOURCES += stdafx.cpp \
    dllmain.cpp \
    nlpsolver.cpp
