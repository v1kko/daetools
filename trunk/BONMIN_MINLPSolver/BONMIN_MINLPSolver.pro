include(../dae.pri)

QT -= core gui
TEMPLATE = lib
CONFIG += staticlib

###############################
# Could be: BONMIN, IPOPT
###############################
CONFIG += BONMIN

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


#win32{
#QMAKE_POST_LINK = copy /y  $${TARGET}.lib $${STATIC_LIBS_DIR}
#}

#unix{
#QMAKE_POST_LINK = cp -f  lib$${TARGET}.a $${STATIC_LIBS_DIR}
#}

#INSTALL_HEADERS = $$system($${COPY_FILES} base_solvers.h $${HEADERS_DIR}/BONMIN_MINLPSolver)


#######################################################
#                Install files
#######################################################
bonmin_headers.path  = $${HEADERS_DIR}/BONMIN_MINLPSolver
bonmin_headers.files = base_solvers.h

bonmin_libs.path         = $${STATIC_LIBS_DIR}
win32::bonmin_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::bonmin_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += bonmin_headers bonmin_libs
