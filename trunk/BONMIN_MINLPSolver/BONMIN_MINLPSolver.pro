include(../dae.pri)

QT -= core gui
TEMPLATE = lib
CONFIG += shared plugin

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
               $${BONMIN_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${BONMIN_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${SOLIBS_RPATH_SL}
LIBS += $${DAE_CONFIG_LIB} \
        $${BOOST_LIBS} \
        $${BONMIN_LIBS} \
        $${MUMPS_LIBS} \
        $${BLAS_LAPACK_LIBS}
}

######################################################################################
#                                IPOPT
######################################################################################
CONFIG(IPOPT, BONMIN|IPOPT):message(IPOPT) {

QMAKE_CXXFLAGS += -DdaeIPOPT
TARGET = cdaeIPOPT_NLPSolver

INCLUDEPATH += $${BOOSTDIR} \
               $${IPOPT_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${IPOPT_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${SOLIBS_RPATH_SL}
LIBS += $${DAE_CONFIG_LIB} \
        $${BOOST_LIBS} \
        $${IPOPT_LIBS} \
        $${MUMPS_LIBS} \
        $${BLAS_LAPACK_LIBS}
}


HEADERS += stdafx.h \
    nlpsolver_class_factory.h \
    nlpsolver.h \
    base_solvers.h \
    ../nlp_common.h
SOURCES += stdafx.cpp \
    dllmain.cpp \
    nlpsolver.cpp

#######################################################
#                Install files
#######################################################
QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

DAE_PROJECT_NAME = $$basename(PWD)

install_headers.path  = $${DAE_INSTALL_HEADERS_DIR}/$${DAE_PROJECT_NAME}
install_headers.files = *.h

install_libs.path  = $${DAE_INSTALL_LIBS_DIR}
install_libs.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

INSTALLS += install_headers install_libs
