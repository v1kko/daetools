include(../dae.pri)

QT -= core gui
TARGET = cdaeNLOPT_NLPSolver
TEMPLATE = lib
CONFIG += shared plugin

INCLUDEPATH += $${BOOSTDIR} \
               $${NLOPT_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${NLOPT_LIBDIR}

LIBS += $${SOLIBS_RPATH_SL}
LIBS += $${DAE_CONFIG_LIB} \
        $${BOOST_LIBS} \
        $${NLOPT_LIBS} \
        $${BLAS_LAPACK_LIBS}

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
