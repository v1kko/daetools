include(../dae.pri)

QT -= core gui
TARGET = cdaeNLOPT_NLPSolver
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
               $${NLOPT_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${NLOPT_LIBDIR}

LIBS += $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${BOOST_LIBS} \
        $${NLOPT_LIBS}

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

#INSTALL_HEADERS = $$system($${COPY_FILES} base_solvers.h $${HEADERS_DIR}/NLOPT_NLPSolver)

#######################################################
#                Install files
#######################################################
nlopt_headers.path  = $${HEADERS_DIR}/NLOPT_NLPSolver
nlopt_headers.files = base_solvers.h

nlopt_libs.path         = $${STATIC_LIBS_DIR}
win32::nlopt_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::nlopt_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += nlopt_headers nlopt_libs
