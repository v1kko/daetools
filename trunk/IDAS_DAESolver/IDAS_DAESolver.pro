include(../dae.pri)
QT -= core \
    gui
TARGET = cdaeIDAS_DAESolver
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
               $${SUNDIALS_INCLUDE} \
               $${MPI_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR}
LIBS +=	$${SUNDIALS_LIBS} \
        $${MPI_LIBS}

SOURCES += stdafx.cpp \
    ida_solver.cpp \
    daesolver.cpp \
    class_factory.cpp \
    base_solvers.cpp
HEADERS += stdafx.h \
    solver_class_factory.h \
    ida_solver.h \
    base_solvers.h \
    dae_array_matrix.h \
    dae_solvers.h \
    ida_la_solver_interface.h


#win32{
#QMAKE_POST_LINK = copy /y  $${TARGET}.lib $${STATIC_LIBS_DIR}
#}

#unix{
#QMAKE_POST_LINK = cp -f  lib$${TARGET}.a $${STATIC_LIBS_DIR}
#}

#INSTALL_HEADERS = $$system($${COPY_FILES} base_solvers.h            $${HEADERS_DIR}/IDAS_DAESolver)
#INSTALL_HEADERS = $$system($${COPY_FILES} solver_class_factory.h    $${HEADERS_DIR}/IDAS_DAESolver)
#INSTALL_HEADERS = $$system($${COPY_FILES} ida_la_solver_interface.h $${HEADERS_DIR}/IDAS_DAESolver)

#######################################################
#                Install files
#######################################################
idas_headers.path   = $${HEADERS_DIR}/IDAS_DAESolver
idas_headers.files  = base_solvers.h \
                      solver_class_factory.h \
                      ida_la_solver_interface.h

idas_libs.path         = $${STATIC_LIBS_DIR}
win32::idas_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::idas_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += idas_headers idas_libs
