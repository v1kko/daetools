include(../dae.pri)
QT -= core gui
TARGET  = cdaeTrilinos_LASolver
TEMPLATE = lib
CONFIG += staticlib

INCLUDEPATH +=  $${BOOSTDIR} \
				$${PYTHON_INCLUDE_DIR} \
				$${PYTHON_SITE_PACKAGES_DIR} \
				$${SUNDIALS_INCLUDE} \
				$${TRILINOS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${BLAS_LIBS} \
		$${LAPACK_LIBS} \
		$${TRILINOS_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    trilinos_amesos_la_solver.cpp

HEADERS += stdafx.h \
    base_solvers.h \
    trilinos_amesos_la_solver.h

#######################################################
#                Install files
#######################################################
#win32{
#QMAKE_POST_LINK = copy /y  $${TARGET}.lib $${STATIC_LIBS_DIR}
#}

#unix{
#QMAKE_POST_LINK = cp -f lib$${TARGET}.a $${STATIC_LIBS_DIR}
#}

trilinos_headers.path  = $${HEADERS_DIR}/LA_SuperLU
trilinos_headers.files = base_solvers.h

trilinos_libs.path         = $${STATIC_LIBS_DIR}
win32::trilinos_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::trilinos_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += trilinos_headers trilinos_libs
