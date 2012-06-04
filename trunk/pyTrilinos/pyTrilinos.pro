include(../dae.pri)
QT -= core gui
TARGET  = pyTrilinos
TEMPLATE = lib

INCLUDEPATH +=  $${BOOSTDIR} \
				$${PYTHON_INCLUDE_DIR} \
				$${PYTHON_SITE_PACKAGES_DIR} \
				$${SUNDIALS_INCLUDE} \
				$${TRILINOS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${DAE_TRILINOS_SOLVER_LIB} \
        $${TRILINOS_LIBS} \
        $${BOOST_PYTHON_LIB} \
		$${BLAS_LIBS} \
		$${LAPACK_LIBS}		

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp

HEADERS += stdafx.h

#######################################################
#                Install files
#######################################################
win32{
QMAKE_POST_LINK = move /y \
    $${DAE_DEST_DIR}/pyTrilinos1.dll \
    $${SOLVERS_DIR}/pyTrilinos.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
    $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
    $${SOLVERS_DIR}/$${TARGET}.$${SHARED_LIB_EXT}
}
