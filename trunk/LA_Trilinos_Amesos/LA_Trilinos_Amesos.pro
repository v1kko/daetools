include(../dae.pri)
QT -= core \
      gui
TARGET  = Trilinos
TEMPLATE = lib

INCLUDEPATH +=  $${BOOSTDIR} \
				$${PYTHON_INCLUDE_DIR} \
				$${PYTHON_SITE_PACKAGES_DIR} \
				$${SUNDIALS_INCLUDE} \
				$${TRILINOS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${BOOST_PYTHON_LIB} \
		$${BLAS_LIBS} \
		$${LAPACK_LIBS} \
		$${TRILINOS_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    trilinos_amesos_la_solver.cpp

HEADERS += stdafx.h \
    trilinos_amesos_la_solver.h

win32{
QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/Trilinos1.dll \
    $${DAE_DEST_DIR}/pyTrilinos.pyd
}

unix{
QMAKE_POST_LINK = cp \
    -f \
    $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
    $${DAE_DEST_DIR}/$${TARGET}.$${SHARED_LIB_EXT}
}
