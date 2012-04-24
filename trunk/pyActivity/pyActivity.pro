include(../dae.pri)
QT -= core \
	gui
TARGET = pyActivity
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NUMPY_INCLUDE_DIR} \
	           $${MPI_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${DAE_CORE_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_ACTIVITY_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_NLPSOLVER_LIB} \
        $${DAE_UNITS_LIB} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${MPI_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32{
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyActivity1.dll \
	$${PYDAE_DIR}/pyActivity.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
        $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
        $${PYDAE_DIR}/$${TARGET}.so
}
