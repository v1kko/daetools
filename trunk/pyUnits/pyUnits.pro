include(../dae.pri)

QT -= core gui
TARGET = pyUnits
TEMPLATE = lib

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NUMPY_INCLUDE_DIR} \
	           $${MPI_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS +=	$${DAE_UNITS_LIB} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}

SOURCES += dllmain.cpp \
	stdafx.cpp \
    dae_python.cpp

HEADERS += stdafx.h \
    docstrings.h \
    python_wraps.h

#######################################################
#                Install files
#######################################################
win32{
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyUnits1.dll \
	$${PYDAE_DIR}/pyUnits.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
        $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
        $${PYDAE_DIR}/$${TARGET}.so
}
