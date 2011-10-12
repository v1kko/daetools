include(../dae.pri)

QT -= core gui
TARGET = pyUnits
TEMPLATE = lib

INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS +=	$${DAE_UNITS_LIB} $${BOOST_PYTHON_LIB} $${BOOST_LIBS}

SOURCES += dllmain.cpp \
	stdafx.cpp \
    dae_python.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32{
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyUnits.dll \
	$${DAE_DEST_DIR}/pyUnits.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/pyUnits.so
}
